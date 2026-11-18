"""Rank road arcs by local metric point-to-segment distance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import math
from collections import defaultdict

from .preprocess import Arc, Observation, Trip, load_trip, project_local


@dataclass(frozen=True)
class Candidate:
    trip_id: str
    observation_id: int
    rank: int
    arc_id: int
    distance_m: float
    ambiguous: bool


@dataclass(frozen=True)
class Prediction:
    trip_id: str
    observation_id: int
    arc_id: int | None
    distance_m: float | None
    candidate_count: int
    status: str


def _point_segment_distance_m(point: Observation, arc: Arc, reference_lat: float) -> float:
    px, py = project_local(point, reference_lat)
    ax, ay = project_local(arc.start, reference_lat)
    bx, by = project_local(arc.end, reference_lat)
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class _ArcGrid:
    """Metric bounding-box index; cell queries expand until top-k is exact."""
    def __init__(self, arcs: tuple[Arc, ...], reference_lat: float, cell_size_m: float = 400.0):
        self.cell = cell_size_m
        self.reference_lat = reference_lat
        self.cells: dict[tuple[int, int], list[Arc]] = defaultdict(list)
        self.bounds: dict[int, tuple[float, float, float, float]] = {}
        self.long_arcs: list[Arc] = []
        for arc in arcs:
            ax, ay = project_local(arc.start, reference_lat)
            bx, by = project_local(arc.end, reference_lat)
            bounds = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
            self.bounds[arc.arc_id] = bounds
            x0, y0 = math.floor(bounds[0] / self.cell), math.floor(bounds[1] / self.cell)
            x1, y1 = math.floor(bounds[2] / self.cell), math.floor(bounds[3] / self.cell)
            # Normally arcs span one or two cells; protect against malformed,
            # unusually long geometries creating a massive bbox expansion.
            if (x1 - x0 + 1) * (y1 - y0 + 1) > 10_000:
                self.long_arcs.append(arc)
            else:
                for gx in range(x0, x1 + 1):
                    for gy in range(y0, y1 + 1):
                        self.cells[(gx, gy)].append(arc)
        self.arc_by_id = {arc.arc_id: arc for arc in arcs}

    def nearest(self, point: Observation, top_k: int) -> list[tuple[float, int]]:
        px, py = project_local(point, self.reference_lat)
        gx, gy = math.floor(px / self.cell), math.floor(py / self.cell)
        checked: set[int] = set()
        scored: list[tuple[float, int]] = [
            (_point_segment_distance_m(point, arc, self.reference_lat), arc.arc_id)
            for arc in self.long_arcs
        ]
        checked.update(arc.arc_id for arc in self.long_arcs)
        max_radius = max((abs(x - gx) for x, _ in self.cells), default=0)
        max_radius = max(max_radius, max((abs(y - gy) for _, y in self.cells), default=0))
        for radius in range(max_radius + 1):
            for x in range(gx - radius, gx + radius + 1):
                for y in range(gy - radius, gy + radius + 1):
                    if radius and max(abs(x - gx), abs(y - gy)) != radius:
                        continue
                    for arc in self.cells.get((x, y), ()):
                        if arc.arc_id not in checked:
                            checked.add(arc.arc_id)
                            scored.append((_point_segment_distance_m(point, arc, self.reference_lat), arc.arc_id))
            scored.sort(key=lambda item: (item[0], item[1]))
            if len(scored) >= top_k:
                left, right = (gx - radius) * self.cell, (gx + radius + 1) * self.cell
                bottom, top = (gy - radius) * self.cell, (gy + radius + 1) * self.cell
                outside_lower_bound = min(px - left, right - px, py - bottom, top - py)
                if scored[top_k - 1][0] <= outside_lower_bound:
                    return scored[:top_k]
        return scored[:top_k]


def rank_candidates(trip: Trip, top_k: int = 5, ambiguity_tolerance_m: float = 1.0) -> tuple[list[Candidate], list[Prediction]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if ambiguity_tolerance_m < 0 or not math.isfinite(ambiguity_tolerance_m):
        raise ValueError("ambiguity_tolerance_m must be finite and non-negative")
    coordinates = [(p.lon, p.lat) for p in trip.observations]
    coordinates.extend((p.lon, p.lat) for p in trip.nodes)
    coordinates.extend((p.lon, p.lat) for arc in trip.arcs for p in (arc.start, arc.end))
    if any(not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90
           for lon, lat in coordinates):
        raise ValueError("trip contains non-finite or out-of-bounds longitude/latitude")
    if not trip.observations:
        return [], []
    # One local tangent plane for the trip's network reduces distortion and
    # keeps point-to-segment distances in metres (the supplied region is small).
    ref_lat = sum(node.lat for node in trip.nodes) / len(trip.nodes) if trip.nodes else sum(o.lat for o in trip.observations) / len(trip.observations)
    grid = _ArcGrid(trip.arcs, ref_lat) if trip.arcs else None
    all_candidates: list[Candidate] = []
    predictions: list[Prediction] = []
    for observation in trip.observations:
        ranked = grid.nearest(observation, top_k) if grid else []
        ambiguous = len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= ambiguity_tolerance_m
        for rank, (distance, arc_id) in enumerate(ranked, 1):
            all_candidates.append(Candidate(trip.trip_id, observation.observation_id, rank, arc_id, distance, ambiguous))
        if ranked:
            predictions.append(Prediction(trip.trip_id, observation.observation_id, ranked[0][1], ranked[0][0], len(ranked), "ok"))
        else:
            predictions.append(Prediction(trip.trip_id, observation.observation_id, None, None, 0, "no_candidates"))
    return all_candidates, predictions


def write_outputs(candidates: list[Candidate], predictions: list[Prediction], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trip_id", "observation_id", "rank", "arc_id", "distance_m", "ambiguous"))
        writer.writerows((c.trip_id, c.observation_id, c.rank, c.arc_id, f"{c.distance_m:.3f}", str(c.ambiguous).lower()) for c in candidates)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trip_id", "observation_id", "arc_id", "distance_m", "candidate_count", "status"))
        writer.writerows((p.trip_id, p.observation_id, "" if p.arc_id is None else p.arc_id,
                          "" if p.distance_m is None else f"{p.distance_m:.3f}", p.candidate_count, p.status) for p in predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip", help="trip directory, e.g. 'Student folder/train/00000001'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("reports/baseline"))
    args = parser.parse_args()
    trip = load_trip(args.trip)
    candidates, predictions = rank_candidates(trip, args.top_k)
    write_outputs(candidates, predictions, args.out)
    matched = sum(p.arc_id is not None for p in predictions)
    print(f"trip={trip.trip_id} observations={len(trip.observations)} arcs={len(trip.arcs)} "
          f"candidates={len(candidates)} predictions={matched}/{len(predictions)} output={args.out}")


if __name__ == "__main__":
    main()
