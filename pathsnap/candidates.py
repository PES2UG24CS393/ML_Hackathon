"""Rank parsed road-network arcs by local metric GPS-to-segment distance."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import math

from .preprocess import ArcGeometry, Trip, load_trip


@dataclass(frozen=True)
class Candidate:
    trip_id: str
    observation_index: int
    rank: int
    arc_id: int
    distance_m: float
    ambiguous: bool


@dataclass(frozen=True)
class Prediction:
    trip_id: str
    observation_index: int
    arc_id: int | None
    distance_m: float | None
    candidate_count: int
    status: str


def _distance(point: tuple[float, float], segment: ArcGeometry) -> float:
    px, py = point
    ax, ay = segment.start_xy
    bx, by = segment.end_xy
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class _SegmentGrid:
    """Metric bbox grid that expands until every possible closer segment is checked."""
    def __init__(self, segments: tuple[ArcGeometry, ...], cell_size_m: float = 400.0):
        self.cell = cell_size_m
        self.cells: dict[tuple[int, int], list[ArcGeometry]] = defaultdict(list)
        self.long_segments: list[ArcGeometry] = []
        for segment in segments:
            ax, ay = segment.start_xy
            bx, by = segment.end_xy
            x0, y0 = math.floor(min(ax, bx) / self.cell), math.floor(min(ay, by) / self.cell)
            x1, y1 = math.floor(max(ax, bx) / self.cell), math.floor(max(ay, by) / self.cell)
            if (x1 - x0 + 1) * (y1 - y0 + 1) > 10_000:
                self.long_segments.append(segment)
            else:
                for gx in range(x0, x1 + 1):
                    for gy in range(y0, y1 + 1):
                        self.cells[(gx, gy)].append(segment)

    def nearest(self, point: tuple[float, float], top_k: int) -> list[tuple[float, int]]:
        px, py = point
        gx, gy = math.floor(px / self.cell), math.floor(py / self.cell)
        checked = {segment.arc_id for segment in self.long_segments}
        scored = [(_distance(point, segment), segment.arc_id) for segment in self.long_segments]
        max_radius = max(
            max((abs(x - gx) for x, _ in self.cells), default=0),
            max((abs(y - gy) for _, y in self.cells), default=0),
        )
        for radius in range(max_radius + 1):
            for x in range(gx - radius, gx + radius + 1):
                for y in range(gy - radius, gy + radius + 1):
                    if radius and max(abs(x - gx), abs(y - gy)) != radius:
                        continue
                    for segment in self.cells.get((x, y), ()):
                        if segment.arc_id not in checked:
                            checked.add(segment.arc_id)
                            scored.append((_distance(point, segment), segment.arc_id))
            scored.sort(key=lambda value: (value[0], value[1]))
            if len(scored) >= top_k:
                left, right = (gx - radius) * self.cell, (gx + radius + 1) * self.cell
                bottom, upper = (gy - radius) * self.cell, (gy + radius + 1) * self.cell
                outside_lower_bound = min(px - left, right - px, py - bottom, upper - py)
                if scored[top_k - 1][0] <= outside_lower_bound:
                    return scored[:top_k]
        return scored[:top_k]


def rank_candidates(trip: Trip, top_k: int = 5, ambiguity_tolerance_m: float = 1.0) -> tuple[list[Candidate], list[Prediction]]:
    """Return ranked candidate arcs and one Top-1 record per source observation."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if ambiguity_tolerance_m < 0 or not math.isfinite(ambiguity_tolerance_m):
        raise ValueError("ambiguity_tolerance_m must be finite and non-negative")
    for lon, lat in ((o.lon, o.lat) for o in trip.observations):
        if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError("trip contains non-finite or out-of-bounds GPS coordinates")
    if not trip.observations:
        return [], []
    segments = trip.network.segments
    if any(not all(math.isfinite(value) for value in (*s.start_xy, *s.end_xy)) for s in segments):
        raise ValueError("trip contains non-finite projected road geometry")
    grid = _SegmentGrid(segments) if segments else None
    candidates: list[Candidate] = []
    predictions: list[Prediction] = []
    for observation, point in zip(trip.observations, trip.observation_xy):
        ranked = grid.nearest(point, top_k) if grid else []
        ambiguous = len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= ambiguity_tolerance_m
        for rank, (distance, arc_id) in enumerate(ranked, 1):
            candidates.append(Candidate(trip.trip_id, observation.obs_id, rank, arc_id, distance, ambiguous))
        if ranked:
            predictions.append(Prediction(trip.trip_id, observation.obs_id, ranked[0][1], ranked[0][0], len(ranked), "ok"))
        else:
            predictions.append(Prediction(trip.trip_id, observation.obs_id, None, None, 0, "no_candidates"))
    return candidates, predictions


def write_outputs(candidates: list[Candidate], predictions: list[Prediction], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trip_id", "observation_index", "rank", "arc_id", "distance_m", "ambiguous"))
        writer.writerows((c.trip_id, c.observation_index, c.rank, c.arc_id, f"{c.distance_m:.3f}", str(c.ambiguous).lower()) for c in candidates)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trip_id", "observation_index", "arc_id", "distance_m", "candidate_count", "status"))
        writer.writerows((p.trip_id, p.observation_index, "" if p.arc_id is None else p.arc_id,
                          "" if p.distance_m is None else f"{p.distance_m:.3f}", p.candidate_count, p.status) for p in predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip", help="trip directory, e.g. 'Student folder/train/00000005'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("reports/baseline"))
    args = parser.parse_args()
    trip = load_trip(args.trip)
    candidates, predictions = rank_candidates(trip, args.top_k)
    write_outputs(candidates, predictions, args.out)
    matched = sum(p.arc_id is not None for p in predictions)
    print(f"trip={trip.trip_id} observations={len(trip.observations)} arcs={len(trip.network.arcs)} "
          f"candidates={len(candidates)} predictions={matched}/{len(predictions)} output={args.out}")


if __name__ == "__main__":
    main()
