"""Inference-safe candidate features and a conservative route-label alignment proposal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, degrees, hypot, isfinite

from .preprocess import Trip


@dataclass(frozen=True)
class CandidateFeature:
    trip_id: str
    obs_id: int
    arc_id: int
    distance_m: float
    heading_diff_deg: float | None
    speed_mps: float | None
    time_gap_s: float | None
    predecessor_connected: bool | None


@dataclass(frozen=True)
class RouteLabel:
    trip_id: str
    obs_id: int
    route_step_id: int | None
    reference_arc_id: int | None
    aligned_distance_m: float | None
    status: str


def feature_rows(trip: Trip, candidate_radius_m: float = 100.0,
                 max_candidates: int = 8, stationary_threshold_m: float = 1.0
                 ) -> list[CandidateFeature]:
    """Build candidate rows from GPS/network only; ``trip.route`` is never read.

    A uniform grid indexes road-segment bounding boxes expanded by the candidate
    radius. Rows are ordered by observation then distance and arc ID for stability.
    """
    if not isfinite(candidate_radius_m) or candidate_radius_m <= 0:
        raise ValueError("candidate_radius_m must be a finite positive number")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    segments = trip.network.segments
    segment_by_id = {segment.arc_id: segment for segment in segments}
    arc_by_id = {arc.arc_id: arc for arc in trip.network.arcs}
    cell = candidate_radius_m
    grid: dict[tuple[int, int], list[int]] = {}
    for index, segment in enumerate(segments):
        a, b = segment.start_xy, segment.end_xy
        x0, x1 = int((min(a[0], b[0]) - cell) // cell), int((max(a[0], b[0]) + cell) // cell)
        y0, y1 = int((min(a[1], b[1]) - cell) // cell), int((max(a[1], b[1]) + cell) // cell)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 400:
            # These very long segments are rare; index them in every query via a sentinel.
            grid.setdefault((-10**12, -10**12), []).append(index)
            continue
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                grid.setdefault((gx, gy), []).append(index)

    # Adjacent observation motion features; undefined on the first row or invalid dt.
    motion: dict[int, tuple[float | None, float | None, float | None]] = {}
    for i, (observation, point) in enumerate(zip(trip.observations, trip.observation_xy)):
        if i == 0:
            motion[observation.obs_id] = (None, None, None)
            continue
        previous = trip.observations[i - 1]
        previous_xy = trip.observation_xy[i - 1]
        dt = observation.timestamp_s - previous.timestamp_s
        if dt <= 0 or not isfinite(dt):
            motion[observation.obs_id] = (None, None, None)
            continue
        dx, dy = point[0] - previous_xy[0], point[1] - previous_xy[1]
        displacement = hypot(dx, dy)
        speed = 0.0 if displacement < stationary_threshold_m else displacement / dt
        heading = None if displacement < stationary_threshold_m else degrees(atan2(dx, dy)) % 360
        motion[observation.obs_id] = (dt, speed, heading)

    per_observation: list[list[tuple[int, float]]] = []
    for point in trip.observation_xy:
        key = (int(point[0] // cell), int(point[1] // cell))
        indices = set(grid.get(key, ()))
        indices.update(grid.get((-10**12, -10**12), ()))
        nearby = []
        for index in indices:
            segment = segments[index]
            distance = _point_segment_distance(point, segment.start_xy, segment.end_xy)
            if distance <= candidate_radius_m:
                nearby.append((segment.arc_id, distance))
        nearby.sort(key=lambda item: (item[1], item[0]))
        per_observation.append(nearby[:max_candidates])

    results: list[CandidateFeature] = []
    for i, (observation, candidates) in enumerate(zip(trip.observations, per_observation)):
        dt, speed, gps_heading = motion[observation.obs_id]
        previous = per_observation[i - 1] if i > 0 else None
        previous_destinations = {
            arc_by_id[arc_id].to_node_id for arc_id, _ in previous
        } if previous is not None else set()
        for arc_id, distance in candidates:
            segment = segment_by_id[arc_id]
            dx = segment.end_xy[0] - segment.start_xy[0]
            dy = segment.end_xy[1] - segment.start_xy[1]
            road_heading = degrees(atan2(dx, dy)) % 360
            heading_diff = None if gps_heading is None else abs((gps_heading - road_heading + 180) % 360 - 180)
            arc = arc_by_id[arc_id]
            connected = None if previous is None or not previous else arc.from_node_id in previous_destinations
            results.append(CandidateFeature(
                trip_id=trip.trip_id, obs_id=observation.obs_id, arc_id=arc_id,
                distance_m=round(distance, 6),
                heading_diff_deg=round(heading_diff, 6) if heading_diff is not None else None,
                speed_mps=round(speed, 6) if speed is not None else None,
                time_gap_s=round(dt, 6) if dt is not None else None,
                predecessor_connected=connected,
            ))
    return results


def align_reference_route(trip: Trip, labels_radius_m: float = 100.0,
                          ambiguity_margin_m: float = 2.0,
                          skip_penalty_m: float = 0.25, *,
                          candidate_rows: list[CandidateFeature] | None = None) -> list[RouteLabel]:
    """Propose monotonic observation-to-route alignment; reject uncertain labels.

    Dynamic programming assigns observations to nondecreasing route-step positions
    using projected point-to-segment distance. The route is accessed only here,
    separately from feature extraction. Rows near repeated/parallel route choices,
    outside the distance gate, or absent from the candidate radius are rejected.
    """
    if not trip.observations:
        return []
    if not trip.route:
        return [RouteLabel(trip.trip_id, obs.obs_id, None, None, None, "no_reference_route")
                for obs in trip.observations]
    segments = {segment.arc_id: segment for segment in trip.network.segments}
    route_arc_ids = [step.arc_id for step in trip.route]
    # Emission cost for every observation and ordered reference-route occurrence.
    distances = [[_point_segment_distance(point, segments[arc_id].start_xy, segments[arc_id].end_xy)
                  for arc_id in route_arc_ids] for point in trip.observation_xy]
    n, m = len(distances), len(route_arc_ids)
    parents: list[list[int]] = [[-1] * m for _ in range(n)]
    previous = [distances[0][j] + j * skip_penalty_m for j in range(m)]
    for i in range(1, n):
        current = [0.0] * m
        best_value, best_index = float("inf"), 0
        for j in range(m):
            value = previous[j] - j * skip_penalty_m
            if value < best_value:
                best_value, best_index = value, j
            parents[i][j] = best_index
            current[j] = distances[i][j] + best_value + j * skip_penalty_m
        previous = current
    route_index = min(range(m), key=previous.__getitem__)
    alignment = [0] * n
    for i in range(n - 1, -1, -1):
        alignment[i] = route_index
        if i:
            route_index = parents[i][route_index]

    if candidate_rows is None:
        candidate_rows = feature_rows(trip, candidate_radius_m=labels_radius_m)
    candidates_by_obs: dict[int, dict[int, float]] = {}
    for row in candidate_rows:
        candidates_by_obs.setdefault(row.obs_id, {})[row.arc_id] = row.distance_m
    output = []
    for i, (obs, chosen_step) in enumerate(zip(trip.observations, alignment)):
        arc_id = route_arc_ids[chosen_step]
        chosen_distance = distances[i][chosen_step]
        alternatives = sorted((d, j) for j, d in enumerate(distances[i]) if j != chosen_step)
        status = "accepted"
        if chosen_distance > labels_radius_m:
            status = "outside_label_radius"
        elif arc_id not in candidates_by_obs.get(obs.obs_id, {}):
            status = "reference_arc_missing_from_candidates"
        elif alternatives and alternatives[0][0] - chosen_distance <= ambiguity_margin_m:
            alternate_step = alternatives[0][1]
            if route_arc_ids[alternate_step] == arc_id:
                status = "ambiguous_repeated_arc"
            else:
                status = "ambiguous_parallel_or_junction"
        else:
            competing_candidates = [distance for candidate_arc, distance
                                    in candidates_by_obs.get(obs.obs_id, {}).items()
                                    if candidate_arc != arc_id]
            if competing_candidates and min(competing_candidates) - chosen_distance <= ambiguity_margin_m:
                status = "ambiguous_parallel_or_junction"
        output.append(RouteLabel(trip.trip_id, obs.obs_id, chosen_step, arc_id,
                                 round(chosen_distance, 6), status))
    return output


def feature_dicts(rows: list[CandidateFeature]) -> list[dict]:
    """Serialize typed feature records without introducing reference labels."""
    return [asdict(row) for row in rows]


def _point_segment_distance(point, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    fraction = 0.0 if length2 == 0 else max(0.0, min(1.0,
        ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2))
    return hypot(point[0] - (a[0] + fraction * dx), point[1] - (a[1] + fraction * dy))
