"""Geometric nearest-road-segment baseline (not a trained ML model)."""

from math import hypot


def _point_segment_distance(point, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0 else max(
        0.0, min(1.0, ((point[0] - a[0]) * vx + (point[1] - a[1]) * vy) / denom)
    )
    return hypot(point[0] - (a[0] + t * vx), point[1] - (a[1] + t * vy))


def nearest_segment_predictions(trip, max_distance_m: float = 250.0):
    """Return one nearest arc record per observation using parser-projected geometry."""
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    segments = trip.network.segments
    cell_size = max(50.0, float(max_distance_m))
    grid = {}
    for index, segment in enumerate(segments):
        a, b = segment.start_xy, segment.end_xy
        ix0, ix1 = int((min(a[0], b[0]) - max_distance_m) // cell_size), int((max(a[0], b[0]) + max_distance_m) // cell_size)
        iy0, iy1 = int((min(a[1], b[1]) - max_distance_m) // cell_size), int((max(a[1], b[1]) + max_distance_m) // cell_size)
        if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 400:
            grid.setdefault(None, []).append(index)
            continue
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                grid.setdefault((ix, iy), []).append(index)

    predictions = []
    for observation, point in zip(trip.observations, trip.observation_xy):
        ix, iy = int(point[0] // cell_size), int(point[1] // cell_size)
        reach = int(max_distance_m // cell_size) + 1
        candidates = set(grid.get(None, ()))
        for gx in range(ix - reach, ix + reach + 1):
            for gy in range(iy - reach, iy + reach + 1):
                candidates.update(grid.get((gx, gy), ()))
        if not candidates:
            candidates = range(len(segments))
        best_arc_id, best_distance = None, float("inf")
        for index in candidates:
            segment = segments[index]
            distance = _point_segment_distance(point, segment.start_xy, segment.end_xy)
            if distance < best_distance:
                best_arc_id, best_distance = segment.arc_id, distance
        if best_distance > max_distance_m:
            best_arc_id, best_distance = None, None
        predictions.append({"obs_id": observation.obs_id, "arc_id": best_arc_id,
                            "distance_m": round(best_distance, 3) if best_distance is not None else None})
    return predictions
