"""Geometric nearest-road-segment baseline (not a trained ML model)."""

from math import isfinite

from .preprocess import Trip


def nearest_segment_predictions(trip: Trip, max_distance_m: float = 250.0):
    """Return a prediction per GPS observation using the parser's metric geometry."""
    if not isfinite(max_distance_m) or max_distance_m < 0:
        raise ValueError("Maximum distance must be finite and nonnegative")

    cell_size = max(50.0, max_distance_m)
    grid = {}
    long_segments = []
    for segment in trip.network.segments:
        a, b = segment.start_xy, segment.end_xy
        lo_x, hi_x = sorted((a[0], b[0]))
        lo_y, hi_y = sorted((a[1], b[1]))
        ix0 = int((lo_x - max_distance_m) // cell_size)
        ix1 = int((hi_x + max_distance_m) // cell_size)
        iy0 = int((lo_y - max_distance_m) // cell_size)
        iy1 = int((hi_y + max_distance_m) // cell_size)
        if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 400:
            long_segments.append(segment)
            continue
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                grid.setdefault((ix, iy), []).append(segment)

    output = []
    for observation, point in zip(trip.observations, trip.observation_xy):
        cell = (int(point[0] // cell_size), int(point[1] // cell_size))
        best_id = None
        best_distance = max_distance_m
        for segment in (*grid.get(cell, ()), *long_segments):
            distance = segment.distance_to(point)
            if distance < best_distance or (
                distance == best_distance and (best_id is None or segment.arc_id < best_id)
            ):
                best_id, best_distance = segment.arc_id, distance
        output.append({
            "obs_id": observation.obs_id,
            "arc_id": best_id,
            "distance_m": round(best_distance, 3) if best_id is not None else None,
        })
    return output
