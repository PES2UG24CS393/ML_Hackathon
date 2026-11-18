"""Geometric nearest-road-segment baseline (not a trained ML model)."""

from math import cos, hypot, radians


def nearest_segment_predictions(trip, max_distance_m: float = 250.0):
    """Return one prediction record per GPS sample using a local metric projection.

    Distances use a trip-centered equirectangular projection, appropriate for these
    small, local trip extents. The maximum-distance gate makes unmatched points explicit.
    """
    lat0 = sum(p["latitude"] for p in trip.track) / len(trip.track)
    lon0 = sum(p["longitude"] for p in trip.track) / len(trip.track)
    sx = 111320.0 * cos(radians(lat0))
    sy = 110574.0

    def xy(lon, lat):
        return ((lon - lon0) * sx, (lat - lat0) * sy)

    nodes = [xy(n["longitude"], n["latitude"]) for n in trip.nodes]
    segments = []
    cell_size = max(50.0, float(max_distance_m))
    grid = {}
    for arc in trip.arcs:
        a = nodes[arc["from_node_id"]]
        b = nodes[arc["to_node_id"]]
        index = len(segments)
        segments.append((arc["arc_id"], a, b))
        lo_x, hi_x = sorted((a[0], b[0]))
        lo_y, hi_y = sorted((a[1], b[1]))
        ix0, ix1 = int((lo_x-max_distance_m)//cell_size), int((hi_x+max_distance_m)//cell_size)
        iy0, iy1 = int((lo_y-max_distance_m)//cell_size), int((hi_y+max_distance_m)//cell_size)
        if (ix1-ix0+1)*(iy1-iy0+1) > 400:
            grid.setdefault(None, []).append(index)
            continue
        for ix in range(ix0, ix1+1):
            for iy in range(iy0, iy1+1):
                grid.setdefault((ix, iy), []).append(index)

    def distance(point, a, b):
        vx, vy = b[0]-a[0], b[1]-a[1]
        denom = vx*vx + vy*vy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((point[0]-a[0])*vx+(point[1]-a[1])*vy)/denom))
        return hypot(point[0]-(a[0]+t*vx), point[1]-(a[1]+t*vy))

    output = []
    for observation in trip.track:
        point = xy(observation["longitude"], observation["latitude"])
        ix, iy = int(point[0]//cell_size), int(point[1]//cell_size)
        reach = int(max_distance_m//cell_size)+1
        candidates = set(grid.get(None, ()))
        for gx in range(ix-reach, ix+reach+1):
            for gy in range(iy-reach, iy+reach+1):
                candidates.update(grid.get((gx, gy), ()))
        # Explicit full scan fallback guarantees a nearest answer when the radius gate misses.
        if not candidates:
            candidates = range(len(segments))
        best_id, best_distance = None, float("inf")
        for index in candidates:
            arc_id, a, b = segments[index]
            d = distance(point, a, b)
            if d < best_distance:
                best_id, best_distance = arc_id, d
        if best_distance > max_distance_m:
            best_id = None
        output.append(dict(observation_index=observation["observation_index"],
                           arc_id=best_id,
                           distance_m=round(best_distance, 3) if best_id is not None else None))
    return output
