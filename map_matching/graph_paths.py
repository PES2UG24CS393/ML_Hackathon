"""Directed, bounded metre-based path searches for candidate arc transitions."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, isfinite
from typing import MutableMapping

from .preprocess import RoadNetwork, Trip


@dataclass(frozen=True)
class PathResult:
    distance_m: float
    arc_ids: tuple[int, ...]


@dataclass(frozen=True)
class CandidateTransition:
    """Travel distance and connector arcs between two projected candidate states."""

    distance_m: float
    connector_arc_ids: tuple[int, ...]
    same_arc_forward: bool = False


def shortest_path(network: RoadNetwork, start_node_id: int, goal_node_id: int,
                  max_distance_m: float, *, cache: MutableMapping | None = None
                  ) -> PathResult | None:
    """Dijkstra on directed arcs, bounded by total projected road length in metres.

    Equal-cost alternatives are resolved by sorted outgoing arc IDs and a stable
    node priority. A caller may pass a per-network dict cache; bounds are part of
    its key so a short failed search cannot poison a later wider search.
    """
    if not isfinite(max_distance_m) or max_distance_m < 0:
        raise ValueError("max_distance_m must be finite and nonnegative")
    count = len(network.nodes)
    if not (0 <= start_node_id < count and 0 <= goal_node_id < count):
        raise ValueError("node ID is outside the road network")
    key = (start_node_id, goal_node_id, float(max_distance_m))
    if cache is not None and key in cache:
        return cache[key]
    if start_node_id == goal_node_id:
        result = PathResult(0.0, ())
        if cache is not None:
            cache[key] = result
        return result

    best = {start_node_id: 0.0}
    previous: dict[int, tuple[int, int]] = {}
    queue = [(0.0, start_node_id)]
    while queue:
        distance, node_id = heappop(queue)
        if distance != best.get(node_id):
            continue
        if node_id == goal_node_id:
            break
        for arc_id in sorted(network.outgoing_arc_ids[node_id]):
            arc = network.arcs[arc_id]
            weight = network.segments[arc_id].length_m
            next_distance = distance + weight
            if next_distance > max_distance_m:
                continue
            old = best.get(arc.to_node_id, float("inf"))
            if next_distance < old:
                best[arc.to_node_id] = next_distance
                previous[arc.to_node_id] = (node_id, arc_id)
                heappush(queue, (next_distance, arc.to_node_id))
    if goal_node_id not in best:
        result = None
    else:
        path = []
        node_id = goal_node_id
        while node_id != start_node_id:
            prior_node, arc_id = previous[node_id]
            path.append(arc_id)
            node_id = prior_node
        result = PathResult(best[goal_node_id], tuple(reversed(path)))
    if cache is not None:
        cache[key] = result
    return result


def transition_between_candidates(trip: Trip, previous_arc_id: int,
                                  previous_point_xy: tuple[float, float],
                                  current_arc_id: int,
                                  current_point_xy: tuple[float, float],
                                  max_distance_m: float, *,
                                  cache: MutableMapping | None = None
                                  ) -> CandidateTransition | None:
    """Measure a directed path from one GPS-projected point to another road state."""
    arcs = trip.network.arcs
    segments = trip.network.segments
    if not (0 <= previous_arc_id < len(arcs) and 0 <= current_arc_id < len(arcs)):
        raise ValueError("arc ID is outside the road network")
    previous = segments[previous_arc_id]
    current = segments[current_arc_id]
    previous_fraction = _segment_fraction(previous.start_xy, previous.end_xy, previous_point_xy)
    current_fraction = _segment_fraction(current.start_xy, current.end_xy, current_point_xy)
    if previous_arc_id == current_arc_id and current_fraction >= previous_fraction:
        distance = (current_fraction - previous_fraction) * previous.length_m
        if distance <= max_distance_m:
            return CandidateTransition(distance, (), same_arc_forward=True)
        return None

    start_remaining = (1.0 - previous_fraction) * previous.length_m
    end_prefix = current_fraction * current.length_m
    middle_bound = max_distance_m - start_remaining - end_prefix
    if middle_bound < 0:
        return None
    middle = shortest_path(
        trip.network, arcs[previous_arc_id].to_node_id,
        arcs[current_arc_id].from_node_id, middle_bound, cache=cache,
    )
    if middle is None:
        return None
    distance = start_remaining + middle.distance_m + end_prefix
    return CandidateTransition(distance, middle.arc_ids)


def _segment_fraction(start, end, point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return 0.0
    return max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2))
