"""Small, dependency-free readers for the PathSnap whitespace trip format.

Source row positions are the stable node/arc/observation IDs. Parsers fail with
line-numbered errors instead of skipping malformed rows and shifting IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math


@dataclass(frozen=True)
class Point:
    lon: float
    lat: float


@dataclass(frozen=True)
class Observation(Point):
    observation_id: int
    timestamp: float


@dataclass(frozen=True)
class Arc:
    arc_id: int
    from_node_id: int
    to_node_id: int
    start: Point
    end: Point


@dataclass(frozen=True)
class Trip:
    trip_id: str
    observations: tuple[Observation, ...]
    nodes: tuple[Point, ...]
    arcs: tuple[Arc, ...]
    route: tuple[int, ...] | None


def _rows(path: str | Path, width: int) -> Iterable[tuple[int, list[str]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != width:
                raise ValueError(f"{path}:{line_no}: expected {width} fields, got {len(fields)}")
            yield line_no, fields


def _coordinate(lon: str, lat: str, where: str) -> Point:
    try:
        x, y = float(lon), float(lat)
    except ValueError as exc:
        raise ValueError(f"{where}: coordinate is not numeric") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{where}: coordinate must be finite")
    if not -180 <= x <= 180 or not -90 <= y <= 90:
        raise ValueError(f"{where}: coordinate outside longitude/latitude bounds")
    return Point(x, y)


def parse_track(path: str | Path) -> tuple[Observation, ...]:
    result = []
    for obs_id, (line_no, fields) in enumerate(_rows(path, 3)):
        point = _coordinate(fields[0], fields[1], f"{path}:{line_no}")
        try:
            timestamp = float(fields[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: timestamp is not numeric") from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"{path}:{line_no}: timestamp must be finite")
        result.append(Observation(point.lon, point.lat, obs_id, timestamp))
    return tuple(result)


def parse_nodes(path: str | Path) -> tuple[Point, ...]:
    return tuple(_coordinate(f[0], f[1], f"{path}:{line}") for line, f in _rows(path, 2))


def parse_arc_rows(path: str | Path) -> tuple[tuple[int, int], ...]:
    result = []
    for line_no, fields in _rows(path, 2):
        try:
            start, end = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: arc endpoint IDs must be integers") from exc
        if start < 0 or end < 0:
            raise ValueError(f"{path}:{line_no}: arc endpoint IDs must be non-negative")
        result.append((start, end))
    return tuple(result)


def parse_route(path: str | Path) -> tuple[int, ...]:
    result = []
    for line_no, fields in _rows(path, 1):
        try:
            arc_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: route arc ID must be an integer") from exc
        if arc_id < 0:
            raise ValueError(f"{path}:{line_no}: route arc ID must be non-negative")
        result.append(arc_id)
    return tuple(result)


def load_trip(directory: str | Path, trip_id: str | None = None) -> Trip:
    directory = Path(directory)
    trip_id = trip_id or directory.name
    track = parse_track(directory / f"{trip_id}.track")
    nodes = parse_nodes(directory / f"{trip_id}.nodes")
    arc_rows = parse_arc_rows(directory / f"{trip_id}.arcs")
    arcs = []
    for arc_id, (start_id, end_id) in enumerate(arc_rows):
        if start_id >= len(nodes) or end_id >= len(nodes):
            raise ValueError(f"{directory}/{trip_id}.arcs:{arc_id + 1}: endpoint ID outside node table")
        arcs.append(Arc(arc_id, start_id, end_id, nodes[start_id], nodes[end_id]))
    route_path = directory / f"{trip_id}.route"
    route = parse_route(route_path) if route_path.exists() else None
    if route is not None and any(arc_id >= len(arcs) for arc_id in route):
        raise ValueError(f"{route_path}: route references an arc outside the arc table")
    return Trip(trip_id, track, nodes, tuple(arcs), route)


def project_local(point: Point, reference_latitude: float) -> tuple[float, float]:
    """Equirectangular local metric projection, suitable for this compact region."""
    radius = 6_371_008.8
    lat0 = math.radians(reference_latitude)
    return (radius * math.radians(point.lon) * math.cos(lat0), radius * math.radians(point.lat))
