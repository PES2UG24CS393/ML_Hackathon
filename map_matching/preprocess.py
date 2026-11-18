"""Parse one PathSnap trip and expose geographic and local metric geometry.

Use load_trip(path_to_trip_directory). IDs are zero-based physical row numbers
within each file. Geographic fields are longitude and latitude in degrees.
A route, when present, is an ordered sequence of RouteStep records and can
repeat an arc. It does not label individual observations.
"""

from dataclasses import dataclass
from math import cos, hypot, isfinite, radians
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Union

PathLike = Union[str, Path]
XY = tuple[float, float]
EARTH_RADIUS_M = 6_371_008.8


class ParseError(ValueError):
    """A source file failed validation."""


class Observation(NamedTuple):
    obs_id: int
    lon: float
    lat: float
    timestamp_s: float


class Node(NamedTuple):
    node_id: int
    lon: float
    lat: float


class Arc(NamedTuple):
    arc_id: int
    from_node_id: int
    to_node_id: int


class RouteStep(NamedTuple):
    row_id: int
    arc_id: int


@dataclass(frozen=True)
class LocalProjection:
    """Local equirectangular plane with x east and y north, both in meters.

    Using the network's mid-latitude for longitude scaling accounts for
    meridian convergence around these compact trip networks. Using one
    origin for roads and GPS makes point-to-segment distances comparable
    within a trip. The spherical approximation is intended for local
    candidate ranking and loses accuracy over large extents.
    """

    lon0: float
    lat0: float

    def project(self, lon: float, lat: float) -> XY:
        delta_lon = (lon - self.lon0 + 180) % 360 - 180
        return (
            EARTH_RADIUS_M * radians(delta_lon) * cos(radians(self.lat0)),
            EARTH_RADIUS_M * radians(lat - self.lat0),
        )


class ArcGeometry(NamedTuple):
    arc_id: int
    start_xy: XY
    end_xy: XY

    @property
    def length_m(self) -> float:
        return hypot(self.end_xy[0] - self.start_xy[0], self.end_xy[1] - self.start_xy[1])

    def distance_to(self, point: XY) -> float:
        """Return the metric distance to the closest point on this segment."""
        start_x, start_y = self.start_xy
        dx = self.end_xy[0] - start_x
        dy = self.end_xy[1] - start_y
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return hypot(point[0] - start_x, point[1] - start_y)
        fraction = ((point[0] - start_x) * dx + (point[1] - start_y) * dy) / length_squared
        fraction = max(0.0, min(1.0, fraction))
        return hypot(point[0] - (start_x + fraction * dx), point[1] - (start_y + fraction * dy))


@dataclass(frozen=True)
class RoadNetwork:
    """Index nodes, arcs and segments by source ID. Adjacency lists outgoing arc IDs."""

    nodes: tuple[Node, ...]
    arcs: tuple[Arc, ...]
    segments: tuple[ArcGeometry, ...]
    outgoing_arc_ids: tuple[tuple[int, ...], ...]
    projection: LocalProjection


@dataclass(frozen=True)
class Trip:
    """Pair each observation with its observation_xy entry by obs_id."""

    trip_id: str
    observations: tuple[Observation, ...]
    observation_xy: tuple[XY, ...]
    network: RoadNetwork
    route: Optional[tuple[RouteStep, ...]]


def _error(path: Path, row_id: int, message: str) -> ParseError:
    return ParseError(f"{path}:{row_id + 1} (row ID {row_id}): {message}")


def _rows(path: Path, width: int) -> Iterator[tuple[int, list[str]]]:
    # Reject malformed rows, including blanks. Skipping one would change IDs.
    with path.open(encoding="utf-8-sig") as source:
        for row_id, line in enumerate(source):
            fields = line.split()
            if len(fields) != width:
                raise _error(path, row_id, f"expected {width} columns, got {len(fields)}")
            yield row_id, fields


def _coordinates(path: Path, row_id: int, fields: list[str]) -> tuple[float, float]:
    try:
        lon, lat = float(fields[0]), float(fields[1])
    except ValueError as exc:
        raise _error(path, row_id, "longitude and latitude must be numbers") from exc
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise _error(path, row_id, "longitude or latitude is non-finite or out of bounds")
    return lon, lat


def _reference(path: Path, row_id: int, value: str, count: int, field: str) -> int:
    try:
        reference = int(value)
    except ValueError as exc:
        raise _error(path, row_id, f"{field} must be an integer") from exc
    if not 0 <= reference < count:
        raise _error(path, row_id, f"{field} {reference} is outside [0, {count})")
    return reference


def parse_track(path: PathLike) -> tuple[Observation, ...]:
    """Read lon, lat, timestamp_s. Require finite, nonnegative, nondecreasing time.

    Duplicate timestamps and gaps are preserved in source order.
    """
    path = Path(path)
    observations = []
    for row_id, fields in _rows(path, 3):
        lon, lat = _coordinates(path, row_id, fields)
        try:
            timestamp = float(fields[2])
        except ValueError as exc:
            raise _error(path, row_id, "timestamp must be a number in seconds") from exc
        if not isfinite(timestamp) or timestamp < 0:
            raise _error(path, row_id, "timestamp must be finite and nonnegative")
        if observations and timestamp < observations[-1].timestamp_s:
            raise _error(path, row_id, "timestamp decreases")
        observations.append(Observation(row_id, lon, lat, timestamp))
    return tuple(observations)


def parse_nodes(path: PathLike) -> tuple[Node, ...]:
    """Read longitude and latitude, assigning each node its original row ID."""
    path = Path(path)
    return tuple(
        Node(row_id, *_coordinates(path, row_id, fields))
        for row_id, fields in _rows(path, 2)
    )


def parse_arcs(path: PathLike, node_count: int) -> tuple[Arc, ...]:
    """Read directed endpoints and validate each against the parsed nodes."""
    path = Path(path)
    return tuple(
        Arc(
            row_id,
            _reference(path, row_id, fields[0], node_count, "from_node_id"),
            _reference(path, row_id, fields[1], node_count, "to_node_id"),
        )
        for row_id, fields in _rows(path, 2)
    )


def parse_route(path: PathLike, arc_count: int) -> tuple[RouteStep, ...]:
    """Read route steps in source order and validate each arc ID."""
    path = Path(path)
    return tuple(
        RouteStep(row_id, _reference(path, row_id, fields[0], arc_count, "arc_id"))
        for row_id, fields in _rows(path, 1)
    )


def load_trip(trip_dir: PathLike, *, include_reference: bool = True) -> Trip:
    """Load files named after their parent trip directory.

    A missing or explicitly excluded route yields None. Invalid rows
    fail the load with a path, line number and zero-based row ID.
    """
    directory = Path(trip_dir)
    trip_id = directory.name
    observations = parse_track(directory / f"{trip_id}.track")
    nodes_path = directory / f"{trip_id}.nodes"
    nodes = parse_nodes(nodes_path)
    if not nodes:
        raise ParseError(f"{nodes_path}: expected at least one node")
    arcs = parse_arcs(directory / f"{trip_id}.arcs", len(nodes))
    route = None
    if include_reference:
        try:
            route = parse_route(directory / f"{trip_id}.route", len(arcs))
        except FileNotFoundError:
            pass

    latitudes = [node.lat for node in nodes]
    projection = LocalProjection(nodes[0].lon, (min(latitudes) + max(latitudes)) / 2)
    node_xy = tuple(projection.project(node.lon, node.lat) for node in nodes)
    outgoing = [[] for _ in nodes]
    segments = []
    for arc in arcs:
        outgoing[arc.from_node_id].append(arc.arc_id)
        segments.append(
            ArcGeometry(arc.arc_id, node_xy[arc.from_node_id], node_xy[arc.to_node_id])
        )
    network = RoadNetwork(
        nodes, arcs, tuple(segments), tuple(tuple(ids) for ids in outgoing), projection
    )
    observation_xy = tuple(projection.project(obs.lon, obs.lat) for obs in observations)
    return Trip(trip_id, observations, observation_xy, network, route)
