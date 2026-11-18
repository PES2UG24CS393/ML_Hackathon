"""Readers for the whitespace-delimited PathSnap trip files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TripData:
    """A trip with stable zero-based observation, node, and arc IDs."""

    trip_id: str
    track: list[dict]
    nodes: list[dict]
    arcs: list[dict]
    route: list[int] | None = None


def _rows(path: Path, width: int, converters, optional: bool = False):
    if optional and not path.exists():
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Required trip file not found: {path}")
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text:
                continue
            fields = text.split()
            if len(fields) != width:
                raise ValueError(f"{path}:{line_no}: expected {width} columns, found {len(fields)}")
            try:
                values.append(tuple(convert(value) for convert, value in zip(converters, fields)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid value: {exc}") from exc
    return values


def load_trip(directory: str | Path, trip_id: str | None = None) -> TripData:
    """Load one trip directory. Files are named ``<trip_id>.<extension>``."""
    directory = Path(directory)
    trip_id = trip_id or directory.name
    prefix = directory / trip_id
    raw_track = _rows(prefix.with_suffix(".track"), 3, (float, float, float))
    raw_nodes = _rows(prefix.with_suffix(".nodes"), 2, (float, float))
    raw_arcs = _rows(prefix.with_suffix(".arcs"), 2, (int, int))
    raw_route = _rows(prefix.with_suffix(".route"), 1, (int,), optional=True)
    if not raw_track:
        raise ValueError(f"Trip {trip_id} has no GPS observations")
    if not raw_nodes or not raw_arcs:
        raise ValueError(f"Trip {trip_id} has an empty road network")

    track = [dict(observation_index=i, longitude=row[0], latitude=row[1], timestamp=row[2])
             for i, row in enumerate(raw_track)]
    nodes = [dict(node_id=i, longitude=row[0], latitude=row[1])
             for i, row in enumerate(raw_nodes)]
    arcs = []
    for arc_id, (from_node_id, to_node_id) in enumerate(raw_arcs):
        if not (0 <= from_node_id < len(nodes) and 0 <= to_node_id < len(nodes)):
            raise ValueError(f"Arc {arc_id} references a node outside 0..{len(nodes)-1}")
        arcs.append(dict(arc_id=arc_id, from_node_id=from_node_id, to_node_id=to_node_id))
    route = [row[0] for row in raw_route] if raw_route is not None else None
    if route is not None:
        bad = [arc_id for arc_id in route if not 0 <= arc_id < len(arcs)]
        if bad:
            raise ValueError(f"Reference route contains invalid arc IDs: {bad[:5]}")
    return TripData(trip_id, track, nodes, arcs, route)
