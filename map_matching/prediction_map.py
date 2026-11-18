"""Folium comparison maps for classifier-only and HMM predicted routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .preprocess import Trip


def write_prediction_map(
    trip: Trip,
    classifier_route_parts: Sequence[Sequence[int]],
    hmm_route_parts: Sequence[Sequence[int]],
    output_html: str | Path,
    *,
    provenance: Mapping[str, object],
    breaks: Sequence[Mapping[str, object]] = (),
    include_reference: bool = False,
) -> Path:
    """Write a layered map and provenance sidecar from supplied prediction outputs.

    Reference geometry is off by default and may only be enabled for permitted
    development/report exports. Route parts must already contain directed graph
    paths; the renderer will not fabricate a connector across a break.
    """
    try:
        import folium
    except ImportError as exc:
        raise ImportError("Prediction maps require folium; install requirements.txt") from exc
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    required = {"trip_id", "model_or_method", "prediction_source", "command"}
    missing = sorted(required - set(provenance))
    if missing:
        raise ValueError(f"provenance is missing required fields: {', '.join(missing)}")
    if provenance["trip_id"] != trip.trip_id:
        raise ValueError("provenance trip_id does not match the rendered trip")
    classifier_parts = _validate_parts(trip, classifier_route_parts, "classifier")
    hmm_parts = _validate_parts(trip, hmm_route_parts, "HMM")

    origin_lon = trip.observations[0].lon if trip.observations else 0.0

    def display_lon(lon):
        return lon + 360 if lon - origin_lon < -180 else lon - 360 if lon - origin_lon > 180 else lon

    def coords_for_part(part):
        return [[display_lon(trip.network.nodes[trip.network.arcs[arc_id].from_node_id].lon),
                 trip.network.nodes[trip.network.arcs[arc_id].from_node_id].lat]
                for arc_id in part] + [[
                    display_lon(trip.network.nodes[trip.network.arcs[part[-1]].to_node_id].lon),
                    trip.network.nodes[trip.network.arcs[part[-1]].to_node_id].lat,
                ]]

    gps_coords = [[display_lon(obs.lon), obs.lat] for obs in trip.observations]
    if gps_coords:
        map_view = folium.Map(location=[gps_coords[0][1], gps_coords[0][0]], zoom_start=14, tiles=None)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(map_view)
    else:
        raise ValueError("cannot render a trip with no GPS observations")

    roads = folium.FeatureGroup(name="Road network", show=True)
    network_lines = []
    for arc in trip.network.arcs:
        start = trip.network.nodes[arc.from_node_id]
        end = trip.network.nodes[arc.to_node_id]
        network_lines.append([[display_lon(start.lon), start.lat], [display_lon(end.lon), end.lat]])
    if network_lines:
        folium.GeoJson({"type": "MultiLineString", "coordinates": network_lines},
                       style_function=lambda _: {"color": "#89939d", "weight": 1, "opacity": 0.38}).add_to(roads)
    roads.add_to(map_view)

    gps = folium.FeatureGroup(name="Raw GPS", show=True)
    point_features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": point},
                       "properties": {"obs_id": trip.observations[i].obs_id}}
                      for i, point in enumerate(gps_coords)]
    folium.GeoJson({"type": "FeatureCollection", "features": point_features},
                   name="GPS points", marker=folium.CircleMarker(radius=3, color="#1769d2",
                   fill=True, fill_opacity=0.85)).add_to(gps)
    for path_coords in _gps_parts(trip, gps_coords):
        folium.PolyLine([(lat, lon) for lon, lat in path_coords], color="#1769d2",
                        weight=2, opacity=0.45, dash_array="4 5").add_to(gps)
    gps.add_to(map_view)

    classifier = folium.FeatureGroup(name="Classifier-only route", show=True)
    _add_route_parts(folium, classifier, classifier_parts, coords_for_part, "#f28e2b", 4)
    classifier.add_to(map_view)

    hmm = folium.FeatureGroup(name="HMM / Viterbi route", show=True)
    _add_route_parts(folium, hmm, hmm_parts, coords_for_part, "#7b2cbf", 5)
    hmm.add_to(map_view)

    breaks_layer = folium.FeatureGroup(name="HMM route breaks", show=True)
    obs_by_id = {obs.obs_id: obs for obs in trip.observations}
    for item in breaks:
        obs_id = item.get("before_obs_id")
        obs = obs_by_id.get(obs_id)
        if obs is None:
            continue
        folium.CircleMarker((obs.lat, display_lon(obs.lon)), radius=7, color="#d62728",
                            fill=True, fill_color="#d62728", fill_opacity=0.9,
                            tooltip=f"Route break before observation {obs_id}: {item.get('reason', 'unspecified')}").add_to(breaks_layer)
    if breaks:
        breaks_layer.add_to(map_view)

    if include_reference and trip.route:
        reference_parts = _split_reference(trip)
        reference = folium.FeatureGroup(name="Reference route (development only)", show=False)
        _add_route_parts(folium, reference, reference_parts, coords_for_part, "#198540", 4)
        reference.add_to(map_view)

    map_view.fit_bounds([(min(point[1] for point in gps_coords), min(point[0] for point in gps_coords)),
                         (max(point[1] for point in gps_coords), max(point[0] for point in gps_coords))],
                        padding=(18, 18), max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(map_view)
    canonical_routes = {"classifier_route_parts": classifier_parts, "hmm_route_parts": hmm_parts,
                        "breaks": list(breaks)}
    route_hash = hashlib.sha256(json.dumps(canonical_routes, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
    metadata = dict(provenance)
    metadata.update(prediction_route_sha256=route_hash, include_reference=include_reference)
    provenance_panel = ("<b>Prediction provenance</b><br>" +
                        "<br>".join(f"{_escape(key)}: {_escape(value)}"
                                   for key, value in metadata.items()))
    map_view.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:18px;left:12px;z-index:999;max-width:420px;'
        'max-height:38vh;overflow:auto;background:white;padding:9px 12px;border:1px solid #aaa;'
        'border-radius:4px;font:12px/1.5 sans-serif" role="note">' + provenance_panel + '</div>'
    ))
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    rendered = map_view.get_root().render()
    output_html.write_text("\n".join(line.rstrip() for line in rendered.splitlines()) + "\n", encoding="utf-8")
    sidecar = output_html.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_html


def _validate_parts(trip: Trip, parts: Sequence[Sequence[int]], layer: str) -> list[list[int]]:
    normalized = []
    arcs = trip.network.arcs
    for part_index, raw_part in enumerate(parts):
        part = []
        for raw_arc in raw_part:
            if not isinstance(raw_arc, int) or not 0 <= raw_arc < len(arcs):
                raise ValueError(f"{layer} part {part_index} has invalid arc ID {raw_arc!r}")
            if not part or part[-1] != raw_arc:
                part.append(raw_arc)
        if not part:
            continue
        for previous, current in zip(part, part[1:]):
            if arcs[previous].to_node_id != arcs[current].from_node_id:
                raise ValueError(f"{layer} part {part_index} is disconnected between arcs {previous} and {current}")
        normalized.append(part)
    return normalized


def _add_route_parts(folium, feature_group, parts, coords_for_part, color, weight):
    for part in parts:
        coordinates = coords_for_part(part)
        folium.PolyLine([(lat, lon) for lon, lat in coordinates], color=color,
                        weight=weight, opacity=0.9).add_to(feature_group)


def _gps_parts(trip: Trip, coordinates, max_gap_s: float = 120.0):
    parts = []
    current = []
    for index, point in enumerate(coordinates):
        if index and trip.observations[index].timestamp_s - trip.observations[index - 1].timestamp_s > max_gap_s:
            if len(current) > 1:
                parts.append(current)
            current = []
        current.append(point)
    if len(current) > 1:
        parts.append(current)
    return parts


def _split_reference(trip: Trip) -> list[list[int]]:
    parts: list[list[int]] = []
    arcs = trip.network.arcs
    for step in trip.route or ():
        arc_id = step.arc_id
        if parts and arcs[parts[-1][-1]].to_node_id != arcs[arc_id].from_node_id:
            parts.append([])
        if not parts:
            parts.append([])
        if not parts[-1] or parts[-1][-1] != arc_id:
            parts[-1].append(arc_id)
    return [part for part in parts if part]


def _escape(value) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
