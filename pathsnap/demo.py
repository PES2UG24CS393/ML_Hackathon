"""Generate a Folium map and nearest-segment predictions for a development trip."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from .baseline import nearest_segment_predictions
from .preprocess import load_trip


def write_map(trip, predictions, path):
    import folium

    # GeoJSON uses longitude/latitude, while Folium bounds use latitude/longitude.
    node_coordinates = [(node.lon, node.lat) for node in trip.network.nodes]
    lines = [
        [node_coordinates[arc.from_node_id], node_coordinates[arc.to_node_id]]
        for arc in trip.network.arcs
    ]
    track = [(obs.lat, obs.lon) for obs in trip.observations]
    map_view = folium.Map(location=track[0], zoom_start=14, tiles=None)
    folium.TileLayer("OpenStreetMap", overlay=True).add_to(map_view)

    def add_lines(name, coordinates, color, weight, opacity, show=True):
        if coordinates:
            folium.GeoJson(
                {"type": "MultiLineString", "coordinates": coordinates},
                name=name,
                style_function=lambda _: {"color": color, "weight": weight, "opacity": opacity},
                show=show,
            ).add_to(map_view)

    add_lines("Road network", lines, "#89939d", 1, 0.45)
    if trip.route:
        add_lines("Reference route", [lines[step.arc_id] for step in trip.route],
                  "#198540", 5, 0.85, show=False)
    selected = sorted({row["arc_id"] for row in predictions if row["arc_id"] is not None})
    add_lines("Nearest segments", [lines[arc_id] for arc_id in selected], "#df3b32", 3, 0.9)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
             "properties": {}}
            for lat, lon in track
        ]},
        name="GPS observations",
        marker=folium.CircleMarker(radius=3, color="#1769d2", fill=True, fill_opacity=0.8),
    ).add_to(map_view)
    latitudes, longitudes = zip(*track)
    bounds = [(min(latitudes), min(longitudes)), (max(latitudes), max(longitudes))]
    map_view.fit_bounds(bounds, padding=(16, 16), max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(map_view)
    legend = ["Gray · Road network", "Blue · GPS observations"]
    if selected:
        legend.append("Red · Nearest segments")
    if trip.route:
        legend.append("Green · Reference route (enable in layers)")
    map_view.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:12px;z-index:999;'
        'background:white;padding:8px 12px;border-radius:4px" role="note">'
        + "<br>".join(legend) + "</div>"
    ))
    html = map_view.get_root().render()
    path.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("trip_dir", type=Path, help="Directory containing <trip_id>.* files")
    parser.add_argument("--output", type=Path, default=Path("outputs/trip"), help="Output filename prefix")
    parser.add_argument("--max-distance-m", type=float, default=250.0)


def run_command(args: argparse.Namespace) -> None:

    trip = load_trip(args.trip_dir)
    if not trip.observations:
        raise ValueError("trip has no GPS observations")
    started = perf_counter()
    predictions = nearest_segment_predictions(trip, args.max_distance_m)
    elapsed = perf_counter() - started
    payload = {
        "trip_id": trip.trip_id,
        "track": [
            dict(observation_index=obs.obs_id, longitude=obs.lon, latitude=obs.lat, timestamp=obs.timestamp_s)
            for obs in trip.observations
        ],
        "nodes": [
            dict(node_id=node.node_id, longitude=node.lon, latitude=node.lat)
            for node in trip.network.nodes
        ],
        "arcs": [arc._asdict() for arc in trip.network.arcs],
        "predictions": predictions,
        # Reference route stays in this development artifact only; it is never used to predict.
        "reference_route": [step.arc_id for step in trip.route] if trip.route is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    html_path = args.output.with_suffix(".html")
    json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    write_map(trip, predictions, html_path)
    matched = sum(p["arc_id"] is not None for p in predictions)
    print(f"Trip {trip.trip_id}: {len(trip.observations)} GPS observations, {len(trip.network.nodes)} nodes, {len(trip.network.arcs)} arcs")
    print(f"Matched within {args.max_distance_m:g} m: {matched}/{len(predictions)} observations")
    first, last = trip.observations[0], trip.observations[-1]
    print(f"Timestamps: {first.timestamp_s:g} to {last.timestamp_s:g} s")
    print("Validation passed: required files, row widths, finite coordinates, coordinate bounds, "
          "timestamp order and arc references")
    route_status = f"{len(trip.route)} steps" if trip.route is not None else "absent"
    print(f"Reference route: {route_status}")
    print(f"Unmatched observations: {len(predictions) - matched}; baseline runtime: {elapsed:.3f} s")
    print(f"Parser contract JSON: {json_path.resolve()}")
    print(f"Interactive map: {html_path.resolve()}")
