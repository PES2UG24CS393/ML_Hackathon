# Parser and baseline handoff contract

The demo uses the shared parser in `pathsnap/preprocess.py` and the geometric baseline in `pathsnap/baseline.py`. `.route` is reference-only ordered arc IDs; a route row is not assumed to label the same-numbered GPS observation.

## Shared parser output

`pathsnap.preprocess.load_trip(directory)` returns a `Trip` with:

- `trip_id`: directory name.
- `observations`: ordered `Observation` records `{obs_id, lon, lat, timestamp_s}`.
- `observation_xy`: parser-projected metric coordinates in the same order as observations.
- `network.nodes`: `Node` records `{node_id, lon, lat}` with zero-based row IDs.
- `network.arcs`: directed `Arc` records `{arc_id, from_node_id, to_node_id}` with zero-based row IDs.
- `network.segments`: `ArcGeometry` records with projected endpoints in metres.
- `route`: ordered `RouteStep` records `{row_id, arc_id}`, or `None` when the route file is absent.

The parser validates coordinate, timestamp, endpoint, and route ID bounds. It retains coordinates in degrees and supplies projected geometry for metric distance calculations.

## Baseline output

`nearest_segment_predictions(trip, max_distance_m=250)` returns one record per GPS observation:

```json
{"obs_id": 0, "arc_id": 12, "distance_m": 7.231}
```

`arc_id` and `distance_m` are `null` if no segment is within the cutoff. This geometric nearest-segment method is a baseline only; it is not a trained ML model and does not use `.route`.

## Development map artifact

The CLI writes `<prefix>.json` and `<prefix>.html`. The JSON keeps the reference route in a separate field for development inspection; it is not passed into prediction. Do not send reference labels to the evaluator UI. The Folium HTML displays separate road, GPS, nearest-segment and optional reference-route layers. The reference layer starts hidden. It loads Leaflet and OpenStreetMap tiles over the internet.
