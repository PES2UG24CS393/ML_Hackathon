# Parser and baseline handoff contract

This is the shared output agreement for issues #1 and #8. It treats `.route` as reference-only ordered arc IDs; it does not claim that route row `i` labels GPS row `i`.

## Parser output

`load_trip(directory, trip_id=None)` returns `TripData` with:

- `trip_id`: directory name by default.
- `track`: ordered dictionaries `{observation_index, longitude, latitude, timestamp}`.
- `nodes`: ordered dictionaries `{node_id, longitude, latitude}`; ID equals the zero-based row number.
- `arcs`: ordered dictionaries `{arc_id, from_node_id, to_node_id}`; ID equals the zero-based row number.
- `route`: ordered list of reference `arc_id` values, or `None` if `.route` is absent.

Coordinates remain longitude/latitude in degrees in parser output. Arc endpoint and route ID bounds are validated. Blank lines are ignored; columns are whitespace-separated and headerless.

## Baseline output

`nearest_segment_predictions(trip, max_distance_m=250)` returns one record per GPS observation:

```json
{"observation_index": 0, "arc_id": 12, "distance_m": 7.231}
```

`arc_id` and `distance_m` are `null` when no road segment is within the configured cutoff. Distances use a trip-centered equirectangular projection. This is a geometric baseline only; it is not one of the required trained models and does not use `.route`.

## Development map artifact

The CLI writes `<prefix>.json` and `<prefix>.html`. The JSON contains parser fields, baseline predictions and a separate `reference_route` field for development visualization/evaluation. Do not send reference labels to the prediction model or evaluator UI. The HTML displays the real network, observed GPS and nearest predicted arcs. It loads Leaflet and OpenStreetMap tiles from the internet.
