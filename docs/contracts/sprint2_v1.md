# PathSnap Sprint 2 contract v1

All trip IDs are eight ASCII digits. Observation and directed arc IDs are nonnegative, zero-based integers from the parser. IDs join tables and stay outside the model feature matrix.

## Inference and supervision

Extract candidates from GPS and the directed road network with a 100 m radius, retaining up to eight per observation. Sort by observation ID, metric segment distance, then arc ID. A missing candidate set emits zero rows. Keep the observation in coverage and decoding.

| Ordered model column | Type and unit | Missing values |
| --- | --- | --- |
| distance_m | Float, meters | Required |
| heading_diff_deg | Float, degrees in [0, 180] | First fix, stationary movement below 1 m, or invalid delta |
| speed_mps | Float, meters per second | First fix or invalid delta |
| time_gap_s | Float, seconds | First fix or invalid delta |
| predecessor_connected | Boolean, encoded as 0 or 1 in matrices | First fix or no previous candidates |

CSV uses empty cells for nulls. JSON uses null. ML matrices use NaN. Connectivity means at least one retained candidate from the previous observation ends at the current arc's start. See [feature definitions](../features.md).

Labels live in a separate table keyed by trip and observation with nullable integer `reference_arc_id`, string `label_status` and string `review_evidence`. Only status `validated` supplies supervised ground truth. A validated observation with a missing reference candidate counts toward observation coverage and Top-1 support. Exclude it from classifier fitting and conditional candidate metrics. A versioned manifest records partition, trip ID, training fold ID 0–4 (null outside training), feature and label versions, and split seed. Freeze grouped training folds before fitting. Hold out student-test for the frozen comparison and keep evaluator inputs isolated.

## Model and decoder

Each factory exposes `build_model(seed: int, params: dict | None = None)`. The fitted estimator supports `fit(X, y)` and `predict_proba(X)`. Use the column identified by class label 1 in `classes_`. Fit preprocessing within each training fold. Score rows have `trip_id`, `obs_id`, `arc_id` and finite `p_positive` in [0, 1]. Classifier ranking ties resolve to smaller arc IDs.

`decode_trip(trip, candidate_scores, config)` returns `matches` in observation order with nullable arc IDs, `route_arc_ids` in traversal order, `route_parts` as separate nonempty ordered arc sequences, and `breaks` with the observation ID where a discontinuity begins and a reason. `route_arc_ids` concatenates the parts. Preserve intermediate arcs and revisits. Missing candidates still receive a match with a null arc. Do not draw connections across parts.

## HTTP

- `GET /health` returns `{"status":"ok","mode":"demo"}` when ready. Mode is `demo` or `release`. Unavailable runtime assets return 503.
- `GET /trips` returns `{"trip_ids":["00000005"]}` with sorted allowlisted IDs for the configured mode.
- `POST /predict` accepts only a JSON `trip_id`. See [request](predict_request.json) and [response](predict_response.json). All positions are WGS84 `[longitude, latitude]`. Each route part has its own line geometry. Elapsed time is in milliseconds. Metrics are null unless enabled and authorized server-side, and then contain only finite aggregate counts or scalars.
- Errors use `{"error":{"code":"invalid_request","message":"Expected an eight-digit trip ID."}}`. Invalid requests return 422, unknown IDs 404, and unavailable assets 503. Reference routes, point labels and filesystem paths stay server-side.

## Ownership and review

| Owner | Files and interface responsibilities |
| --- | --- |
| Rajath | Candidate and label prep, manifests, UI |
| Ritesh | ML package initializers, runner, metrics, logistic regression, random forest, tuning and export |
| Agilesh | Decision tree, XGBoost, directed decoder and prediction maps |
| Rahul | Shared contract and fixtures, KNN, API, deployment, CLI integration and report |

All four review v1. Coordinate breaking changes with affected owners before introducing a new version. Synthetic fixtures exercise interfaces and carry no experimental evidence.
