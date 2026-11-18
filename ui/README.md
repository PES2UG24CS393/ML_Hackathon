# Evaluator UI (Issue #19)

The evaluator is a CPU-only, standard-library Python HTTP server with a self-contained browser client. It lists trips from the prediction API, submits exactly `{ "trip_id": "..." }` for the selected trip, and displays GPS points, any road geometry supplied by the API, predicted route parts, layer controls, auto-fit bounds, status/errors, route breaks, elapsed time, and aggregate metrics. It never reads the local dataset, a reference route, or a local model.

## Run

From the repository root:

```powershell
py -3 -m ui.app --mock
```

Open `http://127.0.0.1:8765` to rehearse with synthetic in-memory data. For the real service:

```powershell
$env:PATHSNAP_API_URL = "http://127.0.0.1:8000"
py -3 -m ui.app --host 0.0.0.0 --port 8765
```

Laptop 2 can open `http://<laptop-1-address>:8765` when both devices can reach the host and its firewall permits the port. The API URL is used server-side by the UI proxy, so the browser only needs the UI host.

## API shape assumption

The adapter follows merged C0 contract v1: `GET /trips` returns `trip_ids`; `POST /predict` accepts only `{ "trip_id": "..." }`. Prediction JSON is passed through. GPS positions are WGS84 `[longitude, latitude]`; the browser plots `gps`, optional `roads`/`road_segments`, `route_parts` with `coordinates`, and `breaks` against the matching observation. It also renders finite aggregate `metrics` (or a no-metrics state) and `elapsed_ms`. The C0 response has no complete road-network geometry field, so the roads layer is populated only when the API supplies it. The UI does not invent or fetch road geometry.

`route_parts` are rendered independently, preserving breaks. The UI makes no claim to be a tile-backed map; it provides a local coordinate plot with auto-fit bounds to keep the client dependency-free. The `--mock` response uses a tiny synthetic fixture only for UI development, never for model predictions. Run `py -3 -m unittest tests.test_ui_s2 -v` for the UI tests.
