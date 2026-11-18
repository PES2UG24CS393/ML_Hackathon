# Sprint 2 API deployment

Install the pinned requirements in a virtual environment on Laptop 1. Run commands
from the repository root. Use the dependencies recorded at model export to load
the chosen model, including the matching XGBoost version if applicable.

For a student-test rehearsal, set the model directory to the trusted export and
the trip root to the student-test partition:

```bash
export PATHSNAP_MODE=demo
export PATHSNAP_TRIP_ROOT="$PWD/data/student_test"
export PATHSNAP_MODEL_DIR=/absolute/path/to/exported-model
python -m uvicorn api.app:app_factory --factory --host 0.0.0.0 --port 8000
```

Use the version 1 bundle exported by the tuning runner. It contains
`pipeline.joblib` (the fitted model and optional probability calibrator) and
`manifest.json` (bundle and feature versions, ordered columns, SHA-256 digest,
and dependency versions). Match the export's dependency versions and use
only trusted artifacts. The Sprint 2 v1 candidate policy uses a 100 m radius
and at most eight candidates. The real API needs the directed decoder module
and a trained bundle; fixture mode works before those assets are available.

On Laptop 2, point the UI's `PATHSNAP_API_URL` at Laptop 1's LAN URL. To smoke
test the connection from Laptop 2, set `PATHSNAP_SERVER` to Laptop 1's LAN IP
and use an ID returned by the second command:

```bash
curl "http://${PATHSNAP_SERVER}:8000/health"
curl "http://${PATHSNAP_SERVER}:8000/trips"
curl -H 'Content-Type: application/json' -d '{"trip_id":"00000007"}' \
  "http://${PATHSNAP_SERVER}:8000/predict"
```

For official evaluation, set `PATHSNAP_MODE=release` and point
`PATHSNAP_TRIP_ROOT` at the supplied reserved partition on Laptop 1. The API
expects ten valid trip folders. Inference reads GPS and the graph. Server-side
evaluation then reads the ordered reference and reports route sequence similarity.
Pointwise accuracy requires independently aligned per-observation reference labels.
Capture successful receiver logs and the UI for all required release checks.

For interface testing, set `PATHSNAP_MODE=fixture` and run the same Uvicorn
command. Fixture mode serves a single synthetic contract response. A real-mode
startup with missing assets reports HTTP 503.
