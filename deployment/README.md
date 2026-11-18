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

The export must contain a fitted probability pipeline in `model.joblib` and its
`manifest.json`. Confirm these filenames and fields with the experiment owner at
handoff. Match the package versions used when the model was saved, and load only
trusted artifacts. The runtime reads these manifest fields:

```json
{
  "feature_columns": ["distance_m", "heading_diff_deg", "speed_mps", "time_gap_s", "predecessor_connected"],
  "candidate_policy": {"radius_m": 100, "max_candidates": 8},
  "decoder_config": {}
}
```

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
