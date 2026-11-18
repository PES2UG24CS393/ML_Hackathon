# Sprint 2 API fixture deployment

Create a virtual environment and install `requirements.txt`. From the repository root on Laptop 1 run:

```bash
PATHSNAP_MODE=fixture python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

On Laptop 2, set `PATHSNAP_SERVER` to the receiver's LAN IP and run:

```bash
curl "http://${PATHSNAP_SERVER}:8000/health"
curl "http://${PATHSNAP_SERVER}:8000/trips"
curl -H 'Content-Type: application/json' -d '{"trip_id":"00000005"}' \
  "http://${PATHSNAP_SERVER}:8000/predict"
```

Check Laptop 1 for the request and timing logs. This opt-in mode uses the single synthetic contract trip and reports mode `demo`. Default startup returns an unavailable health status until real prediction assets are wired into the application. Integration needs the feature pipeline, exported model bundle, HMM decoder, and Laptop 2 verification.
