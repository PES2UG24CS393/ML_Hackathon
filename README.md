# PathSnap

## Quick parser and baseline demo

Python 3.10 or newer is required. The parser, geometric baseline and demo generator use only the Python standard library.

From the repository root in PowerShell:

```powershell
py scripts/demo_trip.py "Student folder\train\00000001" --output outputs\trip_00000001
```

Open `outputs\trip_00000001.html` in a browser. The map loads Leaflet and OpenStreetMap tiles over the internet. The command also writes JSON containing parsed GPS/network data and nearest-segment predictions. Choose another training trip directory to inspect another example.

The baseline finds the closest road segment in a local metric projection, with a 250 m default cutoff. It is a geometry baseline, not a trained ML model. The `.route` values are included in the development JSON as reference data and are not used by the predictor.

See [the parser and baseline contract](docs/parser_baseline_contract.md) before integrating these functions with the model or visualization code.
