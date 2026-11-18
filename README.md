# PathSnap

PathSnap is a machine-learning hackathon project that aims to reconstruct vehicle routes from noisy GPS observations and a directed road network.

- [Problem and architecture](docs/problem_understanding.md) explains the trip files, the relationship between GPS observations and reference routes, and the planned inference outputs.
- [Sprint 1 plan](PathSnap_Sprint_1_Plan.md) lists the work items, acceptance criteria, and evaluation constraints.
- [Candidate features and route-label alignment](docs/features.md) documents the inference schema, null rules, and provisional route alignment.
- [Parser and baseline contract](docs/parser_baseline_contract.md) describes the shared parser fields and nearest-segment output.

## Quick parser and baseline demo

Python 3.10 or newer is required. The parser, geometric baseline and demo generator use only the Python standard library.

From the repository root in PowerShell:

```powershell
py scripts/demo_trip.py "Student folder\train\00000005" --output outputs\trip_00000005
```

Open `outputs\trip_00000005.html` in a browser. The map loads Leaflet and OpenStreetMap tiles over the internet. The command also writes JSON containing parsed GPS/network data and nearest-segment predictions. Choose another training trip directory to inspect another example.

The baseline finds the closest road segment in the parser's local metric projection, with a 250 m default cutoff. It is a geometry baseline, not a trained ML model. The `.route` values are reference-only and are not used by the predictor.

Generate the representative candidate feature table and separate alignment review files with:

```powershell
py scripts/build_features.py "Student folder\train\00000005"
```

Route-derived labels are provisional and are not certified for training; see the feature and alignment note before using them.

The repository contains trip data, course materials, project documentation, and now the initial parser/baseline demo. The complete map-matching pipeline and two-laptop demo remain in development.
