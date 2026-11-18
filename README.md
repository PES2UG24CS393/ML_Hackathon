# PathSnap

PathSnap is a machine-learning hackathon project that aims to reconstruct vehicle routes from noisy GPS observations and a directed road network.

- [Problem and architecture](docs/problem_understanding.md) explains the trip files, the relationship between GPS observations and reference routes, and the planned inference outputs.
- [Sprint 1 plan](PathSnap_Sprint_1_Plan.md) lists the work items, acceptance criteria, and evaluation constraints.
- [Parser and baseline contract](docs/parser_baseline_contract.md) describes the shared parser fields and nearest-segment output.
- [Candidate features and route-label alignment](docs/features.md) documents the inference schema, null rules, and provisional route alignment.
- [Dataset validation findings](docs/dataset.md) reports development data counts, anomalies, and a verified demo trip.
- [Sprint 1 rehearsal and status](docs/sprint-1.md) records the smoke test, demo checklist, and remaining handoffs.

## Run the Sprint 1 demo

Install Python 3.10 or newer. The parser, baseline, and HTML generator use the Python standard library, so no package installation is needed. Run from the repository root using the bundled development training trip `00000005`.

Linux and macOS:

```bash
python3 scripts/demo_trip.py "Student folder/train/00000005" --output outputs/trip_00000005
```

Windows PowerShell:

```powershell
py -3 scripts/demo_trip.py "Student folder/train/00000005" --output outputs/trip_00000005
```

The command reads `.track`, `.nodes`, and `.arcs`, plus the optional `.route` reference. It reports **1,045 GPS observations, 898 nodes, 1,579 arcs, and 1,045 matches within 250 m** for this trip.

- Open [trip_00000005.html](outputs/trip_00000005.html) in a browser with internet access for Leaflet and OpenStreetMap tiles. Gray shows the road network, red highlights selected arcs, and blue shows observed GPS. Use the layer control to toggle GPS and the combined road network and predictions.
- Inspect [trip_00000005.json](outputs/trip_00000005.json) for the parsed data and one prediction per observation. The reference sequence is stored separately in this development artifact.

The baseline selects the nearest segment independently for each observation in the parser’s local metric projection. The 250 m match rate measures coverage within the cutoff. GPS noise, parallel roads, and intersections can lead to the wrong arc, and consecutive predictions can be disconnected. The predictor uses geometry alone, with no training, reference alignment, or route decoding.

## Dataset validation and ranked candidates

Python 3.10 or newer is required; the scripts use only the Python standard library.

```powershell
py -3 scripts/inspect_dataset.py
py -3 -m unittest discover -s tests -v
py -3 -m pathsnap.candidates "Student folder/train/00000005" --top-k 5
```

The validator scans `Student folder/train` and `Student folder/student_test` only. It writes per-trip results to `reports/dataset_inventory.csv` and partition totals to `reports/validation_summary.json`. It does not modify source data or access evaluator data.

The candidate command writes `reports/baseline/candidates.csv` and `reports/baseline/predictions.csv`, keyed by `(trip_id, observation_index)` and preserving source arc IDs. Candidate rows include rank, distance in metres, and an ambiguity flag; predictions include Top-1, count, and status. The demo command writes an HTML map and JSON with the separate nearest-segment baseline. Neither prediction path uses `.route`.

The baseline is geometric, not a trained model or sequence decoder. See the linked parser contract and dataset report for schemas, results, and limitations.

## Candidate features and provisional alignment

Generate the representative candidate feature table and separate alignment review files with:

```powershell
py scripts/build_features.py "Student folder\train\00000005"
```

Route-derived labels are provisional and are not certified for training; see the feature and alignment note before using them.

## Repository layout

| Path | Contents |
| --- | --- |
| `Student folder/train/` and `Student folder/student_test/` | Bundled development trips |
| `pathsnap/` | Trip parser, candidate ranking, baseline, and features |
| `scripts/demo_trip.py` | Raw trip to JSON and HTML entry point |
| `outputs/` | Reproducible sample JSON and interactive map |
| `tests/` | Parser, baseline, and dataset validation tests |
| `docs/` | Data contracts, architecture, and rehearsal notes |
