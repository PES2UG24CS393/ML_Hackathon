# PathSnap

PathSnap is a machine-learning hackathon project that aims to reconstruct vehicle routes from noisy GPS observations and a directed road network.

- [Problem and architecture](docs/problem_understanding.md) explains the trip files, the relationship between GPS observations and reference routes, and the planned inference outputs.
- [Sprint 1 plan](PathSnap_Sprint_1_Plan.md) lists the work items, acceptance criteria, and evaluation constraints.
- [Parser and baseline contract](docs/parser_baseline_contract.md) describes the shared parser fields and nearest-segment output.
- [Dataset validation findings](docs/dataset.md) reports development data counts, anomalies, and a verified demo trip.

## Dataset validation and ranked candidates

Python 3.10 or newer is required; the scripts use only the Python standard library.

```powershell
py -3 scripts/inspect_dataset.py
py -3 -m unittest discover -s tests -v
py -3 -m pathsnap.candidates "Student folder/train/00000005" --top-k 5
py -3 scripts/demo_trip.py "Student folder/train/00000005" --output outputs/trip_00000005
```

The validator scans `Student folder/train` and `Student folder/student_test` only. It writes per-trip results to `reports/dataset_inventory.csv` and partition totals to `reports/validation_summary.json`. It does not modify source data or access evaluator data.

The candidate command writes `reports/baseline/candidates.csv` and `reports/baseline/predictions.csv`, keyed by `(trip_id, observation_index)` and preserving source arc IDs. Candidate rows include rank, distance in metres, and an ambiguity flag; predictions include Top-1, count, and status. The demo command writes an HTML map and JSON with the separate nearest-segment baseline. Neither prediction path uses `.route`.

The baseline is geometric, not a trained model or sequence decoder. See the linked parser contract and dataset report for schemas, results, and limitations.
