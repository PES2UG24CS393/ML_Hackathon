# PathSnap

PathSnap is a machine-learning hackathon project that aims to reconstruct vehicle routes from noisy GPS observations and a directed road network.

- [Problem and architecture](docs/problem_understanding.md) explains the trip files, the relationship between GPS observations and reference routes, and the planned inference outputs.
- [Sprint 1 plan](PathSnap_Sprint_1_Plan.md) lists the work items, acceptance criteria, and evaluation constraints.

The repository includes a parser and local metric road geometry. This branch adds development data validation and Top-K candidate ranking. It is not a trained model or sequence decoder.

## Dataset validation and baseline

Python 3.10+ is sufficient; commands use only the standard library. From the repository root:

```powershell
py -3 scripts/inspect_dataset.py
py -3 -m unittest discover -s tests -v
py -3 -m pathsnap.candidates "Student folder/train/00000005" --top-k 5
```

The validator scans `Student folder/train` and `Student folder/student_test` only. It writes per-trip results to `reports/dataset_inventory.csv` and partition totals to `reports/validation_summary.json`; it does not modify source data or access evaluator data. Use `--data` and `--out` to change the paths.

The candidate command writes `reports/baseline/candidates.csv` and `reports/baseline/predictions.csv`. Both preserve `(trip_id, observation_index)` ordering and original arc IDs. Candidate rows include rank, distance in metres, and ambiguity; prediction rows include Top-1, count, and status. See [dataset validation findings](docs/dataset.md) for the verified demo trip, summary statistics, schemas, and limitations.
