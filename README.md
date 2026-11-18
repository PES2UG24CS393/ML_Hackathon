# PathSnap

PathSnap maps noisy GPS observations to directed road arcs. The current repo includes dataset validation and a nearest-road-segment geometric baseline. The baseline is a reference point matcher, not a trained model or a continuity-aware route decoder.

## Setup

Python 3.10+ is sufficient; the scripts use only the standard library. From the repository root:

```powershell
py -3 scripts/inspect_dataset.py
py -3 -m unittest discover -s tests -v
py -3 -m pathsnap.candidates "Student folder/train/00000005" --top-k 5
```

The validator reads only `Student folder/train` and `Student folder/student_test`; it never modifies source files and does not access the reserved evaluator partition. Its default outputs are `reports/dataset_inventory.csv` (one row per trip) and `reports/validation_summary.json` (partition totals). Override paths with `--data` and `--out`.

The baseline command writes `reports/baseline/candidates.csv` and `reports/baseline/predictions.csv`. Each row is keyed by `(trip_id, observation_id)`; IDs preserve zero-based source row order. Candidate rows also include rank, original arc ID, projected distance in metres, and an ambiguity flag. Prediction rows include Top-1 arc ID, distance, candidate count, and status. Empty arc geometry is represented by a blank arc ID and `no_candidates` status.

## Data contract and method

- `.track`: `longitude latitude timestamp`; the observation ID is its zero-based nonblank row index.
- `.nodes`: `longitude latitude`; node ID is its zero-based row index.
- `.arcs`: `from_node_id to_node_id`; arc ID is its zero-based row index and endpoints are directed.
- `.route`: one arc ID per row in reference order, when present. It is not used to create per-observation predictions.

`pathsnap.preprocess` parses these files, rejects malformed rows instead of silently shifting source IDs, checks coordinate bounds and validates arc and route references. It creates directed arc records with endpoint geometry.

`pathsnap.candidates.rank_candidates` ranks arcs by point-to-segment distance in a local equirectangular metric projection (Earth radius 6,371,008.8 m, reference latitude at the road-network mean). A metric bounding-box grid narrows the search and expands until the requested Top-K is exact. Distances tie-break on original arc ID. This local projection is appropriate for the compact supplied Germany-area dataset; it is not intended for global-scale networks.

No reference `.route` values enter candidate ranking or predictions. The route can be used separately for route-arc coverage diagnostics, but direct observation labels are not available from the file contract. Candidate ranking is reusable as features for later classifiers; this code does not fit a model or infer observation labels.

## Dataset findings

See [docs/dataset.md](docs/dataset.md) for the generated counts, anomaly findings, verified demo trip, file-format notes, and limitations. Re-run the validator after replacing or extending development data to refresh the committed report files.

## Known limitations

- Top-1 is independent for each GPS fix. It may jump between parallel or disconnected roads and ignores temporal gaps and network connectivity.
- Duplicate directed arcs exist in the supplied data; original IDs are preserved and candidates are not deduplicated.
- The parser/geometry portion of Issue #3 was absent from `main` when this work started. `pathsnap.preprocess` provides only the minimal parsing and geometry contract needed for validation and candidate generation, and should be reconciled with the Issue #3 owner's implementation before merge if that work lands concurrently.
- Distance uses a local projection, with small scale distortion; no learned likelihood, HMM, Viterbi decoder, or label alignment is implemented.
