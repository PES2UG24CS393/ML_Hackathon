# Sprint 2 label audit and dataset builder

Run from the repository root:

```powershell
py -3 -m data_prep.build_dataset --data data --out artifacts/data --seed 2026
```

The builder scans only `data/train` and `data/student_test`. It exports four separate CSVs: candidate-only `features_train.csv` and `features_student_test.csv`, plus one per-observation proposal table in `labels_train.csv` and `labels_student_test.csv`. Reference fields are never joined into feature rows. `coverage_by_trip.csv`, `coverage_by_split.csv`, `reference_breaks.csv`, `review_queue.csv`, `sanity_checks.csv`, `folds.json`, and `manifest.json` record provenance and quality.

The review queue samples five anchor observations per trip and includes every candidate miss, ambiguous proposal, and observation aligned near a route discontinuity. The sanity checks independently recompute projected geometry distances and verify route-step monotonicity. They are mechanical evidence, not an independent human sign-off.

## Label review workflow

An absent or unchecked decision means the proposal remains unverified and is not fit-eligible. To record a reviewer decision, copy `artifacts/data/review_decisions_template.csv` and populate:

| Column | Meaning |
|---|---|
| `trip_id`, `obs_id` | Stable training-trip/observation key |
| `decision` | `accepted` or `rejected` |
| `reviewer` | Independent reviewer account; the assigned data owner cannot approve their own row |
| `review_evidence` | Path/URI to the independent sequence/geography review evidence |

Then rerun with `--review-decisions path/to/decisions.csv`. An accepted review is exported as contract status `validated`; it is rejected if the proposed arc is not a candidate, is outside the 100 m label radius, or lacks reviewer/evidence metadata. Student-test decisions are never used as training labels. Ambiguous, missing-candidate, and route-discontinuous cases remain visible in coverage reports.

## Current gate

The builder follows the merged C0 contract v1, with candidate defaults (100 m, Top-8, ascending arc-ID ties). `trip_assignments.csv` records each split and training fold ID (0–4; null for student-test). With no independent decisions supplied, `training_ready` is explicitly false and the label files remain proposals only. The 13 known training reference-route discontinuities (and any student-test discontinuities) are recorded without silently bridging them. No reserved evaluator partition is read.

The `folds.json` file assigns training trip IDs, never observations, to deterministic five-fold validation groups with seed 2026. It does not include student-test or reserved evaluator trips. The reproducible full export contains large per-observation feature and label CSVs; checked-in evidence reports and manifest preserve the aggregate counts and source fingerprint.
