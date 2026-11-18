# Sprint 2 candidate experiments

The harness implements the candidate, label, factory and scoring interfaces in the [Sprint 2 contract](../docs/contracts/sprint2_v1.md). It reads the C2 builder export directly.

Install the numerical, configuration and calibration dependencies in a Python 3.10+ environment.

```sh
python -m pip install numpy scikit-learn pyyaml
python -m ml.trainer --model MODEL --config configs/models/MODEL.yaml
```

A config provides `seed`, `params` for the named factory, an optional `manifest` path and optional `calibrate: true` to fit a sigmoid on training-side out-of-fold probabilities. Override locations with `--manifest` and `--output`. A model factory returns an estimator or pipeline with `fit(X, y)`, `classes_` and `predict_proba(X)`. Its preprocessing belongs inside the factory and is fitted anew on every training split.

## Data handoff

Run from the repository root. The default `artifacts/data/manifest.json` accepts the C2 builder's `sprint2-data-manifest-v1` export. The runner verifies the candidate policy, compares `folds.json` with `trip_assignments.csv`, and derives geometry paths from the export's `source_root`. The separate feature and label CSVs must be generated alongside the manifest. A real run requires `training_ready: true`.

Small development fixtures can instead declare their tables and assignments inline, as shown here. Inline table and geometry paths are relative to the manifest.

```json
{
  "manifest_version": "c2-v1",
  "feature_version": "sprint2_v1",
  "label_version": "review-v1",
  "split_seed": 42,
  "training_ready": true,
  "feature_order": ["distance_m", "heading_diff_deg", "speed_mps", "time_gap_s", "predecessor_connected"],
  "trips": [
    {"trip_id": "00000005", "partition": "train", "fold_id": 0, "trip_dir": "../../data/train/00000005"}
  ],
  "tables": {
    "train": {"features": "features_train.csv", "labels": "labels_train.csv"},
    "student_test": {"features": "features_student_test.csv", "labels": "labels_student_test.csv"}
  }
}
```

Include every trip, with one of five training fold IDs for each training trip and null for each student-test fold. A synthetic manifest requires `"synthetic": true` and the explicit `--synthetic` flag. Only `train` and `student_test` are accepted.

Feature CSV headers are `trip_id,obs_id,arc_id` plus the ordered feature columns above. Label CSV headers are `trip_id,obs_id,reference_arc_id,label_status,review_evidence`. Include every observed fix in the label table, including unlabeled ones and fixes with no candidates. Only reviewed `validated` labels participate in accuracy. A `trip_dir` enables directed continuity via parser geometry. Without geometry continuity has null score and zero support.

## Results

The runner emits `metrics.json`, `cv_scores.csv`, and `student_test_scores.csv`. Metrics include five fold records, pooled out-of-fold results, separate student-test scores, per-trip supports, candidate confusion counts, fit time and per-trip scoring latency. The probability CSVs preserve keys and raw and calibrated scores. Top-1 ties choose the smaller arc ID and candidate misses count as errors. Candidate F1 and log loss condition on validated fixes with their reference arc in the shortlist. Continuity measures directed reachability, which does not impose travel-time feasibility. Report fixture output as synthetic checks only.
