# Versioned model bundle

The actual bundle is produced after reviewed labels and all five real model reports are available. This directory does not contain a placeholder model.

Run the preparation and export from the repository root with the same environment as the verified model runs.

```sh
python -m pip install optuna joblib matplotlib
python -m ml.tune --manifest artifacts/data/manifest.json --runs experiments/results/logistic/metrics.json experiments/results/random_forest/metrics.json experiments/results/knn/metrics.json experiments/results/decision_tree/metrics.json experiments/results/xgboost/metrics.json --space experiments/search_space.json --trials 25 --output experiments/results/tuning-v1 --bundle artifacts/model/v1
```

The candidate model families are compared on identical frozen CV folds. The primary selection metric is trip-macro observation Top-1. Tie-break with lower log loss, then scoring latency. The TPE search uses only training folds. Student-test is loaded for the tuned final comparison after the optimizer selects parameters. The search ranges vary regularization, neighbor count, tree size and boosting speed around the initial model recipes. Confirm the config parameter names against the merged model factories before running.

The output contains five-model comparison tables and plots, confusion matrices, a per-trial journal, and tuned versus untuned metrics. Empty metric cells reflect zero support or unavailable geometry. Inspect supports before interpreting any score. Probabilities are kept as positive-class scores for the planned log-emission transform. Log loss and calibration results are part of the HMM suitability assessment.

The bundle includes the fitted preprocessing and model, optional probability calibrator, ordered feature schema, exact model parameters and dependency versions, source-data manifest fingerprint, and pipeline checksum. The API can read `manifest.json` and load the fitted probability estimator directly from `model.joblib`. The manifest includes `feature_columns` and the v1 candidate policy (100 m radius, at most 8 arcs). From an independent process, `ml.tune.load_bundle(path)` verifies the checksum and returns the same fitted estimator. Pass only candidate feature rows to `ml.trainer.score_candidate_rows(model, None, candidates)`. Load joblib files solely from trusted producers. Keep generated binaries and large prediction files out of Git unless approved for release.
