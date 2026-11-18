# Random forest experiment

Run against the reviewed C2 data export.

```sh
python -m ml.trainer --model random_forest --config configs/models/random_forest.yaml
python -m experiments.model_report experiments/results/random_forest/metrics.json
```

This is XGBoost's standalone random forest mode. It fits 200 parallel randomized trees in one round with row and feature subsampling. The separate XGBoost model in #23 uses sequential boosting rounds. Depth 8 and minimum child weight 5 limit small, highly specialized leaves. Compare training and held-out scores for overfitting. See the [standalone forest documentation](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html).

The second command requires Matplotlib and writes the CV/student-test confusion matrices and training-versus-validation Top-1 plot to `model_evidence.png`. Keep the corresponding fold supports and timing in `metrics.json`.

The reviewed #18 manifest gates the real run. Synthetic fit/probability smoke checks carry no experimental metrics.
