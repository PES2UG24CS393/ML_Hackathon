# Logistic regression experiment

Run against the reviewed C2 data export.

```sh
python -m ml.trainer --model logistic --config configs/models/logistic.yaml
python -m experiments.model_report experiments/results/logistic/metrics.json
```

This is a linear binary-logistic model using deterministic cyclic coordinate descent. L2 regularization (0.001 on the backend's normalized scale) shrinks coefficients after fold-local median imputation and scaling. Start with 100 coordinate-descent rounds and learning rate 0.5. Inspect training versus held-out scores before tuning. The CUDA linear booster requires a compatible XGBoost release, such as the smoke-tested 3.4.1. The backend has deprecated this booster, so pin the validated version for reproducibility.

The second command requires Matplotlib and writes the CV/student-test confusion matrices and training-versus-validation Top-1 plot to `model_evidence.png`. Keep the corresponding fold supports and timing in `metrics.json`.

The reviewed #18 manifest gates the real run. Synthetic fit/probability smoke checks carry no experimental metrics.
