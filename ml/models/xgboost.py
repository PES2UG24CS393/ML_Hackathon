"""CPU XGBoost candidate classifier factory for the shared S2 runner."""

from __future__ import annotations

from collections.abc import Mapping


DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 1,
}


def build_model(seed: int, params: Mapping[str, object] | None = None):
    """Build a reproducible CPU XGBoost estimator with positive class 1."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost model requires the xgboost package") from exc
    options = dict(DEFAULT_PARAMS)
    if params is not None:
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping or None")
        options.update(params)
    # The owned config may tune model parameters, but the actual backend remains CPU.
    options.update(random_state=seed, tree_method="hist", device="cpu", n_jobs=1,
                   objective="binary:logistic", eval_metric="logloss")
    return XGBClassifier(**options)
