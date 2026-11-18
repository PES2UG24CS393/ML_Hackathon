"""Decision Tree candidate classifier factory for the shared S2 runner."""

from __future__ import annotations

from collections.abc import Mapping


DEFAULT_PARAMS = {
    "criterion": "gini",
    "max_depth": 8,
    "min_samples_leaf": 20,
    "min_samples_split": 2,
    "class_weight": None,
}


def build_model(seed: int, params: Mapping[str, object] | None = None):
    """Build a seeded sklearn estimator; class 1 is the positive candidate label."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise ImportError("Decision Tree requires scikit-learn; install the project ML dependencies") from exc
    options = dict(DEFAULT_PARAMS)
    if params is not None:
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping or None")
        options.update(params)
    options["random_state"] = seed
    return DecisionTreeClassifier(**options)
