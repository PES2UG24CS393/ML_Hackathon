"""Standalone CUDA random forest with one round of parallel trees."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .logistic import CudaClassifier


def build_model(seed: int, params: dict | None = None):
    settings = {
        "num_parallel_tree": 200, "max_depth": 8, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bynode": 0.6, "reg_lambda": 1.0, "device": "cuda",
    }
    settings.update(params or {})
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        CudaClassifier(
            booster="gbtree", objective="binary:logistic", tree_method="hist",
            n_estimators=1, learning_rate=1.0, random_state=seed, **settings,
        ),
    )
