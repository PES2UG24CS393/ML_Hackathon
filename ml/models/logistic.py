"""L2-regularized logistic regression using CUDA coordinate descent."""

import json

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class CudaClassifier(XGBClassifier):
    """Fail a requested CUDA run if the backend changes devices during fit."""

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        actual = json.loads(self.get_booster().save_config())["learner"]["generic_param"]["device"]
        if self.device.startswith("cuda") and not actual.startswith("cuda"):
            raise RuntimeError(f"CUDA was requested but XGBoost fitted on {actual}")
        return self


def build_model(seed: int, params: dict | None = None):
    settings = {"reg_lambda": 0.001, "learning_rate": 0.5, "n_estimators": 100, "device": "cuda"}
    settings.update(params or {})
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        CudaClassifier(
            booster="gblinear", updater="coord_descent", feature_selector="cyclic",
            objective="binary:logistic", reg_alpha=0, random_state=seed, **settings,
        ),
    )
