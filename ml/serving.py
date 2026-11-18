"""Inference adapter for exported candidate classifier bundles."""

import numpy as np

from .protocol import FEATURE_ORDER
from .trainer import _log_odds, positive_scores


class CandidateProbabilityModel:
    """Expose the classifier and optional calibrator as one serving estimator."""

    classes_ = np.array([0, 1])
    n_features_in_ = len(FEATURE_ORDER)

    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        scores = positive_scores(self.model, X)
        if self.calibrator is not None:
            scores = positive_scores(self.calibrator, _log_odds(scores))
        return np.column_stack((1 - scores, scores))
