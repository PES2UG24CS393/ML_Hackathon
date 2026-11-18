"""Distance-weighted binary KNN with CUDA neighbor search."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data


class CudaKNN(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=15, batch_size=128):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    def fit(self, X, y):
        import torch

        X, y = validate_data(self, X, y, dtype=np.float32)
        if self.n_neighbors < 1 or self.batch_size < 1:
            raise ValueError("n_neighbors and batch_size must be positive.")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("KNN requires both binary classes.")
        if not torch.cuda.is_available():
            raise RuntimeError("KNN requires a CUDA-enabled PyTorch runtime and GPU.")
        self.training_rows_ = X.copy()
        self.training_labels_ = y.astype(np.float32, copy=True)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float32)
        train = torch.as_tensor(self.training_rows_, device="cuda")
        labels = torch.as_tensor(self.training_labels_, device="cuda")
        k = min(self.n_neighbors, len(train))
        probabilities = []
        for start in range(0, len(X), self.batch_size):
            query = torch.as_tensor(X[start:start + self.batch_size], device="cuda")
            distances = torch.cdist(query, train, compute_mode="donot_use_mm_for_euclid_dist")
            indices = distances.argsort(dim=1, stable=True)[:, :k]
            nearest = distances.gather(1, indices)
            exact = nearest == 0
            inverse = nearest.clamp_min(torch.finfo(nearest.dtype).tiny).reciprocal()
            weights = torch.where(exact.any(dim=1, keepdim=True), exact, inverse)
            positive = (weights * labels[indices]).sum(dim=1) / weights.sum(dim=1)
            probabilities.append(positive.cpu().numpy())
        positive = np.concatenate(probabilities)
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_model(seed: int, params: dict | None = None):
    """Build the deterministic factory used by the shared experiment runner."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        CudaKNN(**(params or {})),
    )
