"""CPU distance-weighted KNN with fold-local preprocessing."""

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DistanceWeightedKNN(KNeighborsClassifier):
    """Keep the configured neighbor count usable on small training folds."""

    def __init__(self, n_neighbors=15, n_jobs=None):
        super().__init__(n_neighbors=n_neighbors, weights="distance", n_jobs=n_jobs)

    def fit(self, X, y):
        result = super().fit(X, y)
        if list(self.classes_) != [0, 1]:
            raise ValueError("KNN requires both binary classes.")
        return result

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        requested = self.n_neighbors if n_neighbors is None else n_neighbors
        available = self.n_samples_fit_ - (X is None)
        return super().kneighbors(X, min(requested, available), return_distance)


def build_model(seed: int, params: dict | None = None):
    """Build a deterministic estimator for the shared experiment runner."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        DistanceWeightedKNN(**(params or {})),
    )
