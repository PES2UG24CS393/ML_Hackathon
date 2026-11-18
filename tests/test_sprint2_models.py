"""Synthetic fixture score smoke tests for S2-A1 model factories."""

import importlib.util
import unittest

from ml.models.decision_tree import build_model as build_decision_tree
from ml.models.xgboost import build_model as build_xgboost


X = [[0.0, 2.0], [0.1, 1.8], [0.2, 1.7], [3.0, 0.1], [3.2, 0.0], [2.8, 0.2]]
Y = [1, 1, 1, 0, 0, 0]


class ModelFactoryTests(unittest.TestCase):
    def assert_positive_class_scores(self, model):
        model.fit(X, Y)
        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities.tolist(), model.predict_proba(X).tolist())
        positive_column = list(model.classes_).index(1)
        positive = [row[positive_column] for row in probabilities]
        self.assertEqual(len(positive), len(Y))
        self.assertTrue(all(0.0 <= float(p) <= 1.0 for p in positive))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn dependency is not installed")
    def test_decision_tree_fit_probability_support_and_seed(self):
        first = build_decision_tree(42, {"max_depth": 3, "min_samples_leaf": 1})
        second = build_decision_tree(42, {"max_depth": 3, "min_samples_leaf": 1})
        first.fit(X, Y)
        second.fit(X, Y)
        self.assertEqual(first.predict_proba(X).tolist(), second.predict_proba(X).tolist())
        self.assert_positive_class_scores(first)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "xgboost dependency is not installed")
    def test_xgboost_cpu_hist_backend_and_probability_support(self):
        model = build_xgboost(42, {"n_estimators": 5, "max_depth": 2})
        self.assertEqual(model.get_params()["tree_method"], "hist")
        self.assertEqual(model.get_params()["device"], "cpu")
        model.fit(X, Y)
        self.assert_positive_class_scores(model)


if __name__ == "__main__":
    unittest.main()
