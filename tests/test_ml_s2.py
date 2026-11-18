import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ml.evaluate import evaluate
from ml.protocol import FEATURE_ORDER, dataset, manifest
from ml.trainer import fit_model, load_inputs, main, score_candidate_rows


class FixtureClassifier:
    def fit(self, X, y):
        self.classes_ = np.array([1, 0])
        self.min_distance_ = float(X[:, 0].min())
        return self

    def predict_proba(self, X):
        p = np.where(X[:, 0] <= self.min_distance_, 0.8, 0.2)
        return np.column_stack((p, 1 - p))


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def fixture_inputs(root):
    trips = []
    tables = {}
    for partition, numbers in (("train", range(1, 6)), ("student_test", [6])):
        candidate_rows, labels = [], []
        for number in numbers:
            identifier = f"{number:08d}"
            trips.append({"trip_id": identifier, "partition": partition,
                          "fold_id": number - 1 if partition == "train" else None})
            candidate_rows.extend([
                [identifier, 0, 4, 2.0, "", "", "", ""],
                [identifier, 0, 5, 9.0, "", "", "", ""],
            ])
            labels.extend([[identifier, 0, 4, "validated", "synthetic"],
                           [identifier, 1, 7, "validated", "synthetic"],
                           [identifier, 2, "", "needs_review", "synthetic"]])
        tables[partition] = {"features": f"{partition}_features.csv",
                             "labels": f"{partition}_labels.csv"}
        write_csv(root / tables[partition]["features"],
                  ["trip_id", "obs_id", "arc_id", *FEATURE_ORDER], candidate_rows)
        write_csv(root / tables[partition]["labels"],
                  ["trip_id", "obs_id", "reference_arc_id", "label_status", "review_evidence"], labels)
    spec = {"synthetic": True, "training_ready": False, "feature_order": FEATURE_ORDER,
            "feature_version": "sprint2_v1", "label_version": "fixture",
            "manifest_version": "fixture", "split_seed": 42, "trips": trips, "tables": tables}
    path = root / "manifest.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


class ExperimentTests(unittest.TestCase):
    def test_ties_misses_and_denominators(self):
        trip = "00000001"
        keys = [(trip, i) for i in range(3)]
        rows = [{"trip_id": trip, "obs_id": 0, "arc_id": arc, "p_positive": 0.8}
                for arc in [5, 4]]
        result = evaluate(rows, keys, {keys[0]: 4, keys[1]: 7})
        self.assertEqual(result["top1"], 0.5)
        self.assertEqual(result["candidate_miss_rate"], 0.5)
        self.assertEqual(result["candidate_support"], 2)
        self.assertEqual(result["confusion_matrix"], [[0, 1], [0, 1]])
        self.assertEqual(result["empty_candidates"], 2)
        self.assertIsNone(result["continuity"])

    def test_cli_fixture_has_disjoint_folds_and_separate_student_test(self):
        with TemporaryDirectory() as path:
            root = Path(path)
            manifest_path = fixture_inputs(root)
            with self.assertRaisesRegex(ValueError, "explicit"):
                manifest(manifest_path)
            config = root / "config.json"
            config.write_text('{"seed": 42}', encoding="utf-8")
            out = root / "out"
            fit_calls = []

            class TrackingClassifier(FixtureClassifier):
                def fit(self, X, y):
                    fit_calls.append((len(X), X.copy()))
                    return super().fit(X, y)

            args = ["--model", "fixture", "--config", str(config), "--manifest",
                    str(manifest_path), "--output", str(out), "--synthetic"]
            with patch("ml.trainer.model_factory",
                       return_value=lambda seed, params: TrackingClassifier()):
                main(args)
            result = json.loads((out / "metrics.json").read_text())
            self.assertEqual([size for size, _ in fit_calls], [8] * 5 + [10])
            self.assertTrue(all(np.isnan(X[:, 1:]).all() for _, X in fit_calls))
            self.assertEqual(result["cv"]["pooled"]["validated"], 10)
            self.assertEqual(result["student_test"]["top1"], 0.5)
            for fold in result["cv"]["folds"]:
                self.assertFalse(set(fold["train_trips"]) & set(fold["validation_trips"]))
                self.assertNotIn("00000006", fold["validation_trips"])
            self.assertTrue((out / "cv_scores.csv").is_file())

    def test_manifest_rejects_cross_partition_duplicates(self):
        with TemporaryDirectory() as path:
            spec_path = fixture_inputs(Path(path))
            spec = json.loads(spec_path.read_text())
            spec["trips"][-1]["trip_id"] = spec["trips"][0]["trip_id"]
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate trip"):
                manifest(spec_path, synthetic=True)

    def test_tuning_is_training_only_and_bundle_scores_round_trip(self):
        from ml.tune import export_bundle, load_bundle, tune

        with TemporaryDirectory() as path:
            root = Path(path)
            spec_path = fixture_inputs(root)
            _, trips, train, _ = load_inputs(spec_path, True, training_only=True)
            folds = {key: item["fold_id"] for key, item in trips["train"].items()}

            def factory(seed, params):
                return FixtureClassifier()

            journal = root / "trials.jsonl"
            space = {"fixture_parameter": {"type": "int", "low": 1, "high": 4}}
            params, top1 = tune(factory, {}, 42, train, folds, False, space, 2, journal)
            self.assertEqual(top1, 0.5)
            self.assertEqual(len(journal.read_text().splitlines()), 2)
            train.partition = "student_test"
            with self.assertRaisesRegex(ValueError, "training partition"):
                tune(factory, {}, 42, train, folds, False, space, 2, root / "invalid.jsonl")
            train.partition = "train"

            fitted = fit_model(factory, params, 42, train, folds)
            bundle = root / "v1"
            export_bundle(bundle, fitted, {"synthetic": True})
            original = score_candidate_rows(*fitted[:2], train.candidates)
            restored = score_candidate_rows(*load_bundle(bundle), train.candidates)
            self.assertEqual(original, restored)
            metadata = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(metadata["feature_order"], list(FEATURE_ORDER))
            self.assertEqual(metadata["candidate_policy"], {"radius_m": 100, "max_candidates": 8})
            child = subprocess.run(
                [sys.executable, "-B", "-c",
                 "import sys; sys.path.insert(0, sys.argv[1]); "
                 "import numpy as np; "
                 "from joblib import load; "
                 "m, _ = load(sys.argv[2]); "
                 "print(m.predict_proba(np.array([[2.0, 0, 0, 0, 0]]))[0, list(m.classes_).index(1)])",
                 str(Path(__file__).parent), str(bundle / "pipeline.joblib")],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(float(child.stdout.strip()), 0.8)

    def test_c2_export_preserves_folds_and_blocks_unreviewed_training(self):
        with TemporaryDirectory() as path:
            root = Path(path)
            spec_path = fixture_inputs(root)
            fixture = json.loads(spec_path.read_text())
            folds = [{"fold_id": trip["fold_id"], "validation_trip_ids": [trip["trip_id"]]}
                     for trip in fixture["trips"] if trip["partition"] == "train"]
            fold_file = root / "folds.json"
            fold_file.write_text(json.dumps(
                {"fold_version": "trip-grouped-v1", "seed": 2026, "folds": folds}
            ))
            write_csv(
                root / "trip_assignments.csv", ("split", "trip_id", "training_fold_id"),
                [(trip["partition"], trip["trip_id"], trip["fold_id"]) for trip in fixture["trips"]],
            )
            outputs = []
            for partition, tables in fixture["tables"].items():
                for kind, name in tables.items():
                    output = f"{kind}_{partition}.csv"
                    (root / name).rename(root / output)
                    outputs.append(output)
            export = {
                "manifest_version": "sprint2-data-manifest-v1", "label_version": "review-v1",
                "contract_version": "sprint2_v1", "contract_frozen": True,
                "evaluator_data_accessed": False, "source_root": str(root / "data"),
                "candidate_settings": {
                    "radius_m": 100.0, "max_candidates": 8, "arc_id_tie_break": "ascending",
                },
                "seed": 2026, "trip_count": {"train": 5, "student_test": 1},
                "folds_file": "folds.json", "trip_assignments_file": "trip_assignments.csv",
                "outputs": outputs, "training_ready": False,
                "readiness_blockers": ["No independently reviewed training labels"],
            }
            spec_path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, "No independently reviewed training labels"):
                manifest(spec_path)
            spec, trips = manifest(spec_path, require_training_ready=False)
            self.assertEqual(spec["split_seed"], 2026)
            self.assertEqual(trips["train"]["00000005"]["fold_id"], 4)
            self.assertIsNone(trips["student_test"]["00000006"]["fold_id"])
            data = dataset(root, spec, "student_test", trips["student_test"])
            self.assertEqual(data.validated, {("00000006", 0): 4, ("00000006", 1): 7})
            self.assertEqual(data.geometry["00000006"], root / "data/student_test/00000006")

            folds[0]["validation_trip_ids"] = ["00000006"]
            fold_file.write_text(json.dumps(
                {"fold_version": "trip-grouped-v1", "seed": 2026, "folds": folds}
            ))
            with self.assertRaisesRegex(ValueError, "assignment files disagree"):
                manifest(spec_path, require_training_ready=False)


if __name__ == "__main__":
    unittest.main()
