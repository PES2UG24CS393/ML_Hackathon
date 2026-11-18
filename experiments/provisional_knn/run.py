"""Run a provisional-label KNN experiment on the development partitions."""

import argparse
import csv
from hashlib import file_digest, sha256 as text_sha256
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np

from data_prep.build_dataset import hash_trip_inputs
from ml.protocol import FEATURE_ORDER, Candidate, Dataset, features, manifest, model_factory, read_config
from ml.trainer import _score, fit_model, matrix, positive_scores, run_cv

PARTITIONS = ("train", "student_test")
ROOT = Path(__file__).resolve().parent


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_proposals(path):
    proposals = {}
    for row in csv_rows(path):
        split = row["split"]
        key = row["trip_id"], int(row["obs_id"])
        if split not in PARTITIONS or key in proposals:
            raise ValueError(f"Unexpected partition or duplicate proposal: {split}, {key}")
        proposals[key] = (split, int(row["proposed_arc"]), int(row["route_step"]), row["input_sha256"])
    return proposals


def load_split(directory, spec, assignments, split, proposals):
    selected = {key: entry for key, entry in proposals.items() if entry[0] == split}
    expected = {key: entry[1] for key, entry in selected.items()}
    seen = set()
    provenance = {}
    tables = spec["tables"][split]
    for kind in ("features", "labels"):
        file = tables[kind]
        if sha256(directory / file) != spec["outputs_sha256"][file]:
            raise ValueError(f"Exported {kind} hash mismatch for {split}")
    for row in csv_rows(directory / tables["labels"]):
        key = row["trip_id"], int(row["obs_id"])
        if key not in selected:
            continue
        _, arc, step, digest = selected[key]
        reference = int(row["reference_arc_id"]), int(row["route_step_id"])
        if row["label_status"] == "rejected" or reference != (arc, step):
            raise ValueError(f"Proposal disagrees with the current export: {key}")
        trip_id = key[0]
        if trip_id not in provenance:
            provenance[trip_id] = hash_trip_inputs(Path(assignments[trip_id]["trip_dir"]))
        if provenance[trip_id] != digest:
            raise ValueError(f"Proposal targets a different source trip: {key}")
        seen.add(key)
    if seen != selected.keys():
        raise ValueError(f"Missing source label rows for {split}")
    candidates = []
    found_targets = set()
    for row in csv_rows(directory / tables["features"]):
        key = row["trip_id"], int(row["obs_id"])
        if key not in selected:
            continue
        arc_id = int(row["arc_id"])
        candidates.append(Candidate(*key, arc_id, features(row)))
        if expected[key] == arc_id:
            found_targets.add(key)
    if found_targets != selected.keys():
        raise ValueError(f"Missing positive candidate rows for {split}")
    return Dataset(split, candidates, {key: "provisional" for key in selected}, expected, {})


def nearest_metrics(data):
    nearest = {}
    for row in data.candidates:
        closest = (row.values[0], row.arc_id)
        if closest < nearest.get(row.observation, (float("inf"), 0)):
            nearest[row.observation] = closest
    per_trip = {}
    for key, arc in data.validated.items():
        hits, total = per_trip.get(key[0], (0, 0))
        per_trip[key[0]] = hits + (nearest[key][1] == arc), total + 1
    total = len(data.validated)
    hits = sum(hits for hits, _ in per_trip.values())
    return {
        "correct": hits, "observations": total, "trip_support": len(per_trip),
        "top1_pooled": hits / total, "top1": sum(hits / size for hits, size in per_trip.values()) / len(per_trip),
    }


def annotate_support(record):
    if isinstance(record, dict):
        return {
            {"validated": "provisional_label_support", "label_coverage": "selected_sample_label_coverage"}.get(key, key):
            annotate_support(value)
            for key, value in record.items()
        }
    if isinstance(record, list):
        return [annotate_support(value) for value in record]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory for each run.")
    spec, assignments = manifest(args.manifest, require_training_ready=False)
    label_file = ROOT / "provisional_labels.csv"
    proposals = load_proposals(label_file)
    train, test = (
        load_split(args.manifest.parent, spec, assignments[split], split, proposals)
        for split in PARTITIONS
    )
    config = read_config("configs/models/knn.yaml")
    params = config["params"]
    seed = spec["split_seed"]
    build = model_factory("knn")
    folds = {trip_id: row["fold_id"] for trip_id, row in assignments["train"].items()}
    cv, _ = run_cv(build, params, seed, train, folds, with_continuity=False)
    model, calibrator, fit = fit_model(build, params, seed, train, folds)
    student, _ = _score((model, calibrator, fit), test, assignments["student_test"], with_continuity=False)
    args.out.mkdir(parents=True)
    bundle = args.out / "knn-provisional.joblib"
    joblib.dump(model, bundle, compress=3)
    sample = matrix(test.candidates)
    np.testing.assert_array_equal(positive_scores(model, sample), positive_scores(joblib.load(bundle), sample))
    report = {
        "model": "knn", "run_kind": "provisional_weak_supervision",
        "official_training_eligible": False, "device": "cpu", "seed": seed,
        "evaluator_data_used": False, "fit_candidate_support": fit["fit_candidate_support"],
        "feature_order": list(FEATURE_ORDER), "params": params,
        "source_fingerprint": spec["data_fingerprint_sha256"],
        "source_manifest_sha256": sha256(args.manifest),
        "provisional_labels_sha256": text_sha256(label_file.read_text(encoding="utf-8").encode()).hexdigest(),
        "model_sha256": sha256(bundle),
        "dependencies": {name: version(name) for name in ("numpy", "scikit-learn", "joblib", "PyYAML")},
        "cv": annotate_support(cv), "student_test": annotate_support(student),
        "nearest_segment": {"training": nearest_metrics(train), "student_test": nearest_metrics(test)},
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, metrics in (("Grouped CV", cv["pooled"]), ("Student test", student)):
        print(f"{label}: {metrics['top1_correct']}/{metrics['validated']} provisional matches; "
              f"trip-macro Top-1 {metrics['top1']:.1%}; {metrics['top1_trip_support']} trips")
    print(bundle)


if __name__ == "__main__":
    main()
