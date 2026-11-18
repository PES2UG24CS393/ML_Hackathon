"""Run the frozen five-fold experiment and separate student-test comparison."""

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .evaluate import evaluate
from .protocol import FEATURE_ORDER, dataset, manifest, model_factory, read_config


def load_inputs(path, synthetic=False, *, training_only=False):
    path = Path(path)
    spec, trips = manifest(path, synthetic=synthetic)
    train = dataset(path.parent, spec, "train", trips["train"])
    student = None if training_only else dataset(
        path.parent, spec, "student_test", trips["student_test"]
    )
    return spec, trips, train, student


def matrix(rows):
    return np.asarray([row.values for row in rows], dtype=np.float32).reshape(-1, len(FEATURE_ORDER))


def eligible_rows(data, trips):
    present = {
        row.observation for row in data.candidates
        if row.trip_id in trips and data.validated.get(row.observation) == row.arc_id
    }
    rows = [row for row in data.candidates if row.observation in present]
    y = np.asarray([int(data.validated[row.observation] == row.arc_id) for row in rows])
    return rows, y


def positive_scores(model, X):
    if not len(X):
        return np.empty(0)
    positive_column, = np.flatnonzero(np.asarray(model.classes_) == 1)
    p = np.asarray(model.predict_proba(X), dtype=float)[:, positive_column]
    if len(p) != len(X) or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("Model did not return a finite probability per candidate")
    return p


def _log_odds(p):
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _fit(factory, params, seed, rows, y):
    if set(y) != {0, 1}:
        raise ValueError("Fitting requires both candidate classes")
    random.seed(seed)
    np.random.seed(seed)
    return factory(seed, params).fit(matrix(rows), y)


def fit_model(factory, params, seed, data, trip_folds, calibrate=False):
    start = perf_counter()
    rows, y = eligible_rows(data, trip_folds)
    calibrator = None
    reason = "disabled"
    if calibrate:
        from sklearn.linear_model import LogisticRegression

        p, actual = [], []
        for fold in sorted(set(trip_folds.values())):
            inside = {key for key, value in trip_folds.items() if value != fold}
            outside = {key for key, value in trip_folds.items() if value == fold}
            inner_rows, inner_y = eligible_rows(data, inside)
            validation_rows, validation_y = eligible_rows(data, outside)
            if set(inner_y) != {0, 1} or not validation_rows:
                continue
            inner = _fit(factory, params, seed, inner_rows, inner_y)
            p.extend(positive_scores(inner, matrix(validation_rows)))
            actual.extend(validation_y)
        reason = "insufficient inner-fold label support"
        if set(actual) == {0, 1}:
            calibrator = LogisticRegression(random_state=seed).fit(_log_odds(np.asarray(p)), actual)
            reason = "sigmoid on inner out-of-fold training scores"

    model = _fit(factory, params, seed, rows, y)
    return model, calibrator, {
        "fit_seconds": perf_counter() - start,
        "fit_candidate_support": len(y),
        "fit_positive_support": int(y.sum()),
        "calibration": reason,
    }


def score_candidate_rows(model, calibrator, candidates):
    raw = positive_scores(model, matrix(candidates))
    scores = positive_scores(calibrator, _log_odds(raw)) if calibrator is not None else raw
    return [
        {"trip_id": row.trip_id, "obs_id": row.obs_id, "arc_id": row.arc_id,
         "p_positive": float(p), "p_raw": float(raw_p)}
        for row, p, raw_p in zip(candidates, scores, raw)
    ]


def _subset_metrics(scores, data, trips, *, with_continuity):
    observations = {key: value for key, value in data.observations.items() if key[0] in trips}
    geometry = (
        {key: value for key, value in data.geometry.items() if key in trips}
        if with_continuity else None
    )
    metrics = evaluate(scores, observations, data.validated, geometry)
    if any(row["p_positive"] != row["p_raw"] for row in scores):
        raw_scores = ({**row, "p_positive": row["p_raw"]} for row in scores)
        metrics["log_loss_raw"] = evaluate(raw_scores, observations, data.validated)["log_loss"]
    else:
        metrics["log_loss_raw"] = metrics["log_loss"]
    return metrics


def _score(fitted, data, trips, *, with_continuity=True):
    model, calibrator, fit_metrics = fitted
    by_trip = defaultdict(list)
    for row in data.candidates:
        if row.trip_id in trips:
            by_trip[row.trip_id].append(row)
    times, scores = {}, []
    for identifier, rows in sorted(by_trip.items()):
        start = perf_counter()
        scores.extend(score_candidate_rows(model, calibrator, rows))
        times[identifier] = perf_counter() - start
    metrics = _subset_metrics(scores, data, trips, with_continuity=with_continuity)
    metrics.update(fit_metrics)
    timing = list(times.values())
    metrics["score_seconds"] = sum(timing)
    metrics["score_trip_median_seconds"] = statistics.median(timing) if timing else None
    metrics["score_trip_p95_seconds"] = float(np.percentile(timing, 95)) if timing else None
    for identifier, elapsed in times.items():
        metrics["per_trip"][identifier]["score_seconds"] = elapsed
    return metrics, scores


def run_cv(factory, params, seed, train, folds, calibrate=False, *, with_continuity=True):
    fold_results, out_of_fold = [], []
    for fold in range(5):
        train_folds = {key: value for key, value in folds.items() if value != fold}
        held_out = {key for key, value in folds.items() if value == fold}
        fitted = fit_model(factory, params, seed, train, train_folds, calibrate)
        metrics, scores = _score(fitted, train, held_out, with_continuity=False)
        train_metrics, _ = _score(fitted, train, train_folds, with_continuity=False)
        metrics.update(fold_id=fold, train_trips=list(sorted(train_folds)),
                       validation_trips=list(sorted(held_out)),
                       train_top1=train_metrics["top1"], train_macro_f1=train_metrics["macro_f1"])
        fold_results.append(metrics)
        out_of_fold.extend(scores)
    pooled = _subset_metrics(out_of_fold, train, folds, with_continuity=with_continuity)
    if pooled["continuity_pairs"]:
        for metrics in fold_results:
            counts = [pooled["per_trip"][key] for key in metrics["validation_trips"]]
            total = sum(item.get("continuity_pairs", 0) for item in counts)
            hits = sum(item.get("continuity_correct", 0) for item in counts)
            metrics.update(continuity=hits / total if total else None, continuity_pairs=total)
    for name in ("macro_f1", "top1"):
        values = [fold[name] for fold in fold_results if fold[name] is not None]
        pooled[f"fold_{name}_mean"] = statistics.fmean(values) if values else None
        pooled[f"fold_{name}_std"] = statistics.pstdev(values) if values else None
    return {"folds": fold_results, "pooled": pooled}, out_of_fold


def write_result(directory, result, **score_sets):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    for name, records in score_sets.items():
        with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, ("trip_id", "obs_id", "arc_id", "p_positive", "p_raw"))
            writer.writeheader()
            writer.writerows(records)


def run_info(manifest_path, spec):
    return {
        "command": [sys.executable, "-m", "ml.trainer", *sys.argv[1:]],
        "python": sys.version,
        "platform": platform.platform(),
        "manifest_version": spec["manifest_version"],
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "label_version": spec["label_version"],
        "feature_version": spec["feature_version"],
        "split_seed": spec["split_seed"],
        "feature_order": list(FEATURE_ORDER),
        "synthetic": spec.get("synthetic", False),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Exercise fixtures only")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    manifest_path = args.manifest or Path(config.get("manifest", "artifacts/data/manifest.json"))
    output = args.output or Path("experiments/results") / args.model
    spec, trips, train, student = load_inputs(manifest_path, args.synthetic)
    seed = int(config.get("seed", spec["split_seed"]))
    factory = model_factory(args.model)
    params = config.get("params", {})
    calibrate = config.get("calibrate", False)
    folds = {key: item["fold_id"] for key, item in trips["train"].items()}
    cv, out_of_fold = run_cv(factory, params, seed, train, folds, calibrate)
    fitted = fit_model(factory, params, seed, train, folds, calibrate)
    test, student_scores = _score(fitted, student, trips["student_test"])
    info = run_info(manifest_path, spec)
    if argv is not None:
        info["command"] = [sys.executable, "-m", "ml.trainer", *argv]
    result = {**info, "model": args.model, "seed": seed, "params": params,
              "calibrate": calibrate, "cv": cv, "student_test": test}
    write_result(output, result, cv_scores=out_of_fold, student_test_scores=student_scores)
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
