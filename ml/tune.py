"""Compare five compatible real runs, tune on training trips and export the winner."""

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib

from .protocol import FEATURE_ORDER, FEATURE_VERSION, dataset, model_factory, read_config
from .serving import CandidateProbabilityModel
from .trainer import _score, fit_model, load_inputs, run_cv, score_candidate_rows, write_result

FAMILIES = {"logistic", "random_forest", "knn", "decision_tree", "xgboost"}
COMPARISON_METRICS = (
    "top1", "macro_f1", "log_loss", "log_loss_raw", "accuracy", "precision", "recall",
    "candidate_miss_rate", "continuity", "candidate_support", "validated",
    "top1_trip_support", "continuity_pairs",
)
CANDIDATE_POLICY = {"radius_m": 100, "max_candidates": 8}


def read_runs(paths, manifest_hash, folds):
    runs = {}
    for path in paths:
        run = read_config(path)
        model = run["model"]
        if model not in FAMILIES or model in runs:
            raise ValueError(f"Unexpected or duplicate family: {model}")
        if run.get("synthetic") is not False or run["manifest_sha256"] != manifest_hash:
            raise ValueError(f"{model} requires real results from the same manifest")
        seen = set()
        for fold in run["cv"]["folds"]:
            number = fold["fold_id"]
            validation = {key for key, value in folds.items() if value == number}
            if not validation or number in seen or set(fold["validation_trips"]) != validation:
                raise ValueError(f"Different validation split for {model}")
            if set(fold["train_trips"]) != set(folds) - validation:
                raise ValueError(f"Different training split for {model}")
            seen.add(number)
        if seen != set(range(5)):
            raise ValueError(f"Incomplete CV evidence for {model}")
        if any(run["cv"]["pooled"][metric] is None for metric in ("top1", "log_loss")):
            raise ValueError(f"Missing supported selection metrics for {model}")
        runs[model] = run
    if set(runs) != FAMILIES:
        raise ValueError(f"Missing model results: {', '.join(sorted(FAMILIES - set(runs)))}")
    first = next(iter(runs.values()))
    for run in runs.values():
        for actual, expected in ((run["cv"]["pooled"], first["cv"]["pooled"]),
                                 (run["student_test"], first["student_test"])):
            for field in ("observations", "validated", "candidate_support", "candidate_misses"):
                if actual[field] != expected[field]:
                    raise ValueError(f"Incompatible {field} support for {run['model']}")
    return runs


def select(runs):
    def key(run):
        pooled = run["cv"]["pooled"]
        timings = [fold["score_trip_median_seconds"] for fold in run["cv"]["folds"]
                   if fold["score_trip_median_seconds"] is not None]
        latency = sum(timings) / len(timings) if timings else float("inf")
        return -pooled["top1"], pooled["log_loss"], latency, run["model"]

    return min(runs.values(), key=key)


def tune(factory, params, seed, train, folds, calibrate, space, trials, journal):
    if train.partition != "train":
        raise ValueError("Bayesian objective requires the training partition")
    if not space or trials < 2:
        raise ValueError("Supply a nonempty search space and at least two trials")
    if journal.exists():
        raise FileExistsError(f"Trial journal already exists: {journal}")
    import optuna

    def objective(trial):
        candidate = dict(params)
        for name, domain in space.items():
            kind = domain["type"]
            if name in {"device", "seed", "random_state"}:
                raise ValueError(f"Search cannot override run infrastructure: {name}")
            if kind == "categorical":
                candidate[name] = trial.suggest_categorical(name, domain["choices"])
            elif kind in ("float", "int"):
                suggest = trial.suggest_float if kind == "float" else trial.suggest_int
                candidate[name] = suggest(name, domain["low"], domain["high"], log=domain.get("log", False))
            else:
                raise ValueError(f"Invalid parameter type for {name}: {kind}")
        cv, _ = run_cv(factory, candidate, seed, train, folds, calibrate, with_continuity=False)
        trial.set_user_attr("fold_top1", [fold["top1"] for fold in cv["folds"]])
        return cv["pooled"]["top1"]

    def log_trial(study, trial):
        with journal.open("a", encoding="utf-8") as handle:
            record = {"number": trial.number, "state": trial.state.name, "seed": seed,
                      "params": trial.params, "top1": trial.value,
                      "fold_top1": trial.user_attrs.get("fold_top1"),
                      "seconds": trial.duration.total_seconds() if trial.duration else None}
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=min(5, trials - 1)
    ))
    study.optimize(objective, n_trials=trials, callbacks=[log_trial], n_jobs=1)
    return {**params, **study.best_params}, study.best_value


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _dependencies():
    installed = {}
    for package in ("numpy", "scikit-learn", "xgboost", "joblib", "optuna", "matplotlib", "pyyaml", "torch"):
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            continue
    return installed


def compare(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(runs.values(), key=lambda run: run["model"])
    table = [
        {"model": run["model"], "cv_fold_top1_std": run["cv"]["pooled"]["fold_top1_std"],
         "cv_fold_macro_f1_std": run["cv"]["pooled"]["fold_macro_f1_std"],
         "cv_fit_seconds": sum(fold["fit_seconds"] for fold in run["cv"]["folds"]),
         "student_fit_seconds": run["student_test"]["fit_seconds"],
         "student_median_score_seconds": run["student_test"]["score_trip_median_seconds"],
         **{f"{split}_{key}": metrics[key] for split, metrics in
            (("cv", run["cv"]["pooled"]), ("student", run["student_test"]))
            for key in COMPARISON_METRICS}}
        for run in ordered
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, table[0].keys())
        writer.writeheader()
        writer.writerows(table)

    names = [row["model"] for row in table]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, metric, label in zip(
        axes, ("cv_top1", "student_top1", "student_log_loss"),
        ("Grouped CV Top-1", "Student-test Top-1", "Student-test log loss"),
    ):
        axis.barh(names, [row[metric] if row[metric] is not None else 0 for row in table])
        for index, row in enumerate(table):
            if row[metric] is None:
                axis.text(0, index, "N/A", va="center")
        axis.set_xlabel(label)
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)

    confusion = {run["model"]: {"cv": run["cv"]["pooled"]["confusion_matrix"],
                                "student_test": run["student_test"]["confusion_matrix"]}
                 for run in ordered}
    _json(output / "confusion_matrices.json", confusion)
    figure, axes = plt.subplots(2, len(names), figsize=(16, 6), layout="constrained")
    for index, name in enumerate(names):
        for axis, split in zip(axes[:, index], ("cv", "student_test")):
            counts = confusion[name][split]
            axis.imshow(counts, cmap="Blues")
            for actual in range(2):
                for predicted in range(2):
                    count = counts[actual][predicted]
                    color = "white" if count > max(map(max, counts)) / 2 else "black"
                    axis.text(predicted, actual, str(count), ha="center", va="center", color=color)
            axis.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
                     title=f"{name} / {split}")
    figure.savefig(output / "confusion_matrices.png", dpi=150)
    plt.close(figure)


def export_bundle(path, fitted, metadata):
    """Publish a complete bundle in one rename at a new versioned destination."""
    if path.exists():
        raise FileExistsError(f"Model bundle already exists: {path}")
    if metadata.get("candidate_policy", CANDIDATE_POLICY) != CANDIDATE_POLICY:
        raise ValueError("Model candidate policy differs from Sprint 2 v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        source = Path(temp) / "bundle"
        source.mkdir()
        model, calibrator, _ = fitted
        model_path = source / "model.joblib"
        joblib.dump(CandidateProbabilityModel(model, calibrator), model_path, compress=3)
        _json(source / "manifest.json", {
            **metadata, "bundle_version": 1, "feature_version": FEATURE_VERSION,
            "feature_columns": list(FEATURE_ORDER), "candidate_policy": CANDIDATE_POLICY,
            "pipeline_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        })
        source.rename(path)


def load_bundle(path):
    """Load a trusted exported bundle for reference-free candidate scoring."""
    path = Path(path)
    spec = read_config(path / "manifest.json")
    if (spec["bundle_version"] != 1 or spec["feature_version"] != FEATURE_VERSION
            or tuple(spec["feature_columns"]) != FEATURE_ORDER
            or spec["candidate_policy"] != CANDIDATE_POLICY):
        raise ValueError("Unsupported bundle or feature schema")
    model_path = path / "model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != spec["pipeline_sha256"]:
        raise ValueError("Pipeline checksum mismatch")
    return joblib.load(model_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs=5, required=True)
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)

    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    spec, trips, train, _ = load_inputs(args.manifest, training_only=True)
    if spec.get("candidate_policy", CANDIDATE_POLICY) != CANDIDATE_POLICY:
        raise ValueError("Data manifest candidate policy differs from Sprint 2 v1")
    folds = {key: item["fold_id"] for key, item in trips["train"].items()}
    runs = read_runs(args.runs, digest, folds)
    winner = select(runs)
    name, seed = winner["model"], winner["seed"]
    search = read_config(args.space)
    space = search[name]
    factory = model_factory(name)
    params, calibrate = winner["params"], winner.get("calibrate", False)
    if args.bundle.exists():
        raise FileExistsError(args.bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    compare(runs, args.output)

    start = perf_counter()
    best_params, objective = tune(factory, params, seed, train, folds, calibrate,
                                  space, args.trials, args.output / "trials.jsonl")
    tuning_seconds = perf_counter() - start
    cv, scores = run_cv(factory, best_params, seed, train, folds, calibrate)
    student = dataset(args.manifest.parent, spec, "student_test", trips["student_test"])
    fitted = fit_model(factory, best_params, seed, train, folds, calibrate)
    student_metrics, student_scores = _score(fitted, student, trips["student_test"])
    metadata = {
        "model": name, "seed": seed, "params": best_params, "calibrate": calibrate,
        "synthetic": False, "manifest_sha256": digest, "manifest_version": spec["manifest_version"],
        "label_version": spec["label_version"], "split_seed": spec["split_seed"],
        "python": platform.python_version(),
        "dependencies": _dependencies(),
        "command": [sys.executable, "-m", "ml.tune", *(sys.argv[1:] if argv is None else argv)],
    }
    result = {**metadata, "search_space": space, "trial_budget": args.trials,
              "tuning_seconds": tuning_seconds, "objective": objective,
              "untuned": {"cv": winner["cv"], "student_test": winner["student_test"]},
              "cv": cv, "student_test": student_metrics}
    write_result(args.output, result, cv_scores=scores, student_test_scores=student_scores)
    export_bundle(args.bundle, fitted, metadata)
    model = load_bundle(args.bundle)
    sample = student.candidates[:32]
    expected = score_candidate_rows(*fitted[:2], sample)
    restored = score_candidate_rows(model, None, sample)
    if [row["p_positive"] for row in restored] != [row["p_positive"] for row in expected]:
        raise RuntimeError("Exported pipeline failed probability parity")
    print(args.bundle / "manifest.json")


if __name__ == "__main__":
    main()
