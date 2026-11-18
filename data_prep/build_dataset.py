"""Build inference features, separate route-label proposals, and QA evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from map_matching.features import (
    FEATURE_CANDIDATE_RADIUS_M,
    FEATURE_MAX_CANDIDATES,
    CandidateFeature,
    feature_rows,
)
from map_matching.label_quality import (
    LABEL_VERSION,
    LabelQuality,
    ReviewDecision,
    label_rows,
    reference_break_rows,
    sanity_checks,
)
from map_matching.preprocess import load_trip

from .readiness import readiness, sha256_file

SPLITS = ("train", "student_test")
FEATURE_SCHEMA = tuple(field.name for field in fields(CandidateFeature))
LABEL_SCHEMA = tuple(field.name for field in fields(LabelQuality))
REVIEW_DECISION_FIELDS = (
    "trip_id", "obs_id", "decision", "reviewer", "review_evidence",
    "reference_arc_id", "route_step_id", "trip_input_sha256",
)
REVIEW_SUMMARY_FIELDS = (
    "split", "trip_id", "fold_id", "observations", "reviewed", "accepted", "rejected",
    "ambiguous", "unreviewed", "candidate_missing", "reference_candidate_missing",
    "fit_observations", "fit_trips", "positive_rows", "negative_rows", "trips_with_negatives",
    "validated_reference_misses",
)
REVIEW_QUEUE_FIELDS = (
    "split", "trip_id", "obs_id", "timestamp_s", "longitude", "latitude",
    "candidate_arc_ids", "proposed_route_step_id", "proposed_reference_arc_id",
    "aligned_distance_m", "alignment_status", "candidate_arc_present",
    "reference_route_discontinuous", "independent_review", "reviewer", "review_evidence",
    "fold_id", "priority", "review_reasons", "previous_gap_s", "trip_input_sha256",
)
BREAK_FIELDS = (
    "split", "trip_id", "break_before_route_step_id", "previous_arc_id", "current_arc_id",
    "previous_to_node_id", "current_from_node_id", "status",
)
TRIP_COVERAGE_FIELDS = (
    "split", "trip_id", "observations", "feature_rows", "observations_with_candidates",
    "empty_candidate_observations", "candidate_coverage", "alignment_proposals",
    "candidate_misses", "candidate_miss_rate", "ambiguous_proposals",
    "validated_labels", "unreviewed_labels", "reference_route_steps",
    "reference_route_breaks", "unique_route_arcs", "unique_route_arcs_in_candidates",
    "route_arc_candidate_coverage", "geometry_sanity_failures", "route_step_order_monotonic",
)


def _empty(value) -> str:
    return "" if value is None else value


def _write_row(writer: csv.DictWriter, record: dict) -> None:
    writer.writerow({key: _empty(value) for key, value in record.items()})


def _write_header(path: Path, columns: Iterable[str]) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    return stream, writer


def load_review_decisions(path: Path | None) -> dict[str, dict[int, tuple[ReviewDecision, str]]]:
    decisions = defaultdict(dict)
    if path is None:
        return decisions
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(REVIEW_DECISION_FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {', '.join(REVIEW_DECISION_FIELDS)}")
        for line_no, row in enumerate(reader, 2):
            try:
                identifier, observation, verdict, reviewer, evidence, arc, step, fingerprint = (
                    row[name].strip() for name in REVIEW_DECISION_FIELDS
                )
                if not re.fullmatch(r"[0-9]{8}", identifier):
                    raise ValueError("invalid trip ID")
                if not re.fullmatch(r"[0-9]+", observation):
                    raise ValueError("invalid observation ID")
                if verdict not in {"accepted", "rejected"}:
                    raise ValueError("decision must be accepted or rejected")
                if bool(arc) != bool(step) or ((arc or verdict == "accepted")
                        and not all(re.fullmatch(r"[0-9]+", value) for value in (arc, step))):
                    raise ValueError("expected the proposed arc and route step IDs")
                if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                    raise ValueError("expected the trip input SHA-256")
                obs_id = int(observation)
                if obs_id in decisions[identifier]:
                    raise ValueError(f"duplicate review decision for {identifier}/{obs_id}")
                decisions[identifier][obs_id] = (
                    ReviewDecision(identifier, obs_id, verdict, reviewer, evidence,
                                   int(arc) if arc else None, int(step) if step else None),
                    fingerprint,
                )
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return decisions


def hash_trip_inputs(trip_dir: Path) -> str:
    if trip_dir.parent.is_symlink() or trip_dir.is_symlink():
        raise ValueError(f"symbolic links are unsupported in trip {trip_dir.name}")
    trip_digest = hashlib.sha256()
    for suffix in (".arcs", ".nodes", ".route", ".track"):
        path = trip_dir / f"{trip_dir.name}{suffix}"
        if path.is_symlink():
            raise ValueError(f"symbolic links are unsupported in trip {trip_dir.name}")
        if not path.exists() and suffix == ".route":
            continue
        fingerprint = sha256_file(path)
        trip_digest.update(f"{path.name}\0{fingerprint}\n".encode())
    return trip_digest.hexdigest()


def update_input_digest(digest, split, trip_id, fingerprint):
    digest.update(f"{split}/{trip_id}\0{fingerprint}\n".encode())


def _folds(train_trip_ids: list[str], seed: int, fold_count: int = 5) -> dict:
    if len(train_trip_ids) < 2:
        raise ValueError("at least two training trips are required to build grouped folds")
    count = min(fold_count, len(train_trip_ids))
    shuffled = sorted(train_trip_ids)
    random.Random(seed).shuffle(shuffled)
    folds = [{"fold_id": i, "validation_trip_ids": []} for i in range(count)]
    for index, trip_id in enumerate(shuffled):
        folds[index % count]["validation_trip_ids"].append(trip_id)
    assigned = [trip_id for fold in folds for trip_id in fold["validation_trip_ids"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(train_trip_ids):
        raise RuntimeError("grouped folds do not form a disjoint cover of training trips")
    return {"fold_version": "trip-grouped-v1", "seed": seed, "folds": folds}


def _trip_review_rows(split, trip, rows, labels, breaks, fold_id, fingerprint):
    candidates = defaultdict(list)
    for row in rows:
        candidates[row.obs_id].append(row.arc_id)
    repeated = Counter(step.arc_id for step in trip.route or ())
    near_break = {step for row in breaks for step in (
        row["break_before_route_step_id"] - 1, row["break_before_route_step_id"]
    )}
    n = len(trip.observations)
    anchors = {0, n // 4, n // 2, 3 * n // 4, n - 1}
    quick_pick_used = False
    for index, (obs, label) in enumerate(zip(trip.observations, labels)):
        arc_ids = candidates[obs.obs_id]
        gap = obs.timestamp_s - trip.observations[index - 1].timestamp_s if index else None
        reasons = []
        if not arc_ids:
            reasons.append("no_candidates")
        if label.candidate_arc_present is False:
            reasons.append("reference_miss")
        if "ambiguous" in label.alignment_status:
            reasons.append(label.alignment_status)
        if label.route_step_id in near_break:
            reasons.append("near_reference_break")
        if repeated[label.reference_arc_id] > 1:
            reasons.append("repeated_route_arc")
        if gap is not None and (gap <= 0 or gap >= 5):
            reasons.append("timestamp_gap")
        if label.reference_route_discontinuous and index in anchors:
            reasons.append("discontinuous_trip_anchor")
        quick_pick = (not reasons and not quick_pick_used and label.alignment_status == "needs_review"
                      and len(arc_ids) >= 2 and arc_ids[0] == label.reference_arc_id)
        quick_pick_used |= quick_pick
        if quick_pick:
            reasons.append("quick_pick")
        if index in anchors:
            reasons.append("trip_anchor")
        if label.reviewer:
            reasons.append("review_recorded")
        if not reasons:
            continue
        priority = ("1_quick" if quick_pick else "2_diagnostic"
                    if set(reasons) - {"trip_anchor", "review_recorded"} else "3_anchor")
        yield {
            "split": split, "trip_id": trip.trip_id, "obs_id": obs.obs_id,
            "timestamp_s": obs.timestamp_s, "longitude": obs.lon, "latitude": obs.lat,
            "candidate_arc_ids": ";".join(map(str, arc_ids)),
            "proposed_route_step_id": label.route_step_id,
            "proposed_reference_arc_id": label.reference_arc_id,
            "aligned_distance_m": label.aligned_distance_m,
            "alignment_status": label.alignment_status,
            "candidate_arc_present": label.candidate_arc_present,
            "reference_route_discontinuous": label.reference_route_discontinuous,
            "independent_review": {"validated": "accepted", "rejected": "rejected"}.get(
                label.label_status, "pending"),
            "reviewer": label.reviewer, "review_evidence": label.review_evidence,
            "fold_id": fold_id, "priority": priority, "review_reasons": ";".join(reasons),
            "previous_gap_s": gap, "trip_input_sha256": fingerprint,
        }


def _review_summary(split, trip_id, fold_id, rows, labels):
    counts = Counter(row.obs_id for row in rows)
    accepted = [label for label in labels if label.label_status == "validated"]
    fit_sizes = [counts[label.obs_id] for label in accepted if label.candidate_arc_present]
    rejected = sum(label.label_status == "rejected" for label in labels)
    return dict(
        split=split, trip_id=trip_id, fold_id=fold_id, observations=len(labels),
        reviewed=len(accepted) + rejected, accepted=len(accepted), rejected=rejected,
        ambiguous=sum("ambiguous" in label.alignment_status for label in labels),
        unreviewed=len(labels) - len(accepted) - rejected,
        candidate_missing=len(labels) - len(counts),
        reference_candidate_missing=sum(label.candidate_arc_present is False for label in labels),
        fit_observations=len(fit_sizes), fit_trips=bool(fit_sizes),
        positive_rows=len(fit_sizes), negative_rows=sum(size - 1 for size in fit_sizes),
        trips_with_negatives=any(size > 1 for size in fit_sizes),
        validated_reference_misses=len(accepted) - len(fit_sizes),
    )


def _trip_metrics(split: str, trip, rows: list[CandidateFeature], labels: list[LabelQuality],
                  checks: list[dict], breaks: list[dict]) -> dict:
    obs_ids = {row.obs_id for row in rows}
    proposal_labels = [row for row in labels if row.reference_arc_id is not None]
    misses = sum(row.candidate_arc_present is False for row in proposal_labels)
    route_arc_ids = {step.arc_id for step in trip.route or ()}
    all_candidate_arc_ids = {row.arc_id for row in rows}
    ambiguous = sum("ambiguous" in row.alignment_status for row in proposal_labels)
    validated = sum(row.label_status == "validated" for row in labels)
    monotonic_steps = [row.route_step_id for row in labels if row.route_step_id is not None]
    monotonic = all(a <= b for a, b in zip(monotonic_steps, monotonic_steps[1:]))
    return {
        "split": split, "trip_id": trip.trip_id, "observations": len(trip.observations),
        "feature_rows": len(rows), "observations_with_candidates": len(obs_ids),
        "empty_candidate_observations": len(trip.observations) - len(obs_ids),
        "candidate_coverage": len(obs_ids) / len(trip.observations) if trip.observations else None,
        "alignment_proposals": len(proposal_labels), "candidate_misses": misses,
        "candidate_miss_rate": misses / len(proposal_labels) if proposal_labels else None,
        "ambiguous_proposals": ambiguous, "validated_labels": validated,
        "unreviewed_labels": sum(row.label_status not in {"validated", "rejected"} for row in labels),
        "reference_route_steps": len(trip.route) if trip.route is not None else None,
        "reference_route_breaks": len(breaks), "unique_route_arcs": len(route_arc_ids),
        "unique_route_arcs_in_candidates": len(route_arc_ids & all_candidate_arc_ids),
        "route_arc_candidate_coverage": len(route_arc_ids & all_candidate_arc_ids) / len(route_arc_ids) if route_arc_ids else None,
        "geometry_sanity_failures": sum(check["geometry_distance_consistent"] is False for check in checks),
        "route_step_order_monotonic": monotonic if monotonic_steps else None,
    }


def _split_metrics(rows: list[dict]) -> dict:
    observations = sum(row["observations"] for row in rows)
    covered = sum(row["observations_with_candidates"] for row in rows)
    proposals = sum(row["alignment_proposals"] for row in rows)
    misses = sum(row["candidate_misses"] for row in rows)
    return {
        "trips": len(rows), "observations": observations,
        "feature_rows": sum(row["feature_rows"] for row in rows),
        "observations_with_candidates": covered,
        "empty_candidate_observations": observations - covered,
        "candidate_coverage": covered / observations if observations else None,
        "alignment_proposals": proposals, "candidate_misses": misses,
        "candidate_miss_rate": misses / proposals if proposals else None,
        "ambiguous_proposals": sum(row["ambiguous_proposals"] for row in rows),
        "validated_labels": sum(row["validated_labels"] for row in rows),
        "reference_route_breaks": sum(row["reference_route_breaks"] for row in rows),
    }


def build_dataset(data_root: Path = Path("data"), output_dir: Path = Path("artifacts/data"),
                  *, seed: int = 2026, review_decisions_path: Path | None = None,
                  student_test_review_decisions_path: Path | None = None,
                  expected_counts: dict[str, int] | None = None) -> dict:
    """Generate separate feature/label tables, folds, manifest and QA evidence."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    if expected_counts is None:
        expected_counts = {"train": 80, "student_test": 10}
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    trip_dirs: dict[str, list[Path]] = {}
    for split in SPLITS:
        partition = data_root / split
        if partition.is_symlink():
            raise ValueError(f"symbolic links are unsupported in the {split} partition")
        if not partition.is_dir():
            raise FileNotFoundError(f"required development split not found: {partition}")
        trip_dirs[split] = sorted(path for path in partition.iterdir() if path.is_dir())
        expected = expected_counts.get(split) if expected_counts is not None else None
        if expected is not None and len(trip_dirs[split]) != expected:
            raise ValueError(f"{split} has {len(trip_dirs[split])} trips; expected {expected}")

    paths = {"train": review_decisions_path, "student_test": student_test_review_decisions_path}
    decisions = {split: load_review_decisions(paths[split]) for split in SPLITS}
    trip_ids = {split: {path.name for path in trip_dirs[split]} for split in SPLITS}
    if trip_ids["train"] & trip_ids["student_test"]:
        raise ValueError("train and student_test trip IDs overlap")
    for split, identifiers in trip_ids.items():
        if any(not re.fullmatch(r"[0-9]{8}", identifier) for identifier in identifiers):
            raise ValueError(f"invalid trip ID in {split}")
        unknown = decisions[split].keys() - identifiers
        if unknown:
            raise ValueError(f"{split} review decisions target other trips: {sorted(unknown)[:5]}")
    folds = _folds(sorted(trip_ids["train"]), seed)
    fold_by_trip_id = {
        trip_id: fold["fold_id"]
        for fold in folds["folds"] for trip_id in fold["validation_trip_ids"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".build-", dir=output_dir) as staging:
        stage = Path(staging)
        manifest = _export(data_root, stage, trip_dirs, decisions, paths, folds, fold_by_trip_id, seed)
        for name in (*manifest["outputs"], "manifest.json"):
            (stage / name).replace(output_dir / name)
    return manifest


def _export(data_root, output_dir, trip_dirs, decisions, paths, folds, fold_by_trip_id, seed):
    input_digest = hashlib.sha256()
    review_summary = []

    streams, writers = {}, {}
    try:
        for split in SPLITS:
            for kind, schema in (
                ("features", FEATURE_SCHEMA),
                ("labels", LABEL_SCHEMA),
            ):
                stream, writer = _write_header(output_dir / f"{kind}_{split}.csv", schema)
                streams[(kind, split)], writers[(kind, split)] = stream, writer

        trip_metrics: list[dict] = []
        break_records: list[dict] = []
        review_records: list[dict] = []
        sanity_records: list[dict] = []
        for split in SPLITS:
            for trip_dir in trip_dirs[split]:
                fingerprint = hash_trip_inputs(trip_dir)
                trip = load_trip(trip_dir)
                feature_rows_for_trip = feature_rows(
                    trip, FEATURE_CANDIDATE_RADIUS_M, FEATURE_MAX_CANDIDATES,
                )
                update_input_digest(input_digest, split, trip.trip_id, fingerprint)
                trip_decisions = {}
                for obs_id, (decision, reviewed_hash) in decisions[split].get(trip.trip_id, {}).items():
                    if reviewed_hash != fingerprint:
                        raise ValueError(f"review targets stale trip inputs: {trip.trip_id}/{obs_id}")
                    trip_decisions[obs_id] = decision
                label_records = label_rows(trip, feature_rows_for_trip, decisions=trip_decisions,
                                           evaluation_only=split == "student_test")
                fold_id = fold_by_trip_id.get(trip.trip_id)
                review_summary.append(_review_summary(split, trip.trip_id, fold_id,
                                                      feature_rows_for_trip, label_records))
                for row in feature_rows_for_trip:
                    _write_row(writers[("features", split)], asdict(row))
                for row in label_records:
                    _write_row(writers[("labels", split)], asdict(row))

                breaks = reference_break_rows(trip, split)
                break_records.extend(breaks)
                n = len(trip.observations)
                offsets = {0, n // 4, n // 2, 3 * n // 4, n - 1} if n else set()
                offsets.update(i for i, label in enumerate(label_records) if label.label_status == "validated")
                checks = sanity_checks(trip, label_records, tuple(sorted(offsets)))
                sanity_records.extend(checks)
                review_records.extend(_trip_review_rows(split, trip, feature_rows_for_trip, label_records,
                                                       breaks, fold_id, fingerprint))
                trip_metrics.append(_trip_metrics(split, trip, feature_rows_for_trip,
                                                  label_records, checks, breaks))
    finally:
        for stream in streams.values():
            stream.close()

    _write_table(output_dir / "coverage_by_trip.csv", TRIP_COVERAGE_FIELDS, trip_metrics)
    split_metrics = {split: _split_metrics([row for row in trip_metrics if row["split"] == split])
                     for split in SPLITS}
    _write_table(output_dir / "coverage_by_split.csv",
                 ("split", *tuple(next(iter(split_metrics.values())).keys())),
                 [{"split": split, **metrics} for split, metrics in split_metrics.items()])
    _write_table(output_dir / "reference_breaks.csv", BREAK_FIELDS, break_records)
    review_records.sort(key=lambda row: (SPLITS.index(row["split"]), row["priority"],
                                         row["fold_id"] if row["fold_id"] is not None else -1,
                                         row["trip_id"], row["obs_id"]))
    _write_table(output_dir / "review_queue.csv", REVIEW_QUEUE_FIELDS, review_records)
    _write_table(output_dir / "review_summary.csv", REVIEW_SUMMARY_FIELDS, review_summary)
    _write_table(output_dir / "sanity_checks.csv",
                 ("trip_id", "obs_id", "route_step_id", "reference_arc_id", "aligned_distance_m",
                  "label_status", "geometry_distance_consistent", "route_step_order_monotonic",
                  "independent_review", "review_evidence"), sanity_records)
    templates = ["review_decisions_template.csv", "student_test_review_decisions_template.csv"]
    for template in templates:
        _write_table(output_dir / template, REVIEW_DECISION_FIELDS, [])
    (output_dir / "folds.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    c0_contract = Path(__file__).resolve().parents[1] / "docs" / "contracts" / "sprint2_v1.md"
    assignments = [
        {"split": split, "trip_id": path.name,
         "training_fold_id": fold_by_trip_id.get(path.name) if split == "train" else None}
        for split in SPLITS for path in trip_dirs[split]
    ]
    _write_table(output_dir / "trip_assignments.csv",
                 ("split", "trip_id", "training_fold_id"), assignments)
    gate, blockers = readiness(review_summary, folds)
    _write_table(output_dir / "review_summary_by_fold.csv", REVIEW_SUMMARY_FIELDS, gate["fold_support"])
    if not c0_contract.is_file():
        blockers.append("C0 contract-v1 has not been merged; final schema freeze is pending")
    if any(row["geometry_sanity_failures"] or row["route_step_order_monotonic"] is False
           for row in trip_metrics if row["split"] == "train"):
        blockers.append("One or more geometry or route-order sanity checks failed")
    manifest = {
        "manifest_version": "sprint2-data-manifest-v1",
        "label_version": LABEL_VERSION,
        "contract_version": "sprint2_v1" if c0_contract.is_file() else "sprint2_v1-proposed",
        "contract_frozen": c0_contract.is_file(),
        "source_root": str(data_root),
        "evaluator_data_accessed": False,
        "data_fingerprint_sha256": input_digest.hexdigest(),
        "data_fingerprint_version": "trip-file-sha256-v2",
        "review_protocol_version": "proposal-bound-v2",
        "review_decisions_sha256": {split: sha256_file(path) if path is not None else None
                                    for split, path in paths.items()},
        "readiness": gate,
        "schema_sha256": hashlib.sha256(json.dumps({
            "feature_schema": FEATURE_SCHEMA, "label_schema": LABEL_SCHEMA,
        }, separators=(",", ":")).encode()).hexdigest(),
        "seed": seed,
        "candidate_settings": {"radius_m": FEATURE_CANDIDATE_RADIUS_M,
                               "max_candidates": FEATURE_MAX_CANDIDATES,
                               "arc_id_tie_break": "ascending"},
        "alignment_settings": {"ambiguity_margin_m": 2.0, "skip_penalty_m": 0.25,
                               "break_policy": "trip proposals with any discontinuity remain excluded pending independent review"},
        "partitions": split_metrics,
        "trip_count": {split: len(trip_dirs[split]) for split in SPLITS},
        "folds_file": "folds.json",
        "trip_assignments_file": "trip_assignments.csv",
        "outputs": ["features_train.csv", "labels_train.csv", "features_student_test.csv",
                    "labels_student_test.csv", "coverage_by_trip.csv", "coverage_by_split.csv",
                    "reference_breaks.csv", "review_queue.csv", "sanity_checks.csv", "folds.json",
                    "trip_assignments.csv", "review_summary.csv", "review_summary_by_fold.csv", *templates],
        "training_ready": not blockers,
        "readiness_blockers": blockers,
    }
    manifest["outputs_sha256"] = {name: sha256_file(output_dir / name)
                                  for name in manifest["outputs"]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_table(path: Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            _write_row(writer, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--student-test-review-decisions", type=Path)
    args = parser.parse_args()
    manifest = build_dataset(args.data, args.out, seed=args.seed,
                             review_decisions_path=args.review_decisions,
                             student_test_review_decisions_path=args.student_test_review_decisions)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
