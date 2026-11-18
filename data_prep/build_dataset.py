"""Build inference features, separate route-label proposals, and QA evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict, fields
from pathlib import Path
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

SPLITS = ("train", "student_test")
FEATURE_SCHEMA = tuple(field.name for field in fields(CandidateFeature))
LABEL_SCHEMA = tuple(field.name for field in fields(LabelQuality))
REVIEW_DECISION_FIELDS = ("trip_id", "obs_id", "decision", "reviewer", "review_evidence")
REVIEW_QUEUE_FIELDS = (
    "split", "trip_id", "obs_id", "timestamp_s", "longitude", "latitude",
    "candidate_arc_ids", "proposed_route_step_id", "proposed_reference_arc_id",
    "aligned_distance_m", "alignment_status", "candidate_arc_present",
    "reference_route_discontinuous", "independent_review", "reviewer", "review_evidence",
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


def load_review_decisions(path: Path | None) -> dict[tuple[str, int], ReviewDecision]:
    if path is None or not path.exists():
        return {}
    decisions: dict[tuple[str, int], ReviewDecision] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(REVIEW_DECISION_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"review decisions must have columns {', '.join(REVIEW_DECISION_FIELDS)}")
        for line_no, row in enumerate(reader, 2):
            try:
                obs_id = int(row["obs_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: obs_id must be an integer") from exc
            decision = ReviewDecision(
                trip_id=row["trip_id"].strip(), obs_id=obs_id,
                decision=row["decision"].strip().lower(), reviewer=row["reviewer"].strip(),
                review_evidence=row["review_evidence"].strip(),
            )
            key = (decision.trip_id, decision.obs_id)
            if key in decisions:
                raise ValueError(f"{path}:{line_no}: duplicate review decision for {key}")
            if decision.decision not in {"accepted", "rejected"}:
                raise ValueError(f"{path}:{line_no}: decision must be accepted or rejected")
            decisions[key] = decision
    return decisions


def _sha256_inputs(data_root: Path) -> str:
    digest = hashlib.sha256()
    for split in SPLITS:
        for trip_dir in sorted((data_root / split).iterdir()):
            if not trip_dir.is_dir():
                continue
            for path in sorted(trip_dir.iterdir()):
                if not path.is_file() or path.suffix not in {".track", ".nodes", ".arcs", ".route"}:
                    continue
                digest.update(f"{split}/{trip_dir.name}/{path.name}\0".encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest()


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


def _trip_review_rows(split: str, trip, feature_rows_for_trip: list[CandidateFeature],
                      labels: list[LabelQuality]) -> list[dict]:
    candidates_by_obs: dict[int, list[CandidateFeature]] = {}
    for row in feature_rows_for_trip:
        candidates_by_obs.setdefault(row.obs_id, []).append(row)
    label_by_obs = {row.obs_id: row for row in labels}
    n = len(trip.observations)
    anchors = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}) if n else []
    selected = set(anchors)
    # Include every sparse miss and ambiguous alignment, plus fixes adjacent to
    # directed reference breaks, so reviewers can inspect the failure modes.
    obs_index = {obs.obs_id: index for index, obs in enumerate(trip.observations)}
    for label in labels:
        if label.candidate_arc_present is False or "ambiguous" in label.alignment_status:
            selected.add(obs_index[label.obs_id])
    for break_row in reference_break_rows(trip, split):
        route_step = break_row["break_before_route_step_id"]
        for label in labels:
            if label.route_step_id in {route_step - 1, route_step}:
                selected.add(obs_index[label.obs_id])
    output = []
    for index in sorted(selected):
        obs = trip.observations[index]
        label = label_by_obs[obs.obs_id]
        candidates = candidates_by_obs.get(obs.obs_id, [])
        output.append({
            "split": split, "trip_id": trip.trip_id, "obs_id": obs.obs_id,
            "timestamp_s": obs.timestamp_s, "longitude": obs.lon, "latitude": obs.lat,
            "candidate_arc_ids": ";".join(str(row.arc_id) for row in candidates),
            "proposed_route_step_id": label.route_step_id,
            "proposed_reference_arc_id": label.reference_arc_id,
            "aligned_distance_m": label.aligned_distance_m,
            "alignment_status": label.alignment_status,
            "candidate_arc_present": label.candidate_arc_present,
            "reference_route_discontinuous": label.reference_route_discontinuous,
            "independent_review": "pending", "reviewer": "", "review_evidence": "",
        })
    return output


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
        if not partition.is_dir():
            raise FileNotFoundError(f"required development split not found: {partition}")
        trip_dirs[split] = sorted(path for path in partition.iterdir() if path.is_dir())
        expected = expected_counts.get(split) if expected_counts is not None else None
        if expected is not None and len(trip_dirs[split]) != expected:
            raise ValueError(f"{split} has {len(trip_dirs[split])} trips; expected {expected}")

    decisions = load_review_decisions(review_decisions_path)
    train_ids = {path.name for path in trip_dirs["train"]}
    unknown = {trip_id for trip_id, _ in decisions} - train_ids
    if unknown:
        raise ValueError(f"review decisions must target training trips only: {sorted(unknown)[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
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
                trip = load_trip(trip_dir)
                feature_rows_for_trip = feature_rows(
                    trip, FEATURE_CANDIDATE_RADIUS_M, FEATURE_MAX_CANDIDATES,
                )
                trip_decisions = ({obs_id: decision for (trip_id, obs_id), decision in decisions.items()
                                   if trip_id == trip.trip_id} if split == "train" else {})
                label_records = label_rows(trip, feature_rows_for_trip, decisions=trip_decisions)
                for row in feature_rows_for_trip:
                    _write_row(writers[("features", split)], asdict(row))
                for row in label_records:
                    _write_row(writers[("labels", split)], asdict(row))

                breaks = reference_break_rows(trip, split)
                break_records.extend(breaks)
                checks = sanity_checks(trip, label_records,
                    tuple(sorted({0, len(trip.observations) // 4, len(trip.observations) // 2,
                                  3 * len(trip.observations) // 4, len(trip.observations) - 1}))
                    if trip.observations else ())
                sanity_records.extend(checks)
                review_records.extend(_trip_review_rows(split, trip, feature_rows_for_trip, label_records))
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
    _write_table(output_dir / "review_queue.csv", REVIEW_QUEUE_FIELDS, review_records)
    _write_table(output_dir / "sanity_checks.csv",
                 ("trip_id", "obs_id", "route_step_id", "reference_arc_id", "aligned_distance_m",
                  "label_status", "geometry_distance_consistent", "route_step_order_monotonic",
                  "independent_review", "review_evidence"), sanity_records)
    if review_decisions_path is None or not Path(review_decisions_path).exists():
        _write_table(output_dir / "review_decisions_template.csv", REVIEW_DECISION_FIELDS, [])

    folds = _folds([path.name for path in trip_dirs["train"]], seed)
    (output_dir / "folds.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    c0_contract = Path(__file__).resolve().parents[1] / "docs" / "contracts" / "sprint2_v1.md"
    fold_by_trip_id = {
        trip_id: fold["fold_id"]
        for fold in folds["folds"] for trip_id in fold["validation_trip_ids"]
    }
    assignments = [
        {"split": split, "trip_id": path.name,
         "training_fold_id": fold_by_trip_id.get(path.name) if split == "train" else None}
        for split in SPLITS for path in trip_dirs[split]
    ]
    _write_table(output_dir / "trip_assignments.csv",
                 ("split", "trip_id", "training_fold_id"), assignments)
    total_validated = split_metrics["train"]["validated_labels"]
    blockers = []
    if not c0_contract.is_file():
        blockers.append("C0 contract-v1 has not been merged; final schema freeze is pending")
    if not total_validated:
        blockers.append("No independently reviewed training labels have status validated")
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
        "data_fingerprint_sha256": _sha256_inputs(data_root),
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
                    "trip_assignments.csv", "review_decisions_template.csv"],
        "training_ready": not blockers,
        "readiness_blockers": blockers,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_table(path: Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            _write_row(writer, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--review-decisions", type=Path)
    args = parser.parse_args()
    manifest = build_dataset(args.data, args.out, seed=args.seed,
                             review_decisions_path=args.review_decisions)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
