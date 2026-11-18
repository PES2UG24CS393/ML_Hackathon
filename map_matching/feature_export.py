"""Generate inference features and a separate reference-route alignment report."""

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

from .features import CandidateFeature, RouteLabel, align_reference_route, feature_rows
from .preprocess import load_trip


def write_csv(path: Path, rows: list[dict], row_type):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(row_type)])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("trip_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--radius-m", type=float, default=100.0)
    parser.add_argument("--max-candidates", type=int, default=8)


def run_command(args: argparse.Namespace) -> None:

    trip = load_trip(args.trip_dir)
    features = feature_rows(trip, args.radius_m, args.max_candidates)
    labels = align_reference_route(trip, args.radius_m, candidate_rows=features)
    feature_path = args.output_dir / f"features_{trip.trip_id}.csv"
    labels_path = args.output_dir / f"alignment_{trip.trip_id}.csv"
    summary_path = args.output_dir / f"alignment_{trip.trip_id}.json"
    write_csv(feature_path, [asdict(row) for row in features], CandidateFeature)
    write_csv(labels_path, [asdict(row) for row in labels], RouteLabel)

    status_counts = Counter(label.status for label in labels)
    offsets = sorted({0, max(0, len(labels)//4), max(0, len(labels)//2),
                      max(0, 3*len(labels)//4), max(0, len(labels)-1)})
    spot_checks = [asdict(labels[index]) for index in offsets if labels]
    summary = {
        "trip_id": trip.trip_id,
        "feature_rows": len(features),
        "observations": len(trip.observations),
        "observations_without_candidates": len(trip.observations) - len({r.obs_id for r in features}),
        "candidate_radius_m": args.radius_m,
        "max_candidates_per_observation": args.max_candidates,
        "route_steps": len(trip.route) if trip.route is not None else None,
        "alignment_status_counts": dict(sorted(status_counts.items())),
        "alignment_rows_accepted_for_manual_review_only": status_counts.get("accepted", 0),
        "spot_check_rows": spot_checks,
        "training_label_readiness": "not ready: route alignment is a proposal; labels need manual map/sequence review and validation on multiple trips",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Trip {trip.trip_id}: {len(features)} candidate rows from {len(trip.observations)} observations")
    print(f"Candidate misses: {summary['observations_without_candidates']}")
    print(f"Alignment statuses: {dict(status_counts)}")
    print(f"Features: {feature_path}")
    print(f"Reference alignment (separate): {labels_path}")
    print(f"Alignment report: {summary_path}")
