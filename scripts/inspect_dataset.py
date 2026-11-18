#!/usr/bin/env python3
"""Inventory and validate development trips without changing source data."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

EXTENSIONS = ("track", "nodes", "arcs", "route")


def inspect_trip(folder: Path, partition: str) -> dict:
    trip_id = folder.name
    row = {"partition": partition, "trip_id": trip_id, "track_rows": 0,
           "node_rows": 0, "arc_rows": 0, "route_rows": 0,
           "route_available": False, "missing_files": "", "empty_files": "",
           "malformed_records": 0, "invalid_coordinates": 0, "invalid_references": 0,
           "non_finite_values": 0,
           "timestamp_reversals": 0, "duplicate_timestamps": 0, "sampling_gaps_gt_2s": 0,
           "self_loops": 0, "duplicate_arcs": 0, "isolated_nodes": 0,
           "weak_components": 0, "status": "valid", "anomalies": ""}
    errors: list[str] = []
    files = {ext: folder / f"{trip_id}.{ext}" for ext in EXTENSIONS}
    missing, empty = [], []
    for ext, path in files.items():
        if not path.is_file():
            missing.append(ext)
            continue
        if path.stat().st_size == 0:
            empty.append(ext)
    row["missing_files"], row["empty_files"] = ";".join(missing), ";".join(empty)
    row["route_available"] = files["route"].is_file() and files["route"].stat().st_size > 0
    required_missing = [ext for ext in missing if ext != "route"]
    if required_missing:
        errors.append("missing:" + ",".join(required_missing))
    if empty:
        errors.append("empty:" + ",".join(empty))

    tables: dict[str, list[list[str]]] = {}
    widths = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}
    for ext, path in files.items():
        valid_rows: list[list[str]] = []
        if not path.is_file():
            tables[ext] = valid_rows
            continue
        try:
            with path.open(encoding="utf-8-sig") as stream:
                for line_no, raw in enumerate(stream, 1):
                    fields = raw.split()
                    if not fields:
                        continue
                    try:
                        if len(fields) != widths[ext]:
                            raise ValueError(f"expected {widths[ext]} columns")
                        nums = [float(token) for token in fields]
                        if not all(math.isfinite(value) for value in nums):
                            row["non_finite_values"] += sum(not math.isfinite(value) for value in nums)
                            if ext in ("track", "nodes"):
                                row["invalid_coordinates"] += 1
                            raise ValueError("non-finite value")
                        if ext in ("track", "nodes") and not (-180 <= nums[0] <= 180 and -90 <= nums[1] <= 90):
                            row["invalid_coordinates"] += 1
                            raise ValueError("coordinate out of bounds")
                        if ext in ("arcs", "route"):
                            try:
                                ids = [int(token) for token in fields]
                            except ValueError as exc:
                                raise ValueError("IDs must be integer tokens") from exc
                            if any(value < 0 for value in ids):
                                raise ValueError("IDs must be non-negative integers")
                        valid_rows.append(fields)
                    except ValueError as exc:
                        row["malformed_records"] += 1
                        errors.append(f"{ext}:{line_no}:{exc}")
        except (OSError, UnicodeError) as exc:
            errors.append(f"unreadable:{ext}:{exc}")
        tables[ext] = valid_rows
        row[{"track": "track_rows", "nodes": "node_rows", "arcs": "arc_rows", "route": "route_rows"}[ext]] = len(valid_rows)

    times = []
    for fields in tables["track"]:
        try:
            times.append(float(fields[2]))
        except ValueError:
            pass
    deltas = [b - a for a, b in zip(times, times[1:])]
    row["timestamp_reversals"] = sum(d < 0 for d in deltas)
    row["duplicate_timestamps"] = sum(count - 1 for count in Counter(times).values() if count > 1)
    row["sampling_gaps_gt_2s"] = sum(d > 2 for d in deltas)

    nodes = tables["nodes"]
    parent = list(range(len(nodes)))
    used = set()
    arcs_seen = Counter()
    for fields in tables["arcs"]:
        a, b = map(int, fields)
        if a >= len(nodes) or b >= len(nodes):
            row["invalid_references"] += 1
            continue
        used.update((a, b))
        if a == b:
            row["self_loops"] += 1
        if arcs_seen[(a, b)]:
            row["duplicate_arcs"] += 1
        arcs_seen[(a, b)] += 1
        # Union-find counts weak components while preserving directed arc rows.
        ra, rb = a, b
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        parent[ra] = rb
    row["isolated_nodes"] = len(nodes) - len(used)
    roots = set()
    for node in range(len(nodes)):
        root = node
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        roots.add(root)
    row["weak_components"] = len(roots)
    for fields in tables["route"]:
        if int(fields[0]) >= len(tables["arcs"]):
            row["invalid_references"] += 1
    if errors:
        row["status"] = "invalid"
        row["anomalies"] = " | ".join(errors[:20])
    elif any(row[key] for key in ("timestamp_reversals", "duplicate_timestamps", "sampling_gaps_gt_2s", "self_loops", "duplicate_arcs", "isolated_nodes", "invalid_references")):
        row["status"] = "valid_with_anomalies"
        row["anomalies"] = ";".join(f"{key}={row[key]}" for key in (
            "timestamp_reversals", "duplicate_timestamps", "sampling_gaps_gt_2s", "self_loops",
            "duplicate_arcs", "isolated_nodes", "invalid_references") if row[key])
    return row


def run(data_root: Path, output_dir: Path) -> dict:
    if not data_root.is_dir():
        raise FileNotFoundError(f"development dataset directory not found: {data_root}")
    rows = []
    for partition in ("train", "student_test"):
        partition_dir = data_root / partition
        if not partition_dir.is_dir():
            raise FileNotFoundError(f"required development partition not found: {partition_dir}")
        rows.extend(inspect_trip(folder, partition) for folder in sorted(partition_dir.iterdir()) if folder.is_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "dataset_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"data_root": str(data_root), "evaluator_partition_accessed": False,
               "partitions": {}, "trip_count": len(rows)}
    for part in ("train", "student_test"):
        subset = [r for r in rows if r["partition"] == part]
        summary["partitions"][part] = {
            "trips": len(subset), "track_observations": sum(r["track_rows"] for r in subset),
            "nodes": sum(r["node_rows"] for r in subset), "arcs": sum(r["arc_rows"] for r in subset),
            "route_rows": sum(r["route_rows"] for r in subset),
            "trips_with_route": sum(r["route_available"] for r in subset),
            "valid": sum(r["status"] == "valid" for r in subset),
            "valid_with_anomalies": sum(r["status"] == "valid_with_anomalies" for r in subset),
            "invalid": sum(r["status"] == "invalid" for r in subset),
            "missing_files": sum(bool(r["missing_files"]) for r in subset),
            "empty_files": sum(bool(r["empty_files"]) for r in subset),
            "anomaly_totals": {key: sum(r[key] for r in subset) for key in (
                "malformed_records", "invalid_coordinates", "non_finite_values", "invalid_references",
                "timestamp_reversals", "duplicate_timestamps", "sampling_gaps_gt_2s",
                "self_loops", "duplicate_arcs", "isolated_nodes")},
            "weak_components": {"total": sum(r["weak_components"] for r in subset),
                                "trips_with_multiple": sum(r["weak_components"] > 1 for r in subset),
                                "maximum_in_one_trip": max((r["weak_components"] for r in subset), default=0)},
        }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Student folder"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.out), indent=2))


if __name__ == "__main__":
    main()
