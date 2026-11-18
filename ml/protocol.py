"""Load the Sprint 2 experiment inputs and enforce split and feature contracts."""

import csv
import json
import math
import re
from dataclasses import dataclass
from hashlib import file_digest
from importlib import import_module
from pathlib import Path

FEATURE_VERSION = "sprint2_v1"
FEATURE_ORDER = (
    "distance_m", "heading_diff_deg", "speed_mps", "time_gap_s", "predecessor_connected"
)
PARTITIONS = ("train", "student_test")
DATA_MANIFEST_VERSION = "sprint2-data-manifest-v1"
CANDIDATE_POLICY = {"radius_m": 100, "max_candidates": 8}


def trip_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{8}", value):
        raise ValueError(f"Invalid trip ID: {value!r}")
    return value


def source_id(value):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"Invalid source ID: {value!r}")
    return int(value)


def read_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def model_factory(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Invalid model name: {name!r}")
    return import_module(f"ml.models.{name}").build_model


def features(row):
    result = []
    for name in FEATURE_ORDER:
        value = row[name]
        if value is None or value == "":
            if name == "distance_m":
                raise ValueError("missing distance_m")
            result.append(math.nan)
            continue
        if name == "predecessor_connected":
            tokens = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
            value = tokens[str(value).lower()]
        number = float(value)
        if not math.isfinite(number) or number < 0 or (name == "heading_diff_deg" and number > 180):
            raise ValueError(f"Invalid {name}: {value!r}")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class Candidate:
    trip_id: str
    obs_id: int
    arc_id: int
    values: tuple

    @property
    def observation(self):
        return self.trip_id, self.obs_id


@dataclass
class Dataset:
    partition: str
    candidates: list
    observations: dict
    validated: dict
    geometry: dict


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _export_file(root, spec, filename):
    path = root / filename
    hashes = spec.get("outputs_sha256")
    if hashes is not None:
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        if hashes.get(filename) != digest:
            raise ValueError(f"Artifact hash mismatch: {filename}")
    return path


def _data_export(path, spec):
    if spec["contract_version"] != FEATURE_VERSION or spec["contract_frozen"] is not True:
        raise ValueError("The data export must use the frozen Sprint 2 v1 contract")
    if spec["evaluator_data_accessed"] is not False:
        raise ValueError("The training export must exclude evaluator data")
    if spec["candidate_settings"] != {**CANDIDATE_POLICY, "arc_id_tie_break": "ascending"}:
        raise ValueError("Unexpected candidate policy in the data export")
    fold_spec = read_config(_export_file(path.parent, spec, spec["folds_file"]))
    if fold_spec["fold_version"] != "trip-grouped-v1" or fold_spec["seed"] != spec["seed"]:
        raise ValueError("The fold file does not match the manifest")
    fold_by_trip = {}
    for fold in fold_spec["folds"]:
        for identifier in fold["validation_trip_ids"]:
            if identifier in fold_by_trip:
                raise ValueError(f"Trip appears more than once in the fold file: {identifier}")
            fold_by_trip[identifier] = fold["fold_id"]
    source = Path(spec["source_root"]).resolve()
    trips = []
    for row in _csv(_export_file(path.parent, spec, spec["trip_assignments_file"])):
        partition = row["split"]
        identifier = trip_id(row["trip_id"])
        if partition not in PARTITIONS:
            raise ValueError(f"Unexpected training or student-test partition: {partition}")
        fold = row["training_fold_id"]
        trips.append({
            "partition": partition,
            "trip_id": identifier,
            "fold_id": source_id(fold) if fold else None,
            "trip_dir": str(source / partition / identifier),
        })
    assigned_folds = {
        item["trip_id"]: item["fold_id"] for item in trips if item["partition"] == "train"
    }
    if assigned_folds != fold_by_trip:
        raise ValueError("The fold and trip assignment files disagree")
    tables = {
        partition: {kind: f"{kind}_{partition}.csv" for kind in ("features", "labels")}
        for partition in PARTITIONS
    }
    if any(name not in spec["outputs"] for files in tables.values() for name in files.values()):
        raise ValueError("The data export must declare the separate feature and label tables")
    for partition in PARTITIONS:
        if sum(item["partition"] == partition for item in trips) != spec["trip_count"][partition]:
            raise ValueError(f"Trip count in assignment file disagrees for {partition}")
    return {
        **spec, "feature_version": FEATURE_VERSION, "feature_order": FEATURE_ORDER,
        "split_seed": spec["seed"], "candidate_policy": CANDIDATE_POLICY,
        "trips": trips, "tables": tables,
    }


def manifest(path, *, synthetic=False, require_training_ready=True):
    path = Path(path)
    spec = read_config(path)
    if spec.get("synthetic", False) is not synthetic:
        raise ValueError("Synthetic fixtures require explicit --synthetic mode")
    if not synthetic and require_training_ready and spec.get("training_ready") is not True:
        reasons = "; ".join(spec.get("readiness_blockers", []))
        message = "Real runs require a training_ready manifest"
        raise ValueError(f"{message}: {reasons}" if reasons else message)
    if spec.get("manifest_version") == DATA_MANIFEST_VERSION:
        spec = _data_export(path, spec)
    if tuple(spec["feature_order"]) != FEATURE_ORDER or spec["feature_version"] != FEATURE_VERSION:
        raise ValueError("Unexpected feature schema version/order")
    by_partition = {name: {} for name in PARTITIONS}
    seen = set()
    for item in spec["trips"]:
        partition = item["partition"]
        identifier = trip_id(item["trip_id"])
        if partition not in by_partition or identifier in seen:
            raise ValueError("Invalid partition or duplicate trip in manifest")
        seen.add(identifier)
        fold = item["fold_id"]
        if partition == "train" and (type(fold) is not int or fold not in range(5)):
            raise ValueError("Training trips require fold IDs 0 through 4")
        if partition == "student_test" and fold is not None:
            raise ValueError("Student-test trips cannot have training folds")
        by_partition[partition][identifier] = item
    if {item["fold_id"] for item in by_partition["train"].values()} != set(range(5)):
        raise ValueError("Manifest must contain five nonempty training folds")
    return spec, by_partition


def dataset(root, spec, partition, trips):
    tables = spec["tables"][partition]
    observations = {}
    validated = {}
    for row in _csv(_export_file(root, spec, tables["labels"])):
        key = trip_id(row["trip_id"]), source_id(row["obs_id"])
        if key[0] not in trips or key in observations:
            raise ValueError(f"Unexpected or duplicate observation: {key}")
        status = row["label_status"]
        if not status:
            raise ValueError(f"Missing label status: {key}")
        observations[key] = status
        if status == "validated":
            if not row["review_evidence"].strip():
                raise ValueError(f"Missing review evidence: {key}")
            validated[key] = source_id(row["reference_arc_id"])
    if {key[0] for key in observations} != set(trips):
        raise ValueError("Label table must cover each manifest trip")
    candidates = []
    seen = set()
    for row in _csv(_export_file(root, spec, tables["features"])):
        identifier = trip_id(row["trip_id"])
        obs_id = source_id(row["obs_id"])
        arc_id = source_id(row["arc_id"])
        key = identifier, obs_id, arc_id
        if key in seen or key[:2] not in observations:
            raise ValueError(f"Unexpected or duplicate candidate: {key}")
        seen.add(key)
        candidates.append(Candidate(*key, features(row)))
    geometry = {
        key: root / item["trip_dir"] for key, item in trips.items() if item.get("trip_dir")
    }
    return Dataset(partition, candidates, observations, validated, geometry)
