"""Load the Sprint 2 experiment inputs and enforce split and feature contracts."""

import csv
import json
import math
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

FEATURE_VERSION = "sprint2_v1"
FEATURE_ORDER = (
    "distance_m", "heading_diff_deg", "speed_mps", "time_gap_s", "predecessor_connected"
)
PARTITIONS = ("train", "student_test")


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


def manifest(path, *, synthetic=False):
    spec = read_config(path)
    if spec.get("synthetic", False) is not synthetic:
        raise ValueError("Synthetic fixtures require explicit --synthetic mode")
    if not synthetic and spec.get("training_ready") is not True:
        raise ValueError("Real runs require a training_ready manifest")
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
    for row in _csv(root / tables["labels"]):
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
    for row in _csv(root / tables["features"]):
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
