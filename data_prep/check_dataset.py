"""Verify generated hashes and load both splits without fitting any model."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from ml.protocol import dataset, manifest

from .build_dataset import SPLITS, hash_trip_inputs, update_input_digest
from .readiness import sha256_file


def check(path):
    spec, trips = manifest(path, require_training_ready=False)
    hashes = spec["outputs_sha256"]
    if set(hashes) != set(spec["outputs"]):
        raise ValueError("Manifest hashes do not cover declared outputs")
    for name, digest in hashes.items():
        if sha256_file(path.parent / name) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")

    source_digest = sha256()
    for split in SPLITS:
        for identifier, assignment in sorted(trips[split].items()):
            trip_dir = Path(assignment["trip_dir"])
            update_input_digest(source_digest, split, identifier, hash_trip_inputs(trip_dir))
    if source_digest.hexdigest() != spec["data_fingerprint_sha256"]:
        raise ValueError("Development inputs differ from the export fingerprint")

    results = {}
    for split, assignments in trips.items():
        loaded = dataset(path.parent, spec, split, assignments)
        actual = dict(trips=len(assignments), observations=len(loaded.observations),
                      feature_rows=len(loaded.candidates), validated_labels=len(loaded.validated))
        if any(spec["partitions"][split][key] != value for key, value in actual.items()):
            raise ValueError(f"Loaded table counts disagree with manifest for {split}")
        results[split] = actual
    return dict(partitions=results, training_ready=spec["training_ready"],
                readiness_blockers=spec["readiness_blockers"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    parser.add_argument("--require-training-ready", action="store_true")
    args = parser.parse_args()
    result = check(args.manifest)
    print(json.dumps(result, indent=2))
    if args.require_training_ready and not result["training_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
