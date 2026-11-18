"""Structural support checks for the frozen grouped training protocol."""

from hashlib import file_digest


MIN_FIT_OBSERVATIONS_PER_TRIP = 2


def sha256_file(path):
    with open(path, "rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def readiness(summaries, folds):
    train = {row["trip_id"]: row for row in summaries if row["split"] == "train"}
    counts = [key for key in next(iter(train.values())) if key not in {"split", "trip_id", "fold_id"}]
    support = []
    for fold in folds["folds"]:
        members = [train[trip_id] for trip_id in fold["validation_trip_ids"]]
        support.append(dict(split="train", fold_id=fold["fold_id"],
                            **{key: sum(row[key] for row in members) for key in counts}))

    blockers = []
    if not any(row["accepted"] for row in train.values()):
        blockers.append("No independently reviewed training labels have status validated")
    thin = [trip_id for trip_id, row in train.items()
            if row["fit_observations"] < MIN_FIT_OBSERVATIONS_PER_TRIP]
    if thin:
        blockers.append(f"{len(thin)} training trips need at least "
                        f"{MIN_FIT_OBSERVATIONS_PER_TRIP} validated candidate-present observations each")
    unsupported = [row["fold_id"] for row in support
                   if row["fit_trips"] < 2 or row["trips_with_negatives"] < 2]
    if len(support) != 5 or unsupported:
        blockers.append(f"Five folds require both candidate classes across at least two trips per fold; unsupported: {unsupported}")
    return {
        "policy": "grouped-support-v1",
        "minimum_fit_observations_per_training_trip": MIN_FIT_OBSERVATIONS_PER_TRIP,
        "training_trips_below_minimum": thin,
        "fold_support": support,
        "student_test_validated": sum(row["accepted"] for row in summaries
                                      if row["split"] == "student_test"),
    }, blockers
