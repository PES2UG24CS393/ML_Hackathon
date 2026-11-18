"""Render evidence from a completed candidate-model experiment."""

import argparse
import json
from pathlib import Path


def render(report_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_path = Path(report_path)
    metrics = json.loads(report_path.read_text(encoding="utf-8"))
    cv = metrics["cv"]
    title = metrics["model"]
    if metrics["synthetic"]:
        title += " (synthetic fixture)"

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    figure.suptitle(title)
    for axis, name, values in zip(
        axes[:2], ("Training CV", "Student-test"),
        (cv["pooled"], metrics["student_test"]),
    ):
        counts = values["confusion_matrix"]
        axis.imshow(counts, cmap="Blues")
        cutoff = max(map(max, counts)) / 2
        for actual, row in enumerate(counts):
            for predicted, count in enumerate(row):
                axis.text(
                    predicted, actual, count, ha="center", va="center",
                    color="white" if count > cutoff else "black",
                )
        axis.set(
            title=f"{name} (n={values['candidate_support']:,})",
            xlabel="Predicted class", ylabel="Reference class",
            xticks=(0, 1), yticks=(0, 1),
        )
    axis = axes[2]
    for field, name in (("train_top1", "Fit trips"), ("top1", "Held-out trips")):
        fold_scores = [(fold["fold_id"] + 1, fold[field]) for fold in cv["folds"]
                       if fold[field] is not None]
        if fold_scores:
            fold_ids, scores = zip(*fold_scores)
            axis.plot(fold_ids, scores, marker="o", label=name)
    axis.set(title="Trip-macro Top-1 by fold", xlabel="Fold", xticks=range(1, 6), ylim=(0, 1.05))
    if axis.lines:
        axis.legend()
    path = report_path.with_name("model_evidence.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    print(render(parser.parse_args().metrics))
