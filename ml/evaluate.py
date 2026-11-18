"""Metrics over candidate rows and source observations."""

from collections import defaultdict
from functools import lru_cache
import math

EPSILON = 1e-6


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _candidate_metrics(truth, probabilities):
    tn = fp = fn = tp = 0
    loss = 0.0
    for actual, probability in zip(truth, probabilities):
        predicted = probability >= 0.5
        if actual:
            tp += predicted
            fn += not predicted
        else:
            fp += predicted
            tn += not predicted
        p = min(1 - EPSILON, max(EPSILON, probability))
        loss -= math.log(p if actual else 1 - p)
    support = tn + fp + fn + tp
    return {
        "candidate_support": support,
        "positive_support": tp + fn,
        "accuracy": _ratio(tp + tn, support),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "macro_f1": (
            ((_ratio(2 * tp, 2 * tp + fp + fn) or 0)
             + (_ratio(2 * tn, 2 * tn + fp + fn) or 0)) / 2
            if support else None
        ),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "log_loss": _ratio(loss, support),
    }


def _continuity(scores, trip_dir):
    from map_matching.preprocess import load_trip

    trip = load_trip(trip_dir, include_reference=False)
    network = trip.network

    @lru_cache(maxsize=4096)
    def reachable(source, target):
        pending = [source]
        visited = {source}
        while pending:
            node = pending.pop()
            if node == target:
                return True
            for arc_id in network.outgoing_arc_ids[node]:
                destination = network.arcs[arc_id].to_node_id
                if destination not in visited:
                    visited.add(destination)
                    pending.append(destination)
        return False

    def fraction(obs_id, arc_id):
        px, py = trip.observation_xy[obs_id]
        segment = network.segments[arc_id]
        ax, ay = segment.start_xy
        bx, by = segment.end_xy
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if not length_squared:
            return 0.0
        return min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))

    continuous = 0
    pairs = zip(trip.observations, trip.observations[1:])
    for first, second in pairs:
        left = scores.get(first.obs_id)
        right = scores.get(second.obs_id)
        if left is None or right is None:
            continue
        if left == right and fraction(first.obs_id, left) <= fraction(second.obs_id, right):
            continuous += 1
            continue
        continuous += reachable(network.arcs[left].to_node_id, network.arcs[right].from_node_id)
    return continuous, max(0, len(trip.observations) - 1)


def evaluate(rows, observations, validated, geometry=None):
    grouped = defaultdict(dict)
    for row in rows:
        p = row["p_positive"]
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Expected finite probability in [0, 1]")
        grouped[row["trip_id"], row["obs_id"]][row["arc_id"]] = p

    per_trip = defaultdict(lambda: {
        "observations": 0, "validated": 0, "candidate_misses": 0,
        "top1_correct": 0, "empty_candidates": 0,
    })
    best_arcs = defaultdict(dict)
    truth, probabilities = [], []
    for key in observations:
        identifier, obs_id = key
        counts = per_trip[identifier]
        candidates = grouped.get(key, {})
        counts["observations"] += 1
        counts["empty_candidates"] += not candidates
        if candidates:
            best_arcs[identifier][obs_id] = min(candidates, key=lambda arc: (-candidates[arc], arc))
        if key not in validated:
            continue
        expected = validated[key]
        counts["validated"] += 1
        if expected not in candidates:
            counts["candidate_misses"] += 1
            continue
        counts["top1_correct"] += best_arcs[identifier][obs_id] == expected
        for arc, probability in candidates.items():
            truth.append(int(arc == expected))
            probabilities.append(probability)
    totals = {
        name: sum(counts[name] for counts in per_trip.values())
        for name in ("observations", "validated", "candidate_misses", "top1_correct", "empty_candidates")
    }
    eligible = []
    continuity_hits = continuity_pairs = 0
    for identifier, counts in per_trip.items():
        counts["top1"] = _ratio(counts["top1_correct"], counts["validated"])
        counts["candidate_miss_rate"] = _ratio(counts["candidate_misses"], counts["validated"])
        if counts["top1"] is not None:
            eligible.append(counts["top1"])
        if geometry and identifier in geometry:
            hits, pairs = _continuity(best_arcs[identifier], geometry[identifier])
            counts.update(continuity_correct=hits, continuity_pairs=pairs)
            continuity_hits += hits
            continuity_pairs += pairs
    return {
        **totals,
        **_candidate_metrics(truth, probabilities),
        "top1": _ratio(sum(eligible), len(eligible)),
        "top1_pooled": _ratio(totals["top1_correct"], totals["validated"]),
        "top1_trip_support": len(eligible),
        "candidate_miss_rate": _ratio(totals["candidate_misses"], totals["validated"]),
        "label_coverage": _ratio(totals["validated"], totals["observations"]),
        "continuity": _ratio(continuity_hits, continuity_pairs),
        "continuity_pairs": continuity_pairs,
        "per_trip": dict(per_trip),
    }
