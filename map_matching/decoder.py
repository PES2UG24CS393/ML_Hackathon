"""Log-space HMM/Viterbi decoder over directed road-segment candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, log
from time import perf_counter
from typing import Mapping, Sequence

from .graph_paths import CandidateTransition, transition_between_candidates
from .preprocess import Trip


@dataclass(frozen=True)
class DecoderConfig:
    max_search_m: float = 3000.0
    max_gap_s: float = 120.0
    max_speed_mps: float = 55.0
    speed_slack_m: float = 60.0
    transition_sigma_m: float = 75.0
    probability_floor: float = 1e-12


@dataclass(frozen=True)
class DecodedMatch:
    trip_id: str
    obs_id: int
    arc_id: int | None
    p_positive: float | None
    break_before: bool = False
    break_reason: str | None = None


@dataclass(frozen=True)
class DecodeResult:
    trip_id: str
    matches: tuple[DecodedMatch, ...]
    route_arc_ids: tuple[int, ...]
    route_parts: tuple[tuple[int, ...], ...]
    breaks: tuple[dict, ...]
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "matches": [asdict(match) for match in self.matches],
            "route_arc_ids": list(self.route_arc_ids),
            "route_parts": [list(part) for part in self.route_parts],
            "breaks": list(self.breaks),
            "elapsed_ms": self.elapsed_ms,
        }


def decode_trip(trip: Trip, candidate_scores: Sequence[Mapping],
                config: DecoderConfig | Mapping | None = None) -> DecodeResult:
    """Decode `(trip_id, obs_id, arc_id, p_positive)` rows without reading references.

    Missing candidate sets, nonpositive timestamps, long temporal gaps, and
    directed-unreachable transitions split route parts. The decoder never
    inserts a road connection across a break.
    """
    started = perf_counter()
    cfg = _coerce_config(config)
    observations = trip.observations
    obs_index = {obs.obs_id: i for i, obs in enumerate(observations)}
    scores_by_index: list[dict[int, float]] = [dict() for _ in observations]
    for row in candidate_scores:
        trip_id = _field(row, "trip_id")
        obs_id = _field(row, "obs_id")
        arc_id = _field(row, "arc_id")
        probability = _field(row, "p_positive")
        if trip_id != trip.trip_id:
            raise ValueError(f"candidate score trip_id {trip_id!r} does not match {trip.trip_id!r}")
        if not isinstance(obs_id, int) or obs_id not in obs_index:
            raise ValueError(f"candidate score has unknown obs_id {obs_id!r}")
        if not isinstance(arc_id, int) or not 0 <= arc_id < len(trip.network.arcs):
            raise ValueError(f"candidate score has unknown arc_id {arc_id!r}")
        probability = float(probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("p_positive must be finite and in [0, 1]")
        index = obs_index[obs_id]
        if arc_id in scores_by_index[index]:
            raise ValueError(f"duplicate score for observation {obs_id}, arc {arc_id}")
        scores_by_index[index][arc_id] = probability

    states = [sorted(scores.items()) for scores in scores_by_index]
    transitions: dict[int, dict[tuple[int, int], tuple[float, CandidateTransition]]] = {}
    split_before: dict[int, str] = {}
    blocks: list[list[int]] = []
    current_block: list[int] = []
    pending_reason: str | None = None
    previous_valid_index: int | None = None
    graph_cache: dict = {}
    for index, candidates in enumerate(states):
        if not candidates:
            if current_block:
                blocks.append(current_block)
                current_block = []
            split_before[index] = "missing_candidates"
            pending_reason = "missing_candidates"
            continue
        if not current_block:
            if index > 0:
                reason = pending_reason or "sequence_break"
                if previous_valid_index is not None:
                    dt = observations[index].timestamp_s - observations[previous_valid_index].timestamp_s
                    if dt <= 0 or not isfinite(dt):
                        reason = "nonpositive_time_delta"
                    elif dt > cfg.max_gap_s:
                        reason = "long_time_gap"
                split_before[index] = reason
            current_block = [index]
            pending_reason = None
            previous_valid_index = index
            continue
        previous_index = current_block[-1]
        dt = observations[index].timestamp_s - observations[previous_index].timestamp_s
        if dt <= 0 or not isfinite(dt):
            reason = "nonpositive_time_delta"
        elif dt > cfg.max_gap_s:
            reason = "long_time_gap"
        else:
            transition_table = _transition_table(
                trip, observations[previous_index], observations[index],
                states[previous_index], candidates, dt, cfg, graph_cache,
            )
            transitions[index] = transition_table
            if transition_table:
                current_block.append(index)
                previous_valid_index = index
                continue
            reason = "no_feasible_directed_transition"
        blocks.append(current_block)
        current_block = [index]
        split_before[index] = reason
        pending_reason = None
        previous_valid_index = index
    if current_block:
        blocks.append(current_block)

    # Viterbi may find no complete path even when every adjacent pair has at
    # least one edge. Split at that first unreachable observation and restart.
    decoded_blocks: list[tuple[list[int], list[int]]] = []
    queue = list(blocks)
    while queue:
        block = queue.pop(0)
        solved, split_at = _viterbi_block(block, states, transitions, cfg.probability_floor)
        if split_at is None:
            decoded_blocks.append((block, solved))
        else:
            left, right = block[:split_at], block[split_at:]
            split_before[right[0]] = "no_viterbi_path_from_previous_states"
            if left:
                queue.insert(0, left)
            if right:
                queue.insert(1 if left else 0, right)
    decoded_blocks.sort(key=lambda item: item[0][0])

    matches: list[DecodedMatch] = []
    for i, obs in enumerate(observations):
        reason = split_before.get(i)
        matches.append(DecodedMatch(trip.trip_id, obs.obs_id, None, None,
                                    break_before=bool(reason), break_reason=reason))
    route_parts: list[tuple[int, ...]] = []
    breaks = [{"before_obs_id": observations[index].obs_id,
               "previous_obs_id": observations[index - 1].obs_id if index else None,
               "reason": reason}
              for index, reason in sorted(split_before.items())]
    for block, selected_arc_ids in decoded_blocks:
        start = block[0]
        reason = split_before.get(start)
        for index, arc_id in zip(block, selected_arc_ids):
            matches[index] = DecodedMatch(
                trip.trip_id, observations[index].obs_id, arc_id,
                dict(states[index])[arc_id], break_before=(index == start and bool(reason)),
                break_reason=reason if index == start else None,
            )
        route = [selected_arc_ids[0]]
        for position in range(1, len(block)):
            index = block[position]
            previous_index = block[position - 1]
            previous_arc = selected_arc_ids[position - 1]
            current_arc = selected_arc_ids[position]
            _, transition = transitions[index][(previous_arc, current_arc)]
            if transition.same_arc_forward:
                continue
            route.extend(transition.connector_arc_ids)
            route.append(current_arc)
        if route:
            route_parts.append(tuple(route))
    flat_route = tuple(arc_id for part in route_parts for arc_id in part)
    return DecodeResult(trip.trip_id, tuple(matches), flat_route, tuple(route_parts),
                        tuple(breaks), round((perf_counter() - started) * 1000.0, 3))


def compare_classifier_and_hmm(
    trip: Trip,
    candidate_scores: Sequence[Mapping],
    decoded: DecodeResult,
    reference_arc_by_obs: Mapping[int, int] | None = None,
    reference_route_arc_ids: Sequence[int] | None = None,
    config: DecoderConfig | Mapping | None = None,
) -> dict:
    """Development-only matched-support metrics; references are explicit inputs.

    References are never read by ``decode_trip``. Callers should use reviewed
    training labels and report student-test metrics separately.
    """
    cfg = _coerce_config(config)
    by_obs: dict[int, dict[int, float]] = {}
    for row in candidate_scores:
        if _field(row, "trip_id") == trip.trip_id:
            by_obs.setdefault(int(_field(row, "obs_id")), {})[int(_field(row, "arc_id"))] = float(_field(row, "p_positive"))
    labels = reference_arc_by_obs or {}
    decoded_by_obs = {match.obs_id: match.arc_id for match in decoded.matches}
    eligible = sorted(obs_id for obs_id in labels if obs_id in by_obs and obs_id in decoded_by_obs)
    classifier_correct = sum(min(by_obs[obs_id], key=lambda arc: (-by_obs[obs_id][arc], arc)) == labels[obs_id]
                             for obs_id in eligible)
    hmm_correct = sum(decoded_by_obs[obs_id] == labels[obs_id] for obs_id in eligible)
    adjacent = [(i, i + 1) for i in range(max(0, len(trip.observations) - 1))]
    match_by_id = {match.obs_id: match.arc_id for match in decoded.matches}
    classifier_top1 = {obs_id: (min(scores, key=lambda arc: (-scores[arc], arc)) if scores else None)
                       for obs_id, scores in by_obs.items()}
    classifier_continuity = _continuity(trip, adjacent, classifier_top1, cfg)
    hmm_continuity = _continuity(trip, adjacent, match_by_id, cfg)
    reference_route = list(reference_route_arc_ids) if reference_route_arc_ids is not None else None
    hmm_sequence = list(decoded.route_arc_ids)
    return {
        "trip_id": trip.trip_id,
        "eligible_observation_count": len(eligible),
        "classifier_only_top1": {"correct": classifier_correct, "support": len(eligible),
                                 "accuracy": classifier_correct / len(eligible) if eligible else None},
        "hmm_top1": {"correct": hmm_correct, "support": len(eligible),
                     "accuracy": hmm_correct / len(eligible) if eligible else None},
        "classifier_only_continuity": classifier_continuity,
        "hmm_continuity": hmm_continuity,
        "hmm_sequence_edit_distance": _sequence_metric(hmm_sequence, reference_route),
        "decode_elapsed_ms": decoded.elapsed_ms,
        "unmatched_observations": sum(match.arc_id is None for match in decoded.matches),
        "break_count": len(decoded.breaks),
        "break_reasons": [item["reason"] for item in decoded.breaks],
    }


def _transition_table(trip, previous_obs, current_obs, previous_states, current_states,
                      dt, config, graph_cache):
    max_travel = min(config.max_search_m, config.max_speed_mps * dt + config.speed_slack_m)
    gps_distance = _distance(trip.observation_xy[previous_obs.obs_id],
                             trip.observation_xy[current_obs.obs_id])
    result = {}
    for previous_arc, _ in previous_states:
        for current_arc, _ in current_states:
            transition = transition_between_candidates(
                trip, previous_arc, trip.observation_xy[previous_obs.obs_id],
                current_arc, trip.observation_xy[current_obs.obs_id], max_travel,
                cache=graph_cache,
            )
            if transition is None:
                continue
            score = -abs(transition.distance_m - gps_distance) / config.transition_sigma_m
            result[(previous_arc, current_arc)] = (score, transition)
    return result


def _viterbi_block(block, states, transitions, probability_floor):
    scores = [log(max(probability, probability_floor)) for _, probability in states[block[0]]]
    backpointers: list[list[int]] = []
    for index in block[1:]:
        transition_table = transitions[index]
        current_scores = []
        parents = []
        for current_arc, probability in states[index]:
            best_score, best_parent = float("-inf"), -1
            for parent_index, (previous_arc, _) in enumerate(states[index - 1]):
                link = transition_table.get((previous_arc, current_arc))
                if link is None or scores[parent_index] == float("-inf"):
                    continue
                candidate = scores[parent_index] + link[0]
                if candidate > best_score:
                    best_score, best_parent = candidate, parent_index
            if best_parent < 0:
                current_scores.append(float("-inf"))
            else:
                current_scores.append(best_score + log(max(probability, probability_floor)))
            parents.append(best_parent)
        if all(score == float("-inf") for score in current_scores):
            return [], block.index(index)
        scores = current_scores
        backpointers.append(parents)
    state_index = max(range(len(scores)), key=lambda i: (scores[i], -states[block[-1]][i][0]))
    selected = [0] * len(block)
    selected[-1] = states[block[-1]][state_index][0]
    for position in range(len(block) - 1, 0, -1):
        state_index = backpointers[position - 1][state_index]
        selected[position - 1] = states[block[position - 1]][state_index][0]
    return selected, None


def _coerce_config(config) -> DecoderConfig:
    if config is None:
        result = DecoderConfig()
    elif isinstance(config, DecoderConfig):
        result = config
    elif isinstance(config, Mapping):
        result = DecoderConfig(**config)
    else:
        raise TypeError("config must be DecoderConfig, a mapping, or None")
    for name in ("max_search_m", "max_gap_s", "max_speed_mps", "transition_sigma_m", "probability_floor"):
        value = getattr(result, name)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isfinite(result.speed_slack_m) or result.speed_slack_m < 0:
        raise ValueError("speed_slack_m must be finite and nonnegative")
    if result.probability_floor >= 1:
        raise ValueError("probability_floor must be less than 1")
    return result


def _field(row, name):
    if isinstance(row, Mapping):
        if name not in row:
            raise ValueError(f"candidate score is missing {name}")
        return row[name]
    if not hasattr(row, name):
        raise ValueError(f"candidate score is missing {name}")
    return getattr(row, name)


def _continuity(trip, pairs, predictions, config):
    segment_by_id = trip.network.segments
    arcs = trip.network.arcs
    continuous = 0
    total = len(pairs)
    reasons = {"unmatched": 0, "time_gap": 0, "unreachable": 0}
    cache = {}
    for left_index, right_index in pairs:
        left = trip.observations[left_index]
        right = trip.observations[right_index]
        left_arc = predictions.get(left.obs_id)
        right_arc = predictions.get(right.obs_id)
        if left_arc is None or right_arc is None:
            reasons["unmatched"] += 1
            continue
        dt = right.timestamp_s - left.timestamp_s
        if dt <= 0 or dt > config.max_gap_s:
            reasons["time_gap"] += 1
            continue
        max_travel = min(config.max_search_m, config.max_speed_mps * dt + config.speed_slack_m)
        path = transition_between_candidates(
            trip, left_arc, trip.observation_xy[left_index], right_arc,
            trip.observation_xy[right_index], max_travel, cache=cache,
        )
        if path is None:
            reasons["unreachable"] += 1
        else:
            continuous += 1
    return {"continuous": continuous, "pairs": total,
            "rate": continuous / total if total else None, "failure_reasons": reasons}


def _sequence_metric(predicted, reference):
    if reference is None:
        return {"normalized_edit_distance": None, "predicted_length": len(predicted), "reference_length": None}
    rows = list(range(len(reference) + 1))
    for i, pred_arc in enumerate(predicted, start=1):
        current = [i]
        for j, ref_arc in enumerate(reference, start=1):
            current.append(min(current[-1] + 1, rows[j] + 1, rows[j - 1] + (pred_arc != ref_arc)))
        rows = current
    distance = rows[-1]
    denominator = max(1, len(reference))
    return {"normalized_edit_distance": distance / denominator,
            "edit_distance": distance, "predicted_length": len(predicted),
            "reference_length": len(reference)}


def _distance(first, second):
    return ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5
