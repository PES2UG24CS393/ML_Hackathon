"""Conservative label validation helpers for route-derived supervision.

Route rows are proposals only. A label is fit-eligible only after an independent
review decision records a reviewer and evidence URI; feature extraction stays
separate and route-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose

from .features import CandidateFeature, RouteLabel, align_reference_route, reference_route_breaks
from .preprocess import Trip

LABEL_VERSION = "sprint2-route-labels-v1"
DATASET_OWNER = "Rajath-N"


@dataclass(frozen=True)
class ReviewDecision:
    trip_id: str
    obs_id: int
    decision: str
    reviewer: str
    review_evidence: str
    reference_arc_id: int | None = None
    route_step_id: int | None = None


@dataclass(frozen=True)
class LabelQuality:
    trip_id: str
    obs_id: int
    route_step_id: int | None
    reference_arc_id: int | None
    aligned_distance_m: float | None
    label_status: str
    alignment_status: str
    candidate_arc_present: bool | None
    reference_route_discontinuous: bool
    review_evidence: str
    reviewer: str


def validate_decision(decision: ReviewDecision, *, candidate_arc_ids: set[int],
                      proposed_arc_id: int | None, aligned_distance_m: float | None,
                      radius_m: float, allow_candidate_miss: bool = False) -> None:
    """Reject validation decisions that lack provenance or candidate support."""
    if decision.decision not in {"accepted", "rejected"}:
        raise ValueError("review decision must be 'accepted' or 'rejected'")
    if not decision.reviewer.strip() or decision.reviewer.strip().casefold() == DATASET_OWNER.casefold():
        raise ValueError("reviewer must be an independent reviewer, not the data owner")
    if not decision.review_evidence.strip():
        raise ValueError("review_evidence must identify a review artifact or record")
    if decision.decision == "accepted":
        if proposed_arc_id is None or aligned_distance_m is None:
            raise ValueError("cannot accept a row without a route-derived arc proposal")
        if proposed_arc_id not in candidate_arc_ids and not allow_candidate_miss:
            raise ValueError("cannot accept a reference arc missing from the candidate set")
        if aligned_distance_m > radius_m and not allow_candidate_miss:
            raise ValueError("cannot accept an alignment outside the frozen label radius")


def label_rows(trip: Trip, candidate_rows: list[CandidateFeature], *,
               radius_m: float = 100.0, ambiguity_margin_m: float = 2.0,
               skip_penalty_m: float = 0.25,
               decisions: dict[int, ReviewDecision] | None = None,
               evaluation_only: bool = False) -> list[LabelQuality]:
    """Return one auditable label/proposal row per source observation."""
    decisions = decisions or {}
    proposals: list[RouteLabel] = align_reference_route(
        trip, labels_radius_m=radius_m, ambiguity_margin_m=ambiguity_margin_m,
        skip_penalty_m=skip_penalty_m, candidate_rows=candidate_rows,
    )
    candidates_by_obs: dict[int, set[int]] = {}
    for row in candidate_rows:
        candidates_by_obs.setdefault(row.obs_id, set()).add(row.arc_id)
    broken_steps = set(reference_route_breaks(trip))
    result: list[LabelQuality] = []
    for proposal in proposals:
        decision = decisions.get(proposal.obs_id)
        status = proposal.status
        evidence = ""
        reviewer = ""
        if decision is not None:
            if decision.trip_id != trip.trip_id or decision.obs_id != proposal.obs_id:
                raise ValueError("review decision key does not match its label row")
            if decision.reference_arc_id is not None and (
                decision.reference_arc_id, decision.route_step_id
            ) != (proposal.reference_arc_id, proposal.route_step_id):
                raise ValueError(f"review targets a stale proposal: {trip.trip_id}/{proposal.obs_id}")
            validate_decision(
                decision,
                candidate_arc_ids=candidates_by_obs.get(proposal.obs_id, set()),
                proposed_arc_id=proposal.reference_arc_id,
                aligned_distance_m=proposal.aligned_distance_m,
                radius_m=radius_m,
                allow_candidate_miss=evaluation_only,
            )
            status = "validated" if decision.decision == "accepted" else "rejected"
            evidence = decision.review_evidence.strip()
            reviewer = decision.reviewer.strip()
        result.append(LabelQuality(
            trip_id=proposal.trip_id,
            obs_id=proposal.obs_id,
            route_step_id=proposal.route_step_id,
            reference_arc_id=proposal.reference_arc_id,
            aligned_distance_m=proposal.aligned_distance_m,
            label_status=status,
            alignment_status=proposal.status,
            candidate_arc_present=(proposal.reference_arc_id in candidates_by_obs.get(proposal.obs_id, set())
                                   if proposal.reference_arc_id is not None else None),
            reference_route_discontinuous=bool(broken_steps),
            review_evidence=evidence,
            reviewer=reviewer,
        ))
    unknown_decisions = set(decisions) - {row.obs_id for row in proposals}
    if unknown_decisions:
        raise ValueError(f"review decisions reference unknown observations: {sorted(unknown_decisions)[:5]}")
    return result


def reference_break_rows(trip: Trip, split: str) -> list[dict]:
    """Describe every directed discontinuity without silently repairing it."""
    if not trip.route:
        return []
    arcs = {arc.arc_id: arc for arc in trip.network.arcs}
    rows = []
    for step_id in reference_route_breaks(trip):
        previous = trip.route[step_id - 1]
        current = trip.route[step_id]
        rows.append({
            "split": split,
            "trip_id": trip.trip_id,
            "break_before_route_step_id": step_id,
            "previous_arc_id": previous.arc_id,
            "current_arc_id": current.arc_id,
            "previous_to_node_id": arcs[previous.arc_id].to_node_id,
            "current_from_node_id": arcs[current.arc_id].from_node_id,
            "status": "discontinuous_reference_route",
        })
    return rows


def sanity_checks(trip: Trip, labels: list[LabelQuality], offsets: tuple[int, ...]) -> list[dict]:
    """Geometry and sequence checks for a stable, reviewable sample set.

    These checks are mechanical evidence; they do not replace the independent
    geographic/sequence reviewer required before training labels are released.
    """
    by_obs = {row.obs_id: row for row in labels}
    arcs = {segment.arc_id: segment for segment in trip.network.segments}
    observation_by_id = {obs.obs_id: (obs, xy) for obs, xy in zip(trip.observations, trip.observation_xy)}
    route_steps = [row.route_step_id for row in labels if row.route_step_id is not None]
    monotonic = all(left <= right for left, right in zip(route_steps, route_steps[1:]))
    result = []
    for index in offsets:
        obs = trip.observations[index]
        row = by_obs[obs.obs_id]
        geometry_match = None
        if row.reference_arc_id is not None and row.aligned_distance_m is not None:
            actual_distance = arcs[row.reference_arc_id].distance_to(observation_by_id[obs.obs_id][1])
            geometry_match = isclose(actual_distance, row.aligned_distance_m, abs_tol=1e-5)
        result.append({
            "trip_id": trip.trip_id,
            "obs_id": obs.obs_id,
            "route_step_id": row.route_step_id,
            "reference_arc_id": row.reference_arc_id,
            "aligned_distance_m": row.aligned_distance_m,
            "label_status": row.label_status,
            "geometry_distance_consistent": geometry_match,
            "route_step_order_monotonic": monotonic if row.route_step_id is not None else None,
            "independent_review": {"validated": "accepted", "rejected": "rejected"}.get(
                row.label_status, "pending"),
            "review_evidence": row.review_evidence,
        })
    return result


def as_label_dicts(rows: list[LabelQuality]) -> list[dict]:
    return [asdict(row) for row in rows]
