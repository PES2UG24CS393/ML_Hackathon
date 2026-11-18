"""Synthetic graph and score fixtures for S2-A2; no dataset labels are used."""

import unittest

from map_matching.decoder import DecoderConfig, compare_classifier_and_hmm, decode_trip
from map_matching.graph_paths import shortest_path
from map_matching.preprocess import (
    Arc, ArcGeometry, LocalProjection, Node, Observation, RoadNetwork, Trip,
)


def toy_trip(xs, times):
    projection = LocalProjection(0.0, 0.0)
    node_lons = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    nodes = tuple(Node(i, lon, 0.0) for i, lon in enumerate(node_lons))
    endpoints = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 1), (5, 6), (4, 2)]
    arcs = tuple(Arc(i, start, end) for i, (start, end) in enumerate(endpoints))
    node_xy = [projection.project(node.lon, node.lat) for node in nodes]
    segments = tuple(ArcGeometry(arc.arc_id, node_xy[arc.from_node_id], node_xy[arc.to_node_id])
                     for arc in arcs)
    outgoing = [[] for _ in nodes]
    for arc in arcs:
        outgoing[arc.from_node_id].append(arc.arc_id)
    network = RoadNetwork(nodes, arcs, segments, tuple(tuple(row) for row in outgoing), projection)
    observations = tuple(Observation(i, lon, 0.0, time) for i, (lon, time) in enumerate(zip(xs, times)))
    xy = tuple(projection.project(obs.lon, obs.lat) for obs in observations)
    return Trip("toy-0001", observations, xy, network, None)


def score(trip_id, obs_id, arc_id, probability):
    return {"trip_id": trip_id, "obs_id": obs_id, "arc_id": arc_id, "p_positive": probability}


class DirectedPathsTests(unittest.TestCase):
    def test_directed_multihop_and_unreachable(self):
        trip = toy_trip([0.0005], [0])
        network = trip.network
        result = shortest_path(network, 1, 3, 250)
        self.assertEqual(result.arc_ids, (1, 2))
        self.assertGreater(result.distance_m, 0)
        self.assertIsNone(shortest_path(network, 3, 1, 250))
        self.assertIsNone(shortest_path(network, 5, 1, 1000))


class DecoderTests(unittest.TestCase):
    def test_viterbi_prefers_directed_coherent_path_over_stronger_local_score(self):
        trip = toy_trip([0.0008, 0.0012, 0.0022], [0, 2, 4])
        rows = [score(trip.trip_id, 0, 0, 0.80), score(trip.trip_id, 0, 3, 0.20),
                score(trip.trip_id, 1, 1, 0.95), score(trip.trip_id, 1, 4, 0.05),
                score(trip.trip_id, 2, 2, 0.90)]
        result = decode_trip(trip, rows)
        self.assertEqual([match.arc_id for match in result.matches], [0, 1, 2])
        self.assertEqual(result.route_parts, ((0, 1, 2),))
        self.assertEqual(result.breaks, ())

    def test_same_arc_forward_motion_does_not_duplicate_route_arc(self):
        trip = toy_trip([0.0002, 0.0008], [0, 10])
        result = decode_trip(trip, [score(trip.trip_id, 0, 0, 0.7), score(trip.trip_id, 1, 0, 0.8)])
        self.assertEqual(result.route_arc_ids, (0,))
        self.assertEqual(result.route_parts, ((0,),))

    def test_backward_motion_on_same_arc_uses_directed_loop_and_keeps_revisit(self):
        trip = toy_trip([0.0018, 0.0012], [0, 3])
        result = decode_trip(trip, [score(trip.trip_id, 0, 1, 0.8), score(trip.trip_id, 1, 1, 0.9)])
        self.assertEqual(result.route_parts, ((1, 4, 1),))

    def test_missing_candidate_and_long_gap_create_explicit_breaks(self):
        trip = toy_trip([0.0005, 0.0015, 0.0025], [0, 1, 500])
        rows = [score(trip.trip_id, 0, 0, 0.9), score(trip.trip_id, 2, 2, 0.9)]
        result = decode_trip(trip, rows, DecoderConfig(max_gap_s=30))
        self.assertIsNone(result.matches[1].arc_id)
        self.assertEqual(result.breaks[0]["reason"], "missing_candidates")
        self.assertEqual(result.breaks[1]["reason"], "long_time_gap")
        self.assertEqual(result.route_parts, ((0,), (2,)))

    def test_disconnected_candidates_split_without_invented_connector(self):
        trip = toy_trip([0.0005, 0.0055], [0, 1])
        rows = [score(trip.trip_id, 0, 0, 0.9), score(trip.trip_id, 1, 5, 0.9)]
        result = decode_trip(trip, rows)
        self.assertEqual(result.route_parts, ((0,), (5,)))
        self.assertEqual(result.breaks[0]["reason"], "no_feasible_directed_transition")

    def test_duplicate_and_invalid_probabilities_rejected(self):
        trip = toy_trip([0.0005], [0])
        row = score(trip.trip_id, 0, 0, 0.5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            decode_trip(trip, [row, row])
        with self.assertRaisesRegex(ValueError, "p_positive"):
            decode_trip(trip, [score(trip.trip_id, 0, 0, 1.1)])

    def test_matched_support_metrics_and_route_edit_distance(self):
        trip = toy_trip([0.0005, 0.0015], [0, 1])
        rows = [score(trip.trip_id, 0, 0, 0.8), score(trip.trip_id, 1, 1, 0.9)]
        decoded = decode_trip(trip, rows)
        metrics = compare_classifier_and_hmm(
            trip, rows, decoded, {0: 0, 1: 1}, [0, 1]
        )
        self.assertEqual(metrics["eligible_observation_count"], 2)
        self.assertEqual(metrics["classifier_only_top1"]["accuracy"], 1.0)
        self.assertEqual(metrics["hmm_top1"]["accuracy"], 1.0)
        self.assertEqual(metrics["hmm_sequence_edit_distance"]["normalized_edit_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
