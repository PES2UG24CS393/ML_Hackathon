import unittest
from pathlib import Path

from map_matching.features import feature_rows
from map_matching.preprocess import (
    Arc, ArcGeometry, LocalProjection, Node, Observation, RoadNetwork, Trip, load_trip,
)


def linear_trip(points, node_lons=(0.0, 0.001), arcs=((0, 1),)):
    projection = LocalProjection(0, 0)
    nodes = tuple(Node(i, lon, 0.0) for i, lon in enumerate(node_lons))
    arc_rows = tuple(Arc(i, start, end) for i, (start, end) in enumerate(arcs))
    segments = tuple(ArcGeometry(a.arc_id,
        projection.project(nodes[a.from_node_id].lon, nodes[a.from_node_id].lat),
        projection.project(nodes[a.to_node_id].lon, nodes[a.to_node_id].lat)) for a in arc_rows)
    outgoing = [[] for _ in nodes]
    for arc in arc_rows:
        outgoing[arc.from_node_id].append(arc.arc_id)
    network = RoadNetwork(nodes, arc_rows, segments, tuple(tuple(x) for x in outgoing), projection)
    observations = tuple(Observation(i * 2, lon, lat, timestamp) for i, (lon, lat, timestamp) in enumerate(points))
    xy = tuple(projection.project(obs.lon, obs.lat) for obs in observations)
    return Trip("synthetic", observations, xy, network, None)


class NoRouteTrip:
    """Proxy that fails if inference feature extraction reads reference labels."""
    def __init__(self, trip):
        self._trip = trip

    def __getattr__(self, name):
        if name == "route":
            raise AssertionError("inference feature extraction read trip.route")
        return getattr(self._trip, name)


class CandidateFeatureConsistencyTests(unittest.TestCase):
    def test_same_arc_forward_stationary_and_reverse_progress(self):
        forward = feature_rows(linear_trip([(0.0002, 0, 1), (0.0008, 0, 3)]))
        self.assertEqual([r.predecessor_connected for r in forward], [None, True])
        self.assertEqual([r.time_gap_s for r in forward], [None, 2.0])

        stationary = feature_rows(linear_trip([(0.0005, 0, 1), (0.0005, 0, 2)]))
        self.assertEqual([r.predecessor_connected for r in stationary], [None, True])
        self.assertEqual(stationary[-1].speed_mps, 0.0)
        self.assertIsNone(stationary[-1].heading_diff_deg)

        reverse = feature_rows(linear_trip([(0.0008, 0, 1), (0.0002, 0, 2)]))
        self.assertEqual(reverse[-1].predecessor_connected, False)

    def test_directed_successor_and_parallel_arc_ties_keep_source_ids(self):
        successor = linear_trip(
            [(0.0005, 0, 1), (0.0015, 0, 2)],
            node_lons=(0.0, 0.001, 0.002), arcs=((0, 1), (1, 2)),
        )
        rows = feature_rows(successor, candidate_radius_m=25.0)
        self.assertEqual([(r.obs_id, r.arc_id) for r in rows], [(0, 0), (2, 1)])
        self.assertEqual(rows[-1].predecessor_connected, True)

        parallel = linear_trip([(0.0005, 0, 1)], arcs=((0, 1), (0, 1)))
        tied = feature_rows(parallel)
        self.assertEqual([r.arc_id for r in tied], [0, 1])

    def test_no_candidates_time_gaps_and_original_observation_ids(self):
        trip = linear_trip([(0.0002, 0, 1), (0.0002, 0, 9), (2.0, 0, 10)])
        rows = feature_rows(trip)
        self.assertEqual([r.obs_id for r in rows], [0, 2])
        self.assertEqual(rows[1].time_gap_s, 8.0)
        self.assertEqual(rows[1].predecessor_connected, True)
        self.assertLessEqual(max(sum(r.obs_id == obs.obs_id for r in rows) for obs in trip.observations), 8)

        gap_trip = linear_trip([(0.0002, 0, 1), (2.0, 0, 9), (0.0003, 0, 10)])
        gap_rows = feature_rows(gap_trip)
        self.assertEqual([r.obs_id for r in gap_rows], [0, 4])
        self.assertEqual(gap_rows[-1].time_gap_s, 1.0)
        self.assertIsNone(gap_rows[-1].predecessor_connected)

    def test_invalid_source_ids_are_rejected_not_reindexed(self):
        trip = linear_trip([(0.0005, 0, 1)])
        arc = Arc(9, 0, 1)
        segment = ArcGeometry(9, trip.network.segments[0].start_xy, trip.network.segments[0].end_xy)
        network = RoadNetwork(trip.network.nodes, (arc,), (segment,), ((9,), ()), trip.network.projection)
        invalid = Trip(trip.trip_id, trip.observations, trip.observation_xy, network, None)
        with self.assertRaisesRegex(ValueError, "arc IDs"):
            feature_rows(invalid)

    def test_features_are_identical_with_reference_access_forbidden(self):
        trip = linear_trip([(0.0002, 0, 1), (0.0005, 0, 2)])
        expected = feature_rows(trip)
        actual = feature_rows(NoRouteTrip(trip))
        self.assertEqual(actual, expected)

    def test_two_real_training_trips_are_deterministic_and_capped(self):
        root = Path(__file__).resolve().parents[1] / "data" / "train"
        for trip_id in ("00000005", "00000012"):
            with self.subTest(trip_id=trip_id):
                trip = load_trip(root / trip_id)
                first = feature_rows(trip)
                self.assertEqual(first, feature_rows(trip))
                self.assertTrue(first)
                self.assertTrue(all(row.trip_id == trip_id for row in first))
                self.assertTrue(all(0 <= row.arc_id < len(trip.network.arcs) for row in first))
                self.assertTrue(all(row.distance_m <= 100.0 for row in first))
                self.assertLessEqual(max(sum(row.obs_id == obs.obs_id for row in first)
                                         for obs in trip.observations), 8)


if __name__ == "__main__":
    unittest.main()
