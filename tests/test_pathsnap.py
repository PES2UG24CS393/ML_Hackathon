import tempfile
import unittest
from pathlib import Path

from pathsnap.preprocess import Arc, ArcGeometry, LocalProjection, Node, Observation, RoadNetwork, Trip, load_trip
from pathsnap.candidates import rank_candidates
from scripts.inspect_dataset import inspect_trip


def make_trip():
    projection = LocalProjection(0, 0)
    nodes = (Node(0, 0, 0), Node(1, 0.001, 0), Node(2, 0, 0.001), Node(3, 0.001, 0.001))
    arcs = (Arc(0, 0, 1), Arc(1, 2, 3))
    segments = tuple(ArcGeometry(a.arc_id, projection.project(nodes[a.from_node_id].lon, nodes[a.from_node_id].lat),
                                 projection.project(nodes[a.to_node_id].lon, nodes[a.to_node_id].lat)) for a in arcs)
    network = RoadNetwork(nodes, arcs, segments, ((0,), (), (1,), ()), projection)
    observations = (Observation(0, 0.0005, 0.0001, 1.0), Observation(1, 0.0005, 0.001, 2.0))
    return Trip("synthetic", observations, tuple(projection.project(o.lon, o.lat) for o in observations), network, None)


class CandidateTests(unittest.TestCase):
    def test_nearest_arc_top_k_order_and_stability(self):
        candidates, predictions = rank_candidates(make_trip(), top_k=2)
        self.assertEqual([p.observation_index for p in predictions], [0, 1])
        self.assertEqual([p.arc_id for p in predictions], [0, 1])
        self.assertEqual([c.rank for c in candidates], [1, 2, 1, 2])
        self.assertTrue(all(c.distance_m >= 0 for c in candidates))

    def test_empty_geometry_and_invalid_top_k(self):
        trip = make_trip()
        trip = Trip(trip.trip_id, trip.observations, trip.observation_xy,
                    RoadNetwork(trip.network.nodes, (), (), tuple(() for _ in trip.network.nodes), trip.network.projection), None)
        candidates, predictions = rank_candidates(trip)
        self.assertEqual(candidates, [])
        self.assertEqual(predictions[0].status, "no_candidates")
        with self.assertRaises(ValueError):
            rank_candidates(trip, top_k=0)

    def test_real_training_trip_parses_and_preserves_observation_order(self):
        trip = load_trip(Path("Student folder/train/00000005"))
        candidates, predictions = rank_candidates(trip, top_k=3)
        self.assertEqual(len(predictions), len(trip.observations))
        self.assertTrue(candidates)
        self.assertTrue(all(p.arc_id is not None and p.arc_id < len(trip.network.arcs) for p in predictions))
        self.assertEqual([p.observation_index for p in predictions], list(range(len(trip.observations))))

    def test_rejects_invalid_coordinates_in_direct_api_input(self):
        trip = make_trip()
        invalid = Observation(0, 181, 0, 1)
        trip = Trip(trip.trip_id, (invalid,), ((0.0, 0.0),), trip.network, None)
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            rank_candidates(trip)


class InventoryTests(unittest.TestCase):
    def test_inventory_detects_missing_and_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "x"
            root.mkdir()
            (root / "x.track").write_text("0 0 1\nbad row\n")
            (root / "x.nodes").write_text("0 0\n")
            (root / "x.arcs").write_text("0 2\n")
            row = inspect_trip(root, "train")
            self.assertEqual(row["malformed_records"], 1)
            self.assertEqual(row["invalid_references"], 1)
            self.assertFalse(row["route_available"])
            self.assertEqual(row["missing_files"], "route")


if __name__ == "__main__":
    unittest.main()
