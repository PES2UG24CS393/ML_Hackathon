import tempfile
import unittest
from pathlib import Path

from pathsnap.preprocess import Arc, Observation, Point, Trip, load_trip
from pathsnap.candidates import rank_candidates
from scripts.inspect_dataset import inspect_trip


class ParserTests(unittest.TestCase):
    def test_source_row_ids_and_route_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "x"
            root.mkdir()
            (root / "t.track").write_text("1 2 3\n1.1 2.1 4\n")
            (root / "t.nodes").write_text("1 2\n1.1 2.1\n")
            (root / "t.arcs").write_text("0 1\n1 0\n")
            (root / "t.route").write_text("1\n0\n")
            trip = load_trip(root, "t")
            self.assertEqual([o.observation_id for o in trip.observations], [0, 1])
            self.assertEqual([a.arc_id for a in trip.arcs], [0, 1])
            self.assertEqual(trip.route, (1, 0))

    def test_rejects_bad_coordinates_and_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t.track").write_text("181 2 3\n")
            with self.assertRaisesRegex(ValueError, "bounds"):
                load_trip(root, "t")


class CandidateTests(unittest.TestCase):
    def test_nearest_arc_top_k_order_and_stability(self):
        obs = (Observation(0.0005, 0.0001, 0, 1.0), Observation(0.0005, 0.001, 1, 2.0))
        nodes = (Point(0, 0), Point(0.001, 0), Point(0, 0.001), Point(0.001, 0.001))
        arcs = (Arc(0, 0, 1, nodes[0], nodes[1]), Arc(1, 2, 3, nodes[2], nodes[3]))
        trip = Trip("synthetic", obs, nodes, arcs, None)
        candidates, predictions = rank_candidates(trip, top_k=2)
        self.assertEqual([p.observation_id for p in predictions], [0, 1])
        self.assertEqual([p.arc_id for p in predictions], [0, 1])
        self.assertEqual([c.rank for c in candidates], [1, 2, 1, 2])
        self.assertTrue(all(c.distance_m >= 0 for c in candidates))

    def test_empty_geometry_and_invalid_top_k(self):
        trip = Trip("empty", (Observation(0, 0, 0, 0),), (), (), None)
        candidates, predictions = rank_candidates(trip)
        self.assertEqual(candidates, [])
        self.assertEqual(predictions[0].status, "no_candidates")
        with self.assertRaises(ValueError):
            rank_candidates(trip, top_k=0)

    def test_rejects_invalid_coordinates_in_direct_api_input(self):
        trip = Trip("bad", (Observation(181, 0, 0, 1),), (), (), None)
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            rank_candidates(trip)

    def test_real_training_trip_runs_without_label_leakage(self):
        trip = load_trip(Path("Student folder/train/00000001"))
        candidates, predictions = rank_candidates(trip, top_k=3)
        self.assertEqual(len(predictions), len(trip.observations))
        self.assertTrue(candidates)
        self.assertTrue(all(p.arc_id is not None and p.arc_id < len(trip.arcs) for p in predictions))
        self.assertEqual([p.observation_id for p in predictions], list(range(len(trip.observations))))


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
