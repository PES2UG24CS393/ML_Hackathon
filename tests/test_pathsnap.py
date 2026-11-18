import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from map_matching.preprocess import Arc, ArcGeometry, LocalProjection, Node, Observation, RoadNetwork, RouteStep, Trip, load_trip
from map_matching.candidates import rank_candidates
from map_matching.baseline import nearest_segment_predictions
from map_matching.demo import write_map
from map_matching.features import align_reference_route, reference_route_breaks
from map_matching.inventory import inspect_trip


def make_trip():
    projection = LocalProjection(0, 0)
    nodes = (Node(0, 0, 0), Node(1, 0.001, 0), Node(2, 0, 0.001), Node(3, 0.001, 0.001))
    arcs = (Arc(0, 0, 1), Arc(1, 2, 3), Arc(2, 0, 1))
    segments = tuple(ArcGeometry(a.arc_id, projection.project(nodes[a.from_node_id].lon, nodes[a.from_node_id].lat),
                                 projection.project(nodes[a.to_node_id].lon, nodes[a.to_node_id].lat)) for a in arcs)
    network = RoadNetwork(nodes, arcs, segments, ((0, 2), (), (1,), ()), projection)
    observations = (Observation(0, 0.0005, 0.0001, 1.0), Observation(1, 0.0005, 0.001, 2.0))
    return Trip("synthetic", observations, tuple(projection.project(o.lon, o.lat) for o in observations), network, None)


class CandidateTests(unittest.TestCase):
    def test_nearest_arc_top_k_order_and_stability(self):
        candidates, predictions = rank_candidates(make_trip(), top_k=2)
        self.assertEqual([p.observation_index for p in predictions], [0, 1])
        self.assertEqual([p.arc_id for p in predictions], [0, 1])
        self.assertEqual([c.rank for c in candidates], [1, 2, 1, 2])
        self.assertEqual([c.arc_id for c in candidates], [0, 2, 1, 0])
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
        trip = load_trip(Path("data/train/00000005"))
        candidates, predictions = rank_candidates(trip, top_k=3)
        self.assertEqual(len(predictions), len(trip.observations))
        self.assertTrue(candidates)
        self.assertTrue(all(p.arc_id is not None and p.arc_id < len(trip.network.arcs) for p in predictions))
        self.assertEqual([p.observation_index for p in predictions], list(range(len(trip.observations))))

    def test_candidate_top_one_agrees_with_existing_baseline(self):
        trip = load_trip(Path("data/train/00000005"))
        _, top_one = rank_candidates(trip, top_k=1)
        existing = nearest_segment_predictions(trip, max_distance_m=100_000)
        self.assertEqual([p.arc_id for p in top_one], [p["arc_id"] for p in existing])

    def test_rejects_invalid_coordinates_in_direct_api_input(self):
        trip = make_trip()
        invalid = Observation(0, 181, 0, 1)
        trip = Trip(trip.trip_id, (invalid,), ((0.0, 0.0),), trip.network, None)
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            rank_candidates(trip)


class MapTests(unittest.TestCase):
    def test_rendered_layers_share_coordinate_order_and_world_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "crossing"
            folder.mkdir()
            (folder / "crossing.track").write_text("179.875 12 1\n-179.875 12.125 2\n")
            (folder / "crossing.nodes").write_text("179.875 12\n-179.875 12.125\n")
            (folder / "crossing.arcs").write_text("0 1\n")
            (folder / "crossing.route").write_text("0\n")
            trip = load_trip(folder)
            path = Path(tmp) / "map.html"
            write_map(trip, nearest_segment_predictions(trip), path)
            html = path.read_text(encoding="utf-8")

        self.assertIn("center: [12.0, 179.875]", html)
        bounds = re.search(r"\.fitBounds\(\s*(\[\[.*?\]\])", html)
        self.assertEqual(json.loads(bounds.group(1)), [[12, 179.875], [12.125, 180.125]])
        layers = [json.loads(value) for value in re.findall(r"_add\((\{.*\})\);", html)]
        geometries = [[feature["geometry"]["coordinates"] for feature in layer["features"]]
                      for layer in layers]
        endpoints = [[179.875, 12], [180.125, 12.125]]
        self.assertEqual(geometries, [[[endpoints]], [[endpoints]], [[endpoints]], endpoints])


class AlignmentTests(unittest.TestCase):
    def test_proposals_require_review_and_flag_discontinuous_reference(self):
        trip = make_trip()
        no_reference = align_reference_route(trip)
        self.assertEqual([row.status for row in no_reference], ["no_reference_route"] * 2)

        continuous = replace(trip, route=(RouteStep(0, 1),))
        self.assertEqual(reference_route_breaks(continuous), ())
        self.assertEqual(align_reference_route(continuous)[1].status, "needs_review")

        discontinuous = replace(trip, route=(RouteStep(0, 0), RouteStep(1, 1)))
        self.assertEqual(reference_route_breaks(discontinuous), (1,))
        proposal = align_reference_route(discontinuous)[1]
        self.assertEqual(proposal.reference_arc_id, 1)
        self.assertEqual(proposal.status, "discontinuous_reference_route")


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

    def test_inventory_rejects_the_same_structural_errors_as_the_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track = root / f"{root.name}.track"
            arcs = root / f"{root.name}.arcs"
            (root / f"{root.name}.nodes").write_text("0 0\n0.1 0.1\n")
            for track_data, arc_data in (
                ("0 0 1\n", "0 2\n"),
                ("0 0 2\n0 0 1\n", "0 1\n"),
                ("0 0 -1\n", "0 1\n"),
            ):
                with self.subTest(track=track_data, arcs=arc_data):
                    track.write_text(track_data)
                    arcs.write_text(arc_data)
                    self.assertEqual(inspect_trip(root, "train")["status"], "invalid")
                    with self.assertRaises(ValueError):
                        load_trip(root)

    def test_inventory_keeps_source_ids_after_malformed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "x"
            root.mkdir()
            (root / "x.track").write_text("0 0 1\n")
            (root / "x.nodes").write_text("0 0\nbad node\n0.1 0.1\n")
            (root / "x.arcs").write_text("bad arc\n0 2\n")
            (root / "x.route").write_text("1\n")
            row = inspect_trip(root, "train")
            self.assertEqual(row["malformed_records"], 2)
            self.assertEqual(row["invalid_references"], 0)
            self.assertEqual(row["arc_rows"], 1)


if __name__ == "__main__":
    unittest.main()
