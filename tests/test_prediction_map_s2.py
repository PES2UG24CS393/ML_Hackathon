"""Synthetic Folium checks for S2-A3; map routes are fixture predictions only."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from map_matching.prediction_map import write_prediction_map
from map_matching.preprocess import (
    Arc, ArcGeometry, LocalProjection, Node, Observation, RoadNetwork, RouteStep, Trip,
)


def map_trip():
    projection = LocalProjection(77.0, 12.0)
    nodes = (Node(0, 77.000, 12.000), Node(1, 77.001, 12.000),
             Node(2, 77.002, 12.000), Node(3, 77.010, 12.010), Node(4, 77.011, 12.010))
    endpoints = ((0, 1), (1, 2), (3, 4))
    arcs = tuple(Arc(i, a, b) for i, (a, b) in enumerate(endpoints))
    node_xy = tuple(projection.project(node.lon, node.lat) for node in nodes)
    segments = tuple(ArcGeometry(arc.arc_id, node_xy[arc.from_node_id], node_xy[arc.to_node_id])
                     for arc in arcs)
    outgoing = ((0,), (1,), (), (2,), ())
    network = RoadNetwork(nodes, arcs, segments, outgoing, projection)
    observations = (Observation(0, 77.0002, 12.0001, 0.0),
                    Observation(1, 77.0012, 12.0001, 1.0),
                    Observation(2, 77.0102, 12.0101, 400.0))
    observation_xy = tuple(projection.project(obs.lon, obs.lat) for obs in observations)
    return Trip("map-toy", observations, observation_xy, network,
                (RouteStep(0, 0), RouteStep(1, 1)))


@unittest.skipUnless(importlib.util.find_spec("folium"), "folium dependency is not installed")
class PredictionMapTests(unittest.TestCase):
    def test_layers_breaks_provenance_and_reference_default(self):
        trip = map_trip()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "comparison.html"
            write_prediction_map(
                trip, [[0, 1]], [[0, 1], [2]], path,
                provenance={"trip_id": "map-toy", "model_or_method": "synthetic fixture",
                            "prediction_source": "tests/test_prediction_map_s2.py",
                            "command": "unit test"},
                breaks=[{"before_obs_id": 2, "reason": "fixture_disconnect"}],
            )
            html = path.read_text(encoding="utf-8")
            self.assertIn("Classifier-only route", html)
            self.assertIn("HMM / Viterbi route", html)
            self.assertIn("HMM route breaks", html)
            self.assertIn("Prediction provenance", html)
            self.assertNotIn("Reference route (development only)", html)
            metadata = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["trip_id"], "map-toy")
            self.assertEqual(len(metadata["prediction_route_sha256"]), 64)

    def test_reference_layer_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "with_reference.html"
            write_prediction_map(
                map_trip(), [[0, 1]], [[0, 1]], path,
                provenance={"trip_id": "map-toy", "model_or_method": "synthetic fixture",
                            "prediction_source": "synthetic", "command": "unit test"},
                include_reference=True,
            )
            self.assertIn("Reference route (development only)", path.read_text(encoding="utf-8"))

    def test_disconnected_arcs_cannot_be_joined_inside_a_route_part(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "disconnected"):
                write_prediction_map(
                    map_trip(), [[0, 2]], [], Path(temp) / "bad.html",
                    provenance={"trip_id": "map-toy", "model_or_method": "fixture",
                                "prediction_source": "synthetic", "command": "unit test"},
                )


if __name__ == "__main__":
    unittest.main()
