import math
from pathlib import Path
import tempfile
import unittest

from pathsnap.preprocess import LocalProjection, ParseError, load_trip


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.trip = Path(temp.name) / "00000042"
        self.trip.mkdir()
        self.write("track", "8.0 49.0 0\n8.0005 49.0 1.5\n8.0006 49.0 1.5\n")
        self.write("nodes", "8.0 49.0\n8.001 49.0\n8.002 49.0\n")
        self.write("arcs", "0 1\n1 0\n0 1\n")

    def write(self, extension, contents):
        (self.trip / f"{self.trip.name}.{extension}").write_text(contents, encoding="utf-8")

    def test_source_ids_directed_geometry_and_optional_route(self):
        trip = load_trip(self.trip)
        self.assertEqual(trip.trip_id, "00000042")
        self.assertIsNone(trip.route)
        self.assertEqual([obs.obs_id for obs in trip.observations], [0, 1, 2])
        self.assertEqual([obs.timestamp_s for obs in trip.observations], [0.0, 1.5, 1.5])
        self.assertEqual([node.node_id for node in trip.network.nodes], [0, 1, 2])
        self.assertEqual([arc.arc_id for arc in trip.network.arcs], [0, 1, 2])
        self.assertEqual(trip.network.outgoing_arc_ids, ((0, 2), (1,), ()))
        first, reverse, parallel = trip.network.segments
        self.assertEqual(first.start_xy, reverse.end_xy)
        self.assertEqual(first.end_xy, reverse.start_xy)
        self.assertEqual(first.end_xy, parallel.end_xy)
        self.assertEqual(first.start_xy, trip.observation_xy[0])
        self.assertAlmostEqual(first.length_m, 73.0, delta=0.3)
        self.assertAlmostEqual(trip.observation_xy[1][0], first.length_m / 2, places=5)

    def test_route_preserves_row_ids_order_and_repetitions(self):
        self.write("route", "2\n1\n2\n")
        route = load_trip(self.trip).route
        self.assertEqual([(step.row_id, step.arc_id) for step in route], [(0, 2), (1, 1), (2, 2)])

    def test_invalid_references(self):
        self.write("route", "0\n3\n")
        with self.assertRaisesRegex(ParseError, r"\.route:2 \(row ID 1\): arc_id 3"):
            load_trip(self.trip)
        self.write("arcs", "0 1\n1 -1\n")
        with self.assertRaisesRegex(ParseError, r"\.arcs:2 \(row ID 1\): to_node_id -1"):
            load_trip(self.trip)
        self.write("arcs", "0 1\n1.0 2\n")
        with self.assertRaisesRegex(ParseError, r"from_node_id must be an integer"):
            load_trip(self.trip)

    def test_invalid_rows_report_original_position(self):
        cases = [
            ("nodes", "8 49\n\n8.001 49\n", "expected 2 columns"),
            ("nodes", "8 49\n8 91\n", "non-finite or out of bounds"),
            ("nodes", "8 49\nnan 49\n", "non-finite or out of bounds"),
            ("track", "8 49 1\n8 49 nan\n", "finite and nonnegative"),
            ("track", "8 49 1\n8 49 -1\n", "finite and nonnegative"),
            ("track", "8 49 1\n8 49 0\n", "timestamp decreases"),
        ]
        for extension, contents, message in cases:
            with self.subTest(contents=contents):
                self.write(extension, contents)
                with self.assertRaisesRegex(ParseError, rf"\.{extension}:2 \(row ID 1\): .*{message}"):
                    load_trip(self.trip)

    def test_projection_wraps_longitude_and_uses_meters(self):
        projection = LocalProjection(179.999, 0.0)
        x, y = projection.project(-179.999, 0.001)
        self.assertAlmostEqual(x, 222.39, places=2)
        self.assertAlmostEqual(y, 111.20, places=2)


class DevelopmentTripTests(unittest.TestCase):
    def test_training_and_student_test_trips(self):
        root = Path(__file__).resolve().parents[1] / "data"
        for partition, trip_id in (("train", "00000001"), ("student_test", "00000000")):
            with self.subTest(trip_id=trip_id):
                trip = load_trip(root / partition / trip_id)
                self.assertEqual(trip.trip_id, trip_id)
                self.assertTrue(trip.observations)
                self.assertTrue(trip.network.segments)
                self.assertTrue(trip.route)
                arc_id = trip.route[0].arc_id
                arc = trip.network.arcs[arc_id]
                node = trip.network.nodes[arc.from_node_id]
                segment = trip.network.segments[arc_id]
                self.assertEqual(segment.arc_id, arc_id)
                self.assertEqual(segment.start_xy, trip.network.projection.project(node.lon, node.lat))
                self.assertTrue(math.isfinite(segment.length_m))
                del trip


if __name__ == "__main__":
    unittest.main()
