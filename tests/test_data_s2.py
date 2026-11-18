import csv
import json
import tempfile
import unittest
from pathlib import Path

from map_matching.features import feature_rows
from map_matching.label_quality import ReviewDecision, label_rows, reference_break_rows
from map_matching.preprocess import (
    Arc, ArcGeometry, LocalProjection, Node, Observation, RoadNetwork, RouteStep, Trip, load_trip,
)
from data_prep.build_dataset import build_dataset


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


class LabelQualityTests(unittest.TestCase):
    def test_independent_decisions_require_candidate_and_evidence(self):
        trip = linear_trip([(0.0002, 0, 1), (0.0008, 0, 2)])
        trip = Trip(trip.trip_id, trip.observations, trip.observation_xy, trip.network,
                    (RouteStep(0, 0),))
        candidates = feature_rows(trip)
        decision = ReviewDecision(trip.trip_id, 0, "accepted", "RiteshReddyy", "review-log.md#row-0")
        rows = label_rows(trip, candidates, decisions={0: decision})
        self.assertEqual(rows[0].label_status, "validated")
        self.assertEqual(rows[0].review_evidence, "review-log.md#row-0")
        self.assertEqual(rows[1].label_status, "needs_review")

        owner_decision = ReviewDecision(trip.trip_id, 0, "accepted", "Rajath-N", "review-log.md#row-0")
        with self.assertRaisesRegex(ValueError, "independent reviewer"):
            label_rows(trip, candidates, decisions={0: owner_decision})
        missing_evidence = ReviewDecision(trip.trip_id, 0, "accepted", "RiteshReddyy", "")
        with self.assertRaisesRegex(ValueError, "review_evidence"):
            label_rows(trip, candidates, decisions={0: missing_evidence})

    def test_reference_break_report_preserves_directed_endpoint_ids(self):
        trip = linear_trip([(0.0005, 0, 1)], node_lons=(0.0, 0.001, 0.002, 0.003),
                           arcs=((0, 1), (2, 3)))
        trip = Trip(trip.trip_id, trip.observations, trip.observation_xy, trip.network,
                    (RouteStep(0, 0), RouteStep(1, 1)))
        breaks = reference_break_rows(trip, "train")
        self.assertEqual(len(breaks), 1)
        self.assertEqual(breaks[0]["break_before_route_step_id"], 1)
        self.assertEqual((breaks[0]["previous_to_node_id"], breaks[0]["current_from_node_id"]), (1, 2))


class DatasetBuilderTests(unittest.TestCase):
    @staticmethod
    def read_csv(path: Path) -> list[dict]:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def write_trip(root: Path, split: str, trip_id: str) -> None:
        folder = root / split / trip_id
        folder.mkdir(parents=True)
        (folder / f"{trip_id}.track").write_text("8.0 49.0 1\n8.0001 49.0 2\n", encoding="utf-8")
        (folder / f"{trip_id}.nodes").write_text("8.0 49.0\n8.001 49.0\n", encoding="utf-8")
        (folder / f"{trip_id}.arcs").write_text("0 1\n", encoding="utf-8")
        (folder / f"{trip_id}.route").write_text("0\n", encoding="utf-8")

    def test_exports_separate_splits_grouped_folds_and_explicit_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            for i in range(5):
                self.write_trip(root, "train", f"{i:08d}")
            self.write_trip(root, "student_test", "00000090")
            output = Path(tmp) / "artifacts"
            manifest = build_dataset(root, output, seed=31,
                                     expected_counts={"train": 5, "student_test": 1})

            self.assertFalse(manifest["training_ready"])
            self.assertIn("No independently reviewed training labels have status validated",
                          manifest["readiness_blockers"])
            self.assertTrue(manifest["contract_frozen"])
            train_features = self.read_csv(output / "features_train.csv")
            student_features = self.read_csv(output / "features_student_test.csv")
            train_labels = self.read_csv(output / "labels_train.csv")
            student_labels = self.read_csv(output / "labels_student_test.csv")
            self.assertTrue(train_features)
            self.assertTrue(student_features)
            self.assertEqual({row["trip_id"] for row in train_features}, {f"{i:08d}" for i in range(5)})
            self.assertEqual({row["trip_id"] for row in student_features}, {"00000090"})
            self.assertTrue(all(row["label_status"] != "validated" for row in train_labels))
            self.assertTrue(all(row["label_status"] != "validated" for row in student_labels))
            folds = json.loads((output / "folds.json").read_text(encoding="utf-8"))["folds"]
            assigned = [trip_id for fold in folds for trip_id in fold["validation_trip_ids"]]
            self.assertEqual(set(assigned), {f"{i:08d}" for i in range(5)})
            self.assertEqual(len(assigned), len(set(assigned)))
            self.assertEqual(len(folds), 5)
            assignments = self.read_csv(output / "trip_assignments.csv")
            self.assertEqual(len(assignments), 6)
            self.assertTrue(all(row["training_fold_id"] for row in assignments
                                if row["split"] == "train"))
            self.assertTrue(all(not row["training_fold_id"] for row in assignments
                                if row["split"] == "student_test"))
            checks = self.read_csv(output / "sanity_checks.csv")
            self.assertTrue(checks)
            self.assertTrue(all(row["geometry_distance_consistent"] == "True" for row in checks))


if __name__ == "__main__":
    unittest.main()
