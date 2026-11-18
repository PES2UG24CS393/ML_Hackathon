"""Load trusted deployment assets and connect scoring, decoding and HTTP."""

import hashlib
import json
import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from .app import AssetsUnavailable, Prediction, create_app
from map_matching.features import feature_rows
from map_matching.preprocess import ParseError, load_trip, parse_route

FEATURES = (
    "distance_m", "heading_diff_deg", "speed_mps", "time_gap_s",
    "predecessor_connected",
)
logger = logging.getLogger("uvicorn.error")


def configured_app(mode):
    from joblib import load
    import numpy as np
    from map_matching.decoder import DecoderConfig, decode_trip

    root = Path(os.environ["PATHSNAP_TRIP_ROOT"]).resolve(strict=True)
    bundle = Path(os.environ["PATHSNAP_MODEL_DIR"]).resolve(strict=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["bundle_version"] != 1
        or manifest["feature_version"] != "sprint2_v1"
        or manifest["feature_order"] != list(FEATURES)
    ):
        raise ValueError("Unsupported model bundle or feature schema.")
    pipeline = bundle / "pipeline.joblib"
    with pipeline.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != manifest["pipeline_sha256"]:
            raise ValueError("Model pipeline checksum mismatch.")
        source.seek(0)
        model, calibrator = load(source)
    decoder_config = DecoderConfig(**manifest.get("decoder_config", {}))
    positive_column = list(model.classes_).index(1)
    if list(model.classes_) != [0, 1] or not callable(model.predict_proba):
        raise ValueError("Expected a fitted binary probability estimator.")
    if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
        raise ValueError("Unexpected trained feature count.")
    if calibrator is not None:
        calibrated_column = list(calibrator.classes_).index(1)

    trips = {
        entry.name: entry
        for entry in root.iterdir()
        if re.fullmatch(r"[0-9]{8}", entry.name)
        and not entry.is_symlink() and entry.is_dir()
        and all(
            path.is_file() and not path.is_symlink()
            for path in (entry / f"{entry.name}.{extension}" for extension in ("track", "nodes", "arcs"))
        )
    }
    if len(trips) != 10:
        raise ValueError("The configured partition must contain ten valid trip folders.")

    def predict(trip_id):
        try:
            trip = load_trip(trips[trip_id], include_reference=False)
        except ParseError as exc:
            raise AssetsUnavailable("Invalid trip files.") from exc
        candidates = feature_rows(trip)
        scores = []
        if candidates:
            matrix = np.asarray(
                [[getattr(row, name) for name in FEATURES] for row in candidates],
                dtype=float,
            )
            probabilities = model.predict_proba(matrix)[:, positive_column]
            if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
                raise AssetsUnavailable("Invalid model probabilities.")
            if calibrator is not None:
                clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
                log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
                probabilities = calibrator.predict_proba(log_odds)[:, calibrated_column]
                if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
                    raise AssetsUnavailable("Invalid calibrated probabilities.")
            scores = [
                {"trip_id": trip_id, "obs_id": row.obs_id, "arc_id": row.arc_id, "p_positive": float(score)}
                for row, score in zip(candidates, probabilities, strict=True)
            ]
        decoded = decode_trip(trip, scores, decoder_config)
        arcs = {arc.arc_id: arc for arc in trip.network.arcs}
        nodes = trip.network.nodes
        parts = []
        for part in decoded.route_parts:
            node_ids = [arcs[part[0]].from_node_id, *(arcs[arc_id].to_node_id for arc_id in part)]
            parts.append({"arc_ids": part, "coordinates": [(nodes[i].lon, nodes[i].lat) for i in node_ids]})
        return Prediction(
            gps=[(obs.lon, obs.lat) for obs in trip.observations],
            matches=[{"obs_id": match.obs_id, "arc_id": match.arc_id} for match in decoded.matches],
            route_arc_ids=decoded.route_arc_ids,
            route_parts=parts,
            breaks=[{"obs_id": item["before_obs_id"], "reason": item["reason"]} for item in decoded.breaks],
        )

    def evaluate(trip_id, prediction):
        path = trips[trip_id] / f"{trip_id}.route"
        if path.is_symlink() or not path.is_file():
            raise AssetsUnavailable("Evaluation reference is unavailable.")
        try:
            # Read the reference only after inference. The archive has ordered
            # route steps, so this metric measures route sequence similarity.
            trip = load_trip(trips[trip_id], include_reference=False)
            reference = [step.arc_id for step in parse_route(path, len(trip.network.arcs))]
        except ParseError as exc:
            raise AssetsUnavailable("Invalid evaluation reference.") from exc
        similarity = SequenceMatcher(None, reference, prediction.route_arc_ids, autojunk=False).ratio()
        return {
            "route_sequence_similarity": similarity,
            "reference_route_steps": len(reference),
            "matched_observations": sum(match.arc_id is not None for match in prediction.matches),
        }

    return create_app(tuple(trips), predict, mode=mode, evaluator=evaluate if mode == "release" else None)


def app_from_environment():
    mode = os.environ.get("PATHSNAP_MODE", "release")
    if mode not in {"demo", "release"}:
        raise ValueError("PATHSNAP_MODE must be demo, release or fixture.")
    try:
        return configured_app(mode)
    except Exception as exc:
        logger.warning("PathSnap assets unavailable (%s).", type(exc).__name__)
        return create_app(mode=mode)
