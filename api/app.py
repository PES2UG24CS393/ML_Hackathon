"""PathSnap Sprint 2 HTTP boundary and explicit fixture mode."""

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

TripID = Annotated[str, Field(pattern=r"^[0-9]{8}$", min_length=8, max_length=8)]
Index = Annotated[int, Field(ge=0)]
Longitude = Annotated[float, Field(ge=-180, le=180)]
Latitude = Annotated[float, Field(ge=-90, le=90)]
Point = tuple[Longitude, Latitude]
Metric = Annotated[float, Field(allow_inf_nan=False)]
logger = logging.getLogger("uvicorn.error")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: TripID


class Match(BaseModel):
    obs_id: Index
    arc_id: Index | None


class RoutePart(BaseModel):
    arc_ids: Annotated[list[Index], Field(min_length=1)]
    coordinates: Annotated[list[Point], Field(min_length=2)]


class RouteBreak(BaseModel):
    obs_id: Index
    reason: str


class Prediction(BaseModel):
    gps: list[Point]
    matches: list[Match]
    route_arc_ids: list[Index]
    route_parts: list[RoutePart]
    breaks: list[RouteBreak]


class PredictionResponse(Prediction):
    trip_id: TripID
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    metrics: dict[str, Metric] | None = None


class AssetsUnavailable(Exception):
    """A required trip or inference asset could not be loaded."""


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def create_app(
    trip_ids: tuple[str, ...] = (),
    predictor: Callable[[str], Prediction] | None = None,
    *,
    mode: Literal["demo", "release"] = "release",
    evaluator: Callable[[str, Prediction], dict[str, float]] | None = None,
) -> FastAPI:
    """Inject trusted inference and, separately, optional server-authorized evaluation."""
    if any(re.fullmatch(r"[0-9]{8}", trip_id) is None for trip_id in trip_ids):
        raise ValueError("Allowlisted trip IDs must be eight ASCII digits.")
    allowed = sorted(set(trip_ids))
    allowed_ids = set(allowed)
    app = FastAPI(title="PathSnap", version="2.0")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _exc):
        return error(422, "invalid_request", "Expected an eight-digit trip ID.")

    @app.get("/health")
    def health():
        if predictor is None:
            return error(503, "assets_unavailable", "Prediction assets are unavailable.")
        return {"status": "ok", "mode": mode}

    @app.get("/trips")
    def trips():
        return {"trip_ids": allowed}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        if request.trip_id not in allowed_ids:
            return error(404, "unknown_trip", "Trip is unavailable in this mode.")
        if predictor is None:
            return error(503, "assets_unavailable", "Prediction assets are unavailable.")
        start = perf_counter()
        try:
            prediction = predictor(request.trip_id)
            metrics = evaluator(request.trip_id, prediction) if evaluator else None
            response = PredictionResponse(
                **prediction.model_dump(include=set(Prediction.model_fields)),
                trip_id=request.trip_id,
                elapsed_ms=(perf_counter() - start) * 1000,
                metrics=metrics,
            )
        except (OSError, AssetsUnavailable):
            logger.warning("Prediction assets unavailable for trip %s", request.trip_id)
            return error(503, "assets_unavailable", "Prediction assets are unavailable.")
        logger.info("Predicted trip %s in %.1f ms (%s)", request.trip_id, response.elapsed_ms, mode)
        return response

    return app


def fixture_app() -> FastAPI:
    fixture = Path(__file__).resolve().parents[1] / "docs/contracts/predict_response.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    result = Prediction.model_validate(data)
    return create_app((data["trip_id"],), lambda _: result.model_copy(deep=True), mode="demo")


def app_factory() -> FastAPI:
    if os.environ.get("PATHSNAP_MODE") == "fixture":
        return fixture_app()
    from .runtime import app_from_environment

    return app_from_environment()
