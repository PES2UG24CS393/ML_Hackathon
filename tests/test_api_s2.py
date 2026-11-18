import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import AssetsUnavailable, Prediction, create_app, fixture_app

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


REQUEST = read_json("docs/contracts/predict_request.json")
RESPONSE = read_json("docs/contracts/predict_response.json")
TRIP_ID = REQUEST["trip_id"]


def test_fixture_round_trip_and_routing_errors():
    client = TestClient(fixture_app())
    assert client.get("/health").json() == {"status": "ok", "mode": "demo"}
    assert client.get("/trips").json() == {"trip_ids": [TRIP_ID]}
    response = client.post("/predict", json=REQUEST)
    assert response.status_code == 200
    payload = response.json()
    assert payload["elapsed_ms"] >= 0
    assert {**payload, "elapsed_ms": RESPONSE["elapsed_ms"]} == RESPONSE

    for bad in ("../00005", "10000000\n", "０００００００５", 5):
        response = client.post("/predict", json={"trip_id": bad})
        assert response.status_code == 422
        assert response.json() == read_json("tests/fixtures/sprint2/api_error.json")
    assert client.post("/predict", json={**REQUEST, "reference": True}).status_code == 422
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404


def test_prediction_precedes_optional_server_evaluation():
    calls = []

    def predictor(trip_id):
        calls.append(("predict", trip_id))
        return Prediction.model_validate(RESPONSE)

    def evaluator(trip_id, prediction):
        assert prediction.route_arc_ids == [4, 9]
        calls.append(("evaluate", trip_id))
        return {"matched": 2}

    client = TestClient(create_app((TRIP_ID,), predictor, evaluator=evaluator))
    assert client.post("/predict", json=REQUEST).json()["metrics"] == {"matched": 2}
    assert calls == [("predict", TRIP_ID), ("evaluate", TRIP_ID)]


def test_missing_assets():
    assert TestClient(create_app()).get("/health").status_code == 503

    def missing(trip_id):
        raise AssetsUnavailable("/private/assets")

    response = TestClient(create_app((TRIP_ID,), missing)).post("/predict", json=REQUEST)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "assets_unavailable"
    assert "/private" not in response.text


def load_tests(loader, tests, pattern):
    for case in (test_fixture_round_trip_and_routing_errors,
                 test_prediction_precedes_optional_server_evaluation, test_missing_assets):
        tests.addTest(unittest.FunctionTestCase(case))
    return tests
