import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ui.app import EvaluatorHandler, HTML


class UIEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream_requests = []

        class Upstream(BaseHTTPRequestHandler):
            trip_payload = {"trip_ids": ["00000005", "00000006"]}
            fail_prediction = False

            def do_GET(self):
                type(self).server_state.append((self.command, self.path, None))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(type(self).trip_payload).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                type(self).server_state.append((self.command, self.path, body))
                if type(self).fail_prediction:
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":{"code":"unavailable","message":"model not ready"}}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"trip_id": body["trip_id"], "gps": [],
                                             "matches": [], "route_arc_ids": [],
                                             "route_parts": [], "breaks": [],
                                             "elapsed_ms": 1.0, "metrics": None}).encode())

            def log_message(self, *args):
                pass

        Upstream.server_state = cls.upstream_requests
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()
        cls.upstream_url = f"http://127.0.0.1:{cls.upstream.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.upstream_thread.join(timeout=2)

    def setUp(self):
        handler = type("TestHandler", (EvaluatorHandler,),
                       {"mock": True, "api_url": self.upstream_url})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request_json(self, path, method="GET", body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(self.url + path, data=data, method=method,
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            return json.loads(response.read())

    def test_mock_trip_list_and_prediction_route_parts(self):
        self.assertEqual(self.request_json("/api/trips"), {"trip_ids": ["00000005"]})
        result = self.request_json("/api/predict", "POST", {"trip_id": "00000005"})
        self.assertEqual(result["trip_id"], "00000005")
        self.assertEqual([part["arc_ids"] for part in result["route_parts"]], [[1], [7]])
        self.assertEqual(result["metrics"]["observation_count"], 3)
        self.assertEqual(result["gps"], [[8.0, 49.0], [8.001, 49.0005], [8.01, 49.01]])
        self.assertEqual(result["breaks"], [{"obs_id": 1, "reason": "no_candidates"}])

    def test_empty_trip_list_is_returned(self):
        handler = self.upstream.RequestHandlerClass
        original = handler.trip_payload
        try:
            handler.trip_payload = {"trip_ids": []}
            self.server.RequestHandlerClass.mock = False
            self.assertEqual(self.request_json("/api/trips"), {"trip_ids": []})
        finally:
            handler.trip_payload = original

    def test_prediction_posts_only_selected_trip_id_to_api(self):
        self.server.RequestHandlerClass.mock = False
        self.request_json("/api/trips")
        result = self.request_json("/api/predict", "POST", {"trip_id": "00000006"})
        self.assertEqual(result["trip_id"], "00000006")
        self.assertEqual(self.upstream_requests[-2:], [
            ("GET", "/trips", None), ("POST", "/predict", {"trip_id": "00000006"}),
        ])

    def test_upstream_api_errors_are_reported(self):
        handler = self.upstream.RequestHandlerClass
        handler.fail_prediction = True
        self.server.RequestHandlerClass.mock = False
        request = Request(self.url + "/api/predict", data=b'{"trip_id":"00000005"}',
                          method="POST", headers={"Content-Type": "application/json"})
        try:
            urlopen(request)
        except HTTPError as error:
            with error:
                self.assertEqual(error.code, 502)
                self.assertIn(b"HTTP 503", error.read())
        else:
            self.fail("upstream failure was not forwarded")
        finally:
            handler.fail_prediction = False

    def test_rejects_extra_prediction_fields(self):
        request = Request(self.url + "/api/predict", data=b'{"trip_id":"x","route":[1]}',
                          method="POST", headers={"Content-Type": "application/json"})
        try:
            urlopen(request)
        except HTTPError as error:
            with error:
                self.assertEqual(error.code, 400)
                self.assertIn(b"exactly trip_id", error.read())
        else:
            self.fail("invalid prediction payload was accepted")

    def test_rejects_non_contract_trip_id(self):
        request = Request(self.url + "/api/predict", data=b'{"trip_id":"mock-trip"}',
                          method="POST", headers={"Content-Type": "application/json"})
        try:
            urlopen(request)
        except HTTPError as error:
            with error:
                self.assertEqual(error.code, 400)
                self.assertIn(b"eight ASCII digits", error.read())
        else:
            self.fail("invalid trip ID was accepted")

    def test_ui_has_layer_controls_and_route_parts_rendered_separately(self):
        self.assertIn('id="gps"', HTML)
        self.assertIn('id="roads"', HTML)
        self.assertIn('id="predicted"', HTML)
        self.assertIn("parts.map(coordsOf)", HTML)
        self.assertIn("for(const b of breakRows)", HTML)
        self.assertIn("result.breaks", HTML)
        self.assertIn("window.addEventListener('resize',draw)", HTML)


if __name__ == "__main__":
    unittest.main()
