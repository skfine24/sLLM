"""HTTP hardening tests for the stdlib front-end (bug fixes #2/#3/#4):
keep-alive integrity on oversized/invalid bodies, chunked rejection, status
mapping for server-side failures, and mid-stream error containment. Uses a
fake engine so it runs without any cached tokenizer.
"""

import http.client
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving import server as srv  # noqa: E402
from serving.server import SaturatedError  # noqa: E402


class _FakeModel:
    recipe = None


class _FakeEngine:
    """Implements only the surface the server calls; failures are switchable."""

    def __init__(self):
        self.model = _FakeModel()
        self.complete_exc = None
        self.stream_mode = "ok"

    def complete_detail(self, prompt, **kw):
        if self.complete_exc is not None:
            raise self.complete_exc
        return {"text": "hello world", "finish_reason": "length",
                "prompt_len": 1, "completion_len": 2}

    def chat_detail(self, messages, **kw):
        return {"text": "hi", "finish_reason": "stop",
                "prompt_len": 1, "completion_len": 1}

    def stream_complete(self, prompt, **kw):
        if self.stream_mode == "raise":
            yield "a", None
            raise RuntimeError("boom-mid-stream")
        for tok in ("a", "b"):
            yield tok, None
        yield "", "length"

    def stream_chat(self, messages, **kw):
        for tok in ("x", "y"):
            yield tok, None
        yield "", "stop"


class TestHTTPHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _FakeEngine()
        cls.httpd, cls.port = srv.create_server(
            cls.engine, host="127.0.0.1", port=0, quiet=True,
            model_name="fake")
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.complete_exc = None
        self.engine.stream_mode = "ok"

    # -- helpers -----------------------------------------------------------
    def _post(self, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def _post_err(self, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=15)
        return ctx.exception

    def _raw(self, method, headers, body=b""):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.putrequest(method, "/v1/completions", skip_host=False,
                            skip_accept_encoding=True)
            for k, v in headers.items():
                conn.putheader(k, v)
            conn.endheaders(body)
            resp = conn.getresponse()
            payload = resp.read()
            return resp.status, dict(resp.getheaders()), payload
        finally:
            conn.close()

    # -- #4 status mapping -------------------------------------------------
    def test_runtime_error_is_500(self):
        self.engine.complete_exc = RuntimeError("gpu decode failed")
        err = self._post_err({"prompt": "hi", "max_tokens": 2})
        self.assertEqual(err.code, 500)

    def test_error_code_field_is_string(self):
        err = self._post_err({"prompt": "hi", "max_tokens": "abc"})
        body = json.loads(err.read().decode("utf-8"))
        self.assertIsInstance(body["error"]["code"], str)

    def test_models_with_query_string(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("GET", "/v1/models?x=1")
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(resp.status, 200)
            self.assertEqual(body["object"], "list")
        finally:
            conn.close()

    def test_health_trailing_slash(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("GET", "/health/")
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(resp.status, 200)
            self.assertEqual(body["status"], "ok")
        finally:
            conn.close()

    def test_connection_cap_returns_503(self):
        # A saturated server (semaphore full) answers 503 and closes instead of
        # spawning/holding another handler thread.
        sem = threading.BoundedSemaphore(1)
        self.httpd.conn_sem = sem
        sem.acquire()  # consume the only slot
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
            try:
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                self.assertEqual(resp.status, 503)
                self.assertEqual(resp.getheader("Connection"), "close")
                resp.read()
            finally:
                conn.close()
        finally:
            sem.release()
            self.httpd.conn_sem = threading.BoundedSemaphore(64)

    def test_saturated_is_429_with_retry_after(self):
        self.engine.complete_exc = SaturatedError("full", retry_after=3)
        err = self._post_err({"prompt": "hi", "max_tokens": 2})
        self.assertEqual(err.code, 429)
        self.assertEqual(err.headers.get("Retry-After"), "3")

    def test_non_numeric_max_tokens_is_400(self):
        err = self._post_err({"prompt": "hi", "max_tokens": "abc"})
        self.assertEqual(err.code, 400)

    # -- #2 keep-alive integrity ------------------------------------------
    def test_oversized_body_closes_connection(self):
        status, headers, _ = self._raw(
            "POST", {"Content-Length": str(srv.MAX_BODY_BYTES + 1),
                     "Content-Type": "application/json"})
        self.assertEqual(status, 413)
        self.assertEqual(headers.get("Connection"), "close")

    def test_chunked_rejected_411(self):
        status, headers, _ = self._raw(
            "POST", {"Transfer-Encoding": "chunked",
                     "Content-Length": "5"}, body=b"hello")
        self.assertEqual(status, 411)
        self.assertEqual(headers.get("Connection"), "close")

    def test_keepalive_success_reuses_connection(self):
        # two sequential requests on ONE connection must both succeed (our
        # close-on-error paths must not poison the normal keep-alive flow)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("POST", "/v1/completions",
                         body=json.dumps({"prompt": "a", "max_tokens": 2}),
                         headers={"Content-Type": "application/json"})
            r1 = conn.getresponse()
            r1.read()
            conn.request("POST", "/v1/completions",
                         body=json.dumps({"prompt": "b", "max_tokens": 2}),
                         headers={"Content-Type": "application/json"})
            r2 = conn.getresponse()
            body = json.loads(r2.read().decode("utf-8"))
            self.assertEqual(r1.status, 200)
            self.assertEqual(r2.status, 200)
            self.assertIn("text", body["choices"][0])
        finally:
            conn.close()

    # -- #3 mid-stream error containment ----------------------------------
    def test_midstream_error_ends_with_done(self):
        self.engine.stream_mode = "raise"
        data = json.dumps({"prompt": "hi", "max_tokens": 4,
                           "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8")
        self.assertIn("data: [DONE]", raw)
        self.assertIn("stream error", raw)
        # server survived: a subsequent request still answers normally
        status, _ = self._post({"prompt": "hi", "max_tokens": 2})
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
