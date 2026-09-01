"""Continuous-batching serving facade tests (design fix #5/#6): the
BatchedServingEngine drives the dev engine end-to-end, preserves per-token
streaming parity with the non-stream path, applies admission control (429) and
size prechecks (400), and releases capacity when a client abandons a stream.
"""

import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.dev_model import build_dev_engine  # noqa: E402
from serving.dev_model import DEFAULT_TOKENIZER_DIR as TOK_DIR  # noqa: E402
from serving.batched_server import BatchedServingEngine  # noqa: E402
from serving.server import create_server, SaturatedError, \
    InvalidRequestError  # noqa: E402


@unittest.skipUnless(os.path.isdir(TOK_DIR), "tokenizer files not cached")
class TestBatchedFacade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inner = build_dev_engine()
        cls.eng = BatchedServingEngine(cls.inner, max_concurrency=2,
                                       state_capacity=2, max_queued=4)

    @classmethod
    def tearDownClass(cls):
        cls.eng.stop()

    # -- generation --------------------------------------------------------
    def test_complete_detail(self):
        d = self.eng.complete_detail("hello", max_new=4, temperature=0.0)
        self.assertIsInstance(d["text"], str)
        self.assertIn(d["finish_reason"], ("stop", "length"))
        self.assertGreater(d["prompt_len"], 0)
        self.assertGreaterEqual(d["completion_len"], 1)
        self.assertLessEqual(d["completion_len"], 4)

    def test_chat_detail(self):
        d = self.eng.chat_detail([{"role": "user", "content": "hi"}],
                                 max_new=3, temperature=0.0)
        self.assertIsInstance(d["text"], str)
        self.assertIn(d["finish_reason"], ("stop", "length"))
        self.assertIn("prompt_text", d)

    # -- streaming parity (stream deltas == non-stream text) ---------------
    def test_stream_complete_matches_nonstream(self):
        parts = []
        reason = None
        for delta, r in self.eng.stream_complete("hello", max_new=6,
                                                 temperature=0.0):
            if delta:
                parts.append(delta)
            if r is not None:
                reason = r
        d = self.eng.complete_detail("hello", max_new=6, temperature=0.0)
        self.assertEqual("".join(parts), d["text"])
        self.assertIn(reason, ("stop", "length"))

    def test_stream_chat_matches_nonstream(self):
        msgs = [{"role": "user", "content": "hi there"}]
        parts = []
        for delta, r in self.eng.stream_chat(msgs, max_new=5, temperature=0.0):
            if delta:
                parts.append(delta)
        d = self.eng.chat_detail(msgs, max_new=5, temperature=0.0)
        self.assertEqual("".join(parts), d["text"])

    # -- admission control -------------------------------------------------
    def test_admission_429_when_at_capacity(self):
        with self.eng._lock:
            prev = self.eng._inflight
            self.eng._inflight = self.eng.max_queued  # pretend fully booked
        try:
            with self.assertRaises(SaturatedError):
                self.eng.complete_detail("hello", max_new=2)
        finally:
            with self.eng._lock:
                self.eng._inflight = prev

    def test_precheck_400_when_too_long(self):
        with self.assertRaises(InvalidRequestError):
            self.eng.complete_detail("hello", max_new=10_000_000)

    def test_precheck_400_max_new_below_one(self):
        with self.assertRaises(InvalidRequestError):
            self.eng.complete_detail("hello", max_new=0)

    # -- abandonment releases capacity -------------------------------------
    def test_abandoning_stream_releases_capacity(self):
        base = self.eng._inflight
        it = self.eng.stream_complete("hello", max_new=6, temperature=0.0)
        next(it)  # start decoding (blocks briefly for the first token)
        self.assertGreater(self.eng._inflight, base)
        it.close()  # client disconnect -> GeneratorExit -> abort + release
        self.assertEqual(self.eng._inflight, base)


@unittest.skipUnless(os.path.isdir(TOK_DIR), "tokenizer files not cached")
class TestBatchedOverHTTP(unittest.TestCase):
    """The facade must satisfy the exact method surface the HTTP server calls,
    including SSE streaming and OpenAI usage accounting."""

    @classmethod
    def setUpClass(cls):
        cls.eng = BatchedServingEngine(build_dev_engine(), max_concurrency=2,
                                       state_capacity=2, max_queued=8)
        cls.httpd, cls.port = create_server(cls.eng, host="127.0.0.1", port=0,
                                            quiet=True, model_name="dev-batched")
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.eng.stop()

    def _post(self, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def test_chat_completion_over_http(self):
        status, obj = self._post({"messages": [{"role": "user",
                                                "content": "hi"}],
                                  "max_tokens": 4, "temperature": 0.0})
        self.assertEqual(status, 200)
        self.assertEqual(obj["choices"][0]["message"]["role"], "assistant")
        self.assertIsInstance(obj["choices"][0]["message"]["content"], str)
        self.assertGreater(obj["usage"]["total_tokens"], 0)

    def test_chat_stream_over_http(self):
        data = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                           "max_tokens": 4, "temperature": 0.0,
                           "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
        self.assertIn("data: [DONE]", raw)
        events = [json.loads(ln[6:]) for ln in raw.splitlines()
                  if ln.startswith("data: ") and ln != "data: [DONE]"]
        self.assertTrue(events)
        self.assertEqual(events[0]["choices"][0]["delta"].get("role"),
                         "assistant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
