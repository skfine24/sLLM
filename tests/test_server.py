"""HTTP front-end smoke tests against the tiny reference model."""

import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.dev_model import build_dev_engine  # noqa: E402
from serving import server as srv  # noqa: E402
from serving.dev_model import DEFAULT_TOKENIZER_DIR as TOK_DIR  # noqa: E402

# same chain as serving.dev_model: Q27B_TOKENIZER_DIR env > config.env >
# dev fallback; the dev engine needs the real tokenizer files.


@unittest.skipUnless(os.path.isdir(TOK_DIR), "tokenizer files not cached")
class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = build_dev_engine()
        cls.httpd, cls.port = srv.create_server(cls.engine, host="127.0.0.1", port=0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post(self, path: str, body: dict):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=10) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read().decode("utf-8"))["status"], "ok")

    def test_chat_completion(self):
        status, obj = self._post("/v1/chat/completions", {
            "model": "dev-tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4, "temperature": 0.0,
        })
        self.assertEqual(status, 200)
        choice = obj["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIsInstance(choice["message"]["content"], str)

    def test_completion(self):
        status, obj = self._post("/v1/completions", {
            "prompt": "hello", "max_tokens": 4, "temperature": 0.0,
        })
        self.assertEqual(status, 200)
        self.assertIsInstance(obj["choices"][0]["text"], str)

    def test_bad_json(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=b"not json",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(ctx.exception.code, 400)

    def test_missing_messages(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/v1/chat/completions", {"max_tokens": 4})
        self.assertEqual(ctx.exception.code, 400)

    def _post_stream(self, path: str, body: dict) -> str:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            self.assertEqual(r.headers.get("Content-Type", ""),
                             "text/event-stream; charset=utf-8")
            return r.read().decode("utf-8")

    def test_chat_stream_sse(self):
        body = _stream_chat_body("hi there", max_tokens=4)
        raw = self._post_stream("/v1/chat/completions", body)
        self.assertIn("data: [DONE]", raw)
        events = [ln[6:] for ln in raw.splitlines() if ln.startswith("data: ")
                  and ln != "data: [DONE]"]
        parsed = [json.loads(e) for e in events]
        self.assertTrue(parsed)
        first = parsed[0]["choices"][0]
        self.assertEqual(first["delta"].get("role"), "assistant")
        # join deltas -> must equal the non-stream text completion
        deltas = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in parsed
            if c["choices"][0].get("delta", {}).get("content"))
        reason = next(c["choices"][0]["finish_reason"] for c in parsed
                      if c["choices"][0]["finish_reason"])
        self.assertIn(reason, ("stop", "length"))
        d = self.engine.chat_detail(messages=[{"role": "user", "content":
                                                "hi there"}],
                                    max_new=4, temperature=0.0)
        self.assertEqual(deltas, d["text"])

    def test_completion_stream_sse(self):
        raw = self._post_stream("/v1/completions", {
            "prompt": "hello", "max_tokens": 4, "temperature": 0.0,
            "stream": True})
        self.assertIn("data: [DONE]", raw)
        events = [json.loads(ln[6:]) for ln in raw.splitlines()
                  if ln.startswith("data: ") and ln != "data: [DONE]"]
        deltas = "".join(c["choices"][0].get("text", "") for c in events)
        d = self.engine.complete_detail("hello", max_new=4, temperature=0.0)
        self.assertEqual(deltas, d["text"])


def _stream_chat_body(content, max_tokens: int, **extra) -> dict:
    return {"model": "dev-tiny", "messages": [{"role": "user",
                                               "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            **extra}


def _build_tool_turn():
    """A valid DSML tool-call completion text, built from the module's own
    (unicode) special-token constants so tests never hardcode those literals."""
    from serving import encoding_dsv4 as enc
    d = enc.dsml_token
    blk = enc.tool_calls_block_name
    block = (
        "<" + d + blk + ">\n"
        "<" + d + "invoke" + ' name="calculator">\n'
        "<" + d + 'parameter name="expr" string="true">40+2</' + d
        + 'parameter>\n'
        "</" + d + "invoke" + ">\n"
        "</" + d + blk + ">"
    )
    return "summary" + "\n\n" + block


class TestToolExecutionLoop(unittest.TestCase):
    """chat_tools DSML loop: parse a tool-call turn, invoke a registered
    callable, feed the result back, regenerate. Uses a scripted 2-turn model
    so the loop mechanics are tested without real DeepSeek weights."""

    def test_loop_invokes_tool_and_regenerates(self):
        from serving.executor import InferenceEngine

        class _FakeModel:
            _is_dsv4 = True

        turns = [_build_tool_turn(), "final answer"]

        class _E(InferenceEngine):
            def _dsv4_text_chat_detail(self, messages, **kw):
                t = turns.pop(0)
                return {"text": t}

        eng = _E(_FakeModel(), object())
        called = {}

        def calculator(**kw):
            called.update(kw)
            return kw.get("expr")

        out = eng.chat_tools([{"role": "user", "content": "compute 40+2"}],
                             tools={"calculator": calculator}, max_turns=3)
        self.assertEqual(called, {"expr": "40+2"})
        self.assertEqual(out["assistant"]["content"], "final answer")
        # loop took 2 turns
        self.assertEqual(out["turns"], 2)
        # the tool result was merged into the message ledger
        flat = json.dumps(out["messages"], ensure_ascii=False)
        self.assertIn("40+2", flat)

    def test_no_tools_returns_plain(self):
        from serving.executor import InferenceEngine

        class _FakeModel:
            _is_dsv4 = True

        class _E(InferenceEngine):
            def _dsv4_text_chat_detail(self, messages, **kw):
                return {"text": "no tool text"}

        eng = _E(_FakeModel(), object())
        out = eng.chat_tools([{"role": "user", "content": "hi"}],
                             tools={}, max_turns=3)
        self.assertEqual(out["assistant"]["content"], "no tool text")
        self.assertEqual(out["turns"], 1)


    def test_unknown_tool_reported(self):
        from serving.executor import InferenceEngine

        class _FakeModel:
            _is_dsv4 = True

        class _E(InferenceEngine):
            def _dsv4_text_chat_detail(self, messages, **kw):
                return {"text": _build_tool_turn()}

        eng = _E(_FakeModel(), object())
        # register a DIFFERENT tool; the model calls "calculator" (unregistered)
        out = eng.chat_tools([{"role": "user", "content": "x"}],
                             tools={"other": lambda: 1}, max_turns=2)
        flat = json.dumps(out["messages"], ensure_ascii=False)
        self.assertIn("unknown tool", flat)


if __name__ == "__main__":
    unittest.main()
