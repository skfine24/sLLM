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


if __name__ == "__main__":
    unittest.main(verbosity=2)
