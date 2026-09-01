"""Recipe launch-section tests (recipe_version/name/defaults/env/command),
the `sllm <recipe>` main entry (TP1/TP2 selection, plan mode), and the
OpenAI-compatible serve mode."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.request

from recipes.schema import Recipe, RecipeError
from serving import main as sllm_main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str) -> Recipe:
    with open(os.path.join(ROOT, "recipes", name), encoding="utf-8") as f:
        return Recipe.from_yaml(f.read())


class TestLaunchSection(unittest.TestCase):
    def test_q4_launch_fields(self):
        r = _load("Qwen3.8-Flash-Next-FP8.yaml")
        self.assertEqual(r.recipe_version, "1")
        self.assertEqual(r.name, "Qwen3.8-Flash-Next-FP8")
        self.assertEqual(r.defaults["nodes"], 2)
        self.assertEqual(r.defaults["tensor_parallel"], r.tp.size)
        self.assertEqual(r.container, "sllm-node:latest")
        self.assertEqual(r.launch_env["NCCL_DEBUG"], "WARN")
        self.assertNotIn("command", r.meta)      # consumed, not meta
        self.assertNotIn("defaults", r.meta)

    def test_render_command(self):
        r = _load("Qwen3.8-Flash-Next-FP8.yaml")
        cmd = r.render_command(recipe="recipes/x.yaml", mode="plan",
                               max_new=32)
        self.assertIn("--nodes 2", cmd)
        self.assertIn("--port 8002", cmd)
        self.assertIn("recipes/x.yaml", cmd)

    def test_tp_consistency_rejected(self):
        doc = {
            "model_id": "x/y", "arch": "qwen3_5",
            "tp": {"size": 2},
            "defaults": {"tensor_parallel": 1},
        }
        with self.assertRaises(RecipeError):
            Recipe.from_dict(doc)

    def test_old_style_minimal_recipe(self):
        r = Recipe.from_dict({"model_id": "x/y", "arch": "qwen3_5"})
        self.assertEqual(r.recipe_version, "1")
        self.assertEqual(r.name, r.model_id)   # name falls back to model_id
        self.assertIsNone(r.command)
        self.assertEqual(r.defaults, {})


class TestSllmMain(unittest.TestCase):
    def test_plan_q4_nodes2(self):
        rc = sllm_main.main(["Qwen3.8-Flash-Next-FP8.yaml", "--mode", "plan"])
        self.assertEqual(rc, 0)

    def test_one_node_rejected_for_tp2(self):
        with self.assertRaises(SystemExit) as cm:
            sllm_main.main(["Qwen3.8-Flash-Next-FP8.yaml", "--tp", "1"])
        self.assertIn("173", str(cm.exception))   # arithmetic, not a guess
        self.assertIn("110", str(cm.exception))   # node budget

    def test_tp1_allowed_when_weights_fit(self):
        # 27B (31 GB) on one node passes the memory arithmetic even though
        # the recipe's designed shape is tp2.
        self.assertEqual(sllm_main._check_nodes(
            _load("Qwen3.8-27B-FP8.yaml"), 1), 1)

    def test_tp_alias(self):
        rc = sllm_main.main(["Qwen2.5-Coder-0.5B.yaml", "--tp", "1",
                             "--mode", "plan"])
        self.assertEqual(rc, 0)

    def test_version_flag_without_recipe(self):
        rc = sllm_main.main(["--version"])
        self.assertEqual(rc, 0)

    def test_resolve_precedence(self):
        r = _load("Qwen3.8-Flash-Next-FP8.yaml")
        self.assertEqual(sllm_main.resolve(r, None, None, None),
                         (2, "0.0.0.0", "8002"))          # recipe defaults
        self.assertEqual(sllm_main.resolve(r, 2, "127.0.0.1", "9999")[1:],
                         ("127.0.0.1", "9999"))           # CLI wins

    def test_tp0_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            sllm_main.main(["Qwen2.5-Coder-0.5B.yaml", "--tp", "0"])
        self.assertIn(">= 1", str(cm.exception))

    def test_run_real_q4_is_gated(self):
        with self.assertRaises(SystemExit) as cm:
            sllm_main.main(["Qwen3.8-Flash-Next-FP8.yaml", "--mode", "run"])
        self.assertIn("C phase", str(cm.exception))

    def test_run_passes_chat_messages(self):
        # regression: run mode used to pass a raw str where chat() expects
        # a list of message dicts (crashed every engine).
        seen = {}

        class Stub:
            def chat(self, messages, max_new=0):
                seen["messages"] = messages
                seen["max_new"] = max_new
                return "ok"

        orig = sllm_main._build_engine
        sllm_main._build_engine = lambda r, d: Stub()
        try:
            rc = sllm_main.run_model(_load("Qwen2.5-Coder-0.5B.yaml"),
                                     None, "ping", 4)
        finally:
            sllm_main._build_engine = orig
        self.assertEqual(rc, 0)
        self.assertEqual(seen["messages"],
                         [{"role": "user", "content": "ping"}])
        self.assertEqual(seen["max_new"], 4)


class TestOpenAIServe(unittest.TestCase):
    """serve mode output shape: OpenAI-compatible objects on the stdlib
    server (dev-tiny engine, real HTTP round-trip on an ephemeral port)."""

    @classmethod
    def setUpClass(cls):
        from serving.dev_model import build_dev_engine
        from serving.server import create_server
        cls.httpd, cls.port = create_server(build_dev_engine(),
                                            host="127.0.0.1", port=0,
                                            model_name="Qwen2.5-Coder-0.5B")
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    def test_models_endpoint(self):
        data = self._get("/v1/models")
        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["id"], "Qwen2.5-Coder-0.5B")
        self.assertEqual(data["data"][0]["owned_by"], "sllm")

    def test_chat_completion_shape(self):
        obj = self._post("/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 4})
        self.assertTrue(obj["id"].startswith("chatcmpl-"))
        self.assertEqual(obj["object"], "chat.completion")
        self.assertEqual(obj["model"], "Qwen2.5-Coder-0.5B")
        self.assertEqual(obj["choices"][0]["message"]["role"], "assistant")
        self.assertIn(obj["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertEqual(
            obj["usage"]["total_tokens"],
            obj["usage"]["prompt_tokens"] + obj["usage"]["completion_tokens"])

    def test_requested_model_echoed(self):
        obj = self._post("/v1/completions",
                         {"model": "anything", "prompt": "a", "max_tokens": 2})
        self.assertEqual(obj["model"], "anything")
        self.assertEqual(obj["object"], "text_completion")

    def test_stream_is_sse(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "x"}],
                             "max_tokens": 2, "temperature": 0.0,
                             "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.headers.get("Content-Type", ""),
                             "text/event-stream; charset=utf-8")
            body = r.read().decode("utf-8")
        self.assertIn("data: [DONE]", body)
        self.assertIn("chat.completion.chunk", body)


if __name__ == "__main__":
    unittest.main()
