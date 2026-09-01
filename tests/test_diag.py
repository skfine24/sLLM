"""Diagonostic layer tests (vLLM-style engine messaging).

Gates:
  * level gating + stderr routing (stdout stays clean)
  * engine startup banner for the arch registry (qwen4_exp, deepseek_v4)
  * per-generation INFO stats + DEBUG token lines from executor.generate
  * HTTP request lines (quiet gates them off)
All via diag -> stderr capture; existing stdout-only behavior untouched.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import unittest

from serving import diag
from serving.dev_model import (build_dev_deepseek_v4_engine,
                               build_dev_qwen4_exp_engine)
from serving.executor import InferenceEngine


def _capture_stderr(fn):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        fn()
    return buf.getvalue()


class TestDiagLevels(unittest.TestCase):
    def setUp(self):
        self._level = diag.get_level()

    def tearDown(self):
        diag.set_level(self._level)
        diag.set_quiet(False)

    def test_info_to_stderr_quiet_gates(self):
        out = _capture_stderr(lambda: diag.info("t", "hello-info"))
        self.assertIn("hello-info", out)
        self.assertIn("[t]", out)
        diag.set_quiet(True)
        out = _capture_stderr(lambda: diag.info("t", "muted"))
        self.assertNotIn("muted", out)

    def test_debug_gated_by_level(self):
        diag.set_level("WARNING")
        out = _capture_stderr(lambda: diag.debug("t", "dbg"))
        self.assertNotIn("dbg", out)
        diag.set_level("DEBUG")
        out = _capture_stderr(lambda: diag.debug("t", "dbg-visible"))
        self.assertIn("dbg-visible", out)

    def test_helpers(self):
        self.assertIn("tokens/s", diag.tps(50, 1.0))
        self.assertIn("GiB", diag.gib(2 ** 30))
        with diag.stopwatch() as sw:
            self.assertGreater(sw.elapsed if False else 0, -1)  # api smoke
        self.assertGreaterEqual(sw.elapsed, 0.0)


class TestEngineBanner(unittest.TestCase):
    def setUp(self):
        self._level = diag.get_level()
        diag.set_level("INFO")

    def tearDown(self):
        diag.set_level(self._level)
        diag.set_quiet(False)

    def test_qwen4_exp_banner_lines(self):
        eng = build_dev_qwen4_exp_engine()
        out = _capture_stderr(eng.show_banner)
        self.assertIn("qwen4_exp", out)
        self.assertIn("architecture", out)
        self.assertIn("weights", out)
        self.assertIn("cache", out)

    def test_deepseek_v4_banner_lines(self):
        eng = build_dev_deepseek_v4_engine()
        out = _capture_stderr(eng.show_banner)
        self.assertIn("deepseek_v4", out)
        self.assertIn("window", out)
        self.assertIn("moe", out)

    def test_describe_is_uniform_sorted_keys(self):
        for build in (build_dev_qwen4_exp_engine, build_dev_deepseek_v4_engine):
            eng = build()
            lines = eng.describe()
            self.assertGreaterEqual(len(lines), 5)
            self.assertTrue(lines[0].startswith("version"))
            self.assertTrue(lines[1].startswith("architecture"))

    def test_banner_shows_version(self):
        eng = build_dev_qwen4_exp_engine()
        out = _capture_stderr(eng.show_banner)
        self.assertIn("version", out)
        from serving.version import VERSION
        self.assertIn(VERSION, out)


class TestGenerationStats(unittest.TestCase):
    def setUp(self):
        self._level = diag.get_level()
        diag.set_level("DEBUG")
        self.eng = build_dev_qwen4_exp_engine()

    def tearDown(self):
        diag.set_level(self._level)
        diag.set_quiet(False)

    def test_generate_emits_info_stats_and_debug_tokens(self):
        out = _capture_stderr(lambda: self.eng.complete("hello there",
                                                        max_new=4))
        self.assertIn("prompt=", out)
        self.assertIn("tokens/s", out)
        self.assertIn("finish=", out)
        self.assertIn("step=0", out)     # DEBUG per-token lines
        self.assertIn("top1=", out)

    def test_stats_do_not_pollute_stdout(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.eng.complete("hello", max_new=2)
        self.assertNotIn("tokens/s", buf.getvalue())


class TestHttpLines(unittest.TestCase):
    def setUp(self):
        self._level = diag.get_level()
        diag.set_level("INFO")

    def tearDown(self):
        diag.set_level(self._level)
        diag.set_quiet(False)

    def _request(self, quiet: bool) -> str:
        from serving.server import create_server
        engine = build_dev_qwen4_exp_engine()
        server, port = create_server(engine, host="127.0.0.1", port=0,
                                     quiet=quiet)
        thread = threading.Thread(target=server.serve_forever,
                                  daemon=True)
        thread.start()
        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            body = json.dumps({"messages": [{"role": "user",
                                             "content": "hi"}],
                               "max_tokens": 2})
            # diag.capture() is thread-safe: the handler thread writes into
            # the same buffer even though it outlives this caller's frame.
            with diag.capture() as buf:
                conn.request("POST", "/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
                resp = conn.getresponse()
                resp.read()
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertEqual(resp.status, 200)
        return buf.getvalue()

    def test_http_line_emitted_when_not_quiet(self):
        out = self._request(quiet=False)
        self.assertIn("POST /v1/chat/completions -> 200", out)

    def test_http_line_silent_when_quiet(self):
        out = self._request(quiet=True)
        self.assertNotIn("/v1/chat/completions", out)


if __name__ == "__main__":
    unittest.main()
