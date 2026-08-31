"""config.env / env_config tests: parser, precedence, and consumer wiring.

Division of responsibility under test: config.env carries SLLM common info
(nodes/toolchain/shared assets); recipes keep model info incl. the weights
location (paths.local_dir)."""

from __future__ import annotations

import os
import tempfile
import unittest

import env_config
from recipes.schema import Recipe
from serving.dev_model import tiny_qwen4_exp_recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestParser(unittest.TestCase):
    def test_file_and_defaults(self):
        d = env_config.parse(os.path.join(ROOT, "config.env"))
        self.assertEqual(d["SLLM_HEAD_IP"], "192.168.0.250")
        self.assertEqual(d["SLLM_WORKER_PAIR_IP"], "10.100.25.2")
        self.assertNotIn("SLLM_MODEL_DIR_QWEN4_EXP", d)   # model info stays
                                                          # in recipes

    def test_precedence_env_file_default(self):
        with tempfile.TemporaryDirectory() as t:
            f = os.path.join(t, "config.env")
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("# c\nexport SLLM_TEST_A=fromfile\nSLLM_TEST_B=\n")
            self.assertEqual(env_config.parse(f)["SLLM_TEST_A"], "fromfile")
            # isolate: read ONLY our temp file, and control the env layer
            old_file = os.environ.get("SLLM_ENV_FILE")
            os.environ["SLLM_ENV_FILE"] = f
            os.environ["SLLM_TEST_A"] = "fromenv"
            try:
                self.assertEqual(env_config.get("SLLM_TEST_A"), "fromenv")
                self.assertEqual(env_config.get("SLLM_TEST_MISSING", "dflt"),
                                 "dflt")
                # empty value in the file = unset -> default wins
                self.assertEqual(env_config.get("SLLM_TEST_B", "d"), "d")
                del os.environ["SLLM_TEST_A"]
                self.assertEqual(env_config.get("SLLM_TEST_A"), "fromfile")
            finally:
                os.environ.pop("SLLM_TEST_A", None)
                if old_file is None:
                    os.environ.pop("SLLM_ENV_FILE", None)
                else:
                    os.environ["SLLM_ENV_FILE"] = old_file

    def test_get_path_expands(self):
        os.environ["SLLM_TEST_P"] = "~/x"
        try:
            self.assertFalse(env_config.get_path("SLLM_TEST_P").startswith("~"))
        finally:
            del os.environ["SLLM_TEST_P"]

    def test_env_file_override_var(self):
        with tempfile.TemporaryDirectory() as t:
            f = os.path.join(t, "alt.env")
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("K=v\n")
            os.environ["SLLM_ENV_FILE"] = f
            try:
                self.assertEqual(env_config.get("K"), "v")
            finally:
                del os.environ["SLLM_ENV_FILE"]


class TestConsumers(unittest.TestCase):
    def test_topology_reads_config_env_with_fallback(self):
        # subprocess isolation: reloading tp.topology in-process would swap
        # the module's class identities and break other test modules.
        import subprocess
        import sys

        # strip SLLM_* so an operator's exported overrides cannot skew the
        # "config.env value" baseline (the subprocess then sees exactly the
        # file, the documented middle layer).
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith("SLLM_")}
        base = subprocess.run(
            [sys.executable, "-c",
             "import tp.topology as t; print(t.HEAD_PAIR_IP)"],
            cwd=ROOT, capture_output=True, text=True, check=True, env=clean)
        self.assertEqual(base.stdout.strip(), "10.100.25.1")  # from config.env
        env = dict(clean, SLLM_HEAD_PAIR_IP="10.9.9.9")
        over = subprocess.run(
            [sys.executable, "-c",
             "import tp.topology as t; print(t.HEAD_PAIR_IP)"],
            cwd=ROOT, capture_output=True, text=True, check=True, env=env)
        self.assertEqual(over.stdout.strip(), "10.9.9.9")     # env wins

    def test_recipes_still_own_weights_location(self):
        for name in ("Qwen3.8-Flash-Next-FP8", "Qwen3.8-27B-FP8",
                     "DeepSeek-V4-Flash-0731", "Qwen2.5-Coder-0.5B"):
            with open(os.path.join(ROOT, "recipes", f"{name}.yaml"),
                      encoding="utf-8") as f:
                r = Recipe.from_yaml(f.read())
            self.assertTrue(r.local_dir, f"{name} must keep paths.local_dir")
        self.assertIsNone(tiny_qwen4_exp_recipe().local_dir)


if __name__ == "__main__":
    unittest.main()
