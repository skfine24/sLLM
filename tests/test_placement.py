"""KV memory placement: plan sizing, host/device backends, graceful fallback.

These run without a GPU: device placement is expected to fall back to the host
backend on this box (the .so / CUDA device are absent), mirroring the
"CPU mode uses only RAM" rule.
"""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.placement import (  # noqa: E402
    build_kv_backend, HostKVBackend, KVMemoryPlan, DeviceKVBackend,
)
from runtime.blocks import OutOfCapacity  # noqa: E402
from serving.dev_model import tiny_standard_recipe, tiny_standard_weights  # noqa: E402


class TestKVMemoryPlan(unittest.TestCase):
    def test_from_recipe_device(self):
        recipe = tiny_standard_recipe()  # memory.kv_placement default "device"
        plan = KVMemoryPlan.from_recipe(recipe, block_size=8)
        self.assertEqual(plan.placement, "device")
        self.assertGreater(plan.bytes_per_token, 0)
        self.assertGreater(plan.num_blocks, 0)
        self.assertEqual(plan.max_tokens, plan.num_blocks * 8)
        self.assertIn("placement", plan.describe())

    def test_host_override_budget(self):
        recipe = tiny_standard_recipe()
        recipe.memory.kv_placement = "host"
        recipe.memory.kv_host_bytes = 4096
        plan = KVMemoryPlan.from_recipe(recipe, block_size=4)
        self.assertEqual(plan.placement, "host")
        self.assertEqual(plan.budget_bytes, 4096)

    def test_explicit_budget_wins(self):
        recipe = tiny_standard_recipe()
        plan = KVMemoryPlan.from_recipe(recipe, block_size=8, kv_avail_bytes=12345)
        self.assertEqual(plan.budget_bytes, 12345)

    def test_bytes_per_token_geometry(self):
        recipe = tiny_standard_recipe()  # model.layers = 2 full attn, 2 KV heads, head_dim 4
        plan = KVMemoryPlan.from_recipe(recipe, block_size=8)
        # device backend stores fp32 buffers -> 4 B/element charged in the plan
        self.assertEqual(plan.bytes_per_token, 2 * 2 * 2 * 4 * 4)
        # host planning keeps the nominal BF16 (2 B) figure
        recipe.memory.kv_placement = "host"
        plan_h = KVMemoryPlan.from_recipe(recipe, block_size=8)
        self.assertEqual(plan_h.bytes_per_token, 2 * 2 * 2 * 4 * 2)


class TestHostKVBackend(unittest.TestCase):
    def setUp(self):
        recipe = tiny_standard_recipe()
        recipe.memory.kv_placement = "host"
        recipe.memory.kv_host_bytes = 1 << 20
        self.plan = KVMemoryPlan.from_recipe(recipe, block_size=2)
        self.backend = HostKVBackend(self.plan)

    def test_store_gather_free_roundtrip(self):
        k = np.zeros((2, 3, 4), dtype=np.float32)
        v = np.ones((2, 3, 4), dtype=np.float32)
        self.backend.reserve(0, 5)
        self.backend.store(0, 0, k, v)
        gk, gv = self.backend.gather(0, 0)
        np.testing.assert_array_equal(gk, k)
        np.testing.assert_array_equal(gv, v)
        self.backend.free(0)
        with self.assertRaises(KeyError):
            self.backend.gather(0, 0)

    def test_capacity_recoverable(self):
        # reserve beyond the planned block budget -> OutOfCapacity (recoverable)
        n = self.plan.num_blocks + 1
        with self.assertRaises(OutOfCapacity):
            self.backend.reserve(1, n)


class TestBackendSelection(unittest.TestCase):
    def test_host_placement_returns_host(self):
        recipe = tiny_standard_recipe()
        recipe.memory.kv_placement = "host"
        backend = build_kv_backend(KVMemoryPlan.from_recipe(recipe, block_size=4))
        self.assertIsInstance(backend, HostKVBackend)
        self.assertFalse(backend.supports_gpu)

    def test_device_placement_falls_back_without_gpu(self):
        # On this box there is no sllm_gpu.so / visible CUDA device, so device
        # placement degrades to the host backend with a warning (CPU mode rule).
        recipe = tiny_standard_recipe()
        recipe.memory.kv_placement = "device"
        plan = KVMemoryPlan.from_recipe(recipe, block_size=4)
        try:
            backend = DeviceKVBackend(plan)
            self.assertTrue(backend.supports_gpu)
        except RuntimeError:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                backend = build_kv_backend(plan)
            self.assertIsInstance(backend, HostKVBackend)
            self.assertTrue(any("device KV placement" in str(x.message) for x in w))


if __name__ == "__main__":
    unittest.main(verbosity=2)
