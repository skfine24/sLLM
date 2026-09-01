"""GPU decode wiring: ReferenceModel(use_gpu=True) runs decode steps on the
GPU kernels (standard + hybrid) and stays identical to the numpy incremental
path. Device-resident decode (resident weights + on-device KV, one sync per
step) is covered for the standard path. Runs only where
kernels/cuda/sllm_gpu.so is built."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ref import incremental as inc  # noqa: E402
from kernels.standard_decode import gpu_standard_decode_step  # noqa: E402
from kernels.hybrid_decode import gpu_hybrid_decode_step  # noqa: E402
from serving.dev_model import (  # noqa: E402
    tiny_recipe, tiny_weights, tiny_standard_recipe, tiny_standard_weights,
)
from serving.executor import ReferenceModel, generate  # noqa: E402

SO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "cuda", "sllm_gpu.so"))
need_gpu = os.path.isfile(SO)


class TestGpuPolicyAuto(unittest.TestCase):
    """GPU-by-default + auto CPU fallback + forced-CPU policy (runs
    EVERYWHERE: no .so/CUDA needed). The default is AUTO: use_gpu=True, which
    the decode loop resolves against _gpu_available() and falls back to numpy
    when no CUDA/.so is present; SLLM_USE_GPU=0 forces CPU."""

    def _model(self, use_gpu_env=None, placement=None):
        saved = {k: os.environ.get(k) for k in ("SLLM_USE_GPU", "SLLM_PLACEMENT")}
        for k in ("SLLM_USE_GPU", "SLLM_PLACEMENT"):
            os.environ.pop(k, None)
        if use_gpu_env is not None:
            os.environ["SLLM_USE_GPU"] = use_gpu_env
        if placement is not None:
            os.environ["SLLM_PLACEMENT"] = placement
        try:
            return ReferenceModel(tiny_standard_recipe(),
                                  tiny_standard_weights(np.random.default_rng(3)))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_auto_default_is_gpu_mode(self):
        m = self._model(None)               # env unset -> AUTO + device
        self.assertTrue(m.use_gpu)
        self.assertEqual(m.gpu_mode, "auto")
        self.assertEqual(m.placement, "device")
        self.assertTrue(m.resident_preferred)

    def test_env_zero_forces_cpu(self):
        m = self._model("0")
        self.assertFalse(m.use_gpu)
        self.assertEqual(m.gpu_mode, "off")

    def test_env_one_forces_gpu(self):
        m = self._model("1")
        self.assertTrue(m.use_gpu)
        self.assertEqual(m.gpu_mode, "force")

    def test_placement_um_forces_cpu(self):
        m = self._model(None, "um")
        self.assertEqual(m.placement, "um")
        self.assertFalse(m.use_gpu)
        self.assertEqual(m.gpu_mode, "off")

    def test_auto_generation_equals_explicit_cpu(self):
        # auto (use_gpu=True) either uses the GPU or falls back to numpy; the
        # emitted sequence must be identical to a forced-CPU run either way.
        ids = list(range(1, 9))
        a = self._model(None)
        b = self._model("0")
        ga = generate(a, None, ids, max_new=6, temperature=0.0, seed=1)
        gb = generate(b, None, ids, max_new=6, temperature=0.0, seed=1)
        self.assertEqual(ga, gb)


@unittest.skipUnless(need_gpu, "sllm_gpu.so not built (no GPU toolchain)")
class TestGpuStandardDecode(unittest.TestCase):
    def setUp(self):
        from kernels import _sllm_cuda as ck
        self.assertTrue(ck.device_count() >= 1)
        self.recipe = tiny_standard_recipe()
        self.weights = tiny_standard_weights(np.random.default_rng(3))
        self.ids = list(range(1, 9))

    def test_step_matches_numpy(self):
        cache_np, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        cache_gpu, _ = inc.prefill(self.ids, self.weights, self.recipe)
        nxt = int(np.argmax(L0[0, -1]))
        for _ in range(4):
            L_np = inc.decode_step(cache_np, self.weights, self.recipe, nxt)
            L_gpu = gpu_standard_decode_step(cache_gpu, self.weights, self.recipe, nxt)
            np.testing.assert_allclose(L_gpu, L_np, rtol=5e-3, atol=5e-3)
            self.assertEqual(int(np.argmax(L_gpu)), int(np.argmax(L_np)))
            nxt = int(np.argmax(L_np))

    def test_engine_gpu_generate_identical(self):
        m_cpu = ReferenceModel(self.recipe, self.weights, use_gpu=False)
        m_gpu = ReferenceModel(self.recipe, self.weights, use_gpu=True)
        a = generate(m_cpu, None, list(self.ids), max_new=12, temperature=0.0, seed=1)
        b = generate(m_gpu, None, list(self.ids), max_new=12, temperature=0.0, seed=1)
        self.assertEqual(a, b)


@unittest.skipUnless(need_gpu, "sllm_gpu.so not built (no GPU toolchain)")
class TestGpuHybridWiring(unittest.TestCase):
    def setUp(self):
        from kernels import _sllm_cuda as ck
        self.assertTrue(ck.device_count() >= 1)
        self.recipe = tiny_recipe()
        self.weights = tiny_weights(np.random.default_rng(42))
        self.ids = list(range(1, 8))

    def test_step_matches_numpy(self):
        cache_np, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        cache_gpu, _ = inc.prefill(self.ids, self.weights, self.recipe)
        nxt = int(np.argmax(L0[0, -1]))
        for _ in range(4):
            L_np = inc.decode_step(cache_np, self.weights, self.recipe, nxt)
            L_gpu = gpu_hybrid_decode_step(cache_gpu, self.weights, self.recipe, nxt)
            np.testing.assert_allclose(L_gpu, L_np, rtol=5e-3, atol=5e-3)
            self.assertEqual(int(np.argmax(L_gpu)), int(np.argmax(L_np)))
            nxt = int(np.argmax(L_np))

    def test_engine_gpu_deterministic_bounded(self):
        m = ReferenceModel(self.recipe, self.weights, use_gpu=True)
        a = generate(m, None, list(self.ids), max_new=10, temperature=0.0, seed=1)
        b = generate(m, None, list(self.ids), max_new=10, temperature=0.0, seed=1)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), len(self.ids) + 10)
        self.assertGreaterEqual(len(a), len(self.ids))


@unittest.skipUnless(need_gpu, "sllm_gpu.so not built (no GPU toolchain)")
class TestDeviceResidentStandard(unittest.TestCase):
    """Device-resident decode: weights uploaded once, KV on the GPU, ONE sync
    per step; logits must match the numpy oracle and the host cache must stay
    authoritative (seamless fallback)."""

    def setUp(self):
        from kernels import _sllm_cuda as ck
        self.assertTrue(ck.device_count() >= 1)
        self.recipe = tiny_standard_recipe()
        self.weights = tiny_standard_weights(np.random.default_rng(3))
        self.ids = list(range(1, 9))

    def _fresh(self, ids):
        cache, L0 = inc.prefill(ids, self.weights, self.recipe)
        return cache, L0

    def test_resident_step_matches_numpy(self):
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable

        table = DeviceWeightTable(self.weights, self.recipe)
        try:
            cache_np, L0 = self._fresh(self.ids)
            cache_dev, _ = self._fresh(self.ids)
            state = DeviceDecodeState(table, cache_dev, self.recipe)
            nxt = int(np.argmax(L0[0, -1]))
            for _ in range(6):
                L_np = inc.decode_step(cache_np, self.weights, self.recipe, nxt)
                L_dev = state.step(nxt)
                np.testing.assert_allclose(L_dev, L_np, rtol=5e-3, atol=5e-3)
                self.assertEqual(int(np.argmax(L_dev)), int(np.argmax(L_np)))
                nxt = int(np.argmax(L_np))
            # host mirror is exact: shapes/positions tracked identically
            self.assertEqual(cache_dev.n_ctx, cache_np.n_ctx)
            for i in cache_np.k:
                self.assertEqual(cache_dev.k[i].shape, cache_np.k[i].shape)
                self.assertEqual(cache_dev.v[i].shape, cache_np.v[i].shape)
            state.free()
        finally:
            table.free()

    def test_kv_capacity_growth(self):
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable

        table = DeviceWeightTable(self.weights, self.recipe)
        try:
            cache, L0 = self._fresh(self.ids)
            state = DeviceDecodeState(table, cache, self.recipe)
            cap0 = state.cap
            self.assertEqual(cap0, len(self.ids))
            nxt = int(np.argmax(L0[0, -1]))
            for _ in range(cap0):  # the first step crosses the capacity boundary
                L = state.step(nxt)
                nxt = int(np.argmax(L))
            self.assertEqual(state.cap, 2 * cap0)
            self.assertEqual(cache.n_ctx, 2 * cap0)
            self.assertEqual(cache.k[0].shape[1], 2 * cap0)
            self.assertTrue(np.all(np.isfinite(L)))
            state.free()
        finally:
            table.free()

    def test_engine_resident_generate_identical(self):
        old = os.environ.get("SLLM_GPU_RESIDENT")
        os.environ["SLLM_GPU_RESIDENT"] = "1"
        try:
            m_cpu = ReferenceModel(self.recipe, self.weights, use_gpu=False)
            m_gpu = ReferenceModel(self.recipe, self.weights, use_gpu=True)
            a = generate(m_cpu, None, list(self.ids), max_new=12, temperature=0.0, seed=1)
            b = generate(m_gpu, None, list(self.ids), max_new=12, temperature=0.0, seed=1)
            self.assertEqual(a, b)
            self.assertFalse(m_gpu._resident_off)  # resident path actually used
            self.assertIsNotNone(m_gpu._dev_table)
        finally:
            if old is None:
                os.environ.pop("SLLM_GPU_RESIDENT", None)
            else:
                os.environ["SLLM_GPU_RESIDENT"] = old

    def test_resident_env_disable_uses_transfer_path(self):
        old = os.environ.get("SLLM_GPU_RESIDENT")
        os.environ["SLLM_GPU_RESIDENT"] = "0"
        try:
            m_cpu = ReferenceModel(self.recipe, self.weights, use_gpu=False)
            m_gpu = ReferenceModel(self.recipe, self.weights, use_gpu=True)
            a = generate(m_cpu, None, list(self.ids), max_new=8, temperature=0.0, seed=1)
            b = generate(m_gpu, None, list(self.ids), max_new=8, temperature=0.0, seed=1)
            self.assertEqual(a, b)
            self.assertIsNone(m_gpu._dev_table)  # resident never built
        finally:
            if old is None:
                os.environ.pop("SLLM_GPU_RESIDENT", None)
            else:
                os.environ["SLLM_GPU_RESIDENT"] = old

    def test_two_sequences_interleaved(self):
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable

        table = DeviceWeightTable(self.weights, self.recipe)
        try:
            ids_a, ids_b = list(range(1, 9)), list(range(3, 11))
            ca_np, La = self._fresh(ids_a)
            cb_np, Lb = self._fresh(ids_b)
            ca, _ = self._fresh(ids_a)
            cb, _ = self._fresh(ids_b)
            sa = DeviceDecodeState(table, ca, self.recipe)
            sb = DeviceDecodeState(table, cb, self.recipe)
            na, nb = int(np.argmax(La[0, -1])), int(np.argmax(Lb[0, -1]))
            for _ in range(5):
                La_np = inc.decode_step(ca_np, self.weights, self.recipe, na)
                Lb_np = inc.decode_step(cb_np, self.weights, self.recipe, nb)
                La_g = sa.step(na)          # interleave the two sequences on
                Lb_g = sb.step(nb)          # one shared weight table
                np.testing.assert_allclose(La_g, La_np, rtol=5e-3, atol=5e-3)
                np.testing.assert_allclose(Lb_g, Lb_np, rtol=5e-3, atol=5e-3)
                self.assertEqual(int(np.argmax(La_g)), int(np.argmax(La_np)))
                self.assertEqual(int(np.argmax(Lb_g)), int(np.argmax(Lb_np)))
                na, nb = int(np.argmax(La_np)), int(np.argmax(Lb_np))
            sa.free()
            sb.free()
        finally:
            table.free()


class TestBf16Conversion(unittest.TestCase):
    """Pure-numpy host helpers (no .so required)."""

    def test_round_to_nearest_even(self):
        from kernels import _sllm_cuda as ck
        x = np.array([1.0, 2.0, -3.5, 0.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9],
                     dtype=np.float32)
        back = (ck.to_bf16(x).astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(back[:4], x[:4])       # exact values
        self.assertEqual(float(back[4]), 1.0)                # tie -> even (down)
        self.assertAlmostEqual(float(back[5]), 1.0 + 2 ** -7, delta=0.0)

    def test_engine_bad_dtype_falls_back(self):
        recipe = tiny_standard_recipe()
        weights = tiny_standard_weights(np.random.default_rng(3))
        m_cpu = ReferenceModel(recipe, weights, use_gpu=False)
        m_gpu = ReferenceModel(recipe, weights, use_gpu=True, gpu_dtype="nope")
        a = generate(m_cpu, None, [1, 2, 3, 4], max_new=6, temperature=0.0)
        b = generate(m_gpu, None, [1, 2, 3, 4], max_new=6, temperature=0.0)
        self.assertEqual(a, b)


def _partial_rotary_recipe():
    """tiny standard recipe with partial_rotary_factor 0.5 (head_dim 4 -> rot 2)."""
    from recipes.schema import Recipe
    d = {
        "model_id": "tiny/qwen2-partrot", "arch": "qwen2", "dtype": "bfloat16",
        "text": {
            "prefix": "model", "tie_word_embeddings": True,
            "hidden_size": 16, "num_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "full_attention": {
                "kernel": "standard_gqa",
                "num_heads": 4, "num_kv_heads": 2, "head_dim": 4,
                "output_gate": False,
                "rope": {"type": "default", "theta": 1e4,
                         "partial_rotary_factor": 0.5},
            },
            "mlp": {"type": "dense", "intermediate_size": 32, "hidden_act": "silu"},
            "vocab_size": 64, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
        },
        "mtp": {"enabled": False}, "vision": {"enabled": False},
        "tp": {"size": 2},
    }
    return Recipe.from_dict(d)


class TestPartialRotaryReference(unittest.TestCase):
    """Standard path must honour recipe rotary_dim (partial RoPE), no GPU."""

    def setUp(self):
        self.recipe = _partial_rotary_recipe()
        self.weights = tiny_standard_weights(np.random.default_rng(5))
        self.ids = list(range(1, 8))

    def test_rotary_dim_and_parity(self):
        from ref import standard as st
        self.assertEqual(self.recipe.rotary_dim(), 2)
        cache, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        full = st.standard_model_forward(np.asarray(self.ids)[None, :],
                                         self.weights, self.recipe)
        np.testing.assert_allclose(L0, full, rtol=1e-6, atol=1e-6)
        self.assertEqual(cache.rot, 2)
        for _ in range(3):
            L = inc.decode_step(cache, self.weights, self.recipe, 3)
            self.assertEqual(L.shape, (64,))
            self.assertTrue(np.all(np.isfinite(L)))


@unittest.skipUnless(need_gpu, "sllm_gpu.so not built (no GPU toolchain)")
class TestResidentDtype(unittest.TestCase):
    """Fused + dtype-tagged resident engine: fp32 stays at fp32 tolerance,
    bf16 (vLLM-style split: bf16 weights/KV, fp32 residual/logits) matches the
    numpy oracle within bf16 tolerance and keeps greedy argmax."""

    def setUp(self):
        from kernels import _sllm_cuda as ck
        self.assertTrue(ck.device_count() >= 1)
        self.recipe = tiny_standard_recipe()
        self.weights = tiny_standard_weights(np.random.default_rng(3))
        self.ids = list(range(1, 9))

    def _steps_match_numpy(self, dtype, rtol, atol):
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable

        table = DeviceWeightTable(self.weights, self.recipe, dtype=dtype)
        try:
            cache_np, L0 = inc.prefill(self.ids, self.weights, self.recipe)
            cache, _ = inc.prefill(self.ids, self.weights, self.recipe)
            state = DeviceDecodeState(table, cache, self.recipe)
            nxt = int(np.argmax(L0[0, -1]))
            for _ in range(5):
                L_np = inc.decode_step(cache_np, self.weights, self.recipe, nxt)
                L = state.step(nxt)
                np.testing.assert_allclose(L, L_np, rtol=rtol, atol=atol)
                self.assertEqual(int(np.argmax(L)), int(np.argmax(L_np)))
                nxt = int(np.argmax(L_np))
            self.assertEqual(cache.n_ctx, cache_np.n_ctx)
            state.free()
        finally:
            table.free()

    def test_resident_fp32_matches_numpy(self):
        self._steps_match_numpy("fp32", 5e-3, 5e-3)

    def test_resident_bf16_matches_numpy(self):
        self._steps_match_numpy("bf16", 3e-2, 3e-2)

    def test_resident_bf16_kv_growth(self):
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable

        table = DeviceWeightTable(self.weights, self.recipe, dtype="bf16")
        try:
            cache, L0 = inc.prefill(self.ids, self.weights, self.recipe)
            state = DeviceDecodeState(table, cache, self.recipe)
            cap0 = state.cap
            nxt = int(np.argmax(L0[0, -1]))
            for _ in range(cap0):
                L = state.step(nxt)
                nxt = int(np.argmax(L))
            self.assertEqual(state.cap, 2 * cap0)
            self.assertEqual(cache.k[0].shape[1], 2 * cap0)
            self.assertTrue(np.all(np.isfinite(L)))
            state.free()
        finally:
            table.free()

    def test_engine_bf16_generate_identical(self):
        m_cpu = ReferenceModel(self.recipe, self.weights, use_gpu=False)
        m_gpu = ReferenceModel(self.recipe, self.weights, use_gpu=True,
                               gpu_dtype="bf16")
        a = generate(m_cpu, None, list(self.ids), max_new=12, temperature=0.0, seed=1)
        b = generate(m_gpu, None, list(self.ids), max_new=12, temperature=0.0, seed=1)
        self.assertEqual(a, b)
        self.assertFalse(m_gpu._resident_off)


@unittest.skipUnless(need_gpu, "sllm_gpu.so not built (no GPU toolchain)")
class TestPartialRotaryGpu(unittest.TestCase):
    """Resident + transfer GPU paths must honour partial rotary too."""

    def setUp(self):
        from kernels import _sllm_cuda as ck
        self.assertTrue(ck.device_count() >= 1)
        self.recipe = _partial_rotary_recipe()
        self.weights = tiny_standard_weights(np.random.default_rng(5))
        self.ids = list(range(1, 8))

    def test_resident_matches_numpy(self):
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable

        table = DeviceWeightTable(self.weights, self.recipe)
        try:
            cache_np, L0 = inc.prefill(self.ids, self.weights, self.recipe)
            cache, _ = inc.prefill(self.ids, self.weights, self.recipe)
            state = DeviceDecodeState(table, cache, self.recipe)
            self.assertEqual(state.rot, 2)
            nxt = int(np.argmax(L0[0, -1]))
            for _ in range(5):
                L_np = inc.decode_step(cache_np, self.weights, self.recipe, nxt)
                L = state.step(nxt)
                np.testing.assert_allclose(L, L_np, rtol=5e-3, atol=5e-3)
                self.assertEqual(int(np.argmax(L)), int(np.argmax(L_np)))
                nxt = int(np.argmax(L_np))
            state.free()
        finally:
            table.free()

    def test_transfer_path_matches_numpy(self):
        cache_np, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        cache_tf, _ = inc.prefill(self.ids, self.weights, self.recipe)
        nxt = int(np.argmax(L0[0, -1]))
        for _ in range(4):
            L_np = inc.decode_step(cache_np, self.weights, self.recipe, nxt)
            L_tf = gpu_standard_decode_step(cache_tf, self.weights, self.recipe, nxt)
            np.testing.assert_allclose(L_tf, L_np, rtol=5e-3, atol=5e-3)
            nxt = int(np.argmax(L_np))


if __name__ == "__main__":
    unittest.main(verbosity=2)
