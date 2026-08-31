"""T0 parity tests for the qwen3_5 numpy reference.

Runs on plain python3 + numpy (torch-free), so it can be executed on the dev
machine and in CI. The key invariant is that the chunked (prefill) and the
recurrent (decode) gated-delta-rule paths produce numerically equal outputs.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ref.qwen3_5 as m  # noqa: E402


def seeded(rng, *shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


class TestGatedDeltaRuleParity(unittest.TestCase):
    def _run(self, b, s, k_heads, kd, v_heads, vd, chunk, use_l2):
        rng = np.random.default_rng(0)
        q = seeded(rng, b, s, k_heads, kd)
        k = seeded(rng, b, s, k_heads, kd)
        v = seeded(rng, b, s, v_heads, vd)
        g = -np.abs(seeded(rng, b, s, v_heads))
        beta = 0.5 + 0.5 * seeded(rng, b, s, v_heads)

        o_r, st_r = m.gated_delta_rule_recurrent(
            q, k, v, g, beta, output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2,
        )
        o_c, st_c = m.gated_delta_rule_chunked(
            q, k, v, g, beta, chunk_size=chunk, output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2,
        )
        self.assertEqual(o_r.shape, o_c.shape)
        self.assertTrue(
            np.allclose(o_r, o_c, atol=1e-3, rtol=1e-2),
            f"chunked!=recurrent outputs max_abs={np.abs(o_r-o_c).max():.3e}",
        )
        self.assertTrue(
            np.allclose(st_r, st_c, atol=1e-3, rtol=1e-2),
            f"chunked!=recurrent states max_abs={np.abs(st_r-st_c).max():.3e}",
        )

    def test_parity_seq_not_multiple_of_chunk(self):
        self._run(b=2, s=100, k_heads=4, kd=16, v_heads=4, vd=16, chunk=32, use_l2=True)

    def test_parity_seq_multiple_of_chunk(self):
        self._run(b=1, s=64, k_heads=2, kd=8, v_heads=2, vd=8, chunk=32, use_l2=True)

    def test_parity_single_token(self):
        self._run(b=1, s=1, k_heads=2, kd=8, v_heads=2, vd=8, chunk=64, use_l2=True)

    def test_parity_no_l2norm(self):
        self._run(b=1, s=40, k_heads=3, kd=8, v_heads=3, vd=8, chunk=16, use_l2=False)

    def test_state_continuity(self):
        """Running chunked with an initial state equals running from scratch
        on the concatenated sequence (chunked prefill chaining property)."""
        rng = np.random.default_rng(7)
        b, s1, s2, k_heads, kd, v_heads, vd = 1, 32, 16, 2, 8, 2, 8
        q = seeded(rng, b, s1 + s2, k_heads, kd)
        k = seeded(rng, b, s1 + s2, k_heads, kd)
        v = seeded(rng, b, s1 + s2, v_heads, vd)
        g = -np.abs(seeded(rng, b, s1 + s2, v_heads))
        beta = 0.5 * seeded(rng, b, s1 + s2, v_heads) + 0.5

        o_full, st_full = m.gated_delta_rule_chunked(
            q, k, v, g, beta, chunk_size=16, output_final_state=True
        )
        o1, st1 = m.gated_delta_rule_chunked(
            q[:, :s1], k[:, :s1], v[:, :s1], g[:, :s1], beta[:, :s1],
            chunk_size=16, output_final_state=True,
        )
        o2, st2 = m.gated_delta_rule_chunked(
            q[:, s1:], k[:, s1:], v[:, s1:], g[:, s1:], beta[:, s1:],
            chunk_size=16, initial_state=st1, output_final_state=True,
        )
        self.assertTrue(np.allclose(st_full, st2, atol=1e-3, rtol=1e-2),
                        f"states differ {np.abs(st_full-st2).max():.3e}")
        self.assertTrue(np.allclose(o_full[:, s1:], o2, atol=1e-3, rtol=1e-2),
                        f"tail outputs differ {np.abs(o_full[:, s1:]-o2).max():.3e}")


class TestNormConv(unittest.TestCase):
    def test_rms_norm_reference(self):
        rng = np.random.default_rng(1)
        x = seeded(rng, 4, 8, 16)
        w = seeded(rng, 16, scale=0.1)
        out = m.rms_norm(x, w, eps=1e-6)
        var = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
        ref = x.astype(np.float32) * (var + 1e-6) ** -0.5
        ref = ref * (1.0 + w.astype(np.float32))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_rms_norm_gated_reference(self):
        rng = np.random.default_rng(2)
        x = seeded(rng, 4, 8, 16)
        w = seeded(rng, 16, scale=0.1)
        gate = seeded(rng, 4, 8, 16)
        out = m.rms_norm_gated(x, w, gate, eps=1e-6)
        var = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
        ref = x.astype(np.float32) * (var + 1e-6) ** -0.5
        ref = ref * w.astype(np.float32) * m.silu(gate.astype(np.float32))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_causal_conv_depthwise_reference(self):
        rng = np.random.default_rng(3)
        b, c, s, k = 2, 6, 10, 4
        x = seeded(rng, b, c, s)
        w = seeded(rng, c, 1, k)
        out = m.causal_conv1d_depthwise(x, w)
        ref = np.zeros_like(out)
        for t in range(s):
            for j in range(k):
                src = t - (k - 1) + j
                if src >= 0:
                    ref[:, :, t] += w[:, 0, j] * x[:, :, src]
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_rotate_half_is_orthogonal(self):
        rng = np.random.default_rng(4)
        x = seeded(rng, 16, 8)
        y = m.rotate_half(x)
        self.assertEqual(x.shape, y.shape)
        # rotate_half applied twice negates
        np.testing.assert_allclose(m.rotate_half(y), -x, rtol=1e-6)


class TestFullAttention(unittest.TestCase):
    def _naive_attention(self, q, k, v, cos, sin, w_scale, need_rotary):
        """Independent, loop-based check for a single query position."""
        b, h, s, d = q.shape
        rotary = cos.shape[-1]
        scores = np.full((b, h, s, s), -np.inf, dtype=np.float64)
        for i in range(s):
            for j in range(i + 1):
                qi = q[:, :, i, :]
                kj = k[:, :, j, :]
                if need_rotary:
                    # partial rotary: query rotated at its own position i,
                    # key at its own position j
                    def _rot(vec, pos):
                        c, sn = cos[:, pos, :], sin[:, pos, :]
                        r, p = vec[..., :rotary], vec[..., rotary:]
                        r2 = r * c[..., None, :] + m.rotate_half(r) * sn[..., None, :]
                        return np.concatenate((r2, p), axis=-1)

                    scores[:, :, i, j] = (_rot(qi, i) * _rot(kj, j)).sum(-1) * w_scale
                else:
                    scores[:, :, i, j] = (qi * kj).sum(-1) * w_scale
        scores = scores - scores.max(-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(-1, keepdims=True)
        return np.matmul(probs, v.astype(np.float64))

    def test_attention_matches_naive_loop(self):
        rng = np.random.default_rng(5)
        b, h, s, d, rotary = 1, 2, 5, 16, 16
        q = seeded(rng, b, h, s, d)
        k = seeded(rng, b, h, s, d)
        v = seeded(rng, b, h, s, d)
        cos, sin = m.compute_cos_sin(1e4, np.arange(s)[None, :], rotary)
        qr, kr = m.apply_rotary_pos_emb(q, k, cos, sin)
        out = m.eager_attention(qr, kr, v, scale=d ** -0.5, causal=True)
        ref = self._naive_attention(q, k, v, cos, sin, d ** -0.5, need_rotary=True)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


class TestGatedDeltaNetForward(unittest.TestCase):
    def test_chunked_vs_recurrent_forward(self):
        rng = np.random.default_rng(6)
        b, s, hidden = 1, 30, 64
        nk, kd, nv = 2, 8, 4
        vd = 8
        key_dim, value_dim = nk * kd, nv * vd
        x = seeded(rng, b, s, hidden)
        w_qkv = seeded(rng, key_dim * 2 + value_dim, hidden, scale=0.1)
        w_conv = seeded(rng, key_dim * 2 + value_dim, 1, 4, scale=0.5)
        w_z = seeded(rng, value_dim, hidden, scale=0.1)
        w_b = seeded(rng, nv, hidden, scale=0.1)
        w_a = seeded(rng, nv, hidden, scale=0.1)
        a_log = np.log(np.abs(seeded(rng, nv, scale=2.0)) + 0.5).astype(np.float32)
        dt_bias = seeded(rng, nv, scale=0.1)
        norm_w = np.ones(vd, dtype=np.float32)
        out_w = seeded(rng, hidden, value_dim, scale=0.1)

        def run(chunked):
            return m.gated_delta_net_forward(
                x, w_qkv, w_conv, w_z, w_b, w_a, a_log, dt_bias, norm_w, out_w,
                num_k_heads=nk, head_k_dim=kd, num_v_heads=nv, head_v_dim=vd,
                conv_kernel_size=4, chunked=chunked, chunk_size=16,
            )

        o_c, st_c = run(chunked=True)
        o_r, st_r = run(chunked=False)
        self.assertTrue(np.allclose(o_c, o_r, atol=1e-3, rtol=1e-2),
                        f"forward out differ {np.abs(o_c-o_r).max():.3e}")
        self.assertTrue(np.allclose(st_c, st_r, atol=1e-3, rtol=1e-2),
                        f"forward state differ {np.abs(st_c-st_r).max():.3e}")


class TestMLP(unittest.TestCase):
    def test_mlp_silu_math(self):
        rng = np.random.default_rng(8)
        x = seeded(rng, 2, 5, 8)
        wg = seeded(rng, 16, 8, scale=0.1)
        wu = seeded(rng, 16, 8, scale=0.1)
        wd = seeded(rng, 8, 16, scale=0.1)
        out = m.mlp_forward(x, wg, wu, wd)
        ref = m.silu(x @ wg.T) * (x @ wu.T)
        ref = ref @ wd.T
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
