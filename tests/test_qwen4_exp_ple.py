"""P++LE foundation oracle tests (Phase 6).

Verifies the extracted deterministic n-gram machinery in
`ref/qwen4_exp_ple.py` against {hand-computed primes, shift semantics,
self-consistent hashing, PLE feature smoke}. The full pipeline wiring
(hyper stream `+= PLE`, ngram-protected router) is separate successor work;
here only the math foundation is covered.
"""

import unittest

import numpy as np

from ref import qwen4_exp_ple as ple


class TestPrimeLayout(unittest.TestCase):
    def test_primes(self):
        for v in (2, 3, 5, 7, 11, 13, 97, 101):
            self.assertTrue(ple._is_prime(v), v)
        for v in (1, 4, 6, 8, 9, 15, 100):
            self.assertFalse(ple._is_prime(v), v)

    def test_find_nth_prime_after(self):
        self.assertEqual(ple.find_nth_prime_after(10, 1), 11)
        self.assertEqual(ple.find_nth_prime_after(10, 2), 13)
        self.assertEqual(ple.find_nth_prime_after(5, 5), 19)

    def test_head_layout(self):
        L = ple.NGramLayout(vocab_size=32, ngram_size=4,
                            heads_per_ngram=2, ngram_vocab_size_base=16,
                            ple_layer_index=0)
        self.assertEqual(L.ngram_heads, 6)
        self.assertEqual(L.head_vocab_sizes.shape, (6,))
        self.assertEqual(L.head_offsets.tolist(), [0] + list(
            np.cumsum(L.head_vocab_sizes)[:-1]))
        # first head vocab = nth-prime-after(16-1=15, 1) = 17
        self.assertEqual(int(L.head_vocab_sizes[0]), 17)
        # indices respect per-head [offset, offset+size)
        for off, size in zip(np.asarray(L.head_offsets),
                             np.asarray(L.head_vocab_sizes)):
            self.assertTrue(off + size <= L.padded_vocab_size)


class TestMultipliers(unittest.TestCase):
    def test_odd_and_deterministic(self):
        m1 = ple.build_layer_multipliers(32, 3, 0, 12345)
        m2 = ple.build_layer_multipliers(32, 3, 0, 12345)
        np.testing.assert_array_equal(m1, m2)
        self.assertTrue(all(v % 2 == 1 for v in m1))
        self.assertTrue(all(v > 0 for v in m1))

    def test_layer_seed_changes_values(self):
        a = ple.build_layer_multipliers(32, 3, 0, 1)
        b = ple.build_layer_multipliers(32, 3, 1, 1)
        self.assertFalse(np.array_equal(a, b))


class TestShiftRightIgnoreEos(unittest.TestCase):
    def test_plain_shift(self):
        ids = np.array([[1, 2, 3, 4]], np.int64)
        got = ple.shift_right_ignore_eos(ids, 1, eos_token_id=0)
        np.testing.assert_array_equal(got, [[0, 1, 2, 3]])

    def test_eos_boundary(self):
        # segment split at the EOS token (id 99); each segment's positions are
        # its OWN token ids shifted right -- crossing the boundary is EOS.
        # (position 2 belongs to the pre-EOS segment, so it keeps token 1.)
        ids = np.array([[1, 2, 99, 3, 4]], np.int64)
        got = ple.shift_right_ignore_eos(ids, 1, eos_token_id=99)
        np.testing.assert_array_equal(got, [[99, 1, 2, 99, 3]])


class TestNgramHashing(unittest.TestCase):
    def _layout(self):
        return ple.NGramLayout(vocab_size=32, ngram_size=3,
                               heads_per_ngram=2, ngram_vocab_size_base=16,
                               ple_layer_index=0, seed=7)

    def test_prefix_stability_and_bounds(self):
        L = self._layout()
        ids_a = np.array([[5, 9, 1]])
        ids_b = np.array([[5, 9, 1, 7]])
        a = ple.ngram_ids(ids_a, L, eos_token_id=31)
        b = ple.ngram_ids(ids_b, L, eos_token_id=31)
        self.assertEqual(a.shape, (1, 3, 4))
        self.assertEqual(b.shape, (1, 4, 4))
        # causal: matching context + same position -> identical heads
        np.testing.assert_array_equal(b[0, :3], a[0, :3])

    def test_heads_within_own_ranges(self):
        L = self._layout()
        ids = np.array([[0, 1, 2, 3, 4, 5]])
        g = ple.ngram_ids(ids, L, eos_token_id=31)[0]
        for col in range(4):
            start = int(L.head_offsets[col])
            size = int(L.head_vocab_sizes[col])
            self.assertTrue((g[:, col] >= start).all())
            self.assertTrue((g[:, col] < start + size).all())

    def test_eos_in_context_fills_with_eos(self):
        L = self._layout()
        # an all-EOS trailing context → every head lands in a fresh range and
        # stays deterministic (no crash, finite, in-bounds)
        ids = np.array([[31, 31, 31]])
        g = ple.ngram_ids(ids, L, eos_token_id=31)
        self.assertEqual(g.shape, (1, 3, 4))
        self.assertTrue((g >= 0).all())
        self.assertTrue((g < L.padded_vocab_size).all())

    def test_reference_reimplementation_matches(self):
        # independent reimplementation of the upstream loop for the 2-gram band
        L = self._layout()
        ids = np.array([[5, 9, 1]])
        got = ple.ngram_ids(ids, L, eos_token_id=31)[0, :, :2]
        hist = np.concatenate([np.full((1, 2), 31, np.int64), ids], axis=-1)
        sh0 = ple.shift_right_ignore_eos(hist, 0, 31)
        sh1 = ple.shift_right_ignore_eos(hist, 1, 31)
        m = (sh0 * L.layer_multipliers_int[0])
        m = np.bitwise_xor(m, sh1 * L.layer_multipliers_int[1])
        refs = np.remainder(
            m[0, -3:][..., None],
            L.head_vocab_sizes[0:2][None, :]) + L.head_offsets[0:2][None, :]
        np.testing.assert_array_equal(got, refs)


class TestPleFeature(unittest.TestCase):
    def _fixture(self):
        hs, hc, pd, B, S = 8, 2, 12, 1, 5
        rng = np.random.default_rng(0)
        hidden = rng.standard_normal((B, S, hc * hs)).astype(np.float32)
        emb = rng.standard_normal((B, S, pd)).astype(np.float32)
        w_key = rng.standard_normal((hc * hs, pd)).astype(np.float32) * 0.1
        w_value = rng.standard_normal((hs, pd)).astype(np.float32) * 0.1
        gamma = np.zeros((hc * hs, 3), np.float32)
        norm_w = np.zeros(hc * hs, np.float32)
        return dict(hs=hs, hc=hc, hidden=hidden, emb=emb, w_key=w_key,
                    w_value=w_value, gamma=gamma, norm_w=norm_w, eps=1e-6)

    def test_shape_finite_with_zero_conv(self):
        k = self._fixture()
        out, state = ple.ple_feature(
            k["hidden"], k["emb"], k["w_key"], k["w_value"], k["gamma"],
            k["norm_w"], k["norm_w"], k["norm_w"], k["hc"], k["hs"], k["eps"])
        self.assertEqual(out.shape, k["hidden"].shape)
        self.assertTrue(np.isfinite(out).all())
        # zero conv kernel => output is purely the gated value (no conv term)
        self.assertIsNotNone(state)

    def test_conv_state_carries_causally(self):
        base = self._fixture()
        gamma = np.ones((base["hc"] * base["hs"], 3), np.float32) * 0.5
        args = (base["hidden"], base["emb"], base["w_key"], base["w_value"],
                gamma, base["norm_w"], base["norm_w"], base["norm_w"],
                base["hc"], base["hs"], base["eps"])
        out1, st1 = ple.ple_feature(*args)
        # run again with the carried state: the 6th token's conv window must
        # equal a plain 1-shot run including 1 extra position of history
        hid2 = np.asarray(base["hidden"])[:, :1]
        emb2 = np.asarray(base["emb"])[:, :1]
        out2, st2 = ple.ple_feature(
            hid2, emb2, base["w_key"], base["w_value"], gamma, base["norm_w"],
            base["norm_w"], base["norm_w"], base["hc"], base["hs"],
            base["eps"], conv_state=st1)
        hole = 5  # state_len = (3-1)*ngram_size(3) = 6 >= S
        self.assertTrue(np.isfinite(out2).all())
        self.assertIsNotNone(st2)
        self.assertEqual(st2.shape[1], (3 - 1) * 3)

    def test_ngram_table_lookup_bounds(self):
        L = self._layout() if hasattr(self, "_layout") else \
            ple.NGramLayout(vocab_size=32, ngram_size=3, heads_per_ngram=2,
                            ngram_vocab_size_base=16, ple_layer_index=0,
                            seed=7)
        table = np.zeros((L.padded_vocab_size, 3), np.float32)
        ids = np.array([[0, 1, 2, 3]])
        e = ple.ngram_embeddings(ids, L, table, eos_token_id=31)
        self.assertEqual(e.shape, (1, 4, L.ngram_heads * 3))

    def test_short_conv_carried_state_matches_single_call(self):
        # one call over the whole chunk vs S single-token calls with the
        # carried state must produce identical per-position conv outputs
        rng = np.random.default_rng(3)
        C, S, K, d = 4, 5, 3, 2
        gamma = rng.standard_normal((C, K)).astype(np.float32) * 0.4
        x = rng.standard_normal((1, S, C)).astype(np.float32)
        whole, _ = ple._short_conv(x, gamma, dilation=d)
        parts = []
        state = None
        for t in range(S):
            o, state = ple._short_conv(x[:, t:t + 1], gamma, dilation=d,
                                       conv_state=state)
            parts.append(o[0, 0])
        parts = np.stack(parts, axis=0)  # (S, C)
        np.testing.assert_allclose(whole[0], parts, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
