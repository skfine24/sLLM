"""Device-resident GPU decode for the qwen3_5 HYBRID family
(Qwen3.8-27B-FP8: GatedDeltaNet linear-attention + gated full-attention).

v1 of the "GPU-resident by default" milestone for the 27B path:

* WEIGHTS                      -> one bf16 upload per model
                                  (HybridDeviceWeightTable, free-memory
                                  guarded before any cudaMalloc)
* full-attention KV + attention -> on-device (fp32), capacity-doubling
* heavy GEMMs (attention proj, O, MLP gate/up/down, lm_head, embed)
                                  -> device-resident, cuBLAS via gemm_ex
* residual / RMSNorm / conv window / GatedDeltaNet params+state
                                  -> host (mirrors the validated transfer
                                     driver kernels/hybrid_decode.py); the
                                     tiny GDN state stays the authoritative
                                     numpy mirror.

Numeric contract = the numpy incremental oracle (argmax/parity). Cluster
gated: needs sllm_gpu.so + CUDA (the parity test skips otherwise). KV on the
GPU is fp32, so no bf16 host<->device copies are involved; weights stay bf16.
"""

from __future__ import annotations

import numpy as np

from kernels import _sllm_cuda as ck
from ref import incremental as inc
from ref import qwen3_5 as qq

_TEXTP = "model.language_model"


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


class HybridDeviceWeightTable:
    """All qwen3_5 weights uploaded ONCE (projections bf16, norms/scalars fp32).
    `rows[key]` records each matrix's OUT dimension for gemm_ex."""

    def __init__(self, weights: dict, recipe, dtype: str = "bf16",
                 slack_bytes: int = 64 << 20):
        if ck.device_count() < 1:
            raise RuntimeError("no CUDA device")
        if dtype not in ("fp32", "bf16"):
            raise ValueError(f"unsupported device dtype {dtype!r}")
        self.dtype = dtype
        self.t = ck.T_BF16 if dtype == "bf16" else ck.T_F32
        self.elem = 2 if dtype == "bf16" else 4
        prefix = recipe.text_prefix
        self._bufs: dict[str, ck.DeviceBuffer] = {}
        self.rows: dict[str, int] = {}

        plan: list[tuple[str, list, bool]] = []

        def add(key: str, *arrs, is32: bool = False) -> None:
            plan.append((key, list(arrs), is32))
            if arrs:
                self.rows[key] = int(np.asarray(arrs[0]).shape[0])

        tied = recipe.tie_word_embeddings
        add("__embed__", weights[f"{prefix}.embed_tokens.weight"])
        if not tied:
            add("__head__", weights["lm_head.weight"])
        add("__final_norm__", weights[f"{prefix}.norm.weight"], is32=True)
        for i in range(recipe.num_layers):
            p = f"{prefix}.layers.{i}"
            if recipe.layer_types[i] == "linear_attention":
                fw = inc._linear_attn_weights(weights, i)
                add(f"{p}.g.iq", fw["w_in_qkv"])
                add(f"{p}.g.z", fw["w_z"])
                add(f"{p}.g.a", fw["w_a"])
                add(f"{p}.g.b", fw["w_b"])
                add(f"{p}.g.out", fw["w_out"])
                add(f"{p}.g.norm", fw["norm_w"], is32=True)
                add(f"{p}.g.conv", fw["w_conv"], is32=True)
                add(f"{p}.g.a_log", fw["a_log"], is32=True)
                add(f"{p}.g.dt", fw["dt_bias"], is32=True)
            else:
                fw = inc._full_attn_weights(weights, i)
                add(f"{p}.f.q", fw["w_q"])
                add(f"{p}.f.k", fw["w_k"])
                add(f"{p}.f.v", fw["w_v"])
                add(f"{p}.f.o", fw["w_o"])
                add(f"{p}.f.qn", fw["q_norm_w"], is32=True)
                add(f"{p}.f.kn", fw["k_norm_w"], is32=True)
            add(f"{p}.in_ln", weights[f"{p}.input_layernorm.weight"], is32=True)
            add(f"{p}.post_ln", weights[f"{p}.post_attention_layernorm.weight"],
                is32=True)
            add(f"{p}.mlp.gate_proj", weights[f"{p}.mlp.gate_proj.weight"])
            add(f"{p}.mlp.up_proj", weights[f"{p}.mlp.up_proj.weight"])
            add(f"{p}.mlp.down_proj", weights[f"{p}.mlp.down_proj.weight"])

        sizes = sum(sum(a.size for a in arrs) * (4 if is32 else self.elem)
                    for _k, arrs, is32 in plan)
        free = ck.mem_free_bytes()
        if free < 0:
            raise RuntimeError("cannot query free device memory "
                               "(cudaMemGetInfo failed); refusing the "
                               "device-resident upload (GB10 guard)")
        if free < sizes + slack_bytes:
            raise RuntimeError(
                f"insufficient free device memory for hybrid resident "
                f"weights: {free} B free < {sizes} B weights "
                f"+ {slack_bytes} B slack")
        for key, arrs, is32 in plan:
            a = _f32(arrs[0] if len(arrs) == 1
                     else np.concatenate([_f32(x) for x in arrs], axis=0))
            if is32 or self.elem == 4:
                self._bufs[key] = ck.to_device(a)
            else:
                buf = ck.alloc_n(a.nbytes // 2)
                buf.upload_raw(ck.to_bf16(a))
                self._bufs[key] = buf
        self.embed = self._bufs["__embed__"]
        self.head = self.embed if tied else self._bufs["__head__"]
        self.final_norm = self._bufs["__final_norm__"]
        if tied:
            self.rows["__head__"] = self.rows["__embed__"]
        self.total_bytes = sizes

    def get(self, key: str):
        return self._bufs.get(key)

    def free(self) -> None:
        for buf in self._bufs.values():
            buf.free()
        self._bufs.clear()
        self.rows.clear()
        self.embed = self.head = self.final_norm = None


def _host_weight(buf) -> np.ndarray:
    return np.frombuffer(buf.copy_host(), dtype=np.float32).reshape(-1).copy()


class HybridDeviceDecodeState:
    """Per-sequence device-resident decode for the hybrid family.

    v1: weights (bf16) + full-attention KV (fp32, capacity-doubling) on the
    GPU; residual / norms / conv window / GatedDeltaNet mirror the transfer
    driver on the host. The numpy IncrementalCache stays authoritative and is
    updated only after a fully successful step.
    """

    def __init__(self, table: HybridDeviceWeightTable, cache, recipe):
        fa = recipe.full_attention
        la = recipe.linear_attention
        self.table = table
        self.cache = cache
        self.recipe = recipe
        self.t = ck.T_F32                 # KV / attention operands stay fp32
        self.elem = 4
        self.L = recipe.num_layers
        nh, kvh, hd = fa.num_heads, fa.num_kv_heads, fa.head_dim
        self.nh, self.kvh, self.hd = nh, kvh, hd
        self.NQ, self.NK = nh * hd, kvh * hd
        self.lin_key_dim = la.num_key_heads * la.key_head_dim
        self.lin_value_dim = la.num_value_heads * la.value_head_dim
        self.rot = recipe.rotary_dim()
        self.eps = cache.eps
        self.theta = cache.theta
        self.scale = float(hd) ** -0.5
        H = self.H = table.rows["__embed__"]
        self.V = table.rows["__head__"]

        self._fa = [i for i, bt in enumerate(recipe.layer_types)
                    if bt == "full_attention"]
        S = cache.n_ctx
        self.cap = max(int(S), 1)
        self._k: dict[int, ck.DeviceBuffer] = {}
        self._v: dict[int, ck.DeviceBuffer] = {}
        for i in self._fa:
            for store, arr in ((self._k, cache.k[i]), (self._v, cache.v[i])):
                store[i] = ck.to_device(_f32(arr))

        self._hx = ck.alloc(self.H)          # reused fp32 1xH embed row
        self._o = ck.alloc(self.V)           # fp32 gemm out (biggest = vocab)
        self._row = ck.alloc(self.NK)        # fp32 kv head slice
        self._qbuf = ck.alloc(self.NQ)
        self._attn = ck.alloc(self.NQ)
        self._scores = ck.alloc(self.nh * self.cap)
        self._logits = ck.alloc(self.V)

    def _gemm(self, src: np.ndarray, key: str) -> np.ndarray:
        """src (1, in) @ W^T with W (rows=rows[key], in); device-resident."""
        ar = np.ascontiguousarray(np.asarray(src, dtype=np.float32)).reshape(1, -1)
        rows = self.table.rows[key]
        buf = ck.to_device(ar)
        try:
            ck.gemm_ex(buf, self.table.get(key), self._o,
                       1, rows, ar.shape[1], self.table.t)
            ck.sync()
            return self._o.copy_host().reshape(1, rows)
        finally:
            buf.free()

    def step(self, last_id: int) -> np.ndarray:
        cache, table, p0 = self.cache, self.table, _TEXTP
        last_id = int(last_id)
        if not 0 <= last_id < self.V:
            raise ValueError(f"token id {last_id} outside vocab [0, {self.V})")
        pos = cache.n_ctx
        max_pos = int(self.recipe.max_position_embeddings)
        if pos >= max_pos:
            raise RuntimeError(f"decode position {pos} reached "
                               f"max_position_embeddings ({max_pos})")
        if pos >= self.cap:
            self._grow(min(max(2 * self.cap, pos + 1), max_pos))

        ck.gather_row_t(table.embed, last_id, self._hx, self.H, table.t)
        ck.sync()
        hidden = self._hx.copy_host().astype(np.float32).reshape(1, self.H)
        cos, sin = inc._pos_cos_sin(self.theta, self.rot, pos)

        fa = self.recipe.full_attention
        nh, kvh, hd = fa.num_heads, fa.num_kv_heads, fa.head_dim
        n_rep = nh // kvh
        new_conv: dict[int, np.ndarray] = {}
        new_k: dict[int, np.ndarray] = {}
        new_v: dict[int, np.ndarray] = {}

        for i, bt in enumerate(self.recipe.layer_types):
            p = f"{p0}.layers.{i}"
            in_norm = _host_weight(table.get(f"{p}.in_ln"))
            post_norm = _host_weight(table.get(f"{p}.post_ln"))
            residual = hidden
            h_in = qq.rms_norm(hidden, in_norm, eps=self.eps)

            if bt == "linear_attention":
                la = self.recipe.linear_attention
                key_dim = la.num_key_heads * la.key_head_dim
                value_dim = la.num_value_heads * la.value_head_dim
                kc = la.conv_kernel_size
                C = 2 * key_dim + value_dim
                mixed = self._gemm(h_in, f"{p}.g.iq").reshape(-1, 1)
                win = np.concatenate([cache.conv_win[i], mixed[None]], axis=-1)
                conv = qq.causal_conv1d_depthwise(
                    win[:, :, -kc:],
                    _host_weight(table.get(f"{p}.g.conv")), activation="silu")
                mixed_conv = np.transpose(conv[:, :, -1:], (0, 2, 1))
                new_conv[i] = win[:, :, -(kc - 1):] if kc > 1 else win[:, :, :0]

                query, key, value = qq._split(mixed_conv,
                                              [key_dim, key_dim, value_dim])
                query = query.reshape(1, 1, -1, la.key_head_dim)
                key = key.reshape(1, 1, -1, la.key_head_dim)
                value = value.reshape(1, 1, la.num_value_heads, la.value_head_dim)
                ratio = la.num_value_heads // la.num_key_heads
                if ratio > 1:
                    query = np.repeat(query, ratio, axis=2)
                    key = np.repeat(key, ratio, axis=2)
                q = qq.l2norm(query[0, 0]).astype(np.float32) / np.float32(
                    np.sqrt(la.key_head_dim))
                k = qq.l2norm(key[0, 0]).astype(np.float32)
                v = np.ascontiguousarray(value[0, 0], dtype=np.float32)

                z = self._gemm(h_in, f"{p}.g.z").reshape(
                    1, 1, la.num_value_heads, la.value_head_dim)
                bvec = self._gemm(h_in, f"{p}.g.b").reshape(-1)
                avec = self._gemm(h_in, f"{p}.g.a").reshape(-1)
                beta = qq.sigmoid(bvec)[0].astype(np.float32)
                g = (-np.exp(_host_weight(table.get(f"{p}.g.a_log")))
                     * qq.softplus(avec.astype(np.float32)
                                   + _host_weight(table.get(f"{p}.g.dt"))))[0]

                out = ck.gated_delta_step(q, k, v, g, beta, cache.state[i][0])
                core = qq.rms_norm_gated(
                    out[None, None],
                    _host_weight(table.get(f"{p}.g.norm")), z,
                    eps=self.eps).reshape(1, 1, -1)
                h = self._gemm(core.reshape(1, -1), f"{p}.g.out")
            else:
                qg = self._gemm(h_in, f"{p}.f.q").reshape(1, nh, hd * 2)
                q2, gate = qq._split(qg, [hd, hd])
                q2 = qq.rms_norm(q2, _host_weight(table.get(f"{p}.f.qn")),
                                 eps=self.eps).reshape(1, nh, 1, hd)
                k = qq.rms_norm(self._gemm(h_in, f"{p}.f.k").reshape(1, kvh, 1, hd),
                                _host_weight(table.get(f"{p}.f.kn")),
                                eps=self.eps)
                v = self._gemm(h_in, f"{p}.f.v").reshape(1, kvh, 1, hd)
                q2, k = qq.apply_rotary_pos_emb(q2, k, cos[None, None],
                                                sin[None, None])
                ki = np.concatenate([cache.k[i], k[0]], axis=1)
                vi = np.concatenate([cache.v[i], v[0]], axis=1)
                new_k[i], new_v[i] = ki, vi
                # write k/v to the on-device KV and run device attention (fp32)
                self._row.upload(_f32(k[0].reshape(-1)))
                ck.kv_write_t(self._k[i], self.cap, pos, self._row, None,
                              kvh, hd)
                self._row.upload(_f32(v[0].reshape(-1)))
                ck.kv_write_t(self._v[i], self.cap, pos, self._row, None,
                              kvh, hd)
                self._qbuf.upload(_f32(q2[0, :, 0, :]))
                ck.attention_decode_t(self._qbuf, self._k[i], self._v[i],
                                      self._scores, self._attn, nh, kvh,
                                      pos + 1, self.cap, hd, self.scale,
                                      self.t)
                ck.sync()
                attn = self._attn.copy_host().astype(np.float32).reshape(1, nh * hd)
                attn = attn * qq.sigmoid(gate.reshape(1, -1))
                h = self._gemm(attn, f"{p}.f.o")

            hidden = ck.elwise_add(residual.astype(np.float32),
                                   h.astype(np.float32)).reshape(1, self.H)
            residual = hidden
            hh = qq.rms_norm(hidden, post_norm, eps=self.eps)
            gs = self._gemm(hh, f"{p}.mlp.gate_proj")
            us = self._gemm(hh, f"{p}.mlp.up_proj")
            hl = self._gemm(qq.silu(gs) * us, f"{p}.mlp.down_proj")
            hidden = ck.elwise_add(residual.astype(np.float32),
                                   hl.astype(np.float32)).reshape(1, self.H)

        hidden = qq.rms_norm(hidden, _host_weight(table.final_norm),
                             eps=self.eps)
        out = self._gemm(hidden, "__head__")

        for i, arr in new_conv.items():
            cache.conv_win[i] = arr
        for i in new_k:
            cache.k[i] = new_k[i]
            cache.v[i] = new_v[i]
        cache.n_ctx += 1
        return out.reshape(-1)

    def _grow(self, cap_new: int) -> None:
        nb = self.kvh * cap_new * self.hd * self.elem
        blk_words = self.hd * self.elem // 4
        if self.hd * self.elem % 4:
            raise ValueError("head row not word-aligned")
        fresh: list[ck.DeviceBuffer] = []
        try:
            for _store in (self._k, self._v):
                for _i in _store:
                    fresh.append(ck.alloc_n(nb))
        except BaseException:
            for buf in fresh:
                try:
                    buf.free()
                except Exception:
                    pass
            raise
        new_k, new_v = {}, {}
        it = iter(fresh)
        try:
            for store, new_store in ((self._k, new_k), (self._v, new_v)):
                for idx, buf in store.items():
                    new = next(it)
                    ck.kv_relayout_w(new, buf, self.kvh, self.cap, cap_new,
                                     blk_words)
                    new_store[idx] = new
        except BaseException:
            for buf in fresh:
                try:
                    buf.free()
                except Exception:
                    pass
            raise
        for store in (self._k, self._v):
            for buf in store.values():
                buf.free()
        self._k, self._v = new_k, new_v
        self.cap = cap_new
        self._scores.free()
        self._scores = ck.alloc(self.nh * cap_new)

    def free(self) -> None:
        for store in (self._k, self._v):
            for buf in store.values():
                buf.free()
            store.clear()
        for name in ("_hx", "_o", "_row", "_qbuf", "_attn", "_scores",
                     "_logits"):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.free()
