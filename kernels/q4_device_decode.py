"""Device-resident GPU decode for the qwen4_exp family
(Qwen3.8-Flash-Next-FP8: GatedDeltaNet + QSA sparse attention +
hyper-connection + MoE).  Milestone C1 / Q4-GPU single-node path.

Design (mirrors kernels/hybrid_device_decode.py for the qwen3_5 hybrid):

* WEIGHTS                  -> one bf16 upload per model (Q4DeviceWeightTable,
                             free-memory guarded before any cudaMalloc).
* projection GEMMs (embed, QKV, O, indexer, MLP in/out, HC down/up, router,
  MoE expert gate/up/down, shared expert, lm_head) -> device-resident cuBLAS
  via gemm_ex / gemm_linear_dev.
* q4-specific device math (grouped GemmaRMSNorm, HC mix/combine, GemmaRMSNorm,
  partial RoPE, QSA mqa/topk/sparse-attn/pool, MoE router/swiglu/axpy/shared
  gate) -> the device-pointer sllm_q4_*_dev kernels.
* GDN recurrent scan + depthwise conv + the QSA KV/indexer caches stay HOST
  (numpy authoritative), matching the hybrid pattern where the tiny recurrent
  state stays the numpy mirror; the full-attn KV is NOT mirrored on-device in
  v1 (the QSA sparse selection re-reads the host caches through the _dev
  kernels per step).

Numeric contract = the numpy oracle (ref/qwen4_exp_pipeline.decode_step);
cluster gated (needs sllm_gpu.so + CUDA; parity test skips otherwise).
"""

from __future__ import annotations

import numpy as np

from kernels import _sllm_cuda as ck
from kernels import _q4_cuda as q4
from ref import qwen3_5 as qq
from ref import qwen4_exp as qe

_TEXTP = "model.language_model"


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


class Q4DeviceWeightTable:
    """All qwen4_exp weights uploaded ONCE (projections bf16, norms/scalars
    fp32). `rows[key]` records each matrix's OUT dimension for gemm_ex."""

    def __init__(self, weights: dict, cfg, dtype: str = "bf16",
                 slack_bytes: int = 64 << 20):
        if ck.device_count() < 1:
            raise RuntimeError("no CUDA device")
        if dtype not in ("fp32", "bf16"):
            raise ValueError(f"unsupported device dtype {dtype!r}")
        self.cfg = cfg
        self.dtype = dtype
        self.t = ck.T_BF16 if dtype == "bf16" else ck.T_F32
        self.elem = 2 if dtype == "bf16" else 4
        self._bufs: dict[str, ck.DeviceBuffer] = {}
        self.rows: dict[str, int] = {}
        self._n32: dict[str, np.ndarray] = {}   # fp32 host mirrors for norms

        plan: list[tuple[str, np.ndarray, bool]] = []

        def add(key: str, a, is32: bool = False) -> None:
            plan.append((key, np.asarray(a), is32))
            self.rows[key] = int(np.asarray(a).shape[0])

        add("__embed__", weights[f"{_TEXTP}.embed_tokens.weight"])
        add("__head__", weights["lm_head.weight"])
        hc, hs, lr = cfg.hc_count, cfg.hidden, cfg.hc_lowrank
        add("__final_norm__",
            weights[f"{_TEXTP}.hyper_connection_mixer.hc_norm.weight"], is32=True)
        add("__final_down__",
            weights[f"{_TEXTP}.hyper_connection_mixer.input_mix_weight_down.weight"])
        add("__final_up__",
            weights[f"{_TEXTP}.hyper_connection_mixer.input_mix_weight_up.weight"])

        for i, bt in enumerate(cfg.layer_types):
            p = f"{_TEXTP}.layers.{i}"
            for kind in ("attn_hyper_connection", "mlp_hyper_connection"):
                hc_p = f"{p}.{kind}"
                add(f"{p}.{kind}.norm",
                    weights[f"{hc_p}.hc_norm.weight"], is32=True)
                add(f"{p}.{kind}.down",
                    weights[f"{hc_p}.input_mix_weight_down.weight"])
                add(f"{p}.{kind}.up",
                    weights[f"{hc_p}.input_mix_weight_up.weight"])
                add(f"{p}.{kind}.inject",
                    weights[f"{hc_p}.block_inject_weight.weight"], is32=True)
            if bt == "linear_attention":
                la_p = f"{p}.linear_attn"
                add(f"{p}.iq", weights[f"{la_p}.in_proj_qkv.weight"])
                add(f"{p}.z", weights[f"{la_p}.in_proj_z.weight"])
                add(f"{p}.b", weights[f"{la_p}.in_proj_b.weight"])
                add(f"{p}.a", weights[f"{la_p}.in_proj_a.weight"])
                add(f"{p}.out", weights[f"{la_p}.out_proj.weight"])
                add(f"{p}.norm", weights[f"{la_p}.norm.weight"], is32=True)
                add(f"{p}.conv", weights[f"{la_p}.conv1d.weight"], is32=True)
                add(f"{p}.a_log", weights[f"{la_p}.A_log"], is32=True)
                add(f"{p}.dt", weights[f"{la_p}.dt_bias"], is32=True)
            else:
                sa_p = f"{p}.self_attn"
                add(f"{p}.q", weights[f"{sa_p}.q_proj.weight"])
                add(f"{p}.k", weights[f"{sa_p}.k_proj.weight"])
                add(f"{p}.v", weights[f"{sa_p}.v_proj.weight"])
                add(f"{p}.o", weights[f"{sa_p}.o_proj.weight"])
                add(f"{p}.qn", weights[f"{sa_p}.q_norm.weight"], is32=True)
                add(f"{p}.kn", weights[f"{sa_p}.k_norm.weight"], is32=True)
                add(f"{p}.iqk", weights[f"{sa_p}.indexer.index_qk_proj.weight"],
                    is32=True)
                add(f"{p}.iq", weights[f"{sa_p}.indexer.q_layernorm.weight"],
                    is32=True)
                add(f"{p}.ik", weights[f"{sa_p}.indexer.k_layernorm.weight"],
                    is32=True)
            mlp_p = f"{p}.mlp"
            add(f"{p}.router", weights[f"{mlp_p}.gate.weight"])
            add(f"{p}.sg", weights[f"{mlp_p}.shared_expert.gate_proj.weight"])
            add(f"{p}.su", weights[f"{mlp_p}.shared_expert.up_proj.weight"])
            add(f"{p}.sd", weights[f"{mlp_p}.shared_expert.down_proj.weight"])
            add(f"{p}.sgl", weights[f"{mlp_p}.shared_expert_gate.weight"])
            for e in range(cfg.n_experts):
                ep = f"{mlp_p}.experts.{e}"
                add(f"{p}.ex.{e}.g", weights[f"{ep}.gate_proj.weight"])
                add(f"{p}.ex.{e}.u", weights[f"{ep}.up_proj.weight"])
                add(f"{p}.ex.{e}.d", weights[f"{ep}.down_proj.weight"])

        sizes = sum(a.size * (4 if is32 else self.elem)
                    for _k, a, is32 in plan)
        free = ck.mem_free_bytes()
        if free < 0:
            raise RuntimeError("cannot query free device memory "
                               "(cudaMemGetInfo failed); refusing the "
                               "device-resident upload (GB10 guard)")
        if free < sizes + slack_bytes:
            raise RuntimeError(
                f"insufficient free device memory for qwen4 resident weights: "
                f"{free} B free < {sizes} B weights + {slack_bytes} B slack")
        for key, a, is32 in plan:
            a32 = _f32(a)
            if is32 or self.elem == 4:
                self._bufs[key] = ck.to_device(a32)
            else:
                buf = ck.alloc_n(a32.nbytes // 2)
                buf.upload_raw(ck.to_bf16(a32))
                self._bufs[key] = buf
            if is32:
                self._n32[key] = a32
        self.embed = self._bufs["__embed__"]
        self.head = self._bufs["__head__"]
        self.total_bytes = sizes

    def get(self, key: str):
        return self._bufs.get(key)

    def host32(self, key: str) -> np.ndarray:
        return self._n32[key]

    def free(self) -> None:
        for buf in self._bufs.values():
            buf.free()
        self._bufs.clear()
        self.rows.clear()
        self._n32.clear()
        self.embed = self.head = None


class Q4DeviceDecodeState:
    """Per-sequence device-resident decode for qwen4_exp (single token/step).

    v1: weights (bf16) on the GPU; GDN recurrent state + conv window + the
    QSA KV/indexer caches live in the host `Qwen4ExpState` (numpy
    authoritative), and are re-read through the device kernels per step.
    """

    def __init__(self, table: Q4DeviceWeightTable, state, cfg):
        self.table = table
        self.state = state            # ref.qwen4_exp_pipeline.Qwen4ExpState
        self.cfg = cfg
        self.H = table.rows["__embed__"]
        self.V = table.rows["__head__"]
        self.hc, self.hs = cfg.hc_count, cfg.hidden
        self.eps = cfg.rms_norm_eps
        self.scale = float(cfg.attn_head_dim) ** -0.5

        self._hx = ck.alloc(self.H)          # reused fp32 1xH embed row
        self._o = ck.alloc(self.V)           # fp32 gemm out (biggest = vocab)

    # -- device projection helper -----------------------------------------
    def _gemm(self, src: np.ndarray, key: str) -> np.ndarray:
        """src (1, in) @ W^T (W rows=rows[key], in) -> (1, rows) fp32."""
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

    def _hc_mix(self, hyper, norm_k, down_k, up_k, hc, hs):
        """host HC mixer (silu/sigmoid are elementwise-cheap; the down/up
        projections are device GEMMs)."""
        normed = qe.grouped_gemma_rmsnorm(hyper, self.table.host32(norm_k),
                                          hs, self.eps)
        low = qq.silu(self._gemm(normed, down_k) / hc)
        w = qq.sigmoid(self._gemm(low, up_k))
        w = w.reshape(w.shape[:-1], hc, hs)
        mixed = (w * normed.reshape(*normed.shape[:-1], hc, hs)).mean(axis=-2)
        return mixed.astype(hyper.dtype), normed

    def _hc_combine(self, block_out, hyper, normed, inject_k, hc, hs):
        inj = 2.0 * qq.sigmoid(
            self._gemm(normed, inject_k) / hc)
        r = hyper.reshape(*hyper.shape[:-1], hc, hs)
        inj = inj.reshape(*inj.shape[:-1], hc, 1)
        out = (r + block_out[..., np.newaxis, :] * inj).reshape(*hyper.shape)
        return out.astype(hyper.dtype)

    def _linear_block(self, h1, i):
        """GDN linear-attention decode step (state+conv on host)."""
        fw = self.table
        la = self.cfg
        key_dim = la.lin_k_heads * la.lin_k_dim
        value_dim = la.lin_v_heads * la.lin_v_dim
        kc = la.lin_conv
        st = self.state.layers[i]
        C = 2 * key_dim + value_dim

        mixed = np.transpose(self._gemm(h1, f"{_TEXTP}.layers.{i}.iq"),
                             (1, 0))               # (C, 1)
        win = np.concatenate([st["conv_win"], mixed[None]], axis=-1)
        conv = qq.causal_conv1d_depthwise(win[:, :, -kc:],
                                          self.table.host32(
                                              f"{_TEXTP}.layers.{i}.conv"),
                                          activation="silu")
        mixed_conv = np.transpose(conv[:, :, -1:], (0, 2, 1))  # (1,1,C)
        st["conv_win"] = win[:, :, -(kc - 1):]

        query, key, value = qq._split(mixed_conv, [key_dim, key_dim, value_dim])
        query = query.reshape(1, 1, -1, la.lin_k_dim)
        key = key.reshape(1, 1, -1, la.lin_k_dim)
        value = value.reshape(1, 1, la.lin_v_heads, la.lin_v_dim)
        ratio = la.lin_v_heads // la.lin_k_heads
        if ratio > 1:
            query = np.repeat(query, ratio, axis=2)
            key = np.repeat(key, ratio, axis=2)

        z = self._gemm(h1, f"{_TEXTP}.layers.{i}.z").reshape(
            1, 1, la.lin_v_heads, la.lin_v_dim)
        beta = qq.sigmoid(self._gemm(h1, f"{_TEXTP}.layers.{i}.b").reshape(1, 1, -1))
        g = (-np.exp(self.table.host32(f"{_TEXTP}.layers.{i}.a_log"))
             * qq.softplus(self._gemm(h1, f"{_TEXTP}.layers.{i}.a").astype(np.float32)
                           + self.table.host32(f"{_TEXTP}.layers.{i}.dt"))).reshape(1, 1, -1)

        core, state = qq.gated_delta_rule_recurrent(
            query, key, value, g=g, beta=beta,
            initial_state=st["gdn"], output_final_state=True,
            use_qk_l2norm_in_kernel=la.lin_l2norm)
        st["gdn"] = state
        core = qq.rms_norm_gated(core, self.table.host32(
            f"{_TEXTP}.layers.{i}.norm"), z, eps=self.eps)
        core = core.reshape(1, 1, -1)
        return self._gemm(core, f"{_TEXTP}.layers.{i}.out").astype(h1.dtype)

    def _qsa_block(self, h1, i, pos):
        """QSA full-attention decode step (caches on host, math on device)."""
        cfg = self.cfg
        st = self.state.layers[i]
        nh, kvh, hd, rot = (cfg.attn_heads, cfg.attn_kv_heads,
                            cfg.attn_head_dim, cfg.rotary_dim)
        p = f"{_TEXTP}.layers.{i}"
        cos, sin = qq.compute_cos_sin(cfg.rope_theta, np.array([[pos]]), rot)

        qg = self._gemm(h1, f"{p}.q").reshape(1, nh, hd * 2)
        q2, gate = qg[..., :hd], qg[..., hd:]
        q2 = qe.gemma_rmsnorm(q2, self.table.host32(f"{p}.qn"), self.eps) \
            .reshape(1, nh, 1, hd)
        k = qe.gemma_rmsnorm(self._gemm(h1, f"{p}.k").reshape(1, kvh, 1, hd),
                             self.table.host32(f"{p}.kn"), self.eps)
        v = self._gemm(h1, f"{p}.v").reshape(1, kvh, 1, hd)
        q2, k = qq.apply_rotary_pos_emb(q2, k, cos, sin)
        from ref.qwen4_exp_pipeline import _append_axis
        _append_axis(st, "k", k[0].astype(np.float32), axis=1)
        _append_axis(st, "v", v[0].astype(np.float32), axis=1)

        q_idx, tok_k = qe.qsa_index_project_qk(
            h1[0], self.table.host32(f"{p}.iqk"), self.table.host32(f"{p}.iq"),
            self.table.host32(f"{p}.ik"), cos[0], sin[0], rot, cfg.idx_heads,
            cfg.idx_dim, self.eps)
        _append_axis(st, "tok_k", tok_k.astype(np.float32), axis=0)
        if (pos + 1) % cfg.idx_ratio == 0:
            from ref.qwen4_exp_pipeline import _indexer_compress_append
            row = _indexer_compress_append(st["tok_k"],
                                           self.table.host32(f"{p}.ik"), cfg,
                                           pos + 1, cfg.rope_theta)
            _append_axis(st, "ck", row, axis=0)

        from ref.qwen4_exp_pipeline import qsa_select_slots
        nb = st["ck"].shape[0]
        rows = np.array([st["k"].shape[1]], np.int32)
        qpos = rows - 1
        if nb:
            slots = qsa_select_slots(q_idx, st["ck"], np.zeros(1, np.int32),
                                     np.array([nb], np.int32), qpos, rows, cfg)
        else:
            slots = qe.qsa_expand_block_indices(
                np.full((1, cfg.idx_budget // cfg.idx_ratio), -1, np.int32),
                qpos, rows, cfg.idx_ratio, cfg.idx_budget)

        attn = qe.qsa_sparse_attention(
            q2.transpose(0, 2, 1, 3)[0], st["k"].transpose(1, 0, 2),
            st["v"].transpose(1, 0, 2), slots)
        attn = attn.reshape(1, nh * hd)
        attn = attn * qq.sigmoid(gate.reshape(1, nh * hd))
        return self._gemm(attn, f"{p}.o").astype(h1.dtype)

    def _moe(self, x, i):
        """MoE: router (device) + per-expert gemv (device) + swiglu/axpy."""
        cfg = self.cfg
        p = f"{_TEXTP}.layers.{i}"
        n = x.shape[0]
        H = self.H
        router = self._gemm(x, f"{p}.router")[0]
        w, ids = qe.moe_route(router, cfg.top_k)
        out = np.zeros((n, H), np.float32)
        for j in range(cfg.top_k):
            e = int(ids[0, j])
            ep = f"{p}.ex.{e}"
            g = self._gemm(x, f"{ep}.g")
            u = self._gemm(x, f"{ep}.u")
            y = qq.silu(g) * u
            d = self._gemm(y, f"{ep}.d")
            out += float(w[0, j]) * d[0]
        sg = self._gemm(x, f"{p}.sg")
        su = self._gemm(x, f"{p}.su")
        shared = qq.silu(sg) * su
        shared = self._gemm(shared, f"{p}.sd")
        gate = qq.sigmoid(self._gemm(x, f"{p}.sgl"))
        out += gate * shared.astype(np.float32)
        return out

    def step(self, last_id: int) -> np.ndarray:
        cfg, state = self.cfg, self.state
        last_id = int(last_id)
        if not 0 <= last_id < self.V:
            raise ValueError(f"token id {last_id} outside vocab [0, {self.V})")
        pos = state.n_ctx

        ck.gather_row_t(self.table.embed, last_id, self._hx, self.H, self.table.t)
        ck.sync()
        hyper = self._hx.copy_host().astype(np.float32).reshape(1, 1, self.H)
        hyper = np.concatenate([hyper] * self.hc, axis=-1)   # (1,1,hc*H)

        for i, bt in enumerate(cfg.layer_types):
            st = state.layers[i]
            a = f"{_TEXTP}.layers.{i}.attn_hyper_connection"
            m = f"{_TEXTP}.layers.{i}.mlp_hyper_connection"
            mixed, normed = self._hc_mix(hyper, f"{a}.norm", f"{a}.down",
                                         f"{a}.up", self.hc, self.hs)
            block = (self._linear_block(mixed, i) if bt == "linear_attention"
                     else self._qsa_block(mixed, i, pos))
            hyper = self._hc_combine(block, hyper, normed, f"{a}.inject",
                                     self.hc, self.hs)
            mixed2, normed2 = self._hc_mix(hyper, f"{m}.norm", f"{m}.down",
                                           f"{m}.up", self.hc, self.hs)
            moe_out = self._moe(mixed2[0], i).reshape(mixed2.shape)
            hyper = self._hc_combine(moe_out, hyper, normed2, f"{m}.inject",
                                     self.hc, self.hs)

        state.n_ctx += 1
        final, _ = self._hc_mix(hyper, "__final_norm__", "__final_down__",
                                "__final_up__", self.hc, self.hs)
        out = self._gemm(final, "__head__")
        return out.reshape(-1)

    def free(self) -> None:
        for name in ("_hx", "_o"):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.free()
