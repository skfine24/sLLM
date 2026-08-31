"""Qwen4_exp model pipeline reference (numpy), milestone Q2.

Wires the Q1 component oracles (`ref/qwen4_exp.py`, which cites
`oracle/upstream/sglang/*`) into a checkpoint-named, single-sequence,
text-only forward: prefill (batched) and incremental decode.

Layer flow (qwen4_exp.py `Qwen4ExpLayerExtensionMixin` / decoder layers):

    hyper (hc*H) -> [PLE: skipped, guarded] -> attn_hc.mix -> block
                 -> attn_hc.combine -> mlp_hc.mix -> MoE -> mlp_hc.combine
    final: hyper_connection_mixer.mix -> lm_head          (no plain `norm`)

Weight names verified against `model.safetensors.index.json` of
`~/models/Qwen3.8-Flash-Next-FP8` (head node, 2026-08-30):

    model.language_model.embed_tokens.weight        lm_head.weight
    model.language_model.hyper_connection_mixer.{hc_norm,input_mix_weight_down,
                                                  input_mix_weight_up}.weight
    model.language_model.layers.{i}.attn_hyper_connection.{hc_norm,
        input_mix_weight_down,input_mix_weight_up,block_inject_weight}.weight
    model.language_model.layers.{i}.mlp_hyper_connection.{...}          (same)
    model.language_model.layers.{i}.linear_attn.{A_log,conv1d.weight,dt_bias,
        in_proj_a.weight,in_proj_b.weight,in_proj_qkv.weight,in_proj_z.weight,
        norm.weight,out_proj.weight}
    model.language_model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
    model.language_model.layers.{i}.self_attn.{q_norm,k_norm}.weight
    model.language_model.layers.{i}.self_attn.indexer.{index_qk_proj.weight,
        q_layernorm.weight,k_layernorm.weight}
    model.language_model.layers.{i}.mlp.gate.weight
    model.language_model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
    model.language_model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight
    model.language_model.layers.{i}.mlp.shared_expert_gate.weight

Shapes verified: q_proj [nh*2*hd, H] (q|gate interleaved per head),
indexer.index_qk_proj [(idx_heads+1)*idx_dim, H],
input_mix_weight_down [lowrank, hc*H], block_inject_weight [hc, hc*H].

Out of scope (Q3+): PLE/ngram (guarded), MTP, vision, TP, fp8 loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import qwen3_5 as qq
from . import qwen4_exp as qe

_TEXT = "model.language_model"


@dataclass
class Qwen4ExpCfg:
    hidden: int
    hc_count: int
    hc_lowrank: int
    layer_types: tuple
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    # gated deltanet (identical structure to qwen3_5, audit doc 06 sec.3)
    lin_k_heads: int = 16
    lin_k_dim: int = 128
    lin_v_heads: int = 48
    lin_v_dim: int = 128
    lin_conv: int = 4
    lin_l2norm: bool = True
    # QSA full attention
    attn_heads: int = 24
    attn_kv_heads: int = 2
    attn_head_dim: int = 256
    rotary_factor: float = 0.25
    # QSA indexer (compressed variant)
    idx_heads: int = 4
    idx_dim: int = 128
    idx_budget: int = 2048
    idx_ratio: int = 4
    # MoE
    n_experts: int = 512
    top_k: int = 10
    moe_inter: int = 640
    shared_inter: int = 640
    # unsupported here
    ple_layer_ids: tuple = field(default=())

    @property
    def rotary_dim(self) -> int:
        return int(self.attn_head_dim * self.rotary_factor)

    @classmethod
    def from_recipe(cls, recipe) -> "Qwen4ExpCfg":
        """Derive the cfg from a parsed recipe (recipes/Qwen3.8-Flash-Next-FP8.yaml shape).

        QSA-only knobs (indexer, hyper-connection, PLE) live in the recipe
        `spec:` meta passthrough; layer type `qsa_attention` maps to the
        pipeline's `full_attention` block (the QSA variant).
        """
        spec = recipe.meta.get("spec", {}) or {}
        hc = spec.get("hc", {}) or {}
        qsa = spec.get("qsa", {}) or {}
        ple = spec.get("ple", {}) or {}
        la, fa, mlp = recipe.linear_attention, recipe.full_attention, recipe.mlp
        return cls(
            hidden=recipe.hidden_size,
            hc_count=int(hc.get("hc_count", 4)),
            hc_lowrank=int(hc.get("hc_lowrank", 320)),
            layer_types=tuple(
                "full_attention" if bt == "qsa_attention" else bt
                for bt in recipe.layer_types),
            rms_norm_eps=float(recipe.rms_norm_eps),
            rope_theta=float(fa.rope.theta),
            lin_k_heads=la.num_key_heads, lin_k_dim=la.key_head_dim,
            lin_v_heads=la.num_value_heads, lin_v_dim=la.value_head_dim,
            lin_conv=la.conv_kernel_size, lin_l2norm=la.qk_l2norm,
            attn_heads=fa.num_heads, attn_kv_heads=fa.num_kv_heads,
            attn_head_dim=fa.effective_head_dim(recipe.hidden_size),
            rotary_factor=float(fa.rope.partial_rotary_factor),
            idx_heads=int(qsa.get("indexer_n_heads", 4)),
            idx_dim=int(qsa.get("indexer_head_dim", 128)),
            idx_budget=int(qsa.get("indexer_budget", 2048)),
            idx_ratio=int(qsa.get("indexer_compress_ratio", 4)),
            n_experts=mlp.num_experts, top_k=mlp.num_experts_per_tok,
            moe_inter=mlp.intermediate_size,
            shared_inter=mlp.shared_expert_intermediate_size,
            ple_layer_ids=tuple(ple.get("layer_ids", ())),
        )

    def validate(self) -> None:
        if self.ple_layer_ids:
            raise NotImplementedError("PLE is deferred to milestone Q5")
        if self.hc_count < 2:
            raise ValueError("qwen4_exp requires hc_count >= 2")
        rot = self.rotary_dim
        if rot <= 0 or rot % 2 or rot > self.idx_dim:
            raise ValueError(
                f"rotary_dim {rot} invalid or larger than index head_dim "
                f"{self.idx_dim} (QSAIndexer constraint)")
        if self.idx_ratio < 2 or self.idx_budget % self.idx_ratio:
            raise ValueError("require idx_ratio >= 2 and budget %% ratio == 0")
        for bt in self.layer_types:
            if bt not in ("linear_attention", "full_attention"):
                raise ValueError(f"unsupported layer type {bt}")


def _sl(axis: int, a: int, b: int, ndim: int):
    return tuple(slice(a, b) if i == axis else slice(None) for i in range(ndim))


def _append_axis(st: dict, key: str, rows: np.ndarray, axis: int) -> None:
    """Append `rows` to st[key] along `axis` with capacity doubling.

    v1 rebuilt the whole cache with np.concatenate on every step (O(S) copy
    per token per layer). st[key] stays an exact-length VIEW of a private
    capacity buffer, so readers and parity tests see identical semantics.
    """
    cur = st[key]
    n, m = cur.shape[axis], rows.shape[axis]
    buf = st.get(key + "_buf")
    cap = 0 if buf is None else buf.shape[axis]
    if n + m > cap:
        newcap = max(n + m, 2 * max(cap, n), 16)
        shape = list(cur.shape)
        shape[axis] = newcap
        buf = np.zeros(shape, cur.dtype)
        if n:
            buf[_sl(axis, 0, n, cur.ndim)] = cur
        st[key + "_buf"] = buf
    buf[_sl(axis, n, n + m, cur.ndim)] = rows
    st[key] = buf[_sl(axis, 0, n + m, cur.ndim)]


class Qwen4ExpState:
    """Per-layer incremental state (GDN/conv window, KV, indexer caches)."""

    def __init__(self, cfg: Qwen4ExpCfg):
        self.cfg = cfg
        self.n_ctx = 0
        self.layers: list[dict] = []
        for bt in cfg.layer_types:
            if bt == "linear_attention":
                c = 2 * cfg.lin_k_heads * cfg.lin_k_dim \
                    + cfg.lin_v_heads * cfg.lin_v_dim
                self.layers.append({
                    "gdn": None,
                    "conv_win": np.zeros((1, c, cfg.lin_conv - 1), np.float32),
                })
            else:
                self.layers.append({
                    "k": np.zeros((cfg.attn_kv_heads, 0, cfg.attn_head_dim), np.float32),
                    "v": np.zeros((cfg.attn_kv_heads, 0, cfg.attn_head_dim), np.float32),
                    "tok_k": np.zeros((0, 1, cfg.idx_dim), np.float32),
                    "ck": np.zeros((0, cfg.idx_dim), np.float32),
                })

    def state_bytes(self) -> int:
        """Live bytes of this sequence's state (includes capacity buffers)."""
        return sum(v.nbytes for lay in self.layers
                   for v in lay.values() if isinstance(v, np.ndarray))


# ---------------------------------------------------------------------------
# weight fetch helpers
# ---------------------------------------------------------------------------

def _hc_w(weights, prefix):
    p = f"{_TEXT}.layers.{prefix}"
    return (weights[f"{p}.hc_norm.weight"],
            weights[f"{p}.input_mix_weight_down.weight"],
            weights[f"{p}.input_mix_weight_up.weight"],
            weights[f"{p}.block_inject_weight.weight"])


def _gdn_w(weights, i):
    p = f"{_TEXT}.layers.{i}.linear_attn"
    return dict(
        w_in_qkv=weights[f"{p}.in_proj_qkv.weight"],
        w_conv=weights[f"{p}.conv1d.weight"],
        w_z=weights[f"{p}.in_proj_z.weight"],
        w_b=weights[f"{p}.in_proj_b.weight"],
        w_a=weights[f"{p}.in_proj_a.weight"],
        a_log=weights[f"{p}.A_log"],
        dt_bias=weights[f"{p}.dt_bias"],
        norm_w=weights[f"{p}.norm.weight"],
        w_out=weights[f"{p}.out_proj.weight"],
    )


def _moe_w(weights, i, cfg: Qwen4ExpCfg):
    p = f"{_TEXT}.layers.{i}.mlp"
    e = cfg.n_experts
    stack = lambda suf: np.stack(  # noqa: E731 (oracle simplicity)
        [np.asarray(weights[f"{p}.experts.{j}.{suf}"], np.float32) for j in range(e)])
    return dict(
        router=weights[f"{p}.gate.weight"],
        eg=stack("gate_proj.weight"), eu=stack("up_proj.weight"),
        ed=stack("down_proj.weight"),
        sg=weights[f"{p}.shared_expert.gate_proj.weight"],
        su=weights[f"{p}.shared_expert.up_proj.weight"],
        sd=weights[f"{p}.shared_expert.down_proj.weight"],
        sgl=weights[f"{p}.shared_expert_gate.weight"],
    )


def _moe_forward(x, mw, cfg: Qwen4ExpCfg):
    """x: (1, S, H) -> (1, S, H) dense-emulated MoE."""
    n = x.shape[1]
    out = qe.moe_block_forward(
        x.reshape(n, -1), mw["router"], mw["eg"], mw["eu"], mw["ed"],
        mw["sg"], mw["sgl"], mw["su"], mw["sd"], cfg.top_k)
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# QSA selection (exposed for tests): compressed keys -> token slots per row
# ---------------------------------------------------------------------------

def qsa_select_slots(q_idx, ck, row_starts, row_ends, qpos, seqlens, cfg):
    """Indexer selection for rows of queries against compressed keys.

    Mirrors `QSAIndexer.select_prefill_tokens` / `select_decode_tokens`
    (identical math; decode == prefill with one row).
    Returns int32 (rows, budget + ratio - 1) in-sequence token ids, -1 padded.
    """
    block_topk = cfg.idx_budget // cfg.idx_ratio
    logits = qe.qsa_mqa_logits(q_idx, ck, row_starts, row_ends)
    blocks = qe.qsa_fast_topk(logits, row_starts, row_ends, block_topk)
    return qe.qsa_expand_block_indices(
        blocks, qpos, seqlens, cfg.idx_ratio, cfg.idx_budget)


def _indexer_compress_append(tok_k, kn_w, cfg, block_end_pos, theta):
    """Compress one complete block [end-ratio, end) and append to `ck`.

    Rope angle comes from the FIRST token of the block
    (`source_rope[group_locs[:, 0]]`, qsa_indexer.py:363-366).
    """
    group = tok_k[block_end_pos - cfg.idx_ratio:block_end_pos, 0]
    pooled = qe.qsa_average_pool_keys(group)
    cos, sin = qq.compute_cos_sin(
        theta, np.array([[block_end_pos - cfg.idx_ratio]]), cfg.rotary_dim)
    return qe.qsa_normalize_compressed_keys(
        pooled[None], kn_w, cos[0], sin[0], cfg.rotary_dim, cfg.idx_dim,
        eps=cfg.rms_norm_eps)[0]


# ---------------------------------------------------------------------------
# attention blocks (operate on `mixed` [1, S, H]; return block output)
# ---------------------------------------------------------------------------

def _linear_block_prefill(h, weights, i, cfg, st):
    fw = _gdn_w(weights, i)
    la = cfg
    out, state = qq.gated_delta_net_forward(
        h, **fw,
        num_k_heads=la.lin_k_heads, head_k_dim=la.lin_k_dim,
        num_v_heads=la.lin_v_heads, head_v_dim=la.lin_v_dim,
        conv_kernel_size=la.lin_conv,
        use_qk_l2norm_in_kernel=la.lin_l2norm,
        chunked=True, chunk_size=64, rms_eps=cfg.rms_norm_eps)
    st["gdn"] = state
    mixed = np.transpose(qq.linear(h, fw["w_in_qkv"]), (0, 2, 1))
    kc = la.lin_conv
    st["conv_win"] = mixed[:, :, -(kc - 1):] if kc > 1 else np.zeros((1, 0, 1))
    return out


def _linear_block_step(h1, weights, i, cfg, st):
    fw = _gdn_w(weights, i)
    la = cfg
    key_dim = la.lin_k_heads * la.lin_k_dim
    value_dim = la.lin_v_heads * la.lin_v_dim
    kc = la.lin_conv

    mixed = np.transpose(qq.linear(h1, fw["w_in_qkv"])[0], (1, 0))  # (C, 1)
    win = np.concatenate([st["conv_win"], mixed[None]], axis=-1)
    conv = qq.causal_conv1d_depthwise(win[:, :, -kc:], fw["w_conv"], activation="silu")
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

    z = qq.linear(h1, fw["w_z"]).reshape(1, 1, la.lin_v_heads, la.lin_v_dim)
    beta = qq.sigmoid(qq.linear(h1, fw["w_b"]).reshape(1, 1, -1))
    g = (-np.exp(fw["a_log"]).astype(np.float32)
         * qq.softplus(qq.linear(h1, fw["w_a"]).astype(np.float32)
                       + fw["dt_bias"])).reshape(1, 1, -1)

    core, state = qq.gated_delta_rule_recurrent(
        query, key, value, g=g, beta=beta,
        initial_state=st["gdn"], output_final_state=True,
        use_qk_l2norm_in_kernel=la.lin_l2norm)
    st["gdn"] = state
    core = qq.rms_norm_gated(core, fw["norm_w"], z, eps=cfg.rms_norm_eps)
    core = core.reshape(1, 1, -1)
    return qq.linear(core, fw["w_out"]).astype(h1.dtype)


def _qsa_pre(weights, i, cfg):
    p = f"{_TEXT}.layers.{i}.self_attn"
    return dict(
        wq=weights[f"{p}.q_proj.weight"], wk=weights[f"{p}.k_proj.weight"],
        wv=weights[f"{p}.v_proj.weight"], wo=weights[f"{p}.o_proj.weight"],
        qn=weights[f"{p}.q_norm.weight"], kn=weights[f"{p}.k_norm.weight"],
        wiqk=weights[f"{p}.indexer.index_qk_proj.weight"],
        iq=weights[f"{p}.indexer.q_layernorm.weight"],
        ik=weights[f"{p}.indexer.k_layernorm.weight"],
    )


def _qsa_block_prefill(h, weights, i, cfg, st):
    """Batched QSA layer over S prompt tokens; fills the incremental caches."""
    aw = _qsa_pre(weights, i, cfg)
    eps = cfg.rms_norm_eps
    nh, kvh, hd, rot = cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim, cfg.rotary_dim
    s = h.shape[1]
    pos = np.arange(s)[None, :]
    cos, sin = qq.compute_cos_sin(cfg.rope_theta, pos, rot)

    qg = qq.linear(h, aw["wq"]).reshape(1, s, nh, hd * 2)
    q2, gate = qg[..., :hd], qg[..., hd:]
    q2 = qe.gemma_rmsnorm(q2, aw["qn"], eps).transpose(0, 2, 1, 3)
    k = qe.gemma_rmsnorm(
        qq.linear(h, aw["wk"]).reshape(1, s, kvh, hd), aw["kn"], eps).transpose(0, 2, 1, 3)
    v = qq.linear(h, aw["wv"]).reshape(1, s, kvh, hd).transpose(0, 2, 1, 3)
    q2, k = qq.apply_rotary_pos_emb(q2, k, cos, sin)
    st["k"], st["v"] = k[0].astype(np.float32), v[0].astype(np.float32)

    # indexer
    q_idx, tok_k = qe.qsa_index_project_qk(
        h[0], aw["wiqk"], aw["iq"], aw["ik"], cos[0], sin[0], rot,
        cfg.idx_heads, cfg.idx_dim, eps)
    st["tok_k"] = tok_k.astype(np.float32)
    nb = s // cfg.idx_ratio
    rows = [_indexer_compress_append(
                st["tok_k"], aw["ik"], cfg, (b + 1) * cfg.idx_ratio,
                cfg.rope_theta)
            for b in range(nb)]
    st["ck"] = (np.concatenate(rows, axis=0) if rows
                else np.zeros((0, cfg.idx_dim), np.float32))

    if nb:
        qpos = np.arange(s, dtype=np.int32)
        ends = ((qpos + 1) // cfg.idx_ratio).astype(np.int32)
        starts = np.zeros(s, dtype=np.int32)
        slots = qsa_select_slots(q_idx, st["ck"], starts, ends, qpos,
                                 np.full(s, s, np.int32), cfg)
    else:
        # no complete block yet: only the partial-block tail is visible
        slots = qe.qsa_expand_block_indices(
            np.full((s, cfg.idx_budget // cfg.idx_ratio), -1, np.int32),
            np.arange(s, dtype=np.int32), np.full(s, s, np.int32),
            cfg.idx_ratio, cfg.idx_budget)

    attn = qe.qsa_sparse_attention(
        q2[0].transpose(1, 0, 2).astype(np.float32),  # (S, nh, hd)
        st["k"].transpose(1, 0, 2), st["v"].transpose(1, 0, 2), slots)
    attn = attn.reshape(1, s, nh * hd)
    attn = attn * qq.sigmoid(gate.reshape(1, s, nh * hd))
    return qq.linear(attn, aw["wo"]).astype(h.dtype)


def _qsa_block_step(h1, weights, i, cfg, st, pos):
    """One incremental QSA decode step at absolute position `pos`."""
    aw = _qsa_pre(weights, i, cfg)
    eps = cfg.rms_norm_eps
    nh, kvh, hd, rot = cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim, cfg.rotary_dim
    cos, sin = qq.compute_cos_sin(cfg.rope_theta, np.array([[pos]]), rot)

    qg = qq.linear(h1, aw["wq"]).reshape(1, nh, hd * 2)
    q2, gate = qg[..., :hd], qg[..., hd:]
    q2 = qe.gemma_rmsnorm(q2, aw["qn"], eps).reshape(1, nh, 1, hd)
    k = qe.gemma_rmsnorm(
        qq.linear(h1, aw["wk"]).reshape(1, kvh, 1, hd), aw["kn"], eps)
    v = qq.linear(h1, aw["wv"]).reshape(1, kvh, 1, hd)
    q2, k = qq.apply_rotary_pos_emb(q2, k, cos, sin)
    _append_axis(st, "k", k[0].astype(np.float32), axis=1)
    _append_axis(st, "v", v[0].astype(np.float32), axis=1)

    q_idx, tok_k = qe.qsa_index_project_qk(
        h1[0], aw["wiqk"], aw["iq"], aw["ik"], cos[0], sin[0], rot,
        cfg.idx_heads, cfg.idx_dim, eps)
    _append_axis(st, "tok_k", tok_k.astype(np.float32), axis=0)
    if (pos + 1) % cfg.idx_ratio == 0:
        row = _indexer_compress_append(
            st["tok_k"], aw["ik"], cfg, pos + 1, cfg.rope_theta)
        _append_axis(st, "ck", row, axis=0)

    # QSA slot coordinates are the layer's OWN cache rows (== absolute pos
    # for the main model, but NOT for an MTP draft state seeded mid-context:
    # rope keeps the absolute `pos`, cache windows use the stored rows).
    nb = st["ck"].shape[0]
    rows = np.array([st["k"].shape[1]], np.int32)
    qpos = rows - 1
    if nb:
        slots = qsa_select_slots(
            q_idx, st["ck"], np.zeros(1, np.int32), np.array([nb], np.int32),
            qpos, rows, cfg)
    else:
        slots = qe.qsa_expand_block_indices(
            np.full((1, cfg.idx_budget // cfg.idx_ratio), -1, np.int32),
            qpos, rows, cfg.idx_ratio, cfg.idx_budget)

    attn = qe.qsa_sparse_attention(
        q2.transpose(0, 2, 1, 3)[0],  # (1, nh, hd)
        st["k"].transpose(1, 0, 2), st["v"].transpose(1, 0, 2), slots)
    attn = attn.reshape(1, nh * hd)
    attn = attn * qq.sigmoid(gate.reshape(1, nh * hd))
    return qq.linear(attn, aw["wo"]).astype(h1.dtype)


# ---------------------------------------------------------------------------
# model-level forward
# ---------------------------------------------------------------------------

def _layer_block(h, weights, i, cfg, st, decode_pos):
    """Attention block dispatch (h: [1, S, H])."""
    if decode_pos is None:
        if cfg.layer_types[i] == "linear_attention":
            return _linear_block_prefill(h, weights, i, cfg, st)
        return _qsa_block_prefill(h, weights, i, cfg, st)
    if cfg.layer_types[i] == "linear_attention":
        return _linear_block_step(h, weights, i, cfg, st)
    return _qsa_block_step(h, weights, i, cfg, st, decode_pos)


def _forward(ids_or_state, weights, cfg, last_id=None, hyper_in=None,
             return_hyper=False):
    """Shared driver: prefill (ids), one decode step (state, last_id), or a
    hyper-injection run (state|None, hyper_in) for the MTP draft model
    (oracle/upstream/sglang/qwen4_exp_mtp.py feeds fused inputs directly).

    `return_hyper` additionally returns the PRE-final-combine hyper tensor
    (1, S, hc*H) — the tensor the MTP consumes as `spec_info.hidden_states`."""
    cfg.validate()
    if hyper_in is not None:
        hyper_in = np.asarray(hyper_in, dtype=np.float32)
    if hyper_in is not None and hyper_in.shape[1] > 1:
        raise NotImplementedError(
            "hyper-injection only supports single-token draft steps (v1); "
            "batched draft-extend is a C-phase feature")
    prefill = last_id is None and hyper_in is None
    state = None
    if prefill:
        ids = np.asarray(ids_or_state, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        state = Qwen4ExpState(cfg)
    elif hyper_in is not None:
        hyper = np.asarray(hyper_in, dtype=np.float32)
        s = hyper.shape[1]
        state = ids_or_state if ids_or_state is not None else Qwen4ExpState(cfg)
    else:
        ids = np.array([[last_id]], dtype=np.int64)
        state = ids_or_state
    if hyper_in is None:
        s = ids.shape[1]
    hc, hs, eps = cfg.hc_count, cfg.hidden, cfg.rms_norm_eps

    if hyper_in is None:
        embed_w = weights[f"{_TEXT}.embed_tokens.weight"]
        emb = embed_w[ids].astype(np.float32)              # (1, S, H)
        hyper = np.concatenate([emb] * hc, axis=-1)        # (1, S, hc*H)

    for i, bt in enumerate(cfg.layer_types):
        st = state.layers[i]
        a_norm, a_dn, a_up, a_inj = _hc_w(weights, f"{i}.attn_hyper_connection")
        m_norm, m_dn, m_up, m_inj = _hc_w(weights, f"{i}.mlp_hyper_connection")

        mixed, normed = qe.hc_mix(hyper, a_norm, a_dn, a_up, hc, hs, eps)
        block = _layer_block(mixed, weights, i, cfg, st,
                             None if prefill else state.n_ctx)
        hyper = qe.hc_combine(block, hyper, normed, a_inj, hc, hs)

        mixed2, normed2 = qe.hc_mix(hyper, m_norm, m_dn, m_up, hc, hs, eps)
        moe_out = _moe_forward(mixed2, _moe_w(weights, i, cfg), cfg).astype(hyper.dtype)
        hyper = qe.hc_combine(moe_out, hyper, normed2, m_inj, hc, hs)

    state.n_ctx += s
    f_norm, f_dn, f_up = (
        weights[f"{_TEXT}.hyper_connection_mixer.hc_norm.weight"],
        weights[f"{_TEXT}.hyper_connection_mixer.input_mix_weight_down.weight"],
        weights[f"{_TEXT}.hyper_connection_mixer.input_mix_weight_up.weight"])
    final, _ = qe.hc_mix(hyper, f_norm, f_dn, f_up, hc, hs, eps)
    logits = final[0] @ weights["lm_head.weight"].T
    if return_hyper:
        return state, logits, hyper
    return state, logits


def prefill(ids, weights, cfg, return_hyper: bool = False):
    """Full forward over the prompt; returns (state, logits [S, V]),
    optionally plus the pre-final hyper tensor (1, S, hc*H) for MTP."""
    return _forward(ids, weights, cfg, return_hyper=return_hyper)


def decode_step(state, weights, cfg, last_id: int) -> np.ndarray:
    """Advance one token using cached state; returns logits (V,)."""
    _, logits = _forward(state, weights, cfg, last_id=int(last_id))
    return logits[-1]


def decode_step_full(state, weights, cfg, last_id: int):
    """decode_step + the post-step hyper row (hc*H,) for MTP drafting."""
    _, logits, hyper = _forward(state, weights, cfg, last_id=int(last_id),
                                return_hyper=True)
    return logits[-1], hyper[0, -1]


def generate_greedy(ids, weights, cfg, max_new: int) -> list[int]:
    """Minimal greedy loop used by tests and early parity work."""
    ids = list(np.asarray(ids, dtype=np.int64).reshape(-1))
    state, logits = prefill(ids, weights, cfg)
    out = []
    for _ in range(max_new):
        nxt = int(np.argmax(logits[-1]))
        out.append(nxt)
        logits = decode_step(state, weights, cfg, nxt)[None]
    return out
