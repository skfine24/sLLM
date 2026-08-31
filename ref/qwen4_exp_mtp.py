"""Qwen4-Exp MTP (speculative draft head) numpy oracle + acceptance loop.

Port of `oracle/upstream/sglang/qwen4_exp_mtp.py` (Qwen4ExpForCausalLMMTP).
Key differences from the qwen3_5 MTP (ref/mtp.py): the draft model is a
FULL one-layer Qwen4ExpModel (QSA attention + HC + MoE, `is_nextn=True`,
no PLE), and with hc_count > 1 the input fusion is the HYBRID 2-fc variant:

    e   = fc_embedding(gemma_rmsnorm(embeds,      pre_fc_norm_embedding))
    hv  = gemma_rmsnorm(hyper, pre_fc_norm_hidden).view(S, hc, H)
    out = (e[:, None, :] + fc_hidden(hv)).reshape(S, hc*H)   # -> new hyper

`hyper` is the MAIN model's pre-final-combine hyper tensor (1, S, hc*H) —
upstream `spec_info.hidden_states`; `Qwen4ExpState`'s final mix runs over
the fused hyper, and the draft's own pre-final hyper chains the next draft
(upstream `_set_hc_logits_hidden_states` / EAGLE-v2 handoff).
hc_count <= 1 falls back to the standard single-fc fusion (mtp.fc, 2H -> H).

Embed and lm_head are SHARED with the main model (upstream: same tensors).
Checkpoint keys `mtp.*` are remapped to driver names by `_mtp_weights_map`.

`generate_greedy_mtp` is the v1 draft/verify loop: it must reproduce plain
greedy EXACTLY (the only correctness gate for spec decode); it does not
speed anything up on the single-token CPU reference — batched multi-token
verification is the GPU (C-phase) feature that makes drafts pay off.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import qwen4_exp as qe
from . import qwen4_exp_pipeline as qp

_TEXT = "model.language_model"


def mtp_cfg(cfg):
    """MTP driver config: one full-attention layer, no PLE (oracle:
    config.num_hidden_layers = 1; layer_types = [full_attention];
    ple_layer_ids = [])."""
    return replace(cfg, layer_types=("full_attention",), ple_layer_ids=())


def mtp_weights_map(weights: dict) -> dict:
    """Remap `mtp.*` checkpoint keys onto the driver's standard names. The
    one-layer draft config only reads layers.0 + the final mixer, which the
    mapped keys overwrite; embed_tokens/lm_head stay shared."""
    d = dict(weights)
    for k, v in weights.items():
        if k.startswith("mtp.layers."):
            d[k.replace("mtp.layers.", f"{_TEXT}.layers.", 1)] = v
        elif k.startswith("mtp.hyper_connection_mixer"):
            d[k.replace("mtp.hyper_connection_mixer",
                        f"{_TEXT}.hyper_connection_mixer", 1)] = v
    return d


class Qwen4ExpMTP:
    """Stateful draft model (one MTP step per call; state rebuilds per
    verification cycle — v1 drafts from verified context only)."""

    def __init__(self, weights: dict, cfg):
        self.cfg = cfg
        self.hc = cfg.hc_count
        self.mcfg = mtp_cfg(cfg)
        self.wmap = mtp_weights_map(weights)
        self.w = weights
        self.hybrid = self.hc > 1
        self.state = None

    def fuse(self, embeds: np.ndarray, hyper: np.ndarray) -> np.ndarray:
        """`_mtp_input_fusion`: embeds (S, H), hyper (S, hc*H) -> fused."""
        eps = self.cfg.rms_norm_eps
        w = self.w
        if self.hybrid:
            e = qe.gemma_rmsnorm(embeds, w["mtp.pre_fc_norm_embedding.weight"],
                                 eps)
            e = e @ w["mtp.fc_embedding.weight"].T
            hv = qe.gemma_rmsnorm(hyper, w["mtp.pre_fc_norm_hidden.weight"],
                                  eps)
            enc = hv.reshape(-1, self.hc, self.cfg.hidden) \
                @ w["mtp.fc_hidden.weight"].T
            return (e[:, None, :] + enc).reshape(hyper.shape)
        cat = np.concatenate([
            qe.gemma_rmsnorm(embeds, w["mtp.pre_fc_norm_embedding.weight"], eps),
            qe.gemma_rmsnorm(hyper, w["mtp.pre_fc_norm_hidden.weight"], eps),
        ], axis=-1)
        return cat @ w["mtp.fc.weight"].T

    def step(self, embed_row: np.ndarray, hyper_row: np.ndarray):
        """Draft one position: returns (logits (V,), next hyper row (hc*H,))."""
        fused = self.fuse(embed_row[None, :].astype(np.float32),
                          hyper_row[None, :].astype(np.float32))
        state, logits, hyper_out = qp._forward(
            self.state, self.wmap, self.mcfg, hyper_in=fused[:, None, :],
            return_hyper=True)
        self.state = state
        return logits[-1], hyper_out[0, -1]


def generate_greedy_mtp(ids, weights, cfg, max_new: int,
                        mtp: Qwen4ExpMTP | None = None, num_draft: int = 2):
    """Greedy generation with MTP drafting. Output is IDENTICAL to
    `qp.generate_greedy` by construction (drafts only skip re-drafting;
    every emitted token is the main model's argmax). Returns
    (tokens, drafted, accepted) for acceptance-rate telemetry."""
    ids = list(np.asarray(ids, dtype=np.int64).reshape(-1))
    mtp = mtp or Qwen4ExpMTP(weights, cfg)
    embed_w = weights[f"{_TEXT}.embed_tokens.weight"]

    state, logits, hyper = qp.prefill(ids, weights, cfg, return_hyper=True)
    out: list[int] = []
    drafted = accepted = 0
    last_tok = ids[-1]                  # last token ACTUALLY in the cache
    last_hyper = hyper[0, -1]            # its pre-final hyper row
    pending = logits[-1]                 # main logits predicting the next id
    pos = len(ids) - 1                   # absolute position of last_tok

    while len(out) < max_new:
        # v1 drafts from verified context only: fresh draft state, but its
        # rope positions CONTINUE the main context (upstream passes the real
        # position ids into the is_nextn model).
        mstate = qp.Qwen4ExpState(mtp.mcfg)
        mstate.n_ctx = pos
        mtp.state = mstate
        drafts, e, h = [], embed_w[last_tok], last_hyper
        for _ in range(num_draft):
            lg, h = mtp.step(e, h)
            d = int(np.argmax(lg))
            drafts.append(d)
            e = embed_w[d]
        for d in drafts:
            p = int(np.argmax(pending))
            drafted += 1
            if d != p:                   # reject: emit the main model's pick
                out.append(p)
                pos += 1
                if len(out) < max_new:
                    pending, last_hyper = qp.decode_step_full(
                        state, weights, cfg, p)
                last_tok = p
                break
            accepted += 1
            out.append(d)
            pos += 1
            if len(out) >= max_new:
                last_tok = d
                break
            pending, last_hyper = qp.decode_step_full(
                state, weights, cfg, d)
            last_tok = d
    return out, drafted, accepted
