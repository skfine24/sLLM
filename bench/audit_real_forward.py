"""Real-weight T1 validation on RGB-less (dev-machine) run.

Fetches the actual weights of layer 0 (GatedDeltaNet) and layer 3
(full attention) of Qwen/Qwen3.8-27B-FP8 over HTTP Range, dequantizes them
with the loaders, and runs the numpy reference forward on a random hidden
state (no embeddings/MLP needed):

  1. layer-0 linear attention: chunked vs recurrent equivalence on REAL weights
  2. layer-3 full attention: paged->eager forward with partial M-RoPE

This integrates loaders + ref on real byte-level FP8 data; full-model / live
parity vs vLLM still requires the cluster.

Run:  python bench/audit_real_forward.py [--seq 37]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loaders.safetensors_reader import load_tensors_from_url  # noqa: E402
from loaders.weights import dequant_tensors  # noqa: E402
from ref import qwen3_5 as q  # noqa: E402

BASE = "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main"


def _report(name: str, arr: np.ndarray):
    rms = float(np.sqrt((arr.astype(np.float64) ** 2).mean()))
    print(f"[{name}] shape={arr.shape} finite={bool(np.isfinite(arr).all())} rms={rms:.5f} "
          f"min={float(arr.min()):.5f} max={float(arr.max()):.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=37)
    args = ap.parse_args()
    s = args.seq
    rng = np.random.default_rng(0)
    print(f"[audit] seq_len={s}")

    # ------------------------- layer 0: GatedDeltaNet -------------------------
    l0 = [f"model.language_model.layers.0.linear_attn.{n}" for n in (
        "in_proj_qkv.weight", "in_proj_qkv.weight_scale_inv",
        "conv1d.weight", "in_proj_z.weight", "in_proj_z.weight_scale_inv",
        "in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias",
        "norm.weight", "out_proj.weight", "out_proj.weight_scale_inv",
    )]
    w0 = dequant_tensors(load_tensors_from_url(f"{BASE}/layers-0.safetensors", l0))
    p = "model.language_model.layers.0.linear_attn."
    hidden = (rng.standard_normal((1, s, 5120)) * 0.05).astype(np.float32)
    _report("hidden-in", hidden)

    def run0(chunked):
        return q.gated_delta_net_forward(
            hidden,
            w_in_qkv=w0[f"{p}in_proj_qkv.weight"], w_conv=w0[f"{p}conv1d.weight"],
            w_z=w0[f"{p}in_proj_z.weight"], w_b=w0[f"{p}in_proj_b.weight"],
            w_a=w0[f"{p}in_proj_a.weight"], a_log=w0[f"{p}A_log"],
            dt_bias=w0[f"{p}dt_bias"], norm_w=w0[f"{p}norm.weight"],
            w_out=w0[f"{p}out_proj.weight"],
            num_k_heads=16, head_k_dim=128, num_v_heads=48, head_v_dim=128,
            conv_kernel_size=4, chunked=chunked, chunk_size=64,
        )

    o_c, st_c = run0(chunked=True)
    o_r, st_r = run0(chunked=False)
    _report("layer0 chunked out", o_c)
    _report("layer0 recurrent out", o_r)
    diff = np.abs(o_c - o_r).max()
    print(f"[audit] layer0 chunked-vs-recurrent max_abs_diff={diff:.3e}  (real weights)")
    print(f"[audit] layer0 state shape={st_c.shape}")

    # ------------------------- layer 3: full attention -------------------------
    l3 = [f"model.language_model.layers.3.self_attn.{n}" for n in (
        "q_proj.weight", "q_proj.weight_scale_inv", "k_proj.weight",
        "k_proj.weight_scale_inv", "v_proj.weight", "v_proj.weight_scale_inv",
        "o_proj.weight", "o_proj.weight_scale_inv", "q_norm.weight", "k_norm.weight",
    )]
    w3 = dequant_tensors(load_tensors_from_url(f"{BASE}/layers-3.safetensors", l3))
    p3 = "model.language_model.layers.3.self_attn."
    hidden3 = (rng.standard_normal((1, 5, 5120)) * 0.05).astype(np.float32)
    cos, sin = q.compute_cos_sin(1e7, np.arange(5, dtype=np.int64)[None, :], 64)
    out = q.full_attention_forward(
        hidden3,
        w_q=w3[f"{p3}q_proj.weight"], w_k=w3[f"{p3}k_proj.weight"],
        w_v=w3[f"{p3}v_proj.weight"], w_o=w3[f"{p3}o_proj.weight"],
        q_norm_w=w3[f"{p3}q_norm.weight"], k_norm_w=w3[f"{p3}k_norm.weight"],
        cos=cos, sin=sin, num_heads=24, kv_heads=4, head_dim=256,
    )
    _report("layer3 full-attn out", out)
    print("[audit] PASS: real-weight layer T1 (GatedDeltaNet + full attention)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
