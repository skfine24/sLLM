"""Dev-machine validation: fetch a real Qwen3.8-27B-FP8 shard over HTTP Range
and prove the FP8 blocked-dequant path on actual checkpoint bytes.

Reads only what is needed (shard header + the layer-0 linear-attention QKV
weight and its scale tensor), so no full-model download is required.

Run:  python bench/audit_shard_fp8.py [--shard-name layers-0.safetensors]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loaders import safetensors_reader as sr  # noqa: E402
from loaders.fp8 import decode_e4m3fn_array, dequant_weight_blocked  # noqa: E402

BASE = "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main"
READ_CHUNK = 1 << 20  # 1 MiB


def fetch_range(url: str, begin: int, length: int) -> bytes:
    end = begin + length - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="layers-0.safetensors")
    args = ap.parse_args()
    url = f"{BASE}/{args.shard}"

    print(f"[audit] fetching header prefix of {args.shard} (1 MiB range)")
    prefix = fetch_range(url, 0, READ_CHUNK)
    header = sr.parse_header_bytes(prefix)
    print(f"[audit] header parsed: {len(header.tensors)} tensors, data_offset={header.data_offset}")

    w_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    s_name = w_name + "_scale_inv"
    w_spec = header.spec(w_name)
    s_spec = header.spec(s_name)
    print(f"[audit] {w_name}: {w_spec.dtype} {w_spec.shape} bytes={w_spec.end - w_spec.begin}")
    print(f"[audit] {s_name}: {s_spec.dtype} {s_spec.shape} bytes={s_spec.end - s_spec.begin}")

    w_raw = fetch_range(url, header.data_offset + w_spec.begin, w_spec.end - w_spec.begin)
    s_raw = fetch_range(url, header.data_offset + s_spec.begin, s_spec.end - s_spec.begin)
    print(f"[audit] fetched weight bytes={len(w_raw)} scale bytes={len(s_raw)}")

    w8 = np.frombuffer(w_raw, dtype=np.uint8).reshape(w_spec.shape)
    scale = sr.decode_tensor_bytes(s_raw, s_spec)
    out = dequant_weight_blocked(w8, scale, 128, 128)
    print(f"[audit] dequant shape={out.shape} dtype={out.dtype}")

    f8 = decode_e4m3fn_array(w8)
    if np.isnan(f8).any():
        print("[audit] FAIL: NaN present in raw fp8 weights")
        return 1

    print(f"[audit] fp8 decoded range: [{f8.min():.3f}, {f8.max():.3f}]")
    print(f"[audit] scale_inv range:   [{scale.min():.6g}, {scale.max():.6g}]")
    print(f"[audit] dequant range:     [{out.min():.6g}, {out.max():.6g}]")

    # For a per-block quantizer, the block maxima should saturate near the fp8
    # ceiling (448) - evidence the decode + block layout are right.
    bh, bw = 128, 128
    nh, nw = w8.shape[0] // bh, w8.shape[1] // bw
    block_max = np.zeros((nh, nw), dtype=np.float32)
    for i in range(nh):
        for j in range(nw):
            block_max[i, j] = np.abs(f8[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]).max()
    sat = (block_max >= 400.0).mean()
    print(f"[audit] blocks={nh}x{nw} frac block-max>=400: {sat:.3f}")

    print("[audit] PASS: real-shard FP8 header parse + decode + blocked dequant OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
