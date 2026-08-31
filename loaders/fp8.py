"""FP8 / low-precision decode utilities for the qwen3_5 checkpoint.

Semantics are pinned to the OCP E4M3FN format (the safetensors dtype F8_E4M3,
matching `torch.float8_e4m3fn`), as verified against `ml_dtypes`:
- sign(1) exp(4, bias 7) mantissa(3)
- subnormals supported: exp == 0 -> (-1)^s * mant * 2^-9
- exp == 15, mant == 0..6 -> normal numbers (1 + mant/8) * 2^8 (256 .. 448)
- exp == 15, mant == 7 -> NaN (single pattern 0x7f / 0xff)
- no infinity

The checkpoint stores per-128x128-block *inverse* scales as BF16
(`weight_scale_inv`), so
    dequant[i, j] = fp8_weight[i, j] * scale_inv[i//128, j//128]

These functions are verified against `ml_dtypes.float8_e4m3fn` in the test
suite (the authoritative reference for the float8 dtype family).
"""

from __future__ import annotations

import numpy as np

_F8_BIAS = 7
_F8_EXP_BITS = 4
_F8_MANT_BITS = 3
_NAN = np.float32(np.nan)


def decode_e4m3fn(byte: int) -> float:
    """Decode a single F8_E4M3 byte to a Python float (NaNs -> nan)."""
    return float(decode_e4m3fn_array(np.array([byte], dtype=np.uint8))[0])


def decode_e4m3fn_array(u8: np.ndarray) -> np.ndarray:
    """Vectorized decode of F8_E4M3 bytes (uint8) to float32."""
    u8 = np.asarray(u8, dtype=np.uint8).astype(np.uint16)
    sign = (u8 >> 7) & 1
    exp = (u8 >> 3) & 0xF
    mant = u8 & 0x7

    pos = np.where(sign == 0, 1.0, -1.0)

    out = np.zeros(u8.shape, dtype=np.float32)
    # exp == 0: subnormal -> mant * 2^(1 - bias - mant_bits) == mant * 2^-9
    sub = exp == 0
    out[sub] = mant[sub].astype(np.float32) * (2.0 ** (1 - _F8_BIAS - _F8_MANT_BITS))
    # NaN: exp == 15 and mant == 0b111 only
    nan_pat = (exp == 15) & (mant == 7)
    # everything else is a normal number, incl. exp==15 & mant==0..6 (256..448)
    normal = ~sub & ~nan_pat
    out[normal] = (1.0 + mant[normal] / 8.0) * (2.0 ** (exp[normal].astype(np.float32) - _F8_BIAS))
    out[nan_pat] = _NAN

    out *= pos
    return out.astype(np.float32)


def decode_bf16_array(u16: np.ndarray) -> np.ndarray:
    """Decode BF16 (as uint16, native endianness) to float32.

    bf16 is the upper 16 bits of fp32, so this is a trivial shift/reinterpret.
    """
    u16 = np.asarray(u16, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dequant_weight_blocked(
    fp8: np.ndarray,
    scale_inv: np.ndarray,
    block_h: int = 128,
    block_w: int = 128,
) -> np.ndarray:
    """Dequantize an (H, W) F8_E4M3 weight with per-(block_h, block_w) inverse
    scales: dequant[i, j] = fp8[i, j] * scale_inv[i//block_h, j//block_w].

    scale_inv should already be decoded (pass BF16 as float32).
    """
    fp8 = np.asarray(fp8, dtype=np.uint8)
    scale_inv = np.asarray(scale_inv, dtype=np.float32)
    h, w = fp8.shape
    expected = (h + block_h - 1) // block_h, (w + block_w - 1) // block_w
    if scale_inv.shape != expected:
        raise ValueError(
            f"scale_inv shape {scale_inv.shape} != expected {expected} "
            f"for block ({block_h}, {block_w}) and weight {fp8.shape}"
        )
    f = decode_e4m3fn_array(fp8)
    # expand block-scale index to a per-element scale matrix.
    # block index i//block_h -> repeat each *block* index block_h times, then
    # truncate to the true row count (handles non-divisible shapes).
    nrow_b = f.shape[0] // block_h + (1 if f.shape[0] % block_h else 0)
    ncol_b = f.shape[1] // block_w + (1 if f.shape[1] % block_w else 0)
    rows = np.repeat(np.arange(nrow_b, dtype=np.int64), block_h)[: h]
    cols = np.repeat(np.arange(ncol_b, dtype=np.int64), block_w)[: w]
    return (f * scale_inv[rows][:, cols]).astype(np.float32)


def dequant_weight_blocked_loop(
    fp8: np.ndarray,
    scale_inv: np.ndarray,
    block_h: int = 128,
    block_w: int = 128,
) -> np.ndarray:
    """Reference implementation of dequant_weight_blocked (straightforward
    double loop) used only by tests as an independent check."""
    fp8 = np.asarray(fp8, dtype=np.uint8)
    h, w = fp8.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = decode_e4m3fn(int(fp8[i, j])) * float(scale_inv[i // block_h, j // block_w])
    return out
