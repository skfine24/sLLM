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


_F8_E5M2_BIAS = 15
_F8_E5M2_MIN_SUBNORMAL = 2.0 ** -16


def decode_e5m2_array(u8: np.ndarray) -> np.ndarray:
    """Decode F8_E5M2 bytes (uint8) to float32.

    E5M2: sign(1) exp(5, bias 15) mantissa(2). Unlike E4M3FN it keeps IEEE
    inf/NaN: exp == 31 -> +/-inf (mant 0) or NaN (mant != 0). Subnormals are
    mant * 2^-16. The scale companion for E5M2 weights (when present) is
    applied by the caller, exactly like the E4M3 path.
    """
    u8 = np.asarray(u8, dtype=np.uint8).astype(np.uint16)
    sign = (u8 >> 7) & 1
    exp = (u8 >> 2) & 0x1F
    mant = u8 & 0x3
    pos = np.where(sign == 0, 1.0, -1.0)
    out = np.zeros(u8.shape, dtype=np.float64)
    sub = exp == 0
    out[sub] = mant[sub].astype(np.float64) * _F8_E5M2_MIN_SUBNORMAL
    inf = (exp == 31) & (mant == 0)
    nan = (exp == 31) & (mant != 0)
    normal = ~sub & ~inf & ~nan
    out[normal] = (1.0 + mant[normal] / 4.0) * (
        2.0 ** (exp[normal].astype(np.float64) - _F8_E5M2_BIAS))
    out[nan] = np.nan
    out[inf] = np.inf
    out *= pos
    return out.astype(np.float32)


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
    scale_inv = np.asarray(scale_inv)
    # provenance guard: a forgotten BF16 decode would hand us uint16 BYTES;
    # astype(float32) would then convert the bit patterns NUMERICALLY
    # (1,2,3 -> 1.0,2.0,3.0) instead of bit-shifting — silently wrong.
    if scale_inv.dtype == np.uint16:
        raise ValueError(
            "scale_inv is uint16: pass the BF16 scale through "
            "decode_bf16_array() (float32) first, do not value-cast it")
    if scale_inv.dtype != np.float32:
        scale_inv = scale_inv.astype(np.float32)
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


# OCP E2M1FN value table (the exact table used by DeepSeek convert.py / kernel).
FP4_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def decode_ue8m0(u8) -> np.ndarray:
    """Decode OCP F8_E8M0 (MICROSOFT_UFloat8_343_... = exponent-only) to float32.

    value = 2^(exponent - 127); 0x00 -> 0.0; 0xFF -> NaN.
    This is the scale format DeepSeek-V4 stores beside E4M3 weights (u8m0).
    """
    u8 = np.asarray(u8, dtype=np.uint8)
    # float64 exponent for headroom (2^128 is inf in float32, the E8M0 max)
    out = (2.0 ** (u8.astype(np.float64) - 127.0)).astype(np.float32)
    out[u8 == 0x00] = 0.0
    out[u8 == 0xFF] = np.nan
    return out


def dequant_fp8_mxfp_weight(fp8, scale_u8, block: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Dequantize an E4M3 weight paired with an E8M0 (ue8m0) block scale.

    dequant[i, j] = e4m3(fp8[i, j]) * 2^(scale_u8[i//bh, j//bw] - 127)
    """
    fp8 = np.asarray(fp8, dtype=np.uint8)
    scale = decode_ue8m0(scale_u8)
    bh, bw = block
    h, w = fp8.shape
    if scale.shape != ((h + bh - 1) // bh, (w + bw - 1) // bw):
        raise ValueError(
            f"e8m0 scale shape {scale.shape} != expected "
            f"{(h + bh - 1) // bh, (w + bw - 1) // bw} for weight {fp8.shape}")
    rows = np.repeat(np.arange(scale.shape[0]), bh)[:h]
    cols = np.repeat(np.arange(scale.shape[1]), bw)[:w]
    return (decode_e4m3fn_array(fp8) * scale[rows][:, cols]).astype(np.float32)


def dequant_fp4_packed_weight(packed, scale_u8, block_w: int = 32) -> np.ndarray:
    """Dequantize a packed E2M1 (fp4) weight to float32.

    `packed` is (N, K//2) int8/uint8: each byte holds two fp4 nibbles
    (low, high) in the OCP E2M1FN layout; `scale_u8` is (N, K//32) E8M0
    per-row-per-32-column scales. Mirror of fp4_gemm in the reference kernel.
    """
    packed = np.asarray(packed)
    if packed.ndim != 2:
        raise ValueError("packed fp4 weights must be 2-D (N, K/2)")
    p = packed.astype(np.uint8)
    n, khalf = p.shape
    low = FP4_TABLE[p & 0x0F]      # (N, K/2)
    high = FP4_TABLE[(p >> 4) & 0x0F]  # (N, K/2)
    w = np.empty((n, khalf * 2), dtype=np.float32)
    w[:, 0::2] = low
    w[:, 1::2] = high
    scale = decode_ue8m0(scale_u8)
    if scale.shape != (n, (w.shape[1] + block_w - 1) // block_w):
        raise ValueError(
            f"fp4 scale shape {scale.shape} != expected "
            f"{(n, (w.shape[1] + block_w - 1) // block_w)} for weight {w.shape}")
    cols = np.repeat(np.arange(scale.shape[1]), block_w)[:w.shape[1]]
    return (w * scale[:, cols]).astype(np.float32)


def dequant_weight_auto(weight, scale, block: tuple[int, int] = (128, 128),
                        fp4_block_w: int = 32) -> np.ndarray:
    """Dispatch to the right block-dequant for a (weight, scale) pair.

    Choice is inferred from geometry/semantics, not config:
      - 2-D E8M0 scale whose rows == weight rows  -> FP4 (packed E2M1, per-row
        32-col blocks)  [unpacked width = 2x packed width]
      - 2-D E8M0 scale whose rows == ceil(rows/128) -> FP8 (E4M3, per 128x128)
      - anything else (F32/BF16 scale) -> legacy inverse-scale block dequant
    """
    s = np.asarray(scale)
    w = np.asarray(weight)
    rows = w.shape[0]
    if s.ndim == 2 and s.dtype == np.uint8:
        k = w.shape[1] * 2 if w.dtype in (np.int8, np.uint8) else w.shape[1]
        if s.shape[0] == rows and s.shape[1] == (k + fp4_block_w - 1) // fp4_block_w:
            return dequant_fp4_packed_weight(w, s, fp4_block_w)
        if s.shape[0] == (rows + block[0] - 1) // block[0]:
            return dequant_fp8_mxfp_weight(w, s, block)
        raise ValueError(
            f"cannot infer quant layout for weight {w.shape} + e8m0 scale {s.shape}")
    if s.ndim == 2 and s.dtype == np.float32:
        if s.shape == ((rows + block[0] - 1) // block[0],
                       (w.shape[1] + block[1] - 1) // block[1]):
            return dequant_weight_blocked(w, s, block[0], block[1])
    return dequant_weight_blocked(w, s, block[0], block[1])


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
