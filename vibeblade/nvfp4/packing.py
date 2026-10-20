"""NVFP4 packing math — E2M1 values, FP8-E4M3 scales, 16-element block dequant.

NVFP4 (per NVIDIA modelopt / TensorRT-LLM):
  - Each element is a 4-bit E2M1 float: [s][e2 e1][m] → ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
  - Elements are grouped in blocks of 16; each block carries one FP8-E4M3
    scale (and optionally a per-tensor FP32 scale)
  - value = fp4_val * block_scale_e4m3 * tensor_scale_fp32

This module is pure numpy at the boundary (no torch dependency) so it can be
unit-tested anywhere and used by the C++ engine later.
"""

from __future__ import annotations

import numpy as np

# E2M1 decode table indexed by the 4-bit code (sign<<3 | exp<<1 | man)
# code = s e1 e0 m  →  value = (-1)^s * 2^(E-1) * (1 + m/2),  E = exp bits
# E=0 (subnormal):  value = (-1)^s * m/4  → {0, 0.25, 0.5, 0.75}
# E=1..3:           value = (-1)^s * {0.5,1,1.5} * 2^(E-1)
E2M1_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)  # exponent bits 0b00..0b11 with mantissa m=0..1 → canonical LUT


def _e2m1_lut() -> np.ndarray:
    """Full 16-entry LUT (8 magnitudes x 2 signs), index = 4-bit code."""
    lut = np.zeros(16, dtype=np.float32)
    for code in range(16):
        sign = -1.0 if code & 0x8 else 1.0
        mag_bits = code & 0x7
        # two exponent bits (mask 0b110), one mantissa bit (0b001)
        exp = (mag_bits >> 1) & 0x3
        man = mag_bits & 0x1
        if exp == 0:
            val = man * 0.5  # subnormal: 0 or 0.5 (bias=1, 2^(1-bias)=1, × m/2)
        else:
            val = (2 ** (exp - 1)) * (1 + 0.5 * man)
        lut[code] = sign * val
    return lut


E2M1_LUT = _e2m1_lut()
# The canonical E2M1 magnitude set (for reference / round-trip checks):
# {0, 0.5, 1, 1.5, 2, 3, 4, 6} — matches E2M1_VALUES above.


def fp8_e4m3_to_float32(codes: np.ndarray) -> np.ndarray:
    """Decode FP8-E4M3 bytes to float32 (no infinities in E4M3; 0xFF = nan)."""
    codes = np.asarray(codes, dtype=np.uint8)
    sign = np.where(codes & 0x80, -1.0, 1.0).astype(np.float32)
    exp_bits = ((codes >> 3) & 0xF).astype(np.int32)
    man_bits = (codes & 0x7).astype(np.float32)

    # E4M3 bias = 7
    normal = exp_bits > 0
    # normal: 2^(e-7) * (1 + m/8);  subnormal: 2^(-6) * (m/8)
    val_normal = np.power(2.0, exp_bits - 7) * (1.0 + man_bits / 8.0)
    val_sub = (2.0 ** -6) * (man_bits / 8.0)
    val = np.where(normal, val_normal, val_sub).astype(np.float32)
    val = np.where(codes == 0x7F, np.float32(np.nan), val)  # E4M3 nan encoding
    return sign * val


def decode_fp4_packed(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8 array (two 4-bit codes per byte) → float32 via E2M1 LUT.

    Low nibble = element 2i, high nibble = element 2i+1.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    lo = E2M1_LUT[packed & 0x0F]
    hi = E2M1_LUT[packed >> 4]
    out = np.empty(packed.shape + (2,), dtype=np.float32)
    out[..., 0] = lo
    out[..., 1] = hi
    return out.reshape(-1)


def encode_fp4(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round-trippable FP4 encode: values → (packed nibbles, codes).

    Quantizes to the nearest E2M1 magnitude (per element) and returns packed
    bytes plus the raw 4-bit codes (sign<<3 | magnitude index).  Only the
    magnitude set {0, .5, 1, 1.5, 2, 3, 4, 6} is representable — used for
    round-trip tests and to build toy checkpoints.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    sign = (x < 0).astype(np.uint8)
    mag = np.abs(x)

    mags = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    idx = np.argmin(np.abs(mag[:, None] - mags[None, :]), axis=1).astype(np.uint8)
    codes = (sign << 3) | idx

    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    if codes.size % 2:
        packed = np.concatenate([packed, np.array([codes[-1]], dtype=np.uint8)])
    return packed, codes


BLOCK = 16  # NVFP4 block size


def dequant_nvfp4(packed: np.ndarray, scales: np.ndarray,
                  tensor_scale: float = 1.0) -> np.ndarray:
    """Full dequant: packed FP4 + per-block FP8 scales (+optional FP32 tensor
    scale) → float32 array.

    packed: (n_packed_bytes,) uint8 → n = 2 * n_packed_bytes elements
    scales: (n_blocks,) uint8 FP8-E4M3 codes
    """
    vals = decode_fp4_packed(packed)
    n = vals.shape[0]
    n_blocks = (n + BLOCK - 1) // BLOCK
    if scales.shape[0] != n_blocks:
        raise ValueError(f"scales len {scales.shape[0]} != blocks {n_blocks} (n={n})")
    block_scales = fp8_e4m3_to_float32(scales)
    expanded = np.repeat(block_scales, BLOCK)[:n]
    return (vals * expanded * np.float32(tensor_scale)).astype(np.float32)


def pack_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize a float matrix to NVFP4 (for toy checkpoints / export).

    Per-block-of-16: scale = amax/6 encoded to FP8-E4M3 (round-trip through
    the same LUT decode used at inference), elements → nearest E2M1 code.

    Returns (packed, scales, tensor_scale). tensor_scale is fixed at 1.0 —
    real modelopt checkpoints carry it in quantization config; we recompute
    from data here.
    """
    shape = np.asarray(w).shape
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    n = w.shape[0]
    n_blocks = (n + BLOCK - 1) // BLOCK
    padded = np.pad(w, (0, n_blocks * BLOCK - n))
    blocks = padded.reshape(n_blocks, BLOCK)

    amax = np.max(np.abs(blocks), axis=1)  # (n_blocks,)
    # scale so amax maps to the max E2M1 magnitude (6.0)
    scale_f = np.where(amax > 0, amax / 6.0, 1.0).astype(np.float32)

    # encode scale to FP8-E4M3 (round-trip: quantize-exponent-mantissa)
    # E4M3: value = 2^(e-7)*(1+m/8); choose e, m greedily
    def to_fp8(vals: np.ndarray) -> np.ndarray:
        vals = np.maximum(vals, 0).astype(np.float32)
        codes = np.zeros(vals.shape, dtype=np.uint8)
        exps = np.clip(np.floor(np.log2(np.maximum(vals, 1e-30)) + 7).astype(np.int32), 1, 15)
        man = np.clip(np.round((vals / np.power(2.0, exps - 7).astype(np.float32) - 1.0) * 8), 0, 7)
        codes = ((exps << 3) | man.astype(np.uint8)).astype(np.uint8)
        # handle saturation: if man rounded to 8 the greedy code is fine (rolls exp)
        codes = np.where(vals == 0, 0, codes)
        return codes

    scale_codes = to_fp8(scale_f)
    scale_rt = fp8_e4m3_to_float32(scale_codes)

    scaled = blocks / np.where(scale_rt != 0, scale_rt, 1.0)[:, None]
    _, codes = encode_fp4(scaled.reshape(-1))
    flat = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    # preserve matrix layout: (out, in) → packed (out, in/2)
    if len(shape) == 2:
        packed = flat.reshape(shape[0], -1)
    else:
        packed = flat
    return packed, scale_codes, 1.0
