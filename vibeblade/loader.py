"""VibeBlade GGUF Loader — mmap-backed weight loading with on-the-fly dequantization.

Supports GGUF v3 format with the two most common quantization types:
  - Q8_0:  8-bit block quantization (1 scale per 32 weights) — good quality
  - Q4_0:  4-bit block quantization (1 scale per 32 weights) — 2× smaller

Memory strategy:
  - Small models (<2 GB): entire file read into RAM (fast)
  - Large models: memory-mapped via mmap; OS pages weights in/out as needed
    This is how llama.cpp runs 35 GB (70B Q4) on 16 GB RAM — the OS only
    keeps hot pages resident, and activation sparsity means ~10% of weights
    are touched per token.

Size math (LLaMA 70B):
  - FP16:    70B × 2 bytes   = 140 GB
  - Q8_0:    70B × 1.0625 B  =  74.4 GB  (1 byte + 2 byte scale per 32)
  - Q4_0:    70B × 0.5625 B  = 39.4 GB  (0.5 byte + 2 byte scale per 32)
  - Q2_K:    ~0.315 B/param  =  22 GB    (k-quants with super-blocks)
  - 16 GB RAM can hold: Q2_K fully, Q4_0 via mmap + sparse skipping
"""

from __future__ import annotations

import mmap
import os
import struct
from typing import Any

import numpy as np

# ── GPU support (optional CuPy) ──
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

# ── GGUF constants ──

GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian uint32

# GGUF tensor type IDs (the ones we handle)
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q4_0 = 2
GGUF_TYPE_Q4_1 = 3
GGUF_TYPE_Q5_0 = 6
GGUF_TYPE_Q5_1 = 7
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_Q8_1 = 9
GGUF_TYPE_Q2_K = 10
GGUF_TYPE_Q3_K = 11
GGUF_TYPE_Q4_K = 12
GGUF_TYPE_Q5_K = 13
GGUF_TYPE_Q6_K = 14
GGUF_TYPE_Q8_K = 15

# GGUF metadata value type IDs
_GGUF_VAL_UINT8 = 0
_GGUF_VAL_INT8 = 1
_GGUF_VAL_UINT16 = 2
_GGUF_VAL_INT16 = 3
_GGUF_VAL_UINT32 = 4
_GGUF_VAL_INT32 = 5
_GGUF_VAL_FLOAT32 = 6
_GGUF_VAL_BOOL = 7
_GGUF_VAL_STRING = 8
_GGUF_VAL_ARRAY = 9
_GGUF_VAL_UINT64 = 10
_GGUF_VAL_INT64 = 11
_GGUF_VAL_FLOAT64 = 12

_STRUCT_FMT: dict[int, str] = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "?", 10: "<Q", 11: "<q", 12: "<d",
}

# Block sizes for quantization types
_QBLOCK_SIZE: dict[int, int] = {
    GGUF_TYPE_Q4_0: 32,
    GGUF_TYPE_Q4_1: 32,
    GGUF_TYPE_Q5_0: 32,
    GGUF_TYPE_Q5_1: 32,
    GGUF_TYPE_Q8_0: 32,
    GGUF_TYPE_Q8_1: 32,
    GGUF_TYPE_Q2_K: 256,
    GGUF_TYPE_Q3_K: 256,
    GGUF_TYPE_Q4_K: 256,
    GGUF_TYPE_Q5_K: 256,
    GGUF_TYPE_Q6_K: 256,
    GGUF_TYPE_Q8_K: 256,
}

# Bytes per block for quantization types
_QBLOCK_BYTES: dict[int, int] = {
    GGUF_TYPE_Q4_0: 18,   # 2 (scale) + 16 (32 nibbles packed)
    GGUF_TYPE_Q4_1: 20,   # 2 (scale) + 2 (min) + 16
    GGUF_TYPE_Q5_0: 22,   # 2 (scale) + 4 (qh) + 16
    GGUF_TYPE_Q5_1: 24,   # 2 (scale) + 2 (min) + 4 (qh) + 16
    GGUF_TYPE_Q8_0: 34,   # 2 (scale) + 32
    GGUF_TYPE_Q8_1: 40,   # 2 (scale) + 2 (sum) + 4 (b) + 32
    GGUF_TYPE_Q2_K: 84,   # 2 + 2 + 16 + 64
 GGUF_TYPE_Q3_K: 178, # 2 + 2 + 16 + 32 + 32 + 32 + 64 (d, mins, scales, h, qs_hi, qs_lo)
    GGUF_TYPE_Q4_K: 144,  # 2 + 2 + 12 + 2 + 2 + 12 + 2 + 2 + 12 + 128
    GGUF_TYPE_Q5_K: 176,  # 2 + 2 + 12 + 2 + 2 + 12 + 4 + 12 + 128 + 32
    GGUF_TYPE_Q6_K: 210,  # 2 + 2 + 16 + 32 + 64 + 64 + 32
    GGUF_TYPE_Q8_K: 292,  # 4 + 4 + 4 + 256 + 32
}

# Type ID → (numpy bytes per element, numpy dtype for raw data)
_RAW_DTYPES: dict[int, tuple[int, np.dtype]] = {
    GGUF_TYPE_F32: (4, np.dtype("float32")),
    GGUF_TYPE_F16: (2, np.dtype("float16")),
}


# ── Dequantization kernels ───────────────────────────────────────────────────
# Per-block kernels (for small quant types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
# and batch kernels (for K-quants: Q2_K through Q8_K) that process ALL blocks
# at once for 100-1000x speedup.

def _dequant_q8_0(block: bytes, block_size: int = 32) -> np.ndarray:
    """Dequantize a Q8_0 block: 2-byte f16 scale + 32 int8 values."""
    scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
    vals = np.frombuffer(block[2:2 + block_size], dtype=np.int8).astype(np.float32)
    return vals * scale


def _dequant_q4_0(block: bytes, block_size: int = 32) -> np.ndarray:
    """Dequantize a Q4_0 block: 2-byte f16 scale + 32 nibbles packed into 16 bytes."""
    scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
    packed = np.frombuffer(block[2:18], dtype=np.uint8)  # 16 bytes = 32 nibbles
    # Unpack: low nibble = packed[i] & 0xF, high nibble = (packed[i] >> 4) & 0xF
    low = (packed & 0x0F).astype(np.float32) - 8.0  # signed: subtract 8
    high = ((packed >> 4) & 0x0F).astype(np.float32) - 8.0
    vals = np.empty(block_size, dtype=np.float32)
    vals[0::2] = low
    vals[1::2] = high
    return vals * scale


def _dequant_q4_1(block: bytes, block_size: int = 32) -> np.ndarray:
    """Dequantize a Q4_1 block: 2-byte scale + 2-byte min + 16 bytes nibbles."""
    scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
    mn = np.frombuffer(block[2:4], dtype=np.float16).astype(np.float32)[0]
    packed = np.frombuffer(block[4:20], dtype=np.uint8)
    low = (packed & 0x0F).astype(np.float32)
    high = ((packed >> 4) & 0x0F).astype(np.float32)
    vals = np.empty(block_size, dtype=np.float32)
    vals[0::2] = low
    vals[1::2] = high
    return mn + vals * scale


def _dequant_q5_0(block: bytes, block_size: int = 32) -> np.ndarray:
    """Dequantize a Q5_0 block: 2-byte f16 scale + 4-byte qh + 16 bytes nibbles."""
    scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
    qh = np.frombuffer(block[2:6], dtype=np.uint8)
    packed = np.frombuffer(block[6:22], dtype=np.uint8)
    low = (packed & 0x0F).astype(np.float32)
    high = ((packed >> 4) & 0x0F).astype(np.float32)
    # qh adds a high bit per nibble: qh[i//2] bit (i%2)*4 gives extra bit
    for i in range(32):
        byte_idx = i // 8
        bit_idx = (i % 8)
        extra = 1 if (qh[byte_idx] >> bit_idx) & 1 else 0
        base = low[i] if i % 2 == 0 else high[i // 2]
        base - 16.0 + extra * 16.0
        # rebuild properly below
    # Cleaner approach
    qh_bits = np.unpackbits(qh)  # 32 bits
    vals = np.empty(block_size, dtype=np.float32)
    vals[0::2] = low
    vals[1::2] = high
    # top bits from qh
    top = qh_bits[:16].astype(np.float32) * 16.0
    vals[0::2] += top
    vals[1::2] += qh_bits[16:32].astype(np.float32) * 16.0
    return (vals - 16.0) * scale


def _dequant_q5_1(block: bytes, block_size: int = 32) -> np.ndarray:
    """Dequantize a Q5_1 block: 2-byte scale + 2-byte min + 4-byte qh + 16 bytes."""
    scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
    mn = np.frombuffer(block[2:4], dtype=np.float16).astype(np.float32)[0]
    qh = np.frombuffer(block[4:8], dtype=np.uint8)
    packed = np.frombuffer(block[8:24], dtype=np.uint8)
    low = (packed & 0x0F).astype(np.float32)
    high = ((packed >> 4) & 0x0F).astype(np.float32)
    vals = np.empty(block_size, dtype=np.float32)
    vals[0::2] = low
    vals[1::2] = high
    qh_bits = np.unpackbits(qh)
    vals[0::2] += qh_bits[:16].astype(np.float32) * 16.0
    vals[1::2] += qh_bits[16:32].astype(np.float32) * 16.0
    return mn + vals * scale


# Placeholder — actual wrapper after _BATCH_DEQUANT dict


def _unpack_6bit_batch(packed: np.ndarray, n_values: int) -> np.ndarray:
    """Unpack n_values 6-bit integers from (n_blocks, nbytes) packed array.

    Fully vectorized — no Python loops over blocks.
    """
    n_blocks = packed.shape[0]
    bits = np.unpackbits(packed.reshape(-1)).reshape(n_blocks, -1)
    needed = n_values * 6
    if bits.shape[1] < needed:
        bits = np.pad(bits, ((0, 0), (0, needed - bits.shape[1])))
    values = np.zeros((n_blocks, n_values), dtype=np.float32)
    for j in range(6):
        values += bits[:, j::6][:, :n_values].astype(np.float32) * (1 << j)
    return values


def _unpack_6bit(packed: np.ndarray, n_values: int) -> np.ndarray:
    """Unpack n_values 6-bit integers from a single packed byte array."""
    return _unpack_6bit_batch(packed.reshape(1, -1), n_values)[0]


# ── Batch K-quant dequantization (processes ALL blocks at once) ──────────────

def _batch_dequant_q2_k(raw: bytes, n_blocks: int) -> np.ndarray:
    """Vectorized Q2_K batch dequant (84 bytes/block, 256 elements/block)."""
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    blocks = raw_arr[:n_blocks * 84].reshape(n_blocks, 84)

    d = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    dmin = blocks[:, 2].astype(np.float32)

    # 16 × 4-bit scales from bytes 3-10 (8 bytes → 16 nibbles)
    sc_packed = blocks[:, 3:11]  # (n_blocks, 8)
    lo = (sc_packed & 0x0F).astype(np.float32)
    hi = ((sc_packed >> 4) & 0x0F).astype(np.float32)
    scales = np.empty((n_blocks, 16), dtype=np.float32)
    scales[:, 0::2] = lo
    scales[:, 1::2] = hi

    # 32 groups of 2-bit quants from bytes 12-43
    qs_packed = blocks[:, 12:44]  # (n_blocks, 32)
    qs_lo = np.empty((n_blocks, 32), dtype=np.float32)
    qs_lo[:, 0::4] = (qs_packed & 0x03).astype(np.float32)
    qs_lo[:, 1::4] = ((qs_packed >> 2) & 0x03).astype(np.float32)
    qs_lo[:, 2::4] = ((qs_packed >> 4) & 0x03).astype(np.float32)
    qs_lo[:, 3::4] = ((qs_packed >> 6) & 0x03).astype(np.float32)

    # 1-bit signs from bytes 76-83
    signs = np.unpackbits(blocks[:, 76:84].reshape(-1)).reshape(
        n_blocks, 64).astype(np.float32)[:, :64]

    # 32 groups × 8 = 256 values: out = (qs_lo + signs) * 4
    qs_expanded = np.repeat(qs_lo, 8, axis=1)  # (n_blocks, 256)
    out = (qs_expanded + signs[:, :256]) * 4.0

    # Apply scales (each of 16 scales repeated 16 times)
    sc_expanded = np.repeat(scales, 16, axis=1)
    return (out * sc_expanded * d[:, np.newaxis]) + dmin[:, np.newaxis]


def _batch_dequant_q3_k(raw: bytes, n_blocks: int) -> np.ndarray:
    """Vectorized Q3_K batch dequant (178 bytes/block, 256 elements/block)."""
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    blocks = raw_arr[:n_blocks * 178].reshape(n_blocks, 178)

    d = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    d = np.where(np.isfinite(d), d, 0.0)

    # scales: 16 bytes → 16 6-bit values (vectorized across all blocks)
    scales = _unpack_6bit_batch(blocks[:, 34:50], 16)  # (n_blocks, 16)

    # h bits: 32 bytes → 256 bits (bit 7→pos 0, bit 0→pos 7, etc.)
    h = blocks[:, 50:82]  # (n_blocks, 32)
    h_shifts = np.arange(7, -1, -1, dtype=np.uint32)  # [7,6,5,...,0]
    h_bits = ((h.astype(np.uint32)[:, :, np.newaxis] >> h_shifts) & 1
              ).astype(np.float32).transpose(0, 2, 1).reshape(n_blocks, 256)

    # qs_hi: 64 bytes → upper 2 bits (4 values per byte)
    hi_bytes = blocks[:, 82:146].astype(np.uint32)  # (n_blocks, 64)
    hi_shifts = np.array([6, 4, 2, 0], dtype=np.uint32)
    qs_hi = ((hi_bytes[:, :, np.newaxis] >> hi_shifts) & 3
             ).astype(np.float32).reshape(n_blocks, 256)

    qs = qs_hi * 2.0 + h_bits

    # Each of 16 scales repeats 16 times
    sc_idx = np.repeat(np.arange(16), 16)
    sc_vals = scales[:, sc_idx]  # (n_blocks, 256)

    return (qs - 4.0) * sc_vals * d[:, np.newaxis]


def _batch_dequant_q4_k(raw: bytes, n_blocks: int) -> np.ndarray:
    """Vectorized Q4_K batch dequant (144 bytes/block, 256 elements/block).

    This is the hot path for Q4_K_M models (most common quant type).
    Processes ALL blocks simultaneously — ~100x faster than block-by-block.
    """
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    blocks = raw_arr[:n_blocks * 144].reshape(n_blocks, 144)

    # d (2 bytes) + dmin (1 byte) + padding (1 byte)
    d = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    dmin = blocks[:, 2].astype(np.float32)

    # scales: 8 bytes → 8 6-bit values (vectorized across all blocks)
    scales = _unpack_6bit_batch(blocks[:, 4:12], 8)  # (n_blocks, 8)

    # quants: 128 bytes → 256 nibbles (interleaved)
    qs = blocks[:, 16:144]  # (n_blocks, 128)
    qs_exp = np.empty((n_blocks, 256), dtype=np.float32)
    qs_exp[:, 0::2] = (qs & 0x0F).astype(np.float32)
    qs_exp[:, 1::2] = ((qs >> 4) & 0x0F).astype(np.float32)

    # Scale index: repeat each of 8 scales 32 times
    sc_idx = np.repeat(np.arange(8), 32)
    sc_vals = scales[:, sc_idx]  # (n_blocks, 256)

    return (qs_exp - 8.0) * sc_vals * d[:, np.newaxis] + dmin[:, np.newaxis]


def _batch_dequant_q5_k(raw: bytes, n_blocks: int) -> np.ndarray:
    """Vectorized Q5_K batch dequant (176 bytes/block, 256 elements/block)."""
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    blocks = raw_arr[:n_blocks * 176].reshape(n_blocks, 176)

    d = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    dmin = blocks[:, 2].astype(np.float32)

    # scales: 8 bytes → 8 6-bit values (vectorized)
    scales = _unpack_6bit_batch(blocks[:, 4:12], 8)  # (n_blocks, 8)

    # qs: 128 bytes (64 low + 64 high nibbles)
    qs = blocks[:, 16:80]
    qs2 = blocks[:, 80:144]
    qs_all = np.empty((n_blocks, 256), dtype=np.float32)
    qs_all[:, 0::2] = np.concatenate([(qs & 0x0F), (qs2 & 0x0F)], axis=1).astype(np.float32)
    qs_all[:, 1::2] = np.concatenate([((qs >> 4) & 0x0F), ((qs2 >> 4) & 0x0F)], axis=1).astype(np.float32)

    # qh: 32 bytes → 1 high bit per value (vectorized)
    qh = blocks[:, 144:176]  # (n_blocks, 32)
    qh_shifts = np.arange(7, -1, -1, dtype=np.uint32)
    qh_bits = ((qh.astype(np.uint32)[:, :, np.newaxis] >> qh_shifts) & 1
               ).astype(np.float32).transpose(0, 2, 1).reshape(n_blocks, 256)

    val = qs_all + qh_bits * 16.0

    sc_idx = np.repeat(np.arange(8), 32)
    sc_vals = scales[:, sc_idx]

    return (val - 16.0) * sc_vals * d[:, np.newaxis] + dmin[:, np.newaxis]


def _batch_dequant_q6_k(raw: bytes, n_blocks: int) -> np.ndarray:
    """Vectorized Q6_K batch dequant (210 bytes/block, 256 elements/block)."""
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    blocks = raw_arr[:n_blocks * 210].reshape(n_blocks, 210)

    d = np.frombuffer(blocks[:, 208:210].tobytes(), dtype=np.float16).astype(np.float32)
    d = np.where(np.isfinite(d), d, 0.0)

    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    sc_raw = blocks[:, 192:208].astype(np.int8)

    # ql: 2 nibbles per byte → 256 values
    ql_low = np.empty((n_blocks, 256), dtype=np.float32)
    ql_low[:, 0::2] = (ql & 0x0F).astype(np.float32)
    ql_low[:, 1::2] = ((ql >> 4) & 0x0F).astype(np.float32)

    # qh: 2-bit high quants (vectorized — no per-block loop)
    shifts = np.array([6, 4, 2, 0], dtype=np.uint32)
    qh_u32 = qh.astype(np.uint32)  # (n_blocks, 64)
    qh_high = ((qh_u32[:, :, np.newaxis] >> shifts) & 3
               ).astype(np.float32).reshape(n_blocks, 256)

    q = ql_low + qh_high * 16.0
    sc = sc_raw.astype(np.float32)
    sc_expanded = np.repeat(sc, 16, axis=1)

    return (q - 32.0) * sc_expanded * d[:, np.newaxis]


def _batch_dequant_q8_k(raw: bytes, n_blocks: int) -> np.ndarray:
    """Vectorized Q8_K batch dequant (292 bytes/block, 256 elements/block)."""
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    blocks = raw_arr[:n_blocks * 292].reshape(n_blocks, 292)

    d = np.frombuffer(blocks[:, 0:4].tobytes(), dtype=np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    dmin = np.frombuffer(blocks[:, 4:8].tobytes(), dtype=np.float32)
    d_s = np.frombuffer(blocks[:, 8:12].tobytes(), dtype=np.float32)

    qs = blocks[:, 12:268].astype(np.int8).astype(np.float32)
    bsums = blocks[:, 268:292].astype(np.int8).astype(np.float32) * d_s[:, np.newaxis]

    sc_idx = np.repeat(np.arange(16), 16)
    bsums_expanded = bsums[:, sc_idx]

    return qs * d[:, np.newaxis] + dmin[:, np.newaxis] + bsums_expanded


# Batch dequant dispatch
_BATCH_DEQUANT = {
    GGUF_TYPE_Q2_K: _batch_dequant_q2_k,
    GGUF_TYPE_Q3_K: _batch_dequant_q3_k,
    GGUF_TYPE_Q4_K: _batch_dequant_q4_k,
    GGUF_TYPE_Q5_K: _batch_dequant_q5_k,
    GGUF_TYPE_Q6_K: _batch_dequant_q6_k,
    GGUF_TYPE_Q8_K: _batch_dequant_q8_k,
}

# Single-block wrappers (for tests and compatibility)
def _dequant_q2_k(block: bytes) -> np.ndarray:
    """Dequantize a single Q2_K block (256 elements)."""
    return _batch_dequant_q2_k(block, 1)[0]

def _dequant_q3_k(block: bytes) -> np.ndarray:
    """Dequantize a single Q3_K block (256 elements)."""
    return _batch_dequant_q3_k(block, 1)[0]

def _dequant_q4_k(block: bytes) -> np.ndarray:
    """Dequantize a single Q4_K block (256 elements)."""
    return _batch_dequant_q4_k(block, 1)[0]

def _dequant_q5_k(block: bytes) -> np.ndarray:
    """Dequantize a single Q5_K block (256 elements)."""
    return _batch_dequant_q5_k(block, 1)[0]

def _dequant_q6_k(block: bytes) -> np.ndarray:
    """Dequantize a single Q6_K block (256 elements)."""
    return _batch_dequant_q6_k(block, 1)[0]

def _dequant_q8_k(block: bytes) -> np.ndarray:
    """Dequantize a single Q8_K block (256 elements)."""
    return _batch_dequant_q8_k(block, 1)[0]


def _dequant_q8_1(block: bytes, block_size: int = 32) -> np.ndarray:
    """Dequantize a Q8_1 block: f16 scale + f16 s + f16 b + 32 int8 values."""
    d = np.frombuffer(block[0:2], dtype=np.float16).astype(np.float32)[0]
    np.frombuffer(block[2:4], dtype=np.float16)  # s (sum, unused in dequant)
    b = np.frombuffer(block[4:8], dtype=np.float16).astype(np.float32)[0]
    vals = np.frombuffer(block[8:8 + block_size], dtype=np.int8).astype(np.float32)
    return vals * d + b

_DEQUANT_FN = {
    GGUF_TYPE_Q8_0: _dequant_q8_0,
    GGUF_TYPE_Q8_1: _dequant_q8_1,
    GGUF_TYPE_Q4_0: _dequant_q4_0,
    GGUF_TYPE_Q4_1: _dequant_q4_1,
    GGUF_TYPE_Q5_0: _dequant_q5_0,
    GGUF_TYPE_Q5_1: _dequant_q5_1,
}

# Supported quantized types → bytes per param (approximate)
_QUANT_BYTES_PER_PARAM: dict[int, float] = {
    GGUF_TYPE_F32: 4.0,
    GGUF_TYPE_F16: 2.0,
    GGUF_TYPE_Q8_0: 1.0625,  # (2 + 32) / 32
    GGUF_TYPE_Q8_1: 1.25,    # (4 + 32 + 4) / 32
    GGUF_TYPE_Q4_0: 0.5625,  # (2 + 16) / 32
    GGUF_TYPE_Q4_1: 0.625,   # (4 + 16) / 32
    GGUF_TYPE_Q5_0: 0.6875,  # (2 + 4 + 16) / 32
    GGUF_TYPE_Q5_1: 0.75,    # (4 + 4 + 16) / 32
    GGUF_TYPE_Q2_K: 0.328125, # 84 / 256
    GGUF_TYPE_Q3_K: 0.4296875, # 110 / 256
    GGUF_TYPE_Q4_K: 0.5625,  # 144 / 256
    GGUF_TYPE_Q5_K: 0.6875,  # 176 / 256
    GGUF_TYPE_Q6_K: 0.8203125, # 210 / 256
    GGUF_TYPE_Q8_K: 1.140625, # 292 / 256
}

_QUANT_NAMES: dict[int, str] = {
    GGUF_TYPE_F32: "F32", GGUF_TYPE_F16: "F16",
    GGUF_TYPE_Q8_0: "Q8_0", GGUF_TYPE_Q8_1: "Q8_1",
    GGUF_TYPE_Q4_0: "Q4_0", GGUF_TYPE_Q4_1: "Q4_1",
    GGUF_TYPE_Q5_0: "Q5_0", GGUF_TYPE_Q5_1: "Q5_1",
    GGUF_TYPE_Q2_K: "Q2_K", GGUF_TYPE_Q3_K: "Q3_K",
    GGUF_TYPE_Q4_K: "Q4_K", GGUF_TYPE_Q5_K: "Q5_K",
    GGUF_TYPE_Q6_K: "Q6_K", GGUF_TYPE_Q8_K: "Q8_K",
}


# ── GGUF Loader ──

class GGUFLoader:
    """Memory-mapped GGUF v3 model loader with on-the-fly dequantization.

    For models < 2 GB, tensors are fully loaded into RAM.
    For larger models, the file is mmap'd and tensors are dequantized
    on demand — the OS handles paging.

    The public interface returns f16 or f32 numpy arrays regardless of
    the on-disk quantization format.
    """

    MMAP_THRESHOLD = 2 * 1024**3  # 2 GB — mmap above this

    def __init__(self, path: str, progress_cb=None) -> None:
        self._path = path
        self._file_size = os.path.getsize(path)
        self._use_mmap = self._file_size > self.MMAP_THRESHOLD
        self._mmap: mmap.mmap | None = None
        self._progress_cb = progress_cb

        self._f = open(path, "rb")
        self._parse_header()
        self._parse_metadata()
        if progress_cb:
            progress_cb("parsing tensor info", 0, self.n_tensors, loading=True)
        self._parse_tensor_infos()
        self._data_start = self._f.tell()

        if self._use_mmap:
            self._f.close()
            fd = os.open(path, os.O_RDONLY)
            self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)

    # ── Header parsing ──

    def _parse_header(self) -> None:
        (magic,) = struct.unpack("<I", self._read(4))
        if magic != GGUF_MAGIC:
            self._f.close()
            raise ValueError(f"Not a GGUF file (magic 0x{magic:08X})")
        (self.version,) = struct.unpack("<I", self._read(4))
        (self.n_tensors,) = struct.unpack("<Q", self._read(8))
        (self.n_kv,) = struct.unpack("<Q", self._read(8))

    def _parse_metadata(self) -> None:
        self.metadata: dict[str, Any] = {}
        for i in range(self.n_kv):
            key = self._read_string()
            (vtype,) = struct.unpack("<I", self._read(4))
            self.metadata[key] = self._read_value(vtype)
            if self._progress_cb and (i + 1) % 100 == 0:
                self._progress_cb("reading metadata", i + 1, self.n_kv, loading=True)

    def _parse_tensor_infos(self) -> None:
        self.tensor_infos: list[dict] = []
        for i in range(self.n_tensors):
            name = self._read_string()
            (n_dims,) = struct.unpack("<I", self._read(4))
            shape: list[int] = []
            for _ in range(n_dims):
                (dim,) = struct.unpack("<Q", self._read(8))
                shape.append(dim)
            shape = tuple(reversed(shape))  # GGUF stores row-major reversed
            (dtype_id,) = struct.unpack("<I", self._read(4))
            (offset,) = struct.unpack("<Q", self._read(8))
            self.tensor_infos.append({
                "name": name,
                "shape": shape,
                "dtype": dtype_id,
                "offset": offset,
            })
            if self._progress_cb and (i + 1) % 50 == 0:
                self._progress_cb("parsing tensor info", i + 1, self.n_tensors, loading=True)

    # ── Low-level readers ──

    def _read(self, n: int) -> bytes:
        data = self._f.read(n)
        if len(data) < n:
            raise ValueError(f"Unexpected EOF reading {n} bytes")
        return data

    def _read_string(self) -> str:
        (length,) = struct.unpack("<Q", self._read(8))
        if length > (1 << 20):
            raise ValueError(f"String too long: {length}")
        return self._read(length).decode("utf-8")

    def _read_value(self, vtype: int) -> Any:
        if vtype == _GGUF_VAL_STRING:
            return self._read_string()
        if vtype == _GGUF_VAL_ARRAY:
            (etype,) = struct.unpack("<I", self._read(4))
            (count,) = struct.unpack("<Q", self._read(8))
            if count > (1 << 18):
                raise ValueError(f"Array too large: {count}")
            return [self._read_value(etype) for _ in range(count)]
        fmt = _STRUCT_FMT.get(vtype)
        if fmt:
            return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]
        raise ValueError(f"Unsupported GGUF value type: {vtype}")

    # ── Tensor access ──

    def tensor_names(self) -> list[str]:
        return [t["name"] for t in self.tensor_infos]

    def tensor_info(self, name: str) -> dict:
        for t in self.tensor_infos:
            if t["name"] == name:
                return t
        raise KeyError(f"Tensor not found: {name}")

    def load_tensor(self, name: str, dtype: np.dtype = np.dtype("float32"),
                    progress_cb=None) -> np.ndarray:
        """
        Load a tensor by name.

        Returns an f32 (or f16 if requested) numpy array.
        For quantized types, this reads the raw blocks and dequantizes.
        """
        info = self.tensor_info(name)
        tid = info["dtype"]
        n_elements = 1
        for d in info["shape"]:
            n_elements *= d
        shape = info["shape"]

        raw = self._read_tensor_bytes(info)

        if tid in (GGUF_TYPE_F32, GGUF_TYPE_F16):
            np_dtype = np.float16 if tid == GGUF_TYPE_F16 else np.float32
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            return arr.astype(dtype) if dtype != np_dtype else arr

        if tid in _DEQUANT_FN or tid in _BATCH_DEQUANT:
            arr = self._dequant_blocks(raw, tid, n_elements,
                                      progress_cb=progress_cb, tensor_name=name)
            return arr.reshape(shape).astype(dtype)

        # Unsupported quant — return raw bytes as best-effort
        raise ValueError(
            f"Unsupported tensor type {tid} ({_QUANT_NAMES.get(tid, '?')}) "
            f"for tensor '{name}'"
        )

    def _read_tensor_bytes(self, info: dict) -> bytes:
        """Read raw tensor bytes from file or mmap."""
        byte_count = self._tensor_byte_count(info)
        offset = self._data_start + info["offset"]

        if self._mmap is not None:
            return self._mmap[offset:offset + byte_count]
        else:
            self._f.seek(offset)
            return self._f.read(byte_count)

    def _tensor_byte_count(self, info: dict) -> int:
        tid = info["dtype"]
        n_elements = 1
        for d in info["shape"]:
            n_elements *= d

        if tid in _RAW_DTYPES:
            return n_elements * _RAW_DTYPES[tid][0]
        if tid in _QBLOCK_SIZE and tid in _QBLOCK_BYTES:
            block_size = _QBLOCK_SIZE[tid]
            block_bytes = _QBLOCK_BYTES[tid]
            n_blocks = n_elements // block_size
            if n_blocks * block_size < n_elements:
                n_blocks += 1  # partial block
            return n_blocks * block_bytes
        raise ValueError(f"Unknown tensor type {tid}")

    def _dequant_blocks(self, raw: bytes, tid: int, n_elements: int,
                        progress_cb=None, tensor_name: str = "") -> np.ndarray:
        """Dequantize raw block data to f32.

        Uses vectorized batch dequant for K-quants (100-1000x faster),
        falls back to block-by-block for small quant types.
        """
        block_size = _QBLOCK_SIZE[tid]
        block_bytes = _QBLOCK_BYTES[tid]
        n_blocks = n_elements // block_size

        # Fast path: vectorized batch dequant for K-quants
        if tid in _BATCH_DEQUANT:
            batch_fn = _BATCH_DEQUANT[tid]
            import sys

            # Chunk large tensors so progress bar updates between chunks
            CHUNK_SIZE = 4096
            if n_blocks <= CHUNK_SIZE:
                # Small tensor — process in one shot
                if progress_cb:
                    progress_cb(tensor_name, n_blocks // 2, n_blocks, loading=True)
                    sys.stdout.flush()
                raw_bytes = bytes(raw[:n_blocks * block_bytes])
                out = batch_fn(raw_bytes, n_blocks).ravel()[:n_elements]
            else:
                # Large tensor — chunk and report progress
                out_parts = []
                for chunk_start in range(0, n_blocks, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, n_blocks)
                    chunk_n = chunk_end - chunk_start
                    start_byte = chunk_start * block_bytes
                    end_byte = chunk_end * block_bytes
                    chunk_raw = bytes(raw[start_byte:end_byte])
                    out_parts.append(batch_fn(chunk_raw, chunk_n))
                    if progress_cb:
                        progress_cb(tensor_name, chunk_end, n_blocks, loading=True)
                        sys.stdout.flush()
                out = np.concatenate(out_parts).ravel()[:n_elements]

            if progress_cb:
                progress_cb(tensor_name, n_blocks, n_blocks, loading=True)
                sys.stdout.flush()
            return out

        # Slow path: per-block dequant for small quant types (Q4_0, Q8_0, etc.)
        fn = _DEQUANT_FN[tid]
        out = np.empty(n_blocks * block_size, dtype=np.float32)
        for i in range(n_blocks):
            start = i * block_bytes
            end = start + block_bytes
            out[i * block_size:(i + 1) * block_size] = fn(
                raw[start:end], block_size
            )
            if progress_cb and (i + 1) % 100 == 0:
                progress_cb(tensor_name, i + 1, n_blocks, loading=True)

        if progress_cb:
            progress_cb(tensor_name, n_blocks, n_blocks, loading=True)
        return out[:n_elements]

    def load_all_tensors(self, dtype: np.dtype = np.dtype("float32"),
                         progress_cb=None) -> dict[str, np.ndarray]:
        """Load all tensors. Returns {name: ndarray}."""
        total = len(self.tensor_infos)
        tensors = {}
        for idx, info in enumerate(self.tensor_infos):
            # Wrap callback to report overall progress (tensor + block level)
            if progress_cb:
                _tensor_idx = idx
                _tensor_total = total
                def _wrapped(name, done, sub_total, loading=False):
                    overall = _tensor_idx + (done / max(sub_total, 1))
                    progress_cb(name, overall, _tensor_total, loading=True)
            else:
                _wrapped = None

            tensors[info["name"]] = self.load_tensor(info["name"], dtype,
                                                     progress_cb=_wrapped)
            if progress_cb:
                import sys
                progress_cb(info["name"], idx + 1, total, loading=True)
                sys.stdout.flush()
        return tensors

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._f and not self._f.closed:
            self._f.close()

    def __enter__(self) -> GGUFLoader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ── Convenience ──

def estimate_model_size_gb(n_params: int, quant_type: int) -> float:
    """Estimate model size in GB for given parameter count and quant type."""
    bpp = _QUANT_BYTES_PER_PARAM.get(quant_type)
    if bpp is None:
        return -1.0
    return n_params * bpp / (1024**3)


# Tensors that are shared (top-level, no blk.N prefix)
_SHARED_TENSORS = frozenset((
    "output_norm.weight",
    "token_embd.weight",
))

def _map_tensor_name(name: str, arch: str) -> str:
    """Map GGUF tensor name to canonical internal name.

    GGUF: {arch}.{component}.{layer}.{part}.weight
    Internal: {component}.{layer}.{part}.weight or blk.{layer}.{component}.{part}.weight

    Shared layers (output_norm, token_embd) have NO layer prefix in internal naming.
    Per-layer components (attn, ffn, ffn_norm) get the blk.N prefix.

    Examples for Qwen3:
    qwen3.token_embd.weight     → token_embd.weight     (shared, no blk.N)
    qwen3.block.31.output_norm.weight → output_norm.weight  (shared, no blk.N)
    qwen3.block.0.attn.q.weight → blk.0.attn.q.weight  (per-layer)
    qwen3.block.0.ffn.gate.weight → blk.0.ffn.gate_proj.weight (per-layer)
    """
    import re

    # Strip architecture prefix
    if arch and name.startswith(arch + "."):
        name = name[len(arch) + 1:]

    # Shared tensors: no blk.N prefix, use bare name
    if name in _SHARED_TENSORS:
        return name

    # Per-layer tensors: block.N → blk.N
    name = re.sub(r"^block\.(\d+)", r"blk.\1", name)

    # ffn.gate → ffn.gate_proj, ffn.up → ffn.up_proj, ffn.down → ffn.down_proj
    name = name.replace(".ffn.gate.", ".ffn.gate_proj.")
    name = name.replace(".ffn.up.", ".ffn.up_proj.")
    name = name.replace(".ffn.down.", ".ffn.down_proj.")

    return name


def load_model(path: str, progress_cb=None, lazy: bool = True,
               max_cached_mb: float = 4096,
               gpu_offload: bool = False) -> dict:
    """Load a GGUF model and return metadata, tensors, and config.

    Args:
        path: Path to GGUF file.
        progress_cb: Optional progress callback(name, done, total, loading=bool).
        lazy: If True, use lazy dequantization (instant load, low memory).
              If False, eagerly load all tensors (slow, high memory).
        max_cached_mb: Max RAM for cached dequantized tensors in lazy mode.
        gpu_offload: If True and CuPy available, move tensors to GPU VRAM.

    Returns:
        dict with metadata, tensors, config.
    """
    loader = GGUFLoader(path, progress_cb=progress_cb)
    metadata = dict(loader.metadata)
    arch = metadata.get("general.architecture", "")
    config = _extract_config(metadata)

    if lazy:
        n_total = len(loader.tensor_infos)
        if progress_cb:
            progress_cb("header", n_total, n_total, loading=True)

        weights = _LazyWeights(loader, arch, max_cached_mb=max_cached_mb,
                               gpu_offload=gpu_offload)
        # Preload shared tensors used every forward pass
        if progress_cb:
            progress_cb("preload", 0, 1, loading=True)
        weights.preload_shared()
        if progress_cb:
            progress_cb("preload", 1, 1, loading=True)
    else:
        tensors_raw = loader.load_all_tensors(progress_cb=progress_cb)
        weights = {_map_tensor_name(k, arch): v for k, v in tensors_raw.items()}
        loader.close()

    return {"metadata": metadata, "tensors": weights, "config": config}


class _LazyWeights:
    """Dict-like that lazily dequantizes tensors from a memory-mapped GGUF file.

    Instead of loading all weights into RAM (140GB for a 35B float32 model),
    tensors are dequantized on-demand from the mmap'd file.  A LRU cache keeps
    recently-used tensors resident while evicting older ones to stay within
    *max_cached_mb*.  Shared tensors (embedding, output norm, output) are
    pinned in cache and never evicted.

    Usage is transparent — code that does ``self.weights["blk.0.attn.q.weight"]``
    works identically whether *weights* is a plain dict or a _LazyWeights.
    """

    def __init__(self, loader: GGUFLoader, arch: str,
                 dtype: np.dtype = np.dtype("float32"),
                 max_cached_mb: float = 4096,
                 gpu_offload: bool = False) -> None:
        self._loader = loader
        self._dtype = dtype
        self._max_cached_mb = max_cached_mb
        self._gpu_offload = gpu_offload and _HAS_CUPY
        self._cache: dict[str, np.ndarray] = {}
        self._pinned: set[str] = set()  # never evict
        self._cached_bytes = 0
        self._access_order: list[str] = []

        # Pre-map canonical names → GGUF tensor info
        self._name_map: dict[str, dict] = {}
        for info in loader.tensor_infos:
            canonical = _map_tensor_name(info["name"], arch)
            self._name_map[canonical] = info

        # Detect architecture features and build virtual name aliases
        # (e.g., Qwen3.6 has fused QKV, attn_gate, post_attention_norm, shexp shared)
        self._aliases: dict[str, str] = {}
        self._has_fused_qkv = any("attn_qkv.weight" in k for k in self._name_map)
        self._build_aliases()

    def _build_aliases(self) -> None:
        """Build virtual name aliases for architecture compatibility.

        Maps canonical names that transformer.py expects to the actual
        tensor names in the GGUF file.  For Qwen3.6-style models:
          - fused attn_qkv → split into attn_q, attn_k, attn_v
          - attn_gate → attn_output
          - post_attention_norm → ffn_norm
          - ffn_gate/up/down_shexp → ffn_gate/up/down (shared expert)
        """
        # Collect all unique suffixes across all blocks
        all_names = list(self._name_map.keys())
        # Find any block prefix (e.g., "blk.0")
        block_prefixes = set()
        for name in all_names:
            parts = name.split(".", 2)
            if len(parts) >= 2 and parts[0].startswith("blk"):
                block_prefixes.add(parts[0] + "." + parts[1])

        if not block_prefixes:
            return

        # Use first block to detect naming patterns
        sample_prefix = sorted(block_prefixes)[0]
        sample_keys = {k[len(sample_prefix) + 1:] for k in all_names
                       if k.startswith(sample_prefix + ".")}

        # attn_output.weight → attn_gate.weight
        if "attn_gate.weight" in sample_keys and "attn_output.weight" not in sample_keys:
            for bp in block_prefixes:
                self._aliases[f"{bp}.attn_output.weight"] = f"{bp}.attn_gate.weight"

        # ffn_norm.weight → post_attention_norm.weight
        if "post_attention_norm.weight" in sample_keys and "ffn_norm.weight" not in sample_keys:
            for bp in block_prefixes:
                self._aliases[f"{bp}.ffn_norm.weight"] = f"{bp}.post_attention_norm.weight"

        # ffn_gate/up/down.weight → ffn_gate/up/down_shexp.weight (shared expert)
        if "ffn_gate_shexp.weight" in sample_keys and "ffn_gate.weight" not in sample_keys:
            for bp in block_prefixes:
                self._aliases[f"{bp}.ffn_gate.weight"] = f"{bp}.ffn_gate_shexp.weight"
                self._aliases[f"{bp}.ffn_up.weight"] = f"{bp}.ffn_up_shexp.weight"
                self._aliases[f"{bp}.ffn_down.weight"] = f"{bp}.ffn_down_shexp.weight"

        # attn_q/k/v.weight → split from attn_qkv.weight
        if "attn_qkv.weight" in sample_keys:
            for bp in block_prefixes:
                self._aliases[f"{bp}.attn_q.weight"] = f"{bp}.attn_qkv.weight"
                self._aliases[f"{bp}.attn_k.weight"] = f"{bp}.attn_qkv.weight"
                self._aliases[f"{bp}.attn_v.weight"] = f"{bp}.attn_qkv.weight"

    # ── dict interface ──

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._cache:
            self._touch(key)
            return self._cache[key]

        # Resolve aliases
        real_key = self._aliases.get(key, key)

        # Fused QKV split: load once, split into Q/K/V, cache all three
        if self._has_fused_qkv and key.endswith(".attn_q.weight"):
            return self._load_split_qkv(key, real_key)
        if self._has_fused_qkv and key.endswith(".attn_k.weight"):
            # Q was loaded first; K and V should be cached already
            k_key = key  # e.g. blk.0.attn_k.weight
            if k_key in self._cache:
                self._touch(k_key)
                return self._cache[k_key]
            # If not cached, trigger Q load (which caches K and V too)
            q_key = key.replace(".attn_k.", ".attn_q.")
            self[q_key]
            return self._cache[k_key]
        if self._has_fused_qkv and key.endswith(".attn_v.weight"):
            v_key = key
            if v_key in self._cache:
                self._touch(v_key)
                return self._cache[v_key]
            q_key = key.replace(".attn_v.", ".attn_q.")
            self[q_key]
            return self._cache[v_key]

        if real_key not in self._name_map:
            parts = key.split(".", 2)
            if len(parts) >= 2:
                prefix = parts[0] + "." + parts[1]
                matches = sorted(n for n in self._name_map if n.startswith(prefix + "."))
                hint = f"\n  Names matching '{prefix}.*': {matches[:20]}"
            else:
                avail = sorted(self._name_map.keys())[:20]
                hint = f"\n  First 20: {avail}"
            raise KeyError(
                f"Tensor '{key}' not found.{hint}"
            )

        info = self._name_map[real_key]
        arr = self._loader.load_tensor(info["name"], self._dtype)

        # Optional GPU offload
        if self._gpu_offload:
            arr = cp.asarray(arr)  # type: ignore[assignment]

        self._cache[key] = arr
        self._cached_bytes += arr.nbytes
        self._access_order.append(key)
        self._evict()

        return arr

    def _load_split_qkv(self, q_key: str, real_key: str) -> np.ndarray:
        """Load fused attn_qkv.weight and split into Q, K, V arrays."""
        if q_key in self._cache:
            self._touch(q_key)
            return self._cache[q_key]

        info = self._name_map[real_key]
        qkv = self._loader.load_tensor(info["name"], self._dtype)

        if self._gpu_offload:
            qkv = cp.asarray(qkv)  # type: ignore[assignment]

        # Infer split sizes from the config metadata
        n_heads = self._loader.metadata.get("qwen3.attention.head_count",
                   self._loader.metadata.get("general.attention.head_count", 32))
        n_kv_heads = self._loader.metadata.get("qwen3.attention.head_count_kv",
                        self._loader.metadata.get("general.attention.head_count_kv", n_heads))
        head_dim = qkv.shape[1] // n_heads  # hidden_dim // num_heads

        q_size = n_heads * head_dim
        k_size = n_kv_heads * head_dim

        q_arr = qkv[:q_size]
        k_arr = qkv[q_size:q_size + k_size]
        v_arr = qkv[q_size + k_size:]

        # Cache all three with their virtual names (keep qkv alive as backing store)
        prefix = q_key.rsplit(".", 1)[0]  # e.g., "blk.0.attn"
        qkv_key = f"{prefix}._qkv_full"
        self._cache[qkv_key] = qkv  # prevent GC of backing array
        self._access_order.append(qkv_key)
        for name, arr in [(f"{prefix}.q.weight", q_arr),
                          (f"{prefix}.k.weight", k_arr),
                          (f"{prefix}.v.weight", v_arr)]:
            self._cache[name] = arr
            self._cached_bytes += arr.nbytes
            self._access_order.append(name)

        self._evict()
        self._touch(q_key)
        return self._cache[q_key]

    def __contains__(self, key: str) -> bool:
        return key in self._name_map or key in self._aliases

    def __len__(self) -> int:
        return len(self._name_map)

    def __iter__(self):
        return iter(self._name_map)

    def __repr__(self) -> str:
        info = self.memory_info()
        gpu = "GPU" if self._gpu_offload else "CPU"
        return (f"_LazyWeights({len(self)} tensors, "
                f"{info['cached_mb']:.0f}/{info['max_mb']:.0f} MB, {gpu})")

    def keys(self):
        return self._name_map.keys()

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def values(self):
        return [self[k] for k in self._name_map]

    def items(self):
        return [(k, self[k]) for k in self._name_map]

    # ── Cache management ──

    def _touch(self, key: str) -> None:
        """Move key to end of LRU order."""
        if key in self._access_order:
            self._access_order.remove(key)
            self._access_order.append(key)

    def _evict(self) -> None:
        """Evict oldest non-pinned tensors to stay within budget."""
        if self._max_cached_mb <= 0:
            return
        max_bytes = self._max_cached_mb * 1024 * 1024
        while self._cached_bytes > max_bytes and len(self._access_order) > 0:
            # Find oldest non-pinned entry
            idx = 0
            while idx < len(self._access_order):
                candidate = self._access_order[idx]
                if candidate not in self._pinned:
                    break
                idx += 1
            else:
                break  # all pinned, can't evict

            evicted_key = self._access_order.pop(idx)
            evicted_arr = self._cache.pop(evicted_key)
            self._cached_bytes -= evicted_arr.nbytes

    def preload_shared(self) -> None:
        """Dequantize and pin shared tensors (used every forward pass)."""
        for name in list(_SHARED_TENSORS):
            if name in self._name_map:
                self[name]  # triggers dequant + cache
                self._pinned.add(name)
        # Also pin output.weight (not in _SHARED_TENSORS but used every pass)
        if "output.weight" in self._name_map:
            self["output.weight"]
            self._pinned.add("output.weight")

    def memory_info(self) -> dict:
        """Return cache memory usage stats."""
        return {
            "cached_mb": round(self._cached_bytes / (1024 * 1024), 1),
            "max_mb": self._max_cached_mb,
            "cached_tensors": len(self._cache),
            "total_tensors": len(self._name_map),
            "pinned_tensors": len(self._pinned),
            "gpu_offload": self._gpu_offload,
        }

    def close(self) -> None:
        """Release cached tensors and close the underlying GGUF loader."""
        self._cache.clear()
        self._pinned.clear()
        self._access_order.clear()
        self._cached_bytes = 0
        if self._loader:
            self._loader.close()
            self._loader = None


def _extract_config(metadata: dict[str, Any]) -> dict[str, Any]:
    arch = metadata.get("general.architecture", "")
    prefix = arch + "." if arch else ""
    config: dict[str, Any] = {}
    for suffix in (
        "context_length", "embedding_length", "block_count",
        "feed_forward_length", "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "attention.value_length", "vocab_size",
        "rope.freq_base", "rope.dimension_count",
        "attention.layer_norm_rms_epsilon",
    ):
        key = prefix + suffix
        if key in metadata:
            config[suffix] = metadata[key]
    if arch:
        config["architecture"] = arch
    return config
