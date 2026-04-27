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
    GGUF_TYPE_Q3_K: 110,  # 2 + 2 + 16 + 32 + 32 + 32
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


# ── Dequantization kernels ──

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
    """Dequantize a Q4_1 block: 2-byte f16 scale + 2-byte f16 min + 16 bytes nibbles."""
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


def _dequant_q2_k(block: bytes) -> np.ndarray:
    """Dequantize a Q2_K super-block (256 values)."""
    # Q2_K: scales (16 × uint8, each represents 0 or 2×scale) + mins (16 × f16) + qs (64 bytes)
    scales = np.frombuffer(block[0:16], dtype=np.uint8).astype(np.float32)
    mins = np.frombuffer(block[16:48], dtype=np.float16).astype(np.float32)
    qs = np.frombuffer(block[48:112], dtype=np.uint8)
    # Each qs byte has two 2-bit values: low = qs & 3, high = (qs >> 2) & 3
    low = (qs & 0x03).astype(np.float32)
    high = ((qs >> 2) & 0x03).astype(np.float32)
    scales.reshape(16, 1)  # (16, 2) via repeat
    # Interleave and reshape to 256
    vals = np.empty(128, dtype=np.float32)
    vals[0::2] = low
    vals[1::2] = high
    # Apply scales and mins per sub-block of 16
    out = np.empty(256, dtype=np.float32)
    for i in range(16):
        sub = vals[i * 8:(i + 1) * 8]
        sc = scales[i] / 16.0  # Q2_K scale encoding
        out[i * 16:(i + 1) * 16] = sub * sc + mins[i]
    # Fill remaining 128 values from second half
    qs2 = np.frombuffer(block[112:176], dtype=np.uint8)
    low2 = (qs2 & 0x03).astype(np.float32)
    high2 = ((qs2 >> 2) & 0x03).astype(np.float32)
    vals2 = np.empty(128, dtype=np.float32)
    vals2[0::2] = low2
    vals2[1::2] = high2
    for i in range(16):
        sub = vals2[i * 8:(i + 1) * 8]
        sc = scales[i] / 16.0
        out[128 + i * 16:128 + (i + 1) * 16] = sub * sc + mins[i]
    return out



def _dequant_q3_k(block: bytes) -> np.ndarray:
    """Dequantize a Q3_K block (256 elements, 110 bytes).

    Layout (from ggml block_q3_K):
      [0:2]    d (f16 scale)
      [2:34]   h (32 bytes, extra high bits for 256 values: 1 bit per 8 values = 32 bytes)
      [34:98]  qs (64 bytes, 2-bit packed quants: 4 values per byte = 256 values)
      [98:110] scales (12 bytes, 16 packed 6-bit sub-block scales)
    """
    d = np.frombuffer(block[0:2], dtype=np.float16).astype(np.float32)[0]
    h = np.frombuffer(block[2:34], dtype=np.uint8)
    qs = np.frombuffer(block[34:98], dtype=np.uint8)
    sc_packed = np.frombuffer(block[98:110], dtype=np.uint8)

    # Unpack 16 6-bit scales from 12 bytes (96 bits)
    scales = np.zeros(16, dtype=np.float32)
    for i in range(16):
        byte_idx = (i * 6) // 8
        bit_off = (i * 6) % 8
        val = int(sc_packed[byte_idx]) >> bit_off
        if byte_idx + 1 < len(sc_packed) and bit_off > 2:
            val |= int(sc_packed[byte_idx + 1]) << (8 - bit_off)
        scales[i] = float(val & 0x3F)

    out = np.empty(256, dtype=np.float32)
    for i in range(256):
        # 2 bits from qs (4 values packed per byte)
        qs_byte = int(qs[i // 4])
        shift = (i % 4) * 2
        q_low = (qs_byte >> shift) & 0x03
        # 1 high bit from h
        h_byte = int(h[i // 8])
        h_bit = (h_byte >> (i % 8)) & 1
        q = q_low + h_bit * 4
        sc_idx = i // 16
        out[i] = (q - 4) * scales[sc_idx] * d

    return out




def _dequant_q4_k(block: bytes) -> np.ndarray:
    """Dequantize a Q4_K block (256 elements, 144 bytes)."""
    d = np.frombuffer(block[0:2], dtype=np.float16).astype(np.float32)[0]
    dmin = float(block[2])

    sc_raw = np.frombuffer(block[4:12], dtype=np.uint8)
    scales = np.zeros(8, dtype=np.float32)
    for i in range(8):
        byte_idx = (i * 6) // 8
        bit_off = (i * 6) % 8
        val = int(sc_raw[byte_idx]) >> bit_off
        if byte_idx + 1 < len(sc_raw) and bit_off > 2:
            val |= int(sc_raw[byte_idx + 1]) << (8 - bit_off)
        scales[i] = float(val & 0x3F)

    out = np.empty(256, dtype=np.float32)
    qs = np.frombuffer(block[16:80], dtype=np.uint8)
    qs2 = np.frombuffer(block[80:144], dtype=np.uint8)

    for half in range(2):
        src = qs if half == 0 else qs2
        sc_base = half * 4
        for i in range(128):
            byte_idx = i // 2
            nib = (int(src[byte_idx]) >> (4 * (i % 2))) & 0x0F
            sc_idx = sc_base + i // 32
            out[half * 128 + i] = (nib - 8) * scales[sc_idx] * d + dmin

    return out


def _dequant_q5_k(block: bytes) -> np.ndarray:
    """Dequantize a Q5_K block (256 elements, 176 bytes)."""
    d = np.frombuffer(block[0:2], dtype=np.float16).astype(np.float32)[0]
    dmin = float(block[2])

    sc_raw = np.frombuffer(block[4:12], dtype=np.uint8)
    scales = np.zeros(8, dtype=np.float32)
    for i in range(8):
        byte_idx = (i * 6) // 8
        bit_off = (i * 6) % 8
        val = int(sc_raw[byte_idx]) >> bit_off
        if byte_idx + 1 < len(sc_raw) and bit_off > 2:
            val |= int(sc_raw[byte_idx + 1]) << (8 - bit_off)
        scales[i] = float(val & 0x3F)

    qh = np.frombuffer(block[144:176], dtype=np.uint8)
    qh_bits = np.unpackbits(qh)

    out = np.empty(256, dtype=np.float32)
    qs = np.frombuffer(block[16:80], dtype=np.uint8)
    qs2 = np.frombuffer(block[80:144], dtype=np.uint8)

    for half in range(2):
        src = qs if half == 0 else qs2
        sc_base = half * 4
        for i in range(128):
            byte_idx = i // 2
            nib = (int(src[byte_idx]) >> (4 * (i % 2))) & 0x0F
            high_bit = int(qh_bits[half * 128 + i])
            val = nib + high_bit * 16
            sc_idx = sc_base + i // 32
            out[half * 128 + i] = (val - 16) * scales[sc_idx] * d + dmin

    return out


def _dequant_q6_k(block: bytes) -> np.ndarray:
    """Dequantize a Q6_K block (256 elements, 210 bytes).

    Layout per super-block of 256:
      [0:128]   uint8 x128 - ql: 4-bit low quants (2 values per byte)
      [128:192] uint8 x64  - qh: 2-bit high quants (4 values per byte)
      [192:208] int8 x16   - sub-block scales (16 sub-blocks of 16 values)
      [208:210] uint16     - d (super-block scale, f16)
    """
    d = np.frombuffer(block[208:210], dtype=np.float16).astype(np.float32)[0]
    ql = np.frombuffer(block[0:128], dtype=np.uint8)
    qh = np.frombuffer(block[128:192], dtype=np.uint8)
    sc = np.frombuffer(block[192:208], dtype=np.int8).astype(np.float32)

    out = np.empty(256, dtype=np.float32)
    for i in range(256):
        ql_byte = int(ql[i // 2])
        if i % 2 == 0:
            ql_low = ql_byte & 0x0F
        else:
            ql_low = (ql_byte >> 4) & 0x0F

        qh_byte = int(qh[i // 4])
        qh_high = (qh_byte >> ((i % 4) * 2)) & 0x03

        q = ql_low | (qh_high << 4)
        s_idx = i // 16
        out[i] = (q - 32) * sc[s_idx] * d

    return out


def _dequant_q8_k(block: bytes) -> np.ndarray:
    """Dequantize a Q8_K block (256 elements, 292 bytes).

    Layout:
      [0:4]    d (float32, super-block scale)
      [4:8]    dmin (float32)
      [8:12]   d_s (float32, scale for sub-block sums)
      [12:268] qs (256 int8 quantized values)
      [268:292] bsums (24 bytes, 16 used: sub-block offset sums)
    """
    d = np.frombuffer(block[0:4], dtype=np.float32)[0]
    dmin = np.frombuffer(block[4:8], dtype=np.float32)[0]
    d_s = np.frombuffer(block[8:12], dtype=np.float32)[0]
    qs = np.frombuffer(block[12:268], dtype=np.int8).astype(np.float32)
    bsums = np.frombuffer(block[268:292], dtype=np.int8).astype(np.float32) * d_s

    out = np.empty(256, dtype=np.float32)
    for i in range(16):
        out[i * 16:(i + 1) * 16] = qs[i * 16:(i + 1) * 16] * d + dmin + bsums[i]
    return out



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
    GGUF_TYPE_Q2_K: _dequant_q2_k,
    GGUF_TYPE_Q3_K: _dequant_q3_k,
    GGUF_TYPE_Q4_K: _dequant_q4_k,
    GGUF_TYPE_Q5_K: _dequant_q5_k,
    GGUF_TYPE_Q6_K: _dequant_q6_k,
    GGUF_TYPE_Q8_K: _dequant_q8_k,
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

    def __init__(self, path: str) -> None:
        self._path = path
        self._file_size = os.path.getsize(path)
        self._use_mmap = self._file_size > self.MMAP_THRESHOLD
        self._mmap: mmap.mmap | None = None

        self._f = open(path, "rb")
        self._parse_header()
        self._parse_metadata()
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
        for _ in range(self.n_kv):
            key = self._read_string()
            (vtype,) = struct.unpack("<I", self._read(4))
            self.metadata[key] = self._read_value(vtype)

    def _parse_tensor_infos(self) -> None:
        self.tensor_infos: list[dict] = []
        for _ in range(self.n_tensors):
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

    def load_tensor(self, name: str, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
        """Load and dequantize a tensor.

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

        if tid in _DEQUANT_FN:
            arr = self._dequant_blocks(raw, tid, n_elements)
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

    def _dequant_blocks(self, raw: bytes, tid: int, n_elements: int) -> np.ndarray:
        """Dequantize raw block data to f32."""
        fn = _DEQUANT_FN[tid]
        block_size = _QBLOCK_SIZE[tid]
        block_bytes = _QBLOCK_BYTES[tid]
        n_blocks = n_elements // block_size

        out = np.empty(n_blocks * block_size, dtype=np.float32)
        for i in range(n_blocks):
            start = i * block_bytes
            end = start + block_bytes
            if tid in (GGUF_TYPE_Q2_K, GGUF_TYPE_Q3_K, GGUF_TYPE_Q4_K,
                       GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_K):
                out[i * block_size:(i + 1) * block_size] = fn(raw[start:end])
            else:
                out[i * block_size:(i + 1) * block_size] = fn(
                    raw[start:end], block_size
                )

        return out[:n_elements]

    def load_all_tensors(self, dtype: np.dtype = np.dtype("float32")) -> dict[str, np.ndarray]:
        """Load all tensors. Returns {name: ndarray}."""
        return {info["name"]: self.load_tensor(info["name"], dtype)
                for info in self.tensor_infos}

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


def load_model(path: str) -> dict:
    """Load a GGUF model and return metadata, tensors, and config."""
    with GGUFLoader(path) as loader:
        metadata = dict(loader.metadata)
        tensors = loader.load_all_tensors()
        config = _extract_config(metadata)
    return {"metadata": metadata, "tensors": tensors, "config": config}


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
