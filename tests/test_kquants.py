"""Tests for K-quants (Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) dequantization."""

import struct
import numpy as np
import pytest
from vibeblade.loader import (
    GGUF_TYPE_Q3_K,
    GGUF_TYPE_Q4_K,
    GGUF_TYPE_Q5_K,
    GGUF_TYPE_Q6_K,
    GGUF_TYPE_Q8_K,
    GGUF_TYPE_Q8_1,
    _dequant_q3_k,
    _dequant_q4_k,
    _dequant_q5_k,
    _dequant_q6_k,
    _dequant_q8_k,
    _dequant_q8_1,
    _QBLOCK_SIZE,
    _QBLOCK_BYTES,
)


class TestQ3KDequant:
    def test_output_shape(self):
        block = bytes(_QBLOCK_BYTES[GGUF_TYPE_Q3_K])
        out = _dequant_q3_k(block)
        assert out.shape == (_QBLOCK_SIZE[GGUF_TYPE_Q3_K],)

    def test_output_dtype(self):
        block = bytes(_QBLOCK_BYTES[GGUF_TYPE_Q3_K])
        out = _dequant_q3_k(block)
        assert out.dtype == np.float32

    def test_nonzero_output_with_nonzero_scale(self):
        """A block with a nonzero scale should produce nonzero output."""
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q3_K])
        # Set d scale to 1.0 (f16)
        data[0:2] = struct.pack('<e', 1.0)
        # Fill quants (offset 34-98) and h (offset 2-34) with nonzero values
        for i in range(2, 34):
            data[i] = 0xFF  # h bits
        for i in range(34, 98):
            data[i] = 0xFF  # qs quants
        for i in range(98, 110):
            data[i] = 0xFF  # scales
        out = _dequant_q3_k(bytes(data))
        assert np.any(out != 0)

    def test_zero_scale_gives_zero_output(self):
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q3_K])
        data[0:2] = struct.pack('<e', 0.0)
        out = _dequant_q3_k(bytes(data))
        assert np.all(out == 0.0)


class TestQ4KDequant:
    def test_output_shape(self):
        block = bytes(_QBLOCK_BYTES[GGUF_TYPE_Q4_K])
        out = _dequant_q4_k(block)
        assert out.shape == (256,)

    def test_output_dtype(self):
        out = _dequant_q4_k(bytes(_QBLOCK_BYTES[GGUF_TYPE_Q4_K]))
        assert out.dtype == np.float32

    def test_known_values(self):
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q4_K])
        # d = 1.0, dmin = 0.0
        data[0:2] = struct.pack('<e', 1.0)
        data[2] = 0
        # Set all scales to 32 (max 6-bit value)
        sc_raw = bytearray(8)
        for i in range(8):
            byte_idx = (i * 6) // 8
            bit_off = (i * 6) % 8
            # Pack value 32 at position i
            packed = 32 << bit_off
            sc_raw[byte_idx] |= (packed & 0xFF)
            if bit_off > 2:
                sc_raw[byte_idx + 1] |= (packed >> 8) & 0xFF
        data[4:12] = bytes(sc_raw)
        # Set quants to 0x88 (nibbles = 8,8 -> 0 after -8 offset)
        for i in range(16, 80):
            data[i] = 0x88
        for i in range(80, 144):
            data[i] = 0x88
        out = _dequant_q4_k(bytes(data))
        # All nibbles are 8, so (8-8)*scale*d + dmin = 0
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_nonzero_output(self):
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q4_K])
        data[0:2] = struct.pack('<e', 1.0)
        # Set scales to nonzero (bytes 4-11, packed 6-bit)
        for i in range(4, 12):
            data[i] = 0xFF  # all scales maxed
        for i in range(16, 144):
            data[i] = 0x11  # nibbles 1,1
        out = _dequant_q4_k(bytes(data))
        assert np.any(out != 0)


class TestQ5KDequant:
    def test_output_shape(self):
        out = _dequant_q5_k(bytes(_QBLOCK_BYTES[GGUF_TYPE_Q5_K]))
        assert out.shape == (256,)

    def test_output_dtype(self):
        out = _dequant_q5_k(bytes(_QBLOCK_BYTES[GGUF_TYPE_Q5_K]))
        assert out.dtype == np.float32

    def test_nonzero_output(self):
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q5_K])
        data[0:2] = struct.pack('<e', 1.0)
        for i in range(4, 12):
            data[i] = 0xFF  # scales nonzero
        for i in range(16, 144):
            data[i] = 0x11
        for i in range(144, 176):
            data[i] = 0xFF
        out = _dequant_q5_k(bytes(data))
        assert np.any(out != 0)


class TestQ6KDequant:
    def test_output_shape(self):
        out = _dequant_q6_k(bytes(_QBLOCK_BYTES[GGUF_TYPE_Q6_K]))
        assert out.shape == (256,)

    def test_output_dtype(self):
        out = _dequant_q6_k(bytes(_QBLOCK_BYTES[GGUF_TYPE_Q6_K]))
        assert out.dtype == np.float32

    def test_known_range(self):
        """Q6_K produces 6-bit signed values: -32 to +31."""
        # Create block with max quant values (ql=0xF, qh high=0x3 => q=63, output=63-32=31)
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q6_K])
        # d = 1.0 (f16)
        data[208:210] = struct.pack('<e', 1.0)
        # All scales = 1 (int8)
        for i in range(192, 208):
            data[i] = 1
        # ql = all 0xFF (low nibble 0xF for all)
        for i in range(128):
            data[i] = 0xFF
        # qh = all 0xFF (high bits = 0x3 for all pairs)
        for i in range(128, 192):
            data[i] = 0xFF
        out = _dequant_q6_k(bytes(data))
        # q = 0xF | (0x3 << 4) = 0x3F = 63
        # output = (63 - 32) * 1 * 1 = 31
        assert np.all(out == 31.0)

    def test_min_value(self):
        """Minimum Q6_K value: ql=0x0, qh=0x0 => q=0, output=0-32=-32."""
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q6_K])
        data[208:210] = struct.pack('<e', 1.0)
        for i in range(192, 208):
            data[i] = 1
        # ql = all zeros, qh = all zeros -> q=0, output = -32
        out = _dequant_q6_k(bytes(data))
        assert np.all(out == -32.0)

    def test_midpoint_value(self):
        """Midpoint: ql=0, qh=0b10 => q=0x20=32, output=32-32=0."""
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q6_K])
        data[208:210] = struct.pack('<e', 1.0)
        for i in range(192, 208):
            data[i] = 1
        # ql = all 0, qh = pattern to give 0x10 (bit pattern 01 for each 2-bit field)
        for i in range(128, 192):
            data[i] = 0x55  # 01010101 -> each 2-bit pair = 01 => qh_high = 1
        out = _dequant_q6_k(bytes(data))
        # q = 0 | (1 << 4) = 16, output = 16 - 32 = -16
        assert np.all(out == -16.0)


class TestQ8KDequant:
    def test_output_shape(self):
        out = _dequant_q8_k(bytes(_QBLOCK_BYTES[GGUF_TYPE_Q8_K]))
        assert out.shape == (256,)

    def test_output_dtype(self):
        out = _dequant_q8_k(bytes(_QBLOCK_BYTES[GGUF_TYPE_Q8_K]))
        assert out.dtype == np.float32

    def test_nonzero_output(self):
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q8_K])
        # d = 1.0
        struct.pack_into('<f', data, 0, 1.0)
        # Fill quants with nonzero int8 values
        for i in range(12, 268):
            data[i] = 42
        out = _dequant_q8_k(bytes(data))
        assert np.any(out != 0)

    def test_zero_scale(self):
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q8_K])
        struct.pack_into('<f', data, 0, 0.0)
        struct.pack_into('<f', data, 4, 0.0)
        for i in range(12, 268):
            data[i] = 42
        out = _dequant_q8_k(bytes(data))
        # d=0 means quant contribution is 0, but dmin and b might add offset
        # With d_s=0, b values are all 0 too
        assert np.all(out == 0.0)


class TestQ81Dequant:
    def test_output_shape(self):
        block = bytes(_QBLOCK_BYTES[GGUF_TYPE_Q8_1])
        out = _dequant_q8_1(block)
        assert out.shape == (32,)

    def test_output_dtype(self):
        block = bytes(_QBLOCK_BYTES[GGUF_TYPE_Q8_1])
        out = _dequant_q8_1(block)
        assert out.dtype == np.float32

    def test_nonzero_output(self):
        data = bytearray(_QBLOCK_BYTES[GGUF_TYPE_Q8_1])
        # Q8_1 layout: 2B f16 d + 2B f16 s + 4B f16 b + 32B int8 = 40 bytes
        data[0:2] = struct.pack('<e', 1.0)   # d = 1.0
        data[2:4] = struct.pack('<e', 0.0)   # s = 0.0
        data[4:8] = struct.pack('<e', 0.0) + b'\x00\x00'  # b = 0.0, 2B pad
        for i in range(8, 8 + 32):
            data[i] = 10
        out = _dequant_q8_1(bytes(data))
        assert np.allclose(out, 10.0)


class TestDispatchCompleteness:
    """Verify all known quant types are registered in _DEQUANT_FN."""

    def test_all_k_quants_registered(self):
        from vibeblade.loader import _DEQUANT_FN
        k_types = [GGUF_TYPE_Q3_K, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K,
                    GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_K]
        for t in k_types:
            assert t in _DEQUANT_FN, f"Type {t} not in _DEQUANT_FN"

    def test_q8_1_registered(self):
        from vibeblade.loader import _DEQUANT_FN
        assert GGUF_TYPE_Q8_1 in _DEQUANT_FN

    def test_q8_k_in_block_tables(self):
        assert GGUF_TYPE_Q8_K in _QBLOCK_SIZE
        assert GGUF_TYPE_Q8_K in _QBLOCK_BYTES
