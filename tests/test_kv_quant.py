"""Tests for VibeBlade KIVI — asymmetric 2-bit KV cache quantization."""

import numpy as np
import pytest

from vibeblade.kv_quant import (
    KIVICache,
    quantize_kv_2bit,
    dequantize_kv_2bit,
    pack_2bit,
    unpack_2bit,
)


class TestQuantize2Bit:
    def test_per_channel_quantize(self):
        x = np.random.randn(8, 16).astype(np.float32)
        q, scale, kv_min = quantize_kv_2bit(x, axis=1)
        assert q.dtype == np.uint8
        assert np.all(q >= 0) and np.all(q <= 3)
        assert scale.shape == (16,)  # one scale per channel (feature dim)
        assert kv_min.shape == (16,)

    def test_per_token_quantize(self):
        x = np.array([[1.0, 5.0], [3.0, 7.0]], dtype=np.float32)
        q, scale, kv_min = quantize_kv_2bit(x, axis=0)
        assert q.shape == x.shape
        assert scale.shape == (2,)  # one scale per token (row)
        assert kv_min.shape == (2,)

    def test_roundtrip(self):
        x = np.random.randn(4, 16).astype(np.float32)
        for axis in [0, 1]:
            q, scale, kv_min = quantize_kv_2bit(x, axis)
            x_recon = dequantize_kv_2bit(q, scale, axis, kv_min)
            assert x_recon.shape == x.shape
            # Should be approximately close (4 levels)
            error = np.max(np.abs(x_recon.astype(np.float32) - x))
            assert error < 1.5  # max quantization error for 2-bit

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            quantize_kv_2bit(np.zeros((2, 3)), axis=2)


class TestPackUnpack2Bit:
    def test_pack_unpack_roundtrip(self):
        values = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
        packed = pack_2bit(values)
        assert packed.shape == (2,)  # 8 values -> 2 bytes
        unpacked = unpack_2bit(packed, 8)
        np.testing.assert_array_equal(values, unpacked)

    def test_odd_length(self):
        values = np.array([1, 2, 3], dtype=np.uint8)
        packed = pack_2bit(values)
        unpacked = unpack_2bit(packed, 3)
        np.testing.assert_array_equal(values, unpacked)


class TestKIVICache:
    def test_init(self):
        cache = KIVICache(2, 4, 16, 128)
        assert cache.compression_ratio() > 1.0

    def test_update_and_get(self):
        cache = KIVICache(2, 4, 16, 64)
        k = np.random.randn(4, 16).astype(np.float16)
        v = np.random.randn(4, 16).astype(np.float16)

        cache.update(0, k, v, position=0)
        assert cache.length == 1

        k_out, v_out = cache.get(0)
        assert k_out.shape == (4, 1, 16)
        assert v_out.shape == (4, 1, 16)

    def test_compression_ratio(self):
        cache = KIVICache(2, 4, 16, 1024)
        # 2-bit packed should be ~8x smaller than fp16 (16 bits / 2 bits)
        # Plus scale overhead, so expect > 4x
        assert cache.compression_ratio() > 4.0

    def test_clear(self):
        cache = KIVICache(1, 2, 8, 32)
        cache.update(0, np.ones((2, 8), dtype=np.float16), np.ones((2, 8), dtype=np.float16), 0)
        cache.clear()
        assert cache.length == 0

    def test_multi_token(self):
        cache = KIVICache(1, 2, 8, 64)
        for i in range(10):
            k = np.full((2, 8), float(i), dtype=np.float16)
            v = np.full((2, 8), float(i + 100), dtype=np.float16)
            cache.update(0, k, v, i)

        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 10, 8)
        # First token should be close to 0 (per-channel quant uses global min/max)
        # so there will be quantization error but it should be small
        assert float(k_out[0, 0, 0]) < 1.0
        # Last token should be close to 9
        assert float(k_out[0, 9, 0]) > 7.0
