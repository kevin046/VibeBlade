"""Tests for VibeBlade SmoothQuant."""

import numpy as np
import pytest

from vibeblade.smoothquant import (
    compute_smooth_factor,
    smooth_weights,
    smooth_activations,
    quantize_smoothed_w8a8,
    dequantize_w8,
    SmoothQuantizer,
)


class TestSmoothFactor:
    def test_basic(self):
        x = np.random.randn(8, 16).astype(np.float32)
        s = compute_smooth_factor(x, alpha=0.5)
        assert s.shape == (16,)
        assert np.all(s > 0)

    def test_alpha_zero(self):
        x = np.random.randn(8, 16).astype(np.float32)
        s = compute_smooth_factor(x, alpha=0.0)
        np.testing.assert_allclose(s, np.ones(16))


class TestSmoothWeights:
    def test_shape_preserved(self):
        w = np.random.randn(16, 32).astype(np.float32)
        s = np.random.rand(16).astype(np.float32) + 0.1
        w_smooth = smooth_weights(w, s)
        assert w_smooth.shape == w.shape


class TestSmoothActivations:
    def test_shape_preserved(self):
        x = np.random.randn(8, 16).astype(np.float32)
        s = np.random.rand(16).astype(np.float32) + 0.1
        x_smooth = smooth_activations(x, s)
        assert x_smooth.shape == x.shape


class TestQuantizeSmoothedW8A8:
    def test_output_types(self):
        w = np.random.randn(16, 32).astype(np.float32)
        x = np.random.randn(8, 16).astype(np.float32)
        w_q, w_s, x_s, s = quantize_smoothed_w8a8(w, x)
        assert w_q.dtype == np.int8
        assert w_s.dtype == np.float32
        assert x_s.dtype == np.float32
        assert s.dtype == np.float32

    def test_w8_range(self):
        w = np.random.randn(16, 32).astype(np.float32) * 5
        x = np.random.randn(8, 16).astype(np.float32)
        w_q, _, _, _ = quantize_smoothed_w8a8(w, x)
        assert np.all(w_q >= -128)
        assert np.all(w_q <= 127)

    def test_roundtrip_error(self):
        w = np.random.randn(64, 128).astype(np.float32)
        x = np.random.randn(32, 64).astype(np.float32)
        w_q, w_s, _, s = quantize_smoothed_w8a8(w, x)
        w_recon = dequantize_w8(w_q, w_s)
        error = np.max(np.abs(w_recon.astype(np.float32) - smooth_weights(w, s)))
        # 8-bit quantization error should be small
        assert error < 0.1


class TestSmoothQuantizer:
    def test_calibrate_and_forward(self):
        sq = SmoothQuantizer(alpha=0.5)
        w = np.random.randn(64, 128).astype(np.float32)
        x = np.random.randn(8, 64).astype(np.float32)

        sq.calibrate_layer(0, w, x)
        out = sq.forward_layer(0, x)
        assert out.shape == (8, 128)
        assert out.dtype == np.float16

    def test_memory_savings(self):
        sq = SmoothQuantizer(alpha=0.5)
        w = np.random.randn(64, 128).astype(np.float32)
        x = np.random.randn(8, 64).astype(np.float32)
        sq.calibrate_layer(0, w, x)
        assert sq.memory_savings() > 0.5  # int8 vs float32 = ~4x

    def test_uncalibrated_raises(self):
        sq = SmoothQuantizer()
        with pytest.raises(KeyError):
            sq.forward_layer(0, np.random.randn(8, 64).astype(np.float32))
