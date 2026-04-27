"""Tests for vibeblade.quant — RotorQuant 4-bit quantisation module."""

from __future__ import annotations

import math
import numpy as np
import pytest

from vibeblade.quant import (
    build_so4_rotor,
    dequantize_4bit,
    pack_nibbles,
    quantization_error,
    quantize_4bit,
    unpack_nibbles,
)


# ---------------------------------------------------------------------------
# pack_nibbles / unpack_nibbles
# ---------------------------------------------------------------------------

class TestPackUnpack:
    def test_pack_unpack_roundtrip(self):
        """Pack then unpack should recover original values."""
        rng = np.random.default_rng(42)
        values = rng.integers(0, 16, size=100).astype(np.uint8)
        packed = pack_nibbles(values)
        recovered = unpack_nibbles(packed, values.size)
        np.testing.assert_array_equal(values, recovered)

    def test_pack_nibbles_shape(self):
        """n values → ceil(n/2) bytes."""
        assert pack_nibbles(np.zeros(10, dtype=np.uint8)).shape == (5,)
        assert pack_nibbles(np.zeros(11, dtype=np.uint8)).shape == (6,)
        assert pack_nibbles(np.zeros(1, dtype=np.uint8)).shape == (1,)

    def test_pack_nibbles_values(self):
        """Low and high nibbles are stored correctly."""
        # values: [0x3, 0xB] → byte = (0xB << 4) | 0x3 = 0xB3
        packed = pack_nibbles(np.array([3, 11], dtype=np.uint8))
        assert packed[0] == 0xB3

    def test_pack_nibbles_odd_length(self):
        """Odd-length input: last byte's high nibble is zero-padded."""
        packed = pack_nibbles(np.array([7], dtype=np.uint8))
        assert packed[0] == 0x07  # high nibble = 0

    def test_pack_nibbles_invalid_values(self):
        """Values > 15 should raise ValueError."""
        with pytest.raises(ValueError):
            pack_nibbles(np.array([16], dtype=np.uint8))

    def test_unpack_nibbles_insufficient_bytes(self):
        """Asking for more values than packed bytes can hold should raise."""
        with pytest.raises(ValueError):
            unpack_nibbles(np.array([0xAB], dtype=np.uint8), n=3)


# ---------------------------------------------------------------------------
# build_so4_rotor
# ---------------------------------------------------------------------------

class TestBuildSO4Rotor:
    def test_build_so4_rotor_det(self):
        """Rotation matrix should have det ≈ 1."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            w = rng.standard_normal(4).astype(np.float32)
            R = build_so4_rotor(w)
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-5, f"det={det}"

    def test_build_so4_rotor_inverse(self):
        """R @ R^T should be approximately the identity."""
        rng = np.random.default_rng(456)
        for _ in range(20):
            w = rng.standard_normal(4).astype(np.float32)
            R = build_so4_rotor(w)
            identity = R @ R.T
            np.testing.assert_allclose(identity, np.eye(4), atol=1e-6)

    def test_build_so4_rotor_identity_for_zero(self):
        """Zero weights should produce identity rotation."""
        R = build_so4_rotor(np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(R, np.eye(4))

    def test_build_so4_rotor_bad_shape(self):
        """Non-(4,) input should raise ValueError."""
        with pytest.raises(ValueError):
            build_so4_rotor(np.zeros(5))


# ---------------------------------------------------------------------------
# quantize_4bit / dequantize_4bit
# ---------------------------------------------------------------------------

class TestQuantizeDequantize:
    def test_quantize_dequantize_roundtrip(self):
        """dequantize(quantize(w)) should approximate w."""
        rng = np.random.default_rng(789)
        w = rng.standard_normal(256).astype(np.float32)
        packed, scales, rotors = quantize_4bit(w, group_size=32)
        reconstructed = dequantize_4bit(packed, scales, rotors, n=w.size)
        err = quantization_error(w, reconstructed)
        # 4-bit quantisation with rotation should be reasonably accurate.
        assert err < 0.15, f"RMS error {err} too large"

    def test_quantize_shape_mismatch(self):
        """Weight length not divisible by group_size should raise ValueError."""
        w = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="divisible by group_size"):
            quantize_4bit(w, group_size=32)

    def test_quantize_bad_group_size(self):
        """group_size not divisible by 4 should raise ValueError."""
        w = np.random.randn(20).astype(np.float32)
        with pytest.raises(ValueError, match="divisible by 4"):
            quantize_4bit(w, group_size=6)

    def test_group_size_edge_cases(self):
        """Test with group_size 4, 8, and 32 — all should roundtrip."""
        rng = np.random.default_rng(321)
        for gs in [4, 8, 32]:
            n = gs * 4  # 4 groups
            w = rng.standard_normal(n).astype(np.float32)
            packed, scales, rotors = quantize_4bit(w, group_size=gs)
            reconstructed = dequantize_4bit(packed, scales, rotors, n=n)
            err = quantization_error(w, reconstructed)
            assert err < 0.15, f"group_size={gs}: RMS error {err} too large"
            assert reconstructed.shape == (n,)

    def test_constant_weights(self):
        """All-constant weights should reconstruct perfectly (error ≈ 0)."""
        w = np.full(32, 5.0, dtype=np.float32)
        packed, scales, rotors = quantize_4bit(w, group_size=32)
        reconstructed = dequantize_4bit(packed, scales, rotors, n=w.size)
        err = quantization_error(w, reconstructed)
        assert err < 1e-5, f"Constant weights: RMS error {err}"

    def test_output_shapes(self):
        """Verify returned shapes match expected dimensions."""
        n = 128
        group_size = 32
        sub_groups = group_size // 4
        num_groups = n // group_size
        w = np.random.randn(n).astype(np.float32)
        packed, scales, rotors = quantize_4bit(w, group_size=group_size)

        assert packed.ndim == 1
        assert packed.dtype == np.uint8
        assert packed.size == math.ceil(n / 2)

        assert scales.shape == (num_groups, 2, sub_groups)
        assert scales.dtype == np.float32

        assert rotors.shape == (num_groups, sub_groups, 4, 4)
        assert rotors.dtype == np.float32


# ---------------------------------------------------------------------------
# quantization_error
# ---------------------------------------------------------------------------

class TestQuantizationError:
    def test_quantization_error_zero(self):
        """Identical arrays should give error 0.0."""
        x = np.random.randn(50).astype(np.float32)
        assert quantization_error(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_quantization_error_known(self):
        """Known offset: RMS of [0, 3, 4] vs [1, 1, 7] = sqrt((1+4+9)/3)."""
        a = np.array([0.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([1.0, 1.0, 7.0], dtype=np.float32)
        expected = math.sqrt((1 + 4 + 9) / 3)
        assert quantization_error(a, b) == pytest.approx(expected)

    def test_quantization_error_shape_mismatch(self):
        """Different-shaped arrays should raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            quantization_error(np.zeros(3), np.zeros(4))
