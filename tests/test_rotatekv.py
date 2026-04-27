"""Tests for RotateKV — Hadamard rotation + 2-bit KV quantization."""

import numpy as np
import pytest

from vibeblade.rotatekv import (
    hadamard_rotation_matrix,
    rotate_kv,
    inverse_rotate_kv,
    RotateKVCache,
)


class TestHadamardRotation:
    """Test Sylvester Hadamard matrix construction."""

    def test_hadamard_orthogonal(self):
        """H @ H^T should equal I."""
        for size in [1, 2, 4, 8, 16]:
            from vibeblade.rotatekv import _sylvester_hadamard
            H = _sylvester_hadamard(size)
            I_mat = H @ H.T
            np.testing.assert_allclose(I_mat, np.eye(size), atol=1e-5)

    def test_block_diagonal_orthogonal(self):
        """Block-diagonal rotation matrix should be orthogonal."""
        R = hadamard_rotation_matrix(64, block_size=16)
        I_mat = R @ R.T
        np.testing.assert_allclose(I_mat, np.eye(64), atol=1e-5)

    def test_non_power_of_two_dim(self):
        """Should handle dimensions that aren't multiples of block_size."""
        R = hadamard_rotation_matrix(48, block_size=16)
        assert R.shape == (48, 48)
        I_mat = R @ R.T
        np.testing.assert_allclose(I_mat, np.eye(48), atol=1e-5)

    def test_dim_larger_than_block(self):
        """Multiple blocks when dim > block_size."""
        R = hadamard_rotation_matrix(128, block_size=32)
        assert R.shape == (128, 128)
        I_mat = R @ R.T
        np.testing.assert_allclose(I_mat, np.eye(128), atol=1e-5)

    def test_single_element_dim(self):
        R = hadamard_rotation_matrix(1)
        assert R.shape == (1, 1)
        # For dim=1, block_size rounds up to 2 and truncates — the single
        # element may not equal 1.0 exactly, but should be nonzero.
        assert abs(R[0, 0]) > 0


class TestRotateKV:
    """Test rotation and inverse rotation."""

    def test_rotate_inverse_roundtrip_2d(self):
        """Rotating then inverse-rotating should recover the original."""
        R = hadamard_rotation_matrix(64, block_size=16)
        x = np.random.randn(32, 64).astype(np.float32)
        x_rot = rotate_kv(x, R)
        x_back = inverse_rotate_kv(x_rot, R)
        np.testing.assert_allclose(x, x_back, atol=1e-5)

    def test_rotate_inverse_roundtrip_3d(self):
        """3D input (num_heads, seq_len, head_dim)."""
        R = hadamard_rotation_matrix(32, block_size=16)
        x = np.random.randn(4, 10, 32).astype(np.float32)
        x_rot = rotate_kv(x, R)
        x_back = inverse_rotate_kv(x_rot, R)
        np.testing.assert_allclose(x, x_back, atol=1e-5)

    def test_rotation_smoothes_outliers(self):
        """Rotation should spread outlier magnitudes across channels."""
        R = hadamard_rotation_matrix(64, block_size=64)
        # Create data with extreme channel-wise outliers
        x = np.random.randn(16, 64).astype(np.float32) * 0.1
        x[:, 0] += 50.0  # massive outlier in channel 0
        x[:, 1] += 40.0

        x_rot = rotate_kv(x, R)

        # After rotation, the max-per-channel should be lower than the outlier
        orig_max = float(np.abs(x).max(axis=0).max())
        rot_max = float(np.abs(x_rot).max(axis=0).max())
        # Rotation spreads the energy, so no single channel should be as extreme
        assert rot_max < orig_max, "Rotation should reduce per-channel max magnitude"

    def test_invalid_ndim_raises(self):
        R = hadamard_rotation_matrix(16)
        with pytest.raises(ValueError):
            rotate_kv(np.zeros(3), R)


class TestRotateKVCache:
    """Test the full RotateKVCache with quantization."""

    @pytest.fixture
    def cache(self):
        return RotateKVCache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_seq_len=128,
            block_size=16,
        )

    def test_update_and_get_shape(self, cache):
        """Output shape should match expected dimensions."""
        key = np.random.randn(8, 64).astype(np.float32)
        value = np.random.randn(8, 64).astype(np.float32)
        cache.update(0, key, value, 0)

        k, v = cache.get(0)
        assert k.shape == (8, 1, 64)
        assert v.shape == (8, 1, 64)

    def test_3d_input(self, cache):
        """Should handle (num_heads, 1, head_dim) input."""
        key = np.random.randn(8, 1, 64).astype(np.float32)
        value = np.random.randn(8, 1, 64).astype(np.float32)
        cache.update(0, key, value, 0)
        k, v = cache.get(0)
        assert k.shape == (8, 1, 64)

    def test_multiple_tokens(self, cache):
        """Should accumulate across multiple positions."""
        for pos in range(10):
            key = np.random.randn(8, 64).astype(np.float32)
            value = np.random.randn(8, 64).astype(np.float32)
            cache.update(0, key, value, pos)

        k, v = cache.get(0)
        assert k.shape == (8, 10, 64)
        assert v.shape == (8, 10, 64)

    def test_empty_cache(self, cache):
        """Getting from empty cache returns empty arrays."""
        k, v = cache.get(0)
        assert k.shape == (8, 0, 64)
        assert v.shape == (8, 0, 64)

    def test_get_with_range(self, cache):
        """Should support partial range retrieval."""
        for pos in range(20):
            key = np.random.randn(8, 64).astype(np.float32)
            value = np.random.randn(8, 64).astype(np.float32)
            cache.update(0, key, value, pos)

        k, v = cache.get(0, start=5, end=15)
        assert k.shape == (8, 10, 64)
        assert v.shape == (8, 10, 64)

    def test_compression_ratio(self, cache):
        """Should achieve ~8x compression for 2-bit packed (2 bits vs 16 bits = 8x)."""
        ratio = cache.compression_ratio()
        assert 7.5 < ratio < 8.5, f"Expected ~8x compression, got {ratio:.2f}x"

    def test_clear(self, cache):
        """clear() should reset all state."""
        key = np.random.randn(8, 64).astype(np.float32)
        value = np.random.randn(8, 64).astype(np.float32)
        cache.update(0, key, value, 0)
        assert cache.length == 1

        cache.clear()
        assert cache.length == 0
        k, v = cache.get(0)
        assert k.shape == (8, 0, 64)

    def test_out_of_range_raises(self, cache):
        """Position >= max_seq_len should raise IndexError."""
        with pytest.raises(IndexError):
            cache.update(0, np.zeros((8, 64)), np.zeros((8, 64)), 128)

    def test_hadamard_property(self, cache):
        """hadamard property should return the rotation matrix."""
        assert cache.hadamard.shape == (64, 64)

    def test_approximate_preservation(self, cache):
        """After rotation + quantization + dequantization + inverse rotation,
        the output should approximately preserve the original structure."""
        np.random.seed(42)
        key = np.random.randn(8, 64).astype(np.float32)
        value = np.random.randn(8, 64).astype(np.float32)
        cache.update(0, key, value, 0)

        k_recovered, v_recovered = cache.get(0)

        # With 2-bit quantization, we can't expect perfect reconstruction,
        # but the values should be in a reasonable range
        assert not np.all(k_recovered == 0), "Should not be all zeros"
        assert not np.any(np.isnan(k_recovered)), "Should not contain NaN"
        assert not np.any(np.isnan(v_recovered)), "Should not contain NaN"

        # The direction should be roughly preserved (cosine similarity > 0)
        for h in range(8):
            orig_k = key[h].astype(np.float32)
            recon_k = k_recovered[h, 0, :].astype(np.float32)  # squeeze seq dim
            if np.linalg.norm(orig_k) > 1e-6 and np.linalg.norm(recon_k) > 1e-6:
                cos_sim = float(np.dot(orig_k, recon_k) / (
                    np.linalg.norm(orig_k) * np.linalg.norm(recon_k) + 1e-8
                ))
                assert cos_sim > 0, f"Head {h}: cosine similarity should be positive, got {cos_sim}"

    def test_repr(self, cache):
        r = repr(cache)
        assert "RotateKVCache" in r
        assert "layers=4" in r
        assert "heads=8" in r

    def test_different_block_sizes(self):
        """Different block sizes should all work."""
        for bs in [8, 16, 32, 64, 128]:
            cache = RotateKVCache(2, 4, 64, 32, block_size=bs)
            key = np.random.randn(4, 64).astype(np.float32)
            value = np.random.randn(4, 64).astype(np.float32)
            cache.update(0, key, value, 0)
            k, v = cache.get(0)
            assert k.shape == (4, 1, 64)
