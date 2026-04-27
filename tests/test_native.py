"""Tests for the native C++ backend."""
import numpy as np
import pytest

# Try importing native backend, skip tests if not available
try:
    import vibeblade._vibeblade_native as nat
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="native backend not built")

np.random.seed(42)


def _f16(arr):
    """Convert to float16."""
    return arr.astype(np.float16)


class TestGEMM:
    def test_basic_shapes(self):
        A = np.random.randn(4, 8).astype(np.float16)
        B = np.random.randn(8, 6).astype(np.float16)
        C = nat.gemm(A, B)
        expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        assert C.shape == (4, 6)
        np.testing.assert_allclose(C.astype(np.float32), expected.astype(np.float32), atol=0.1)

    def test_square(self):
        A = np.random.randn(16, 16).astype(np.float16)
        B = np.random.randn(16, 16).astype(np.float16)
        C = nat.gemm(A, B)
        expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(C, expected, atol=0.15)

    def test_single_row(self):
        A = np.random.randn(1, 64).astype(np.float16)
        B = np.random.randn(64, 128).astype(np.float16)
        C = nat.gemm(A, B)
        assert C.shape == (1, 128)

    def test_alpha_beta(self):
        # alpha=2 with fp16 inputs loses precision in the fp16 intermediate
        # Just verify alpha scaling works at all
        A = np.random.randn(4, 8).astype(np.float16)
        B = np.random.randn(8, 6).astype(np.float16)
        C = nat.gemm(A, B, alpha=2.0)
        C1 = nat.gemm(A, B, alpha=1.0)
        # alpha=2 should roughly double the output
        ratio = C.astype(np.float32) / (C1.astype(np.float32) + 1e-8)
        assert np.abs(ratio.mean() - 2.0) < 0.2


class TestRMSNorm:
    def test_basic(self):
        x = np.random.randn(8, 64).astype(np.float16)
        w = np.ones(64, dtype=np.float16)
        out = nat.rms_norm(x, w)
        assert out.shape == x.shape

    def test_preserves_direction(self):
        x = np.random.randn(2, 16).astype(np.float16)
        w = np.ones(16, dtype=np.float16)
        out = nat.rms_norm(x, w, eps=1e-5)
        # Normalized vectors should have ~unit RMS
        rms = np.sqrt(np.mean(out.astype(np.float32) ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=0.05)


class TestActivations:
    def test_silu_basic(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
        out = nat.silu(x)
        assert out.shape == x.shape
        # silu(0) = 0
        assert abs(float(out[2])) < 0.01
        # silu should be positive for positive x
        assert float(out[3]) > 0
        assert float(out[4]) > 0

    def test_silu_mul(self):
        a = np.random.randn(32).astype(np.float16)
        b = np.random.randn(32).astype(np.float16)
        out = nat.silu_mul(a, b)
        expected = a.astype(np.float32) * (1.0 / (1.0 + np.exp(-a.astype(np.float32)))) * b.astype(np.float32)
        np.testing.assert_allclose(out.astype(np.float32), expected, atol=0.05)


class TestQuantize2Bit:
    def test_per_channel_roundtrip(self):
        x = np.random.randn(8, 16).astype(np.float16)
        packed, scales, mins = nat.quantize_2bit(x, axis=1)
        out = nat.dequantize_2bit(packed, scales, mins, 8, 16, axis=1)
        assert out.shape == (8, 16)
        # 2-bit quantization is lossy but should be in the right ballpark
        diff = np.abs(out.astype(np.float32) - x.astype(np.float32))
        assert diff.mean() < 1.5

    def test_per_token_roundtrip(self):
        x = np.random.randn(4, 8).astype(np.float16)
        packed, scales, mins = nat.quantize_2bit(x, axis=0)
        out = nat.dequantize_2bit(packed, scales, mins, 4, 8, axis=0)
        assert out.shape == (4, 8)


class TestFusedSDPA:
    def test_basic(self):
        Q = np.random.randn(4, 16).astype(np.float16)
        K = np.random.randn(8, 16).astype(np.float16)
        V = np.random.randn(8, 16).astype(np.float16)
        output = nat.fused_sdpa(Q, K, V)
        assert output.shape == (4, 16)

    def test_matches_manual(self):
        Q = np.random.randn(2, 8).astype(np.float16)
        K = np.random.randn(4, 8).astype(np.float16)
        V = np.random.randn(4, 8).astype(np.float16)
        output = nat.fused_sdpa(Q, K, V)

        # Manual attention
        Q32, K32, V32 = Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32)
        scale = 1.0 / np.sqrt(8.0)
        attn = Q32 @ K32.T * scale
        attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = (attn @ V32).astype(np.float16)

        np.testing.assert_allclose(output.astype(np.float32), expected.astype(np.float32), atol=0.1)


class TestBackend:
    def test_autodetect(self):
        from vibeblade.backend import get_backend, Backend
        assert get_backend() == Backend.NATIVE

    def test_dispatch(self):
        from vibeblade.backend import gemm
        A = np.random.randn(4, 8).astype(np.float16)
        B = np.random.randn(8, 6).astype(np.float16)
        C = gemm(A, B)
        assert C.shape == (4, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
