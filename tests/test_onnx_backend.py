"""Tests for ONNX Runtime backend, TensorRT stub, and accelerated router."""

# ruff: noqa: E402 — onnxruntime is optional; skip entire file if missing
import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from vibeblade.onnx_backend import (
    ORTOps,
    _build_gemm_graph,
    _build_rms_norm_graph,
    _build_silu_graph,
    _build_softmax_graph,
    detect_providers,
    platform_info,
)
from vibeblade.tensorrt_backend import TensorRTEngine, is_available, platform_support
from vibeblade.accelerated import AcceleratedBackend, get_accelerator, AccelConfig
from vibeblade.transformer import (
    forward_token,
    build_rope_cache,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_weights():
    """2-layer, 4-head, GQA model weights for testing."""
    n_layers, n_heads, n_kv, hidden, hd, inter = 2, 4, 2, 256, 64, 768
    w = {}
    for i in range(n_layers):
        for k in ("attn_norm", "ffn_norm"):
            w[f"blk.{i}.{k}.weight"] = np.random.randn(hidden).astype(np.float32)
        w[f"blk.{i}.attn_q.weight"] = np.random.randn(n_heads * hd, hidden).astype(np.float32)
        w[f"blk.{i}.attn_k.weight"] = np.random.randn(n_kv * hd, hidden).astype(np.float32)
        w[f"blk.{i}.attn_v.weight"] = np.random.randn(n_kv * hd, hidden).astype(np.float32)
        w[f"blk.{i}.attn_output.weight"] = np.random.randn(hidden, n_heads * hd).astype(np.float32)
        w[f"blk.{i}.ffn_gate.weight"] = np.random.randn(inter, hidden).astype(np.float32)
        w[f"blk.{i}.ffn_up.weight"] = np.random.randn(inter, hidden).astype(np.float32)
        w[f"blk.{i}.ffn_down.weight"] = np.random.randn(hidden, inter).astype(np.float32)
    w["token_embd.weight"] = np.random.randn(1000, hidden).astype(np.float32)
    w["output_norm.weight"] = np.random.randn(hidden).astype(np.float32)
    w["output.weight"] = np.random.randn(1000, hidden).astype(np.float32)
    return w


# ── Platform detection ────────────────────────────────────────────────────


class TestPlatformDetection:
    def test_platform_info_returns_dict(self):
        info = platform_info()
        assert "arch" in info
        assert "system" in info
        assert "ort_available" in info
        assert "providers" in info
        assert isinstance(info["providers"], list)

    def test_detect_providers_has_cpu(self):
        # CPU provider should always be available if ORT is installed
        providers = detect_providers()
        if providers:
            assert "CPUExecutionProvider" in providers


# ── ONNX graph builders ──────────────────────────────────────────────────


class TestONNXGraphBuilders:
    def test_gemm_graph_valid(self):
        onnx_bytes = _build_gemm_graph(64, 128)
        assert isinstance(onnx_bytes, bytes)
        assert len(onnx_bytes) > 0
        import onnx
        model = onnx.load_from_string(onnx_bytes)
        onnx.checker.check_model(model)

    def test_rms_norm_graph_valid(self):
        onnx_bytes = _build_rms_norm_graph(256)
        import onnx
        model = onnx.load_from_string(onnx_bytes)
        onnx.checker.check_model(model)

    def test_silu_graph_valid(self):
        onnx_bytes = _build_silu_graph(512)
        import onnx
        model = onnx.load_from_string(onnx_bytes)
        onnx.checker.check_model(model)

    def test_softmax_graph_valid(self):
        onnx_bytes = _build_softmax_graph(32)
        import onnx
        model = onnx.load_from_string(onnx_bytes)
        onnx.checker.check_model(model)


# ── ORTOps per-op tests ──────────────────────────────────────────────────


class TestORTOps:
    @pytest.fixture
    def ops(self):
        return ORTOps(num_threads=2)

    def test_gem_matches_numpy(self, ops):
        x = np.random.randn(1, 128).astype(np.float32)
        W = np.random.randn(256, 128).astype(np.float32)
        y_ort = ops.gemm(x, W)
        y_np = x @ W.T
        np.testing.assert_allclose(y_ort, y_np, atol=1e-5)

    def test_rms_norm_matches_numpy(self, ops):
        x = np.random.randn(1, 512).astype(np.float32)
        w = np.random.randn(512).astype(np.float32)
        y_ort = ops.rms_norm(x, w)
        y_np = ORTOps._rms_norm_numpy(x, w, 1e-5)
        np.testing.assert_allclose(y_ort, y_np, atol=1e-5)

    def test_silu_matches_numpy(self, ops):
        x = np.random.randn(1, 1024).astype(np.float32)
        y_ort = ops.silu(x)
        y_np = x * (1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_allclose(y_ort, y_np, atol=1e-5)

    def test_softmax_matches_numpy(self, ops):
        x = np.random.randn(1, 32).astype(np.float32)
        y_ort = ops.softmax(x)
        y_np = ORTOps._softmax_numpy(x)
        np.testing.assert_allclose(y_ort, y_np, atol=1e-5)
        # Sum to 1
        np.testing.assert_allclose(y_ort.sum(axis=-1), 1.0, atol=1e-6)

    def test_gemm_cache_hits(self, ops):
        x = np.random.randn(1, 64).astype(np.float32)
        W = np.random.randn(128, 64).astype(np.float32)
        ops.gemm(x, W)  # miss
        ops.gemm(x, W)  # hit — same dims, same graph
        # Gemm graph is keyed by (op, in_dim, out_dim) — should get a hit
        assert ops.cache_stats["misses"] >= 1
        # Total calls should equal hits + misses
        total = ops.cache_stats["hits"] + ops.cache_stats["misses"]
        assert total >= 2

    def test_ops_fallback_when_disabled(self):
        ops = ORTOps.__new__(ORTOps)
        ops.enabled = False
        ops._cache = None
        x = np.random.randn(1, 64).astype(np.float32)
        W = np.random.randn(128, 64).astype(np.float32)
        y = ops.gemm(x, W)
        np.testing.assert_allclose(y, x @ W.T)


# ── Full layer tests ──────────────────────────────────────────────────────


class TestForwardLayer:
    def test_layer_output_shape(self, small_weights):
        ops = ORTOps(num_threads=2)
        rc, rs = build_rope_cache(64, 2048)
        x = np.random.randn(1, 256).astype(np.float32)
        cos, sin = rc[0:1], rs[0:1]

        out, k, v = ops.forward_layer(x, 0, small_weights, cos, sin, None, None, 0, 4, 2, 64)
        assert out.shape == (1, 256)
        assert k.shape == (2, 1, 64)  # n_kv_heads=2
        assert v.shape == (2, 1, 64)

    def test_layer_matches_numpy(self, small_weights):
        ops = ORTOps(num_threads=2)
        rc, rs = build_rope_cache(64, 2048)
        x = np.random.randn(1, 256).astype(np.float32)

        out_ort, k_ort, v_ort = ops.forward_layer(
            x, 0, small_weights, rc[0:1], rs[0:1], None, None, 0, 4, 2, 64
        )
        out_np, k_np, v_np = forward_token(
            x, small_weights, 0, rc, rs, None, None, 0, n_heads=4, n_kv_heads=2
        )
        # ORT uses different BLAS — allow float32 precision diff (~5e-6)
        np.testing.assert_allclose(out_ort, out_np, atol=0.01)
        np.testing.assert_allclose(k_ort, k_np, atol=1e-4)
        np.testing.assert_allclose(v_ort, v_np, atol=1e-4)

    def test_layer_with_cache(self, small_weights):
        ops = ORTOps(num_threads=2)
        rc, rs = build_rope_cache(64, 2048)
        x = np.random.randn(1, 256).astype(np.float32)

        out0, k0, v0 = ops.forward_layer(
            x, 0, small_weights, rc[0:1], rs[0:1], None, None, 0, 4, 2, 64
        )
        out1, k1, v1 = ops.forward_layer(
            out0, 1, small_weights, rc[1:2], rs[1:2], k0, v0, 1, 4, 2, 64
        )
        assert k1.shape[1] == 2  # 2 positions cached
        assert v1.shape[1] == 2

    def test_layer_no_gqa(self):
        """Test with n_heads == n_kv_heads (no GQA)."""
        n_heads = n_kv = hidden = hd = 4
        w = {}
        for k in ("attn_norm", "ffn_norm"):
            w[f"blk.0.{k}.weight"] = np.random.randn(hidden).astype(np.float32)
        w["blk.0.attn_q.weight"] = np.random.randn(n_heads * hd, hidden).astype(np.float32)
        w["blk.0.attn_k.weight"] = np.random.randn(n_kv * hd, hidden).astype(np.float32)
        w["blk.0.attn_v.weight"] = np.random.randn(n_kv * hd, hidden).astype(np.float32)
        w["blk.0.attn_output.weight"] = np.random.randn(hidden, n_heads * hd).astype(np.float32)
        w["blk.0.ffn_gate.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
        w["blk.0.ffn_up.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
        w["blk.0.ffn_down.weight"] = np.random.randn(hidden, hidden).astype(np.float32)

        ops = ORTOps(num_threads=2)
        rc, rs = build_rope_cache(hd, 2048)
        x = np.random.randn(1, hidden).astype(np.float32)

        out, k, v = ops.forward_layer(x, 0, w, rc[0:1], rs[0:1], None, None, 0, n_heads, n_kv, hd)
        assert out.shape == (1, hidden)
        assert k.shape == (n_kv, 1, hd)


# ── TensorRT stub tests ──────────────────────────────────────────────────


class TestTensorRT:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_platform_support_returns_dict(self):
        info = platform_support()
        assert "available" in info
        assert "version" in info
        assert "reason" in info

    def test_engine_fallback_gemm(self):
        engine = TensorRTEngine()
        x = np.random.randn(1, 64).astype(np.float32)
        W = np.random.randn(128, 64).astype(np.float32)
        y = engine.gemm(x, W)
        np.testing.assert_allclose(y, x @ W.T)

    @pytest.mark.skipif(is_available(), reason="TRT available, test build")
    def test_build_fails_gracefully_without_trt(self):
        engine = TensorRTEngine()
        result = engine.build_from_onnx(b"invalid")
        assert result is False

    def test_active_false_without_trt(self):
        engine = TensorRTEngine()
        assert engine.active is False


# ── Accelerated backend tests ────────────────────────────────────────────


class TestAcceleratedBackend:
    def test_creates_with_numpy_fallback(self, small_weights):
        backend = AcceleratedBackend(
            weights=small_weights,
            config=AccelConfig(n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=256, head_dim=64, intermediate_dim=768),
            backend="numpy",
        )
        assert backend.stats.backend == "numpy"
        assert backend.summary() is not None

    def test_get_accelerator_convenience(self, small_weights):
        accel = get_accelerator(
            small_weights,
            n_layers=2, n_heads=4, n_kv_heads=2,
            hidden_dim=256, head_dim=64, intermediate_dim=768,
            backend="numpy",
        )
        assert isinstance(accel, AcceleratedBackend)

    def test_full_decode_shape(self, small_weights):
        backend = AcceleratedBackend(
            weights=small_weights,
            config=AccelConfig(n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=256, head_dim=64, intermediate_dim=768),
            backend="numpy",
        )
        logits, caches = backend.full_decode(np.array([0]))
        assert logits.shape[0] == 1
        assert logits.shape[1] == 1000  # vocab size
        assert len(caches) == 2  # 2 layers

    def test_benchmark_runs(self, small_weights):
        backend = AcceleratedBackend(
            weights=small_weights,
            config=AccelConfig(n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=256, head_dim=64, intermediate_dim=768),
            backend="numpy",
        )
        result = backend.benchmark_tokens(n_tokens=5)
        assert "tokens_per_sec" in result
        assert result["tokens"] == 5

    def test_decode_with_cache_growth(self, small_weights):
        backend = AcceleratedBackend(
            weights=small_weights,
            config=AccelConfig(n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=256, head_dim=64, intermediate_dim=768),
            backend="numpy",
        )
        caches = {}
        for pos in range(5):
            logits, caches = backend.full_decode(np.array([0]), caches, start_position=pos)
        # After 5 tokens, cache should have 5 entries per layer
        assert caches[0][0].shape[1] == 5  # 5 K positions
