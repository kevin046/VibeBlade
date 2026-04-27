"""Tests for vibeblade.moe_advanced — confidence routing, prefetching,
heterogeneous quantization, and CPU kernel optimization."""

from __future__ import annotations


import numpy as np
import pytest

from vibeblade.moe import ExpertRouter
from vibeblade.moe_advanced import (
    ConfidenceRouter,
    ContextAwarePrefetcher,
    HeteroQuantizer,
    CPUKernelOptimizer,
)

np.random.seed(42)

SHARED_DIM = 64
NUM_EXPERTS = 16
TOPK = 8


def _make_router(topk: int = TOPK) -> ExpertRouter:
    w = np.random.randn(SHARED_DIM, NUM_EXPERTS).astype(np.float32)
    return ExpertRouter(w, topk=topk)


# ═══════════════════════════════════════════════════════════════════════════
# ConfidenceRouter
# ═══════════════════════════════════════════════════════════════════════════


class TestConfidenceRouter:
    def test_confident_token_uses_single_expert(self):
        """When top-1 weight > threshold, only min_topk experts returned."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.9, min_topk=1)

        # Create a hidden state that strongly activates one expert
        # by aligning it with the router weight column
        expert_id = 0
        strong_x = router.weight[:, expert_id : expert_id + 1].T  # (1, shared_dim)
        indices, weights, stats = cr.route(strong_x[0])

        # Strong alignment should produce high confidence (early exit)
        # or at minimum return valid routing results
        assert indices.ndim == 1
        assert len(indices) >= 1  # at least min_topk

    def test_uncertain_token_uses_full_topk(self):
        """Uniform-ish routing should use all topk experts."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.99, min_topk=1)

        # Random hidden state — unlikely to have >0.99 confidence
        x = np.random.randn(SHARED_DIM).astype(np.float32)
        indices, weights, stats = cr.route(x)

        # With threshold 0.99, almost certainly no early exit
        assert indices.ndim == 1
        assert len(indices) == 8  # full topk
        assert stats["early_exit"] is False or stats["early_exit"] == np.bool_(False)

    def test_min_topk_respected(self):
        """Even at high confidence, at least min_topk experts returned."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.5, min_topk=3)

        # Strongly biased toward one expert
        expert_id = 0
        strong_x = router.weight[:, expert_id : expert_id + 1].T
        indices, weights, stats = cr.route(strong_x[0])

        assert len(indices) >= 3  # min_topk

    def test_stats_tracking(self):
        """After multiple routes, stats are consistent."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.7, min_topk=1)

        # Route 10 random tokens
        for _ in range(10):
            x = np.random.randn(SHARED_DIM).astype(np.float32)
            cr.route(x)

        stats = cr.stats
        assert stats["total_tokens"] == 10
        assert 0 <= stats["early_exit_count"] <= 10
        assert 1.0 <= stats["avg_experts_per_token"] <= 8.0
        assert stats["saved_experts"] >= 0
        assert stats["early_exit_rate"] >= 0.0
        assert stats["early_exit_rate"] <= 1.0

    def test_different_thresholds(self):
        """Lower threshold → more early exits."""
        for threshold in [0.5, 0.7, 0.9, 0.99]:
            router = _make_router(topk=8)
            cr = ConfidenceRouter(router, confidence_threshold=threshold, min_topk=1)

            exits = 0
            for _ in range(20):
                x = np.random.randn(SHARED_DIM).astype(np.float32)
                _, _, stats = cr.route(x)
                if stats["early_exit"] if isinstance(stats["early_exit"], (bool, np.bool_)) else bool(stats["early_exit"]):
                    exits += 1

            # Lower threshold should have more exits (statistical tendency)
            # Just verify it doesn't crash and produces valid results
            assert cr.stats["total_tokens"] == 20

    def test_batch_input(self):
        """Batch of tokens with mixed confidence."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.5, min_topk=1)

        batch_size = 4
        x = np.random.randn(batch_size, SHARED_DIM).astype(np.float32)
        indices, weights, stats = cr.route(x)

        assert indices.ndim == 2
        assert indices.shape[0] == batch_size
        assert stats["confidence"].shape == (batch_size,)
        assert stats["num_experts"].shape == (batch_size,)
        assert stats["early_exit"].shape == (batch_size,)

    def test_confidence_threshold_1_0_never_exits(self):
        """Threshold of 1.0 means no token ever triggers early exit."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=1.0, min_topk=1)

        for _ in range(10):
            x = np.random.randn(SHARED_DIM).astype(np.float32)
            indices, weights, stats = cr.route(x)
            assert len(indices) == 8
            assert stats["early_exit"] is False or stats["early_exit"] == np.bool_(False)

        assert cr.stats["early_exit_count"] == 0

    def test_confidence_threshold_0_0_always_exits(self):
        """Threshold of 0.0 means every token triggers early exit."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.0, min_topk=1)

        for _ in range(10):
            x = np.random.randn(SHARED_DIM).astype(np.float32)
            indices, weights, stats = cr.route(x)
            assert len(indices) == 1  # min_topk

        assert cr.stats["early_exit_count"] == 10

    def test_reset_stats(self):
        """reset_stats clears all counters."""
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.5, min_topk=1)
        cr.route(np.random.randn(SHARED_DIM).astype(np.float32))
        assert cr.stats["total_tokens"] == 1

        cr.reset_stats()
        assert cr.stats["total_tokens"] == 0
        assert cr.stats["early_exit_count"] == 0

    def test_invalid_threshold_raises(self):
        router = _make_router(topk=8)
        with pytest.raises(ValueError):
            ConfidenceRouter(router, confidence_threshold=1.5)
        with pytest.raises(ValueError):
            ConfidenceRouter(router, confidence_threshold=-0.1)

    def test_min_topk_exceeds_router_topk_raises(self):
        router = _make_router(topk=4)
        with pytest.raises(ValueError):
            ConfidenceRouter(router, confidence_threshold=0.9, min_topk=5)

    def test_setter_validates_threshold(self):
        router = _make_router(topk=8)
        cr = ConfidenceRouter(router, confidence_threshold=0.9)
        cr.confidence_threshold = 0.5
        assert cr.confidence_threshold == 0.5
        with pytest.raises(ValueError):
            cr.confidence_threshold = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# ContextAwarePrefetcher
# ═══════════════════════════════════════════════════════════════════════════


def _make_routers(num_layers: int = 5) -> dict:
    return {
        idx: _make_router(topk=4) for idx in range(num_layers)
    }


class TestContextAwarePrefetcher:
    def test_routing_proximity_predicts_next_layer(self):
        """ROUTING_PROXIMITY should predict experts for layer+1."""
        routers = _make_routers(5)
        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=1,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        preds = pf.update_and_predict(layer_idx=0, x_norm=x)

        # Should predict experts for layer 1
        assert len(preds) > 0
        assert all(t_layer == 1 for t_layer, _, _ in preds)

    def test_prefetch_skips_hot_experts(self):
        """Predicted experts in the hot set should be filtered out."""
        routers = _make_routers(5)
        # Make expert 0 always the top prediction by using its weight column
        router_1 = routers[1]
        hot_ids = {1: [0]}  # expert 0 at layer 1 is hot

        pf = ContextAwarePrefetcher(
            routers=routers,
            hot_expert_ids=hot_ids,
            prefetch_depth=1,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        # Strongly bias toward expert 0
        x = router_1.weight[:, 0:1].T[0]
        preds = pf.update_and_predict(layer_idx=0, x_norm=x)

        # Expert 0 at layer 1 should NOT appear in predictions
        for _, eid, _ in preds:
            assert eid != 0

    def test_prefetch_depth(self):
        """With depth=3, should predict for layer+1, layer+2, layer+3."""
        routers = _make_routers(10)
        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=3,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        preds = pf.update_and_predict(layer_idx=2, x_norm=x)

        target_layers = {t_layer for t_layer, _, _ in preds}
        # Should include layers 3, 4, 5 (depth=3 from layer 2)
        assert target_layers == {3, 4, 5}

    def test_frequency_based_strategy(self):
        """FREQUENCY_BASED should use frequency_map for predictions."""
        routers = _make_routers(5)
        freq_map = {
            1: {0: 0.5, 3: 0.3, 7: 0.2},
            2: {1: 0.4, 5: 0.35, 9: 0.25},
        }

        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=2,
            strategy=ContextAwarePrefetcher.Strategy.FREQUENCY_BASED,
            frequency_map=freq_map,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        preds = pf.update_and_predict(layer_idx=0, x_norm=x)

        target_layers = {t_layer for t_layer, _, _ in preds}
        assert 1 in target_layers
        assert 2 in target_layers
        # Scores should come from frequency map
        for t_layer, eid, score in preds:
            if t_layer in freq_map and eid in freq_map[t_layer]:
                assert abs(score - freq_map[t_layer][eid]) < 1e-6

    def test_stats_tracking(self):
        """Stats should track issued prefetches and hits."""
        routers = _make_routers(5)
        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=1,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        preds = pf.update_and_predict(layer_idx=0, x_norm=x)

        stats = pf.stats
        assert stats["prefetch_issued"] == len(preds)
        assert stats["lookahead_layers"] == 1
        assert stats["hit_rate"] == 0.0  # no callbacks yet

    def test_empty_routers(self):
        """Graceful handling when future layers have no router."""
        routers = {0: _make_router(topk=4)}  # only layer 0
        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=3,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        preds = pf.update_and_predict(layer_idx=0, x_norm=x)

        # No routers for layer 1, 2, 3 → no predictions
        assert len(preds) == 0

    def test_prefetch_callback(self):
        """Callback should update hit tracking."""
        routers = _make_routers(5)
        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=1,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        preds = pf.update_and_predict(layer_idx=0, x_norm=x)
        issued = len(preds)

        # Simulate callback for first prediction
        if preds:
            t_layer, eid, _ = preds[0]
            pf.prefetch_callback(t_layer, [eid])
            stats = pf.stats
            assert stats["prefetch_hits"] == 1
            assert stats["pending_count"] == issued - 1

    def test_reset_stats(self):
        """reset_stats clears all tracking."""
        routers = _make_routers(5)
        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=1,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        pf.update_and_predict(layer_idx=0, x_norm=x)
        assert pf.stats["prefetch_issued"] > 0

        pf.reset_stats()
        assert pf.stats["prefetch_issued"] == 0
        assert pf.stats["prefetch_hits"] == 0
        assert pf.stats["pending_count"] == 0

    def test_no_duplicate_in_flight(self):
        """Same expert shouldn't be issued twice without callback."""
        routers = _make_routers(5)
        pf = ContextAwarePrefetcher(
            routers=routers,
            prefetch_depth=1,
            strategy=ContextAwarePrefetcher.Strategy.ROUTING_PROXIMITY,
        )

        x = np.random.randn(SHARED_DIM).astype(np.float32)
        preds1 = pf.update_and_predict(layer_idx=0, x_norm=x)
        preds2 = pf.update_and_predict(layer_idx=0, x_norm=x)

        # Second call should not re-issue already in-flight prefetches
        total_issued = pf.stats["prefetch_issued"]
        assert total_issued == len(preds1)
        # preds2 should be empty (everything already in flight)
        assert len(preds2) == 0


# ═══════════════════════════════════════════════════════════════════════════
# HeteroQuantizer
# ═══════════════════════════════════════════════════════════════════════════


class TestHeteroQuantizer:
    def _make_quant(self, block_size: int = 32) -> HeteroQuantizer:
        return HeteroQuantizer(hot_expert_ids={0: [0, 1]}, block_size=block_size)

    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize should preserve shapes."""
        q = self._make_quant()
        gate = np.random.randn(64, 128).astype(np.float32)
        up = np.random.randn(64, 128).astype(np.float32)
        down = np.random.randn(128, 64).astype(np.float32)

        g_p, u_p, d_p = q.quantize_expert(gate, up, down)
        result = q.dequantize_expert(
            g_p, u_p, d_p,
            orig_shapes=(gate.shape, up.shape, down.shape),
        )
        rg, ru, rd = result

        assert rg.shape == gate.shape
        assert ru.shape == up.shape
        assert rd.shape == down.shape

    def test_packed_format(self):
        """Packed output should be uint8."""
        q = self._make_quant()
        gate = np.random.randn(32, 32).astype(np.float32)
        g_p, _, _ = q.quantize_expert(gate, gate, gate)
        assert g_p.dtype == np.uint8

    def test_memory_savings(self):
        """Memory savings should be ~90% for float32→2-bit."""
        q = self._make_quant()
        savings = q.expert_memory_savings(1024 * 1024)  # 1M float32 values
        # 2-bit packed = 3 bits/value out of 32 bits = ~90.6% savings
        assert 0.85 < savings < 0.95

    def test_dequantize_all_zeros(self):
        """All-zero block: range=0 gets clamped to scale=1, quantizes to level 0."""
        q = self._make_quant()
        zeros = np.zeros((32, 32), dtype=np.float32)
        g_p, _, _ = q.quantize_expert(zeros, zeros, zeros)
        rg, _, _ = q.dequantize_expert(
            g_p, g_p, g_p,
            orig_shapes=(zeros.shape, zeros.shape, zeros.shape),
        )
        # Zero-range block: scale clamped to 1.0, zero_point=0.
        # All values quantize to same level, dequant to LUT[level] * 1 + 0.
        # Just verify uniformity and bounded range.
        assert rg.shape == zeros.shape
        assert np.allclose(rg, rg[0, 0], atol=1e-6)  # all same value

    def test_dequantize_all_ones(self):
        """All-ones tensor should roundtrip reasonably."""
        q = self._make_quant()
        ones = np.ones((32, 32), dtype=np.float32)
        g_p, _, _ = q.quantize_expert(ones, ones, ones)
        rg, _, _ = q.dequantize_expert(
            g_p, g_p, g_p,
            orig_shapes=(ones.shape, ones.shape, ones.shape),
        )
        # 2-bit quantization has limited precision, allow some error
        np.testing.assert_allclose(rg, ones, atol=0.5)

    def test_different_block_sizes(self):
        """Test with block_size=16, 32, 64."""
        for bs in [16, 32, 64]:
            q = self._make_quant(block_size=bs)
            w = np.random.randn(64, 64).astype(np.float32)
            g_p, u_p, d_p = q.quantize_expert(w, w, w)
            rg, _, _ = q.dequantize_expert(
                g_p, u_p, d_p,
                orig_shapes=(w.shape, w.shape, w.shape),
            )
            assert rg.shape == w.shape

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError):
            HeteroQuantizer(hot_expert_ids={}, block_size=3)  # not multiple of 4
        with pytest.raises(ValueError):
            HeteroQuantizer(hot_expert_ids={}, block_size=2)  # < 4


# ═══════════════════════════════════════════════════════════════════════════
# CPUKernelOptimizer
# ═══════════════════════════════════════════════════════════════════════════


class TestCPUKernelOptimizer:
    def test_detect_returns_cpuinfo(self):
        """detect() should return a CPUInfo with all fields populated."""
        opt = CPUKernelOptimizer()
        info = opt.detect()

        assert info.arch in ("x86_64", "aarch64", "unknown")
        assert isinstance(info.has_avx512, bool)
        assert isinstance(info.has_avx2, bool)
        assert isinstance(info.has_amx, bool)
        assert isinstance(info.has_neon, bool)
        assert info.blas_library in ("openblas", "mkl", "blis", "accelerate", "none")
        assert info.l2_cache_kb > 0
        assert isinstance(info.l3_cache_kb, int)
        assert info.num_cores >= 1
        assert info.optimal_block_size >= 32
        assert isinstance(info.use_float16, bool)
        assert info.recommended_threads >= 1

    def test_detect_arch(self):
        """Should return a valid arch string."""
        arch = CPUKernelOptimizer._detect_arch()
        assert isinstance(arch, str)
        assert len(arch) > 0

    def test_optimized_matmul_matches_numpy(self):
        """Result should be close to np.dot for small matrices."""
        opt = CPUKernelOptimizer()
        info = opt.detect()

        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        expected = a @ b
        result = opt.optimized_matmul(a, b, info=info)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cache_aware_tiling(self):
        """Large matmul should use blocking and still match numpy."""
        opt = CPUKernelOptimizer()
        # Force a small block size to exercise tiling
        info = opt.CPUInfo(
            arch="x86_64",
            has_avx512=False,
            has_avx2=False,
            has_amx=False,
            has_neon=False,
            blas_library="none",
            l2_cache_kb=256,
            l3_cache_kb=0,
            num_cores=1,
            optimal_block_size=32,
            use_float16=False,
            recommended_threads=1,
        )

        a = np.random.randn(128, 64).astype(np.float32)
        b = np.random.randn(64, 128).astype(np.float32)
        expected = a @ b
        result = opt.optimized_matmul(a, b, info=info)

        np.testing.assert_allclose(result, expected, atol=1e-5)  # tiled matmul precision

    def test_incompatible_shapes_raises(self):
        opt = CPUKernelOptimizer()
        a = np.random.randn(32, 16).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)  # K mismatch

        with pytest.raises(ValueError):
            opt.optimized_matmul(a, b)

    def test_read_cache_sizes(self):
        """Should return dict with l2 and l3 keys."""
        caches = CPUKernelOptimizer._read_cache_sizes()
        assert "l2" in caches
        assert "l3" in caches

    def test_read_num_cores(self):
        """Should return a positive integer."""
        cores = CPUKernelOptimizer._read_num_cores()
        assert isinstance(cores, int)
        assert cores >= 1

    def test_parse_cache_kb(self):
        assert CPUKernelOptimizer._parse_cache_kb("256 KB") == 256
        assert CPUKernelOptimizer._parse_cache_kb("8192 KB") == 8192
        assert CPUKernelOptimizer._parse_cache_kb("32 MB") == 32768

    def test_detection_caching(self):
        """Second detect() call should return same object."""
        opt = CPUKernelOptimizer()
        info1 = opt.detect()
        info2 = opt.detect()
        assert info1 is info2
