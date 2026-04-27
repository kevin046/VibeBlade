"""Tests for vibeblade.sparse — TurboSparse activation sparsity module."""

from __future__ import annotations

import numpy as np
import pytest

from vibeblade.sparse import (
    batch_drelu,
    compute_sparsity,
    drelu_activation,
    predict_activations,
    sparse_ffn_silu,
    sparse_matmul,
    SparsePredictor,
    topk_activation_mask,
)


class TestDreluActivation:
    def test_drelu_basic(self):
        """Positive values pass through; negative values become zero."""
        x = np.array([-2.0, -1.0, 0.0, 0.5, 3.0])
        out = drelu_activation(x)
        expected = np.array([0.0, 0.0, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(out, expected)

    def test_drelu_threshold(self):
        """Custom threshold should only pass values strictly above it."""
        x = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
        out = drelu_activation(x, threshold=1.0)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.5, 2.0])
        np.testing.assert_allclose(out, expected)


class TestPredictActivations:
    def test_predict_activations(self):
        """Mask should be True only where x > threshold."""
        x = np.array([-1.0, 0.0, 0.5, 1.5])
        mask = predict_activations(x)
        expected = np.array([False, False, True, True])
        assert mask.shape == (4,)
        np.testing.assert_array_equal(mask, expected)


class TestComputeSparsity:
    def test_compute_sparsity_all_zero(self):
        """All-zero input should give sparsity 1.0."""
        x = np.zeros(100)
        assert compute_sparsity(x) == pytest.approx(1.0)

    def test_compute_sparsity_all_positive(self):
        """All-positive input should give sparsity 0.0."""
        x = np.ones(100)
        assert compute_sparsity(x) == pytest.approx(0.0)


class TestSparseMatmul:
    def test_sparse_matmul_matches_dense(self):
        """When all neurons are active (mask all True), result == dense matmul."""
        rng = np.random.RandomState(42)
        activations = rng.randn(4, 8)
        weights = rng.randn(8, 6)
        mask = np.ones(8, dtype=bool)

        result = sparse_matmul(activations, weights, mask)
        dense = activations @ weights
        np.testing.assert_allclose(result, dense)

    def test_sparse_matmul_skips_zeros(self):
        """With ~50% mask, sparse result should differ from dense."""
        rng = np.random.RandomState(123)
        activations = rng.randn(4, 8)
        weights = rng.randn(8, 6)

        # alternating mask ≈ 50%
        mask = np.array([True, False, True, False, True, False, True, False])

        sparse_result = sparse_matmul(activations, weights, mask)
        dense_result = activations @ weights

        # The sparse result should not equal the dense result
        assert not np.allclose(sparse_result, dense_result)

        # Verify it matches the manually computed version
        expected = (activations * mask[np.newaxis, :]) @ weights
        np.testing.assert_allclose(sparse_result, expected)


class TestTopkActivationMask:
    def test_topk_activation_mask(self):
        """Each row must have exactly k True entries."""
        rng = np.random.RandomState(7)
        x = rng.randn(5, 10)
        k = 3
        mask = topk_activation_mask(x, k)

        assert mask.shape == (5, 10)
        for row in mask:
            assert np.count_nonzero(row) == k

    def test_topk_activation_mask_keeps_largest(self):
        """The kept entries should correspond to the largest values per row."""
        x = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        mask = topk_activation_mask(x, k=2)

        # Top-2 values are 5.0 and 4.0 (indices 1 and 4)
        expected = np.array([[False, True, False, False, True]])
        np.testing.assert_array_equal(mask, expected)


class TestBatchDrelu:
    def test_batch_drelu_shapes(self):
        """Output shapes must match input for (batch, seq, hidden)."""
        x = np.random.randn(3, 5, 7)
        activated, mask = batch_drelu(x)

        assert activated.shape == (3, 5, 7)
        assert mask.shape == (3, 5, 7)
        assert activated.dtype == np.float64
        assert mask.dtype == bool

    def test_batch_drelu_correctness(self):
        """Values ≤ 0 should be zeroed in activated output."""
        x = np.array([[[ -1.0, 0.0, 2.0]]])  # (1, 1, 3)
        activated, mask = batch_drelu(x)

        expected_act = np.array([[[0.0, 0.0, 2.0]]])
        expected_mask = np.array([[[False, False, True]]])
        np.testing.assert_allclose(activated, expected_act)
        np.testing.assert_array_equal(mask, expected_mask)


class TestSparseFFNSilu:
    """Tests for PowerInfer-style sparse FFN (sparse_ffn_silu)."""

    def test_sparse_ffn_returns_correct_shape(self):
        """Output shape must match dense FFN: (seq, hidden)."""
        rng = np.random.RandomState(42)
        seq, hidden, inter = 2, 64, 128
        x = rng.randn(seq, hidden).astype(np.float32)
        gate_w = rng.randn(inter, hidden).astype(np.float32)
        up_w = rng.randn(inter, hidden).astype(np.float32)
        down_w = rng.randn(hidden, inter).astype(np.float32)

        out, stats = sparse_ffn_silu(x, gate_w, up_w, down_w, sparse_ratio=0.1)
        assert out.shape == (seq, hidden)
        assert "active_neurons" in stats
        assert "total_neurons" in stats
        assert "sparsity_ratio" in stats

    def test_sparse_ratio_controls_active_count(self):
        """Lower sparse_ratio → fewer active neurons → more sparsity."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 64).astype(np.float32)
        gate_w = rng.randn(256, 64).astype(np.float32)
        up_w = rng.randn(256, 64).astype(np.float32)
        down_w = rng.randn(64, 256).astype(np.float32)

        _, stats_10 = sparse_ffn_silu(x, gate_w, up_w, down_w, sparse_ratio=0.1)
        _, stats_50 = sparse_ffn_silu(x, gate_w, up_w, down_w, sparse_ratio=0.5)
        _, stats_100 = sparse_ffn_silu(x, gate_w, up_w, down_w, sparse_ratio=1.0)

        # More restrictive ratio → more sparsity
        assert stats_10["sparsity_ratio"] > stats_50["sparsity_ratio"]
        # 100% = dense, 0 sparsity
        assert stats_100["sparsity_ratio"] == 0.0
        assert stats_100["active_neurons"] == 256

    def test_sparse_ffn_single_token_decode(self):
        """Single token decode (seq=1) should work correctly."""
        rng = np.random.RandomState(7)
        x = rng.randn(1, 32).astype(np.float32)
        gate_w = rng.randn(64, 32).astype(np.float32)
        up_w = rng.randn(64, 32).astype(np.float32)
        down_w = rng.randn(32, 64).astype(np.float32)

        out, stats = sparse_ffn_silu(x, gate_w, up_w, down_w, sparse_ratio=0.25)
        assert out.shape == (1, 32)
        # 25% of 64 = 16 neurons
        assert stats["active_neurons"] == 16

    def test_sparse_ffn_approximates_dense(self):
        """At high sparse_ratio, sparse output should be close to dense.

        At sparse_ratio=1.0, all neurons are kept so results must be identical.
        """
        rng = np.random.RandomState(99)
        x = rng.randn(1, 16).astype(np.float32)
        gate_w = rng.randn(32, 16).astype(np.float32)
        up_w = rng.randn(32, 16).astype(np.float32)
        down_w = rng.randn(16, 32).astype(np.float32)

        # Dense FFN via ffn_silu
        from vibeblade.transformer import ffn_silu
        dense_out = ffn_silu(x, gate_w, up_w, down_w)

        # Sparse FFN at 100% = dense
        sparse_out, stats = sparse_ffn_silu(x, gate_w, up_w, down_w, sparse_ratio=1.0)
        np.testing.assert_allclose(sparse_out, dense_out, rtol=1e-5)


class TestSparsePredictor:
    """Tests for the SparsePredictor (offline calibration mode)."""

    def test_predictor_online_mode(self):
        """Online mode should return a mask without calibration."""
        pred = SparsePredictor(n_layers=2, intermediate_dim=64, sparse_ratio=0.1)
        rng = np.random.RandomState(42)
        x = rng.randn(1, 32).astype(np.float32)
        gate_w = rng.randn(64, 32).astype(np.float32)

        mask = pred.predict(0, x, gate_w)
        assert mask.shape == (64,)
        assert mask.dtype == bool
        assert mask.sum() == 6  # 10% of 64

    def test_predictor_calibration(self):
        """After calibration, should use heavy-hitter sets."""
        pred = SparsePredictor(n_layers=2, intermediate_dim=64, sparse_ratio=0.1)
        rng = np.random.RandomState(42)
        gate_w = rng.randn(64, 32).astype(np.float32)

        # Run some calibration updates
        for _ in range(10):
            x = rng.randn(1, 32).astype(np.float32)
            pred.calibrate_update(0, x, gate_w)

        assert not pred.is_calibrated
        pred.calibrate_finish()
        assert pred.is_calibrated

        # After calibration, predict should use heavy hitters
        mask = pred.predict(0, rng.randn(1, 32).astype(np.float32), gate_w)
        assert mask.shape == (64,)
        assert mask.sum() >= 1

    def test_predictor_saves_compute(self):
        """Predicted mask should be smaller than full intermediate dim."""
        pred = SparsePredictor(n_layers=1, intermediate_dim=1024, sparse_ratio=0.05)
        rng = np.random.RandomState(42)
        x = rng.randn(1, 512).astype(np.float32)
        gate_w = rng.randn(1024, 512).astype(np.float32)

        mask = pred.predict(0, x, gate_w)
        assert mask.sum() <= 102  # 10% of 1024 (5% rounds up via max(1,...))
        assert mask.sum() >= 1


class TestSparseForwardDecodeIntegration:
    """Integration tests: sparse forward_decode_single produces valid logits."""

    def test_sparse_decode_returns_4_values(self):
        """forward_decode_single with sparse_ratio < 1 should return 4 values."""
        from vibeblade.transformer import (
            forward_prefill, forward_decode_single, build_rope_cache,
        )

        rng = np.random.RandomState(42)
        hidden, n_heads, vocab = 64, 4, 100
        inter = 128

        # Build tiny model weights
        weights = {}
        weights["token_emb.weight"] = rng.randn(vocab, hidden).astype(np.float32)
        weights["output_norm.weight"] = np.ones(hidden, dtype=np.float32)
        weights["output.weight"] = rng.randn(vocab, hidden).astype(np.float32) * 0.01
        for suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                       "attn_output.weight", "attn_norm.weight", "ffn_norm.weight"):
            if suffix.endswith("norm.weight"):
                weights[f"blk.0.{suffix}"] = np.ones(hidden, dtype=np.float32)
            elif suffix == "attn_output.weight":
                weights[f"blk.0.{suffix}"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
            elif suffix.startswith("attn_q"):
                weights[f"blk.0.{suffix}"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
            elif suffix.startswith("attn_k") or suffix.startswith("attn_v"):
                weights[f"blk.0.{suffix}"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
            else:
                weights[f"blk.0.{suffix}"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01

        # FFN weights
        weights["blk.0.ffn_gate.weight"] = rng.randn(inter, hidden).astype(np.float32) * 0.01
        weights["blk.0.ffn_up.weight"] = rng.randn(inter, hidden).astype(np.float32) * 0.01
        weights["blk.0.ffn_down.weight"] = rng.randn(hidden, inter).astype(np.float32) * 0.01

        cos, sin = build_rope_cache(hidden // n_heads, 32)

        # Prefill
        prompt = np.array([1, 5, 10])
        logits, kv_k, kv_v = forward_prefill(
            prompt, weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, cos_cache=cos, sin_cache=sin,
            n_heads=n_heads, n_kv_heads=n_heads,
        )

        # Sparse decode
        result = forward_decode_single(
            42, 3, weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, kv_caches_k=kv_k, kv_caches_v=kv_v,
            cos_cache=cos, sin_cache=sin, n_heads=n_heads, n_kv_heads=n_heads,
            sparse_ratio=0.1,
        )

        assert len(result) == 4
        logits, kv_k2, kv_v2, sp_stats = result
        assert logits.shape == (vocab,)
        assert "layers" in sp_stats
        assert sp_stats["layers"][0]["sparsity_ratio"] > 0.0

    def test_dense_decode_still_works(self):
        """forward_decode_single with sparse_ratio=1.0 (dense) returns empty stats."""
        from vibeblade.transformer import (
            forward_prefill, forward_decode_single, build_rope_cache,
        )

        rng = np.random.RandomState(7)
        hidden, n_heads, vocab = 32, 2, 50
        weights = {}
        weights["token_emb.weight"] = rng.randn(vocab, hidden).astype(np.float32)
        weights["output_norm.weight"] = np.ones(hidden, dtype=np.float32)
        weights["output.weight"] = rng.randn(vocab, hidden).astype(np.float32) * 0.01
        weights["blk.0.attn_norm.weight"] = np.ones(hidden, dtype=np.float32)
        weights["blk.0.ffn_norm.weight"] = np.ones(hidden, dtype=np.float32)
        weights["blk.0.attn_q.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
        weights["blk.0.attn_k.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
        weights["blk.0.attn_v.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
        weights["blk.0.attn_output.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
        weights["blk.0.ffn_gate.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
        weights["blk.0.ffn_up.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01
        weights["blk.0.ffn_down.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.01

        cos, sin = build_rope_cache(hidden // n_heads, 16)

        logits_p, kv_k, kv_v = forward_prefill(
            np.array([1]), weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, cos_cache=cos, sin_cache=sin,
            n_heads=n_heads, n_kv_heads=n_heads,
        )

        logits_d, kv_k2, kv_v2, sp_stats = forward_decode_single(
            5, 1, weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, kv_caches_k=kv_k, kv_caches_v=kv_v,
            cos_cache=cos, sin_cache=sin, n_heads=n_heads, n_kv_heads=n_heads,
            sparse_ratio=1.0,  # dense
        )

        assert logits_d.shape == (vocab,)
        assert sp_stats == {}  # no sparse stats when dense

