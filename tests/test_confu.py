"""Tests for ConFu — contemplate-token speculative decoding."""

import numpy as np

from vibeblade.confu import (
    ContemplateTokenLayer,
    ConFuDraftModel,
    ConFuSpeculator,
    ConFuStats,
)


class TestContemplateTokenLayer:
    """Test contemplate token generation."""

    def test_output_shape_2d(self):
        layer = ContemplateTokenLayer(hidden_dim=256, contemplate_dim=64)
        hidden = np.random.randn(10, 256).astype(np.float32)
        out = layer.forward(hidden)
        assert out.shape == (10, 64)

    def test_output_shape_1d(self):
        layer = ContemplateTokenLayer(hidden_dim=128, contemplate_dim=32)
        hidden = np.random.randn(128).astype(np.float32)
        out = layer.forward(hidden[np.newaxis, :])
        assert out.shape == (1, 32)

    def test_default_contemplate_dim(self):
        layer = ContemplateTokenLayer(hidden_dim=512)
        assert layer.get_contemplate_dim() == 128  # 512 // 4

    def test_reproducible_with_seed(self):
        layer1 = ContemplateTokenLayer(hidden_dim=64, seed=42)
        layer2 = ContemplateTokenLayer(hidden_dim=64, seed=42)
        x = np.random.randn(5, 64).astype(np.float32)
        np.testing.assert_array_almost_equal(layer1.forward(x), layer2.forward(x))

    def test_output_nonzero(self):
        layer = ContemplateTokenLayer(hidden_dim=64, contemplate_dim=16)
        x = np.random.randn(3, 64).astype(np.float32) + 5.0
        out = layer.forward(x)
        assert not np.all(out == 0), "Contemplate tokens should be nonzero for nonzero input"


class TestConFuDraftModel:
    """Test the lightweight draft model."""

    def test_draft_output(self):
        model = ConFuDraftModel(
            hidden_dim=256, num_heads=4, vocab_size=1000, seed=42
        )
        emb = np.random.randn(256).astype(np.float32)
        hidden = np.random.randn(256).astype(np.float32)
        contemplate = np.random.randn(64).astype(np.float32)

        token_id, probs = model.draft(emb, hidden, contemplate)
        assert isinstance(token_id, int)
        assert 0 <= token_id < 1000
        assert probs.shape == (1000,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_draft_without_contemplate(self):
        model = ConFuDraftModel(
            hidden_dim=256, num_heads=4, vocab_size=500, seed=42
        )
        emb = np.random.randn(256).astype(np.float32)
        hidden = np.random.randn(256).astype(np.float32)

        token_id, probs = model.draft(emb, hidden, None)
        assert isinstance(token_id, int)
        assert 0 <= token_id < 500

    def test_generate_contemplate(self):
        model = ConFuDraftModel(
            hidden_dim=256, num_heads=4, vocab_size=1000, seed=42
        )
        hidden = np.random.randn(10, 256).astype(np.float32)
        contemplate = model.generate_contemplate(hidden)
        assert contemplate.shape == (10, 64)

    def test_generate_contemplate_1d(self):
        model = ConFuDraftModel(
            hidden_dim=256, num_heads=4, vocab_size=1000, seed=42
        )
        hidden = np.random.randn(256).astype(np.float32)
        contemplate = model.generate_contemplate(hidden)
        assert contemplate.shape == (64,)


class TestConFuSpeculator:
    """Test the full speculation + verification pipeline."""

    def _make_target_fn(self, vocab_size=1000, hidden_dim=256):
        """Create a mock target model function."""
        rng = np.random.RandomState(42)
        rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.01

        def target_fn(token_ids):
            # Mock: produce logits based on last token embedding
            logits = np.zeros(vocab_size, dtype=np.float32)
            last_token = token_ids[-1] if len(token_ids) > 0 else 0
            logits[last_token % vocab_size] = 2.0  # bias toward last token
            logits[(last_token + 1) % vocab_size] = 1.0  # slight bias toward next
            hidden = np.random.randn(hidden_dim).astype(np.float32) * 0.1
            return logits, hidden

        return target_fn

    def test_speculate_returns_correct_types(self):
        target_fn = self._make_target_fn()
        spec = ConFuSpeculator(
            target_model_fn=target_fn,
            hidden_dim=256,
            num_layers=12,
            speculate_k=5,
            vocab_size=1000,
            seed=42,
        )
        token_ids = np.array([1, 2, 3], dtype=np.int64)
        accepted, mask, stats = spec.speculate(token_ids)

        assert accepted.ndim == 1
        assert mask.ndim == 1
        assert len(mask) == 5
        assert isinstance(stats, ConFuStats)

    def test_stats_tracking(self):
        target_fn = self._make_target_fn()
        spec = ConFuSpeculator(
            target_model_fn=target_fn,
            hidden_dim=256,
            num_layers=12,
            speculate_k=3,
            vocab_size=1000,
            seed=42,
        )

        for _ in range(5):
            token_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
            spec.speculate(token_ids)

        stats = spec.stats
        assert stats.total_drafts == 15  # 5 iterations * 3 drafts
        assert stats.contemplate_token_count == 5
        assert stats.accepted_tokens >= 0
        assert stats.rejected_tokens >= 0

    def test_acceptance_rate(self):
        stats = ConFuStats(total_drafts=100, accepted_tokens=88)
        assert abs(stats.acceptance_rate - 0.88) < 1e-6

    def test_acceptance_rate_zero_drafts(self):
        stats = ConFuStats()
        assert stats.acceptance_rate == 0.0

    def test_speedup_ratio(self):
        stats = ConFuStats(total_drafts=10, accepted_tokens=8)
        # (8 + 1) / (10 + 1) = 9/11 ≈ 0.818
        assert abs(stats.speedup_ratio - 9.0 / 11.0) < 1e-6

    def test_speedup_ratio_zero(self):
        stats = ConFuStats()
        assert stats.speedup_ratio == 1.0

    def test_rejection_rate(self):
        stats = ConFuStats(total_drafts=100, accepted_tokens=75)
        assert abs(stats.rejection_rate - 0.25) < 1e-6

    def test_reset_stats(self):
        target_fn = self._make_target_fn()
        spec = ConFuSpeculator(
            target_model_fn=target_fn,
            hidden_dim=256,
            num_layers=12,
            speculate_k=3,
            vocab_size=1000,
            seed=42,
        )
        spec.speculate(np.array([1, 2, 3]))
        assert spec.stats.total_drafts > 0

        spec.reset_stats()
        assert spec.stats.total_drafts == 0

    def test_single_token_input(self):
        target_fn = self._make_target_fn()
        spec = ConFuSpeculator(
            target_model_fn=target_fn,
            hidden_dim=256,
            num_layers=12,
            speculate_k=2,
            vocab_size=1000,
            seed=42,
        )
        accepted, mask, _ = spec.speculate(np.array([1]))
        assert accepted.ndim == 1
        assert len(mask) == 2

    def test_with_draft_embeddings(self):
        target_fn = self._make_target_fn()
        spec = ConFuSpeculator(
            target_model_fn=target_fn,
            hidden_dim=256,
            num_layers=12,
            speculate_k=3,
            vocab_size=1000,
            seed=42,
        )

        def emb_fn(token_id):
            return np.random.randn(256).astype(np.float32) * 0.1

        accepted, mask, stats = spec.speculate(np.array([1, 2, 3]), emb_fn)
        assert accepted.ndim == 1
        assert stats.total_drafts == 3


class TestConFuStatsEdgeCases:
    """Edge case testing for ConFuStats."""

    def test_all_accepted(self):
        stats = ConFuStats(total_drafts=10, accepted_tokens=10)
        assert stats.acceptance_rate == 1.0
        assert stats.rejection_rate == 0.0

    def test_all_rejected(self):
        stats = ConFuStats(total_drafts=10, accepted_tokens=0, rejected_tokens=10)
        assert stats.acceptance_rate == 0.0
        assert stats.rejection_rate == 1.0
