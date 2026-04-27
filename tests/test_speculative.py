"""Tests for VibeBlade EAGLE Speculative Decoding."""

import numpy as np

from vibeblade.speculative import (
    EAGLEDraftHead,
    EAGLEDecoder,
    SpeculativeVerifier,
)


class TestEAGLEDraftHead:
    def test_init(self):
        head = EAGLEDraftHead(64, 1000, num_heads=4, max_draft_tokens=3)
        assert head.vocab_size == 1000
        assert head.num_heads == 4

    def test_draft_single(self):
        head = EAGLEDraftHead(64, 100, num_heads=1, max_draft_tokens=5)
        hidden = np.random.randn(64).astype(np.float32)
        tokens = head.draft_single(hidden)
        assert len(tokens) == 5
        assert all(0 <= t < 100 for t in tokens)

    def test_draft_tokens_multiple_heads(self):
        head = EAGLEDraftHead(32, 50, num_heads=3, max_draft_tokens=4)
        hidden = np.random.randn(32).astype(np.float32)
        sequences = head.draft_tokens(hidden)
        assert len(sequences) == 3
        for seq in sequences:
            assert len(seq) == 4

    def test_extract_features(self):
        head = EAGLEDraftHead(64, 100)
        hidden = np.random.randn(64).astype(np.float32)
        features = head.extract_features(hidden)
        assert features.shape == (64,)

    def test_extract_features_with_prev_emb(self):
        head = EAGLEDraftHead(64, 100)
        hidden = np.random.randn(64).astype(np.float32)
        prev = np.random.randn(64).astype(np.float32)
        features = head.extract_features(hidden, prev_token_emb=prev)
        assert features.shape == (64,)


class TestSpeculativeVerifier:
    def test_greedy_accept_correct(self):
        draft = [1, 2, 3]
        target_logits = [
            np.array([-10, 100, -10, -10], dtype=np.float32),  # argmax=1
            np.array([-10, -10, 100, -10], dtype=np.float32),  # argmax=2
            np.array([-10, -10, -10, 100], dtype=np.float32),  # argmax=3
        ]
        draft_logits = target_logits[:]
        accepted, all_ok, n = SpeculativeVerifier.verify(draft, target_logits, draft_logits, temperature=0)
        assert all_ok
        assert n == 3
        assert accepted == [1, 2, 3]

    def test_greedy_reject_wrong(self):
        draft = [1, 2, 3]
        target_logits = [
            np.array([-10, 100, -10, -10], dtype=np.float32),  # argmax=1 -> accept
            np.array([100, -10, -10, -10], dtype=np.float32),  # argmax=0 -> reject
        ]
        draft_logits = target_logits[:]
        accepted, all_ok, n = SpeculativeVerifier.verify(draft, target_logits, draft_logits, temperature=0)
        assert not all_ok
        assert n == 1

    def test_empty_draft(self):
        accepted, all_ok, n = SpeculativeVerifier.verify([], [], [], temperature=0)
        assert accepted == []
        assert all_ok

    def test_stochastic_verification(self):
        """Temperature > 0 should use rejection sampling, not always deterministic."""
        np.random.seed(42)
        draft = [5, 10, 15]
        target_logits = [
            np.random.randn(50).astype(np.float32) for _ in range(3)
        ]
        draft_logits = [
            np.random.randn(50).astype(np.float32) for _ in range(3)
        ]
        accepted, _, n = SpeculativeVerifier.verify(draft, target_logits, draft_logits, temperature=1.0)
        # Should accept some but maybe not all
        assert 0 < n <= 3


class TestEAGLEDecoder:
    def test_init(self):
        head = EAGLEDraftHead(64, 100)
        decoder = EAGLEDecoder(head)
        assert decoder.acceptance_rate == 0.0

    def test_speculate_step(self):
        head = EAGLEDraftHead(64, 100, num_heads=1, max_draft_tokens=3)
        decoder = EAGLEDecoder(head, temperature=1.0)

        hidden = np.random.randn(64).astype(np.float32)

        # Mock target model
        def target_fn(tokens):
            vocab_size = 100
            logits = [np.random.randn(vocab_size).astype(np.float32) for _ in tokens]
            new_hidden = np.random.randn(64).astype(np.float32)
            return logits, new_hidden, np.random.randn(64).astype(np.float32)

        accepted, new_hidden = decoder.speculate_step(hidden, None, target_fn)
        assert isinstance(accepted, list)
        assert new_hidden.shape == (64,)
