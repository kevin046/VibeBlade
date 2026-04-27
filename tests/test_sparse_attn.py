"""Tests for VibeBlade MInference — Dynamic Sparse Attention."""

import numpy as np

from vibeblade.sparse_attn import (
    AttentionPattern,
    assign_pattern,
    generate_a_shape_mask,
    generate_block_sparse_mask,
    generate_vertical_slash_mask,
    MInferenceScheduler,
)


class TestAssignPattern:
    def test_lower_layers(self):
        p = assign_pattern(0, 0, num_layers=12, num_heads=8)
        assert p == AttentionPattern.A_SHAPE

    def test_upper_layers(self):
        p = assign_pattern(11, 0, num_layers=12, num_heads=8)
        assert p == AttentionPattern.BLOCK_SPARSE

    def test_returns_enum(self):
        p = assign_pattern(5, 3, num_layers=12, num_heads=8)
        assert isinstance(p, AttentionPattern)

    def test_all_layers_assigned(self):
        for layer_idx in range(12):
            for h in range(8):
                p = assign_pattern(layer_idx, h, 12, 8)
                assert isinstance(p, AttentionPattern)


class TestAShapeMask:
    def test_shape(self):
        mask = generate_a_shape_mask(32, 4)
        assert mask.shape == (4, 32, 32)
        assert mask.dtype == bool

    def test_has_active_entries(self):
        mask = generate_a_shape_mask(16, 2)
        assert np.any(mask)

    def test_local_window(self):
        seq_len = 20
        mask = generate_a_shape_mask(seq_len, 1, window_size=4)
        # Each position should attend to its 4 recent tokens + self
        for q in range(seq_len):
            active = np.where(mask[0, q])[0]
            expected = list(range(max(0, q - 3), q + 1))
            for pos in expected:
                assert pos in active


class TestBlockSparseMask:
    def test_shape(self):
        mask = generate_block_sparse_mask(32, 4, block_size=8)
        assert mask.shape == (4, 32, 32)

    def test_has_active_entries(self):
        mask = generate_block_sparse_mask(16, 2, block_size=4)
        assert np.any(mask)


class TestVerticalSlashMask:
    def test_shape(self):
        mask = generate_vertical_slash_mask(32, 4)
        assert mask.shape == (4, 32, 32)

    def test_has_active_entries(self):
        mask = generate_vertical_slash_mask(16, 2)
        assert np.any(mask)


class TestMInferenceScheduler:
    def test_init(self):
        sched = MInferenceScheduler(12, 8, 64)
        assert sched.num_layers == 12
        assert sched.num_heads == 8

    def test_get_pattern(self):
        sched = MInferenceScheduler(12, 8)
        for layer_idx in range(12):
            for h in range(8):
                p = sched.get_pattern(layer_idx, h)
                assert isinstance(p, AttentionPattern)

    def test_get_mask(self):
        sched = MInferenceScheduler(6, 4, 32)
        masks = sched.get_mask(16)
        assert isinstance(masks, dict)
        assert len(masks) == 6
        mask = masks[0]
        assert mask.shape == (4, 16, 16)
        assert mask.dtype == bool

    def test_mask_cache(self):
        sched = MInferenceScheduler(4, 2)
        m1 = sched.get_mask(16)
        m2 = sched.get_mask(16)
        assert m1 is m2  # same object reference

    def test_sparsity_ratio(self):
        sched = MInferenceScheduler(12, 8, 64, block_size=32, window_size=8)
        ratio = sched.sparsity_ratio(64)
        # Should achieve significant sparsity
        assert ratio > 0.3

    def test_sparse_attention_output(self):
        sched = MInferenceScheduler(4, 2, 8)
        q = np.random.randn(2, 4, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        mask = sched.get_layer_mask(layer_idx=0, seq_len=4)

        out = sched.sparse_attention(q, k, v, mask, layer_idx=0)
        assert out.shape == (2, 4, 8)
        assert not np.any(np.isnan(out))

    def test_clear_cache(self):
        sched = MInferenceScheduler(4, 2)
        sched.get_mask(16)
        sched.get_mask(32)
        sched.clear_cache()
        assert len(sched._mask_cache) == 0

    def test_repr(self):
        sched = MInferenceScheduler(12, 8)
        r = repr(sched)
        assert "MInferenceScheduler" in r
        assert "12" in r
