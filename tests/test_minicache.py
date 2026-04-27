"""Tests for VibeBlade MiniCache — depth-dimension KV cache compression."""

import numpy as np

from vibeblade.minicache import MiniCache
from vibeblade.transformer import build_rope_cache


class TestMiniCacheBasic:
    def test_init(self):
        cache = MiniCache(8, 4, 16, 128, compression_ratio=4)
        assert cache.compression_ratio == 4
        assert cache.memory_savings() > 0

    def test_update_and_get(self):
        cache = MiniCache(6, 2, 8, 32, compression_ratio=2, start_layer=2)
        k = np.random.randn(2, 8).astype(np.float16)
        v = np.random.randn(2, 8).astype(np.float16)

        cache.update(0, k, v, position=0)
        cache.update(3, k, v, position=0)

        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 1, 8)

        k_out3, _ = cache.get(3)
        assert k_out3.shape == (2, 1, 8)

    def test_clear(self):
        cache = MiniCache(4, 2, 8, 64)
        cache.update(0, np.ones((2, 8), dtype=np.float16), np.ones((2, 8), dtype=np.float16), 0)
        cache.clear()
        assert cache.length == 0

    def test_memory_savings(self):
        cache = MiniCache(16, 4, 16, 256, compression_ratio=4)
        # Should save significant memory since only ~6 layers stored (4 uncompressed + 3 reps)
        assert cache.memory_savings() > 0.3


class TestMiniCacheCompression:
    def test_representative_layers_stored(self):
        cache = MiniCache(12, 2, 8, 64, compression_ratio=4, start_layer=4)
        # Layers 0-3 are uncompressed (start_layer=4)
        # Layers 4-11: reps at 4, 8, so 2 representative layers in compressed zone
        # Total reps: 4 + 2 = 6
        assert len(cache._representative_layers) == 6

    def test_different_compression_ratios(self):
        for ratio in [2, 4, 8]:
            cache = MiniCache(32, 4, 16, 128, compression_ratio=ratio, start_layer=8)
            assert cache.memory_savings() > 0
            assert cache.memory_savings() < 0.95

    def test_interpolated_layers_return_values(self):
        cache = MiniCache(8, 2, 8, 32, compression_ratio=4, start_layer=2)
        k = np.random.randn(2, 8).astype(np.float16)
        v = np.random.randn(2, 8).astype(np.float16)

        # Write to a representative layer
        cache.update(4, k, v, position=0)
        # Read from an interpolated layer (same group)
        k_interp, v_interp = cache.get(5)
        assert k_interp.shape == (2, 1, 8)
        # Should have data (interpolated from layer 4 or zero)
        assert k_interp.shape == (2, 1, 8)


class TestMiniCacheBulkLoad:
    def test_bulk_load_populates_cache(self):
        cache = MiniCache(4, 2, 8, 32, compression_ratio=4)
        keys = np.random.randn(2, 5, 8).astype(np.float16)
        values = np.random.randn(2, 5, 8).astype(np.float16)
        cache.bulk_load(0, keys, values, start_pos=0)
        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 5, 8)
        assert v_out.shape == (2, 5, 8)

    def test_bulk_load_with_start_pos(self):
        cache = MiniCache(4, 2, 8, 32, compression_ratio=4)
        keys = np.random.randn(2, 3, 8).astype(np.float16)
        values = np.random.randn(2, 3, 8).astype(np.float16)
        cache.bulk_load(1, keys, values, start_pos=10)
        k_out, _ = cache.get(1)
        assert k_out.shape == (2, 13, 8)

    def test_bulk_load_truncates_at_max_seq(self):
        cache = MiniCache(4, 2, 8, 8, compression_ratio=4)  # max_seq_len=8
        keys = np.random.randn(2, 20, 8).astype(np.float16)
        values = np.random.randn(2, 20, 8).astype(np.float16)
        cache.bulk_load(0, keys, values, start_pos=0)
        k_out, _ = cache.get(0)
        assert k_out.shape == (2, 8, 8)  # truncated


class TestMiniCacheForwardIntegration:
    """Integration tests: MiniCache wired into forward_prefill/forward_decode_single."""

    @staticmethod
    def _make_tiny_model(vocab=64, hidden=32, n_heads=4, n_layers=4, seq=8):
        """Create a tiny model config + random weights for testing."""
        head_dim = hidden // n_heads
        np.random.seed(42)
        weights: dict[str, np.ndarray] = {}
        weights["token_embd.weight"] = np.random.randn(vocab, hidden).astype(np.float32) * 0.1
        weights["output_norm.weight"] = np.ones(hidden, dtype=np.float32)
        weights["output.weight"] = np.random.randn(vocab, hidden).astype(np.float32) * 0.01
        for i in range(n_layers):
            pfx = f"blk.{i}"
            weights[f"{pfx}.attn_norm.weight"] = np.ones(hidden, dtype=np.float32)
            weights[f"{pfx}.ffn_norm.weight"] = np.ones(hidden, dtype=np.float32)
            weights[f"{pfx}.attn_q.weight"] = np.random.randn(n_heads * head_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.attn_k.weight"] = np.random.randn(n_heads * head_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.attn_v.weight"] = np.random.randn(n_heads * head_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.attn_output.weight"] = np.random.randn(hidden, n_heads * head_dim).astype(np.float32) * 0.1
            inter_dim = hidden * 2
            weights[f"{pfx}.ffn_gate.weight"] = np.random.randn(inter_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.ffn_up.weight"] = np.random.randn(inter_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.ffn_down.weight"] = np.random.randn(hidden, inter_dim).astype(np.float32) * 0.1
        cos, sin = build_rope_cache(head_dim, seq)
        return weights, cos, sin, hidden, n_heads, n_heads, seq

    def test_prefill_populates_minicache(self):
        """forward_prefill with minicache should populate the cache."""
        from vibeblade.transformer import forward_prefill
        weights, cos, sin, hidden, n_heads, n_kv_heads, seq = self._make_tiny_model()
        mc = MiniCache(4, n_heads, hidden // n_heads, seq, compression_ratio=2, start_layer=1)
        token_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        logits, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, 4, cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads, minicache=mc,
        )
        assert logits.shape == (4, 64)
        # All layers should have entries in minicache
        for layer_idx in range(4):
            k_out, v_out = mc.get(layer_idx)
            assert k_out.shape == (n_heads, 4, hidden // n_heads)

    def test_decode_uses_minicache(self):
        """forward_decode_single with minicache should use compressed KV."""
        from vibeblade.transformer import forward_prefill, forward_decode_single
        weights, cos, sin, hidden, n_heads, n_kv_heads, seq = self._make_tiny_model(seq=16)
        n_layers = 4
        mc = MiniCache(n_layers, n_heads, hidden // n_heads, seq, compression_ratio=2, start_layer=1)
        token_ids = np.array([1, 2, 3], dtype=np.int32)

        # Prefill (populates minicache)
        logits_prefill, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads, minicache=mc,
        )

        # Decode one token with minicache
        logits, new_kv_k, new_kv_v, stats = forward_decode_single(
            5, 3,
            weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, kv_k, kv_v,
            cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            sparse_ratio=1.0, minicache=mc,
        )
        assert logits.shape == (64,)
        assert "minicache" in stats
        assert stats["minicache"]["layers_compressed"] == n_layers
        assert stats["minicache"]["layers_total"] == n_layers

    def test_decode_with_sparse_and_minicache(self):
        """Both sparse FFN and minicache should work together."""
        from vibeblade.transformer import forward_prefill, forward_decode_single
        weights, cos, sin, hidden, n_heads, n_kv_heads, seq = self._make_tiny_model(seq=16)
        n_layers = 4
        mc = MiniCache(n_layers, n_heads, hidden // n_heads, seq, compression_ratio=4, start_layer=1)
        token_ids = np.array([1, 2], dtype=np.int32)

        logits_prefill, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads, minicache=mc,
        )

        logits, new_kv_k, new_kv_v, stats = forward_decode_single(
            3, 2,
            weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, kv_k, kv_v,
            cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            sparse_ratio=0.5, minicache=mc,
        )
        assert logits.shape == (64,)
        assert stats["mode"] == "powerinfer"
        assert "minicache" in stats
        assert len(stats["layers"]) == n_layers

    def test_decode_without_minicache_unchanged(self):
        """forward_decode_single without minicache should return no minicache stats."""
        from vibeblade.transformer import forward_decode_single
        weights, cos, sin, hidden, n_heads, n_kv_heads, seq = self._make_tiny_model(seq=16)
        n_layers = 2
        # Create dummy KV caches
        head_dim = hidden // n_heads
        kv_k = [np.zeros((n_heads, 1, head_dim), dtype=np.float32) for _ in range(n_layers)]
        kv_v = [np.zeros((n_heads, 1, head_dim), dtype=np.float32) for _ in range(n_layers)]

        logits, _, _, stats = forward_decode_single(
            1, 1,
            weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, kv_k, kv_v,
            cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            sparse_ratio=1.0, minicache=None,
        )
        assert "minicache" not in stats
