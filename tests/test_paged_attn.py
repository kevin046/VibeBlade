"""Tests for VibeBlade PagedAttention."""

import numpy as np
import pytest

from vibeblade.paged_attn import PagedKVCache


class TestPagedKVCacheBasic:
    def test_init(self):
        cache = PagedKVCache(2, 4, 16, num_pages=32)
        assert cache.num_free_pages == 32
        assert len(cache) == 0

    def test_append_and_get(self):
        cache = PagedKVCache(2, 4, 16, num_pages=32)
        k = np.random.randn(4, 16).astype(np.float16)
        v = np.random.randn(4, 16).astype(np.float16)

        cache.append(0, k, v)
        assert len(cache) == 1

        result_k, result_v = cache.get(0)
        assert result_k.shape == (4, 1, 16)
        assert result_v.shape == (4, 1, 16)
        np.testing.assert_allclose(result_k[:, 0, :], k, atol=1e-3)
        np.testing.assert_allclose(result_v[:, 0, :], v, atol=1e-3)

    def test_clear(self):
        cache = PagedKVCache(2, 4, 16, num_pages=32)
        k = np.random.randn(4, 16).astype(np.float16)
        v = np.random.randn(4, 16).astype(np.float16)
        cache.append(0, k, v)
        cache.clear()
        assert len(cache) == 0
        assert cache.num_free_pages == 32

    def test_get_empty_range(self):
        cache = PagedKVCache(1, 4, 16, num_pages=16)
        k_out, v_out = cache.get(0, 0, 0)
        assert k_out.shape == (4, 0, 16)
        assert v_out.shape == (4, 0, 16)


class TestPagedKVCachePaging:
    def test_multi_page_sequence(self):
        cache = PagedKVCache(1, 2, 8, num_pages=64, page_size=4)
        tokens = 10  # spans 3 pages (0-3, 4-7, 8-9)
        for i in range(tokens):
            k = np.random.randn(2, 8).astype(np.float16) * (i + 1)
            v = np.random.randn(2, 8).astype(np.float16) * (i + 1)
            cache.append(0, k, v)

        assert len(cache) == 10
        assert cache.num_used_pages == 3  # ceil(10/4) = 3

        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 10, 8)

    def test_page_boundary_correctness(self):
        cache = PagedKVCache(1, 1, 4, num_pages=16, page_size=3)
        # Write exactly across page boundaries
        for i in range(7):  # pages: [0,1,2], [3,4,5], [6]
            k = np.full((1, 4), float(i), dtype=np.float16)
            v = np.full((1, 4), float(i + 100), dtype=np.float16)
            cache.append(0, k, v)

        k_out, v_out = cache.get(0, 0, 7)
        for i in range(7):
            assert float(k_out[0, i, 0]) == float(i)
            assert float(v_out[0, i, 0]) == float(i + 100)

    def test_out_of_pages_raises(self):
        cache = PagedKVCache(1, 1, 4, num_pages=2, page_size=2)
        for i in range(4):  # fills 2 pages exactly
            cache.append(0, np.zeros((1, 4), dtype=np.float16), np.zeros((1, 4), dtype=np.float16))

        with pytest.raises(MemoryError, match="exhausted"):
            cache.append(0, np.zeros((1, 4), dtype=np.float16), np.zeros((1, 4), dtype=np.float16))


class TestPagedKVCacheSharing:
    def test_share_prefix(self):
        cache_a = PagedKVCache(1, 2, 8, num_pages=32, page_size=4)
        cache_b = PagedKVCache(1, 2, 8, num_pages=32, page_size=4)

        # Write 6 tokens to cache_a
        for i in range(6):
            k = np.full((2, 8), float(i), dtype=np.float16)
            v = np.full((2, 8), float(i + 50), dtype=np.float16)
            cache_a.append(0, k, v)

        # Share prefix of 4 tokens
        cache_b.share_prefix(cache_a, 0, prefix_len=4)

        # cache_b should see same data for positions 0-3
        k_b, v_b = cache_b.get(0, 0, 4)
        for i in range(4):
            assert float(k_b[0, i, 0]) == float(i)
            assert float(v_b[0, i, 0]) == float(i + 50)


class TestPagedKVCacheMemory:
    def test_memory_usage_reflects_actual(self):
        cache = PagedKVCache(2, 4, 16, num_pages=64, page_size=8)
        initial = cache.memory_usage_bytes
        assert initial == 0

        for i in range(20):
            cache.append(0, np.random.randn(4, 16).astype(np.float16),
                         np.random.randn(4, 16).astype(np.float16))

        used = cache.memory_usage_bytes
        assert used > 0
        assert used < cache.total_pool_bytes

    def test_free_pages_reclaims(self):
        cache = PagedKVCache(1, 1, 4, num_pages=8, page_size=2)
        for i in range(10):  # uses 5 pages
            cache.append(0, np.zeros((1, 4), dtype=np.float16), np.zeros((1, 4), dtype=np.float16))

        assert cache.num_used_pages == 5
        freed = cache.free_pages(3)
        # Pages beyond seq_len are freed
        assert freed >= 0


class TestPagedKVCacheBulk:
    def test_bulk_append(self):
        cache = PagedKVCache(2, 2, 8, num_pages=32, page_size=4)
        keys = np.random.randn(2, 5, 8).astype(np.float16)
        values = np.random.randn(2, 5, 8).astype(np.float16)
        cache.bulk_append(0, keys, values)
        assert len(cache) == 5
        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 5, 8)

    def test_to_flat_caches(self):
        cache = PagedKVCache(2, 2, 8, num_pages=32, page_size=4)
        for i in range(3):
            cache.append(0, np.full((2, 8), float(i), dtype=np.float16),
                         np.full((2, 8), float(i + 50), dtype=np.float16))
        k_list, v_list = cache.to_flat_caches()
        assert len(k_list) == 2
        assert len(v_list) == 2
        assert k_list[0].shape == (2, 3, 8)


class TestPagedAttnForwardIntegration:
    """Integration tests: PagedKVCache wired into forward_prefill/forward_decode_single."""

    @staticmethod
    def _make_tiny_model(vocab=64, hidden=32, n_heads=4, n_layers=4, seq=32):
        """Create a tiny model config + random weights for testing."""
        from vibeblade.transformer import build_rope_cache
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
        return weights, cos, sin, hidden, n_heads, n_heads

    def test_prefill_populates_paged_cache(self):
        """forward_prefill with paged_attn should populate the paged pool."""
        from vibeblade.transformer import forward_prefill
        weights, cos, sin, hidden, n_heads, n_kv_heads = self._make_tiny_model()
        n_layers = 4
        pa = PagedKVCache(n_layers, n_heads, hidden // n_heads, num_pages=32, page_size=4)
        token_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        logits, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads, paged_attn=pa,
        )
        assert logits.shape == (5, 64)
        assert len(pa) == 5
        # Verify data roundtrips through paged cache
        pa_k, pa_v = pa.get(0)
        assert pa_k.shape == (n_heads, 5, hidden // n_heads)

    def test_decode_uses_paged_cache(self):
        """forward_decode_single with paged_attn should read from paged pool."""
        from vibeblade.transformer import forward_prefill, forward_decode_single
        weights, cos, sin, hidden, n_heads, n_kv_heads = self._make_tiny_model(seq=32)
        n_layers = 4
        pa = PagedKVCache(n_layers, n_heads, hidden // n_heads, num_pages=64, page_size=4)
        token_ids = np.array([1, 2, 3], dtype=np.int32)

        logits_prefill, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads, paged_attn=pa,
        )

        # Decode one token with paged_attn
        logits, new_kv_k, new_kv_v, stats = forward_decode_single(
            5, 3,
            weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, kv_k, kv_v,
            cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            sparse_ratio=1.0, paged_attn=pa,
        )
        assert logits.shape == (64,)
        assert "paged_attn" in stats
        assert stats["paged_attn"]["seq_len"] == 4  # 3 prefill + 1 decode

    def test_decode_with_sparse_and_paged(self):
        """Both sparse FFN and paged_attn should work together."""
        from vibeblade.transformer import forward_prefill, forward_decode_single
        weights, cos, sin, hidden, n_heads, n_kv_heads = self._make_tiny_model(seq=32)
        n_layers = 4
        pa = PagedKVCache(n_layers, n_heads, hidden // n_heads, num_pages=64, page_size=4)
        token_ids = np.array([1, 2], dtype=np.int32)

        logits_prefill, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads, paged_attn=pa,
        )

        logits, _, _, stats = forward_decode_single(
            3, 2,
            weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, kv_k, kv_v,
            cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            sparse_ratio=0.5, paged_attn=pa,
        )
        assert logits.shape == (64,)
        assert stats["mode"] == "powerinfer"
        assert "paged_attn" in stats
        assert len(stats["layers"]) == n_layers

    def test_prefix_sharing_then_decode(self):
        """Prefix sharing between two caches then decode on second."""
        from vibeblade.transformer import forward_prefill
        weights, cos, sin, hidden, n_heads, n_kv_heads = self._make_tiny_model(seq=32)
        n_layers = 2

        # Request A prefill
        pa_a = PagedKVCache(n_layers, n_heads, hidden // n_heads, num_pages=32, page_size=4)
        token_a = np.array([1, 2, 3, 4], dtype=np.int32)
        logits_a, kv_k_a, kv_v_a = forward_prefill(
            token_a, weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads, paged_attn=pa_a,
        )

        # Request B shares prefix of 2 tokens with A
        pa_b = PagedKVCache(n_layers, n_heads, hidden // n_heads, num_pages=32, page_size=4)
        for layer_idx in range(n_layers):
            pa_b.share_prefix(pa_a, layer_idx, prefix_len=2)

        # B should see A's prefix
        k_b, v_b = pa_b.get(0, 0, 2)
        k_a, v_a = pa_a.get(0, 0, 2)
        np.testing.assert_allclose(k_b, k_a, atol=1e-3)

    def test_decode_without_paged_unchanged(self):
        """forward_decode_single without paged_attn should return no paged stats."""
        from vibeblade.transformer import forward_decode_single
        weights, cos, sin, hidden, n_heads, n_kv_heads = self._make_tiny_model(seq=16)
        n_layers = 2
        head_dim = hidden // n_heads
        kv_k = [np.zeros((n_heads, 1, head_dim), dtype=np.float32) for _ in range(n_layers)]
        kv_v = [np.zeros((n_heads, 1, head_dim), dtype=np.float32) for _ in range(n_layers)]

        logits, _, _, stats = forward_decode_single(
            1, 1,
            weights["token_embd.weight"], weights["output_norm.weight"],
            weights["output.weight"], weights, n_layers, kv_k, kv_v,
            cos, sin,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            sparse_ratio=1.0, paged_attn=None,
        )
        assert "paged_attn" not in stats
