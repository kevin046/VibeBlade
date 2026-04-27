"""Tests for VibeBlade Transformer components.

Tests the real forward pass: rms_norm, rope, attention, ffn_silu,
forward_prefill, and forward_decode_single using actual weight tensors.
"""

import numpy as np

from vibeblade.transformer import (
    rms_norm,
    rope,
    build_rope_cache,
    attention,
    ffn_silu,
    forward_token,
    forward_prefill,
    forward_decode_single,
)


# --- RMSNorm ---

class TestRMSNorm:
    def test_rms_norm_output_shape(self):
        weight = np.ones(64, dtype=np.float32)
        x = np.random.randn(3, 5, 64).astype(np.float32)
        out = rms_norm(x, weight)
        assert out.shape == x.shape

    def test_rms_norm_preserves_direction(self):
        """Scaling a vector shouldn't change its direction (only scale)."""
        weight = np.ones(32, dtype=np.float32)
        vec = np.random.randn(32).astype(np.float32)
        scaled = 5.0 * vec
        out1 = rms_norm(vec[np.newaxis, :], weight)
        out2 = rms_norm(scaled[np.newaxis, :], weight)
        cos_sim = np.dot(out1[0], out2[0]) / (
            np.linalg.norm(out1[0]) * np.linalg.norm(out2[0]) + 1e-8
        )
        assert np.allclose(cos_sim, 1.0, atol=1e-5)


# --- RoPE ---

class TestRotaryEmbedding:
    def test_rotary_embedding_shape(self):
        cos, sin = build_rope_cache(32, 64)
        x = np.random.randn(4, 32).astype(np.float32)
        out = rope(x, cos[:1], sin[:1])
        assert out.shape == x.shape

    def test_rotary_embedding_different_positions(self):
        cos, sin = build_rope_cache(32, 64)
        x = np.random.randn(1, 32).astype(np.float32)
        out0 = rope(x, cos[0:1], sin[0:1])
        out5 = rope(x, cos[5:6], sin[5:6])
        assert not np.allclose(out0, out5)

    def test_rope_cache_shape(self):
        cos, sin = build_rope_cache(16, 128, 500000.0)
        assert cos.shape == (128, 8)  # (max_seq, head_dim/2)
        assert sin.shape == (128, 8)


# --- Attention ---

class TestAttention:
    def test_attention_output_shape(self):
        np.random.seed(42)
        seq, hidden, n_heads = 5, 64, 4
        hidden // n_heads
        q = np.random.randn(seq, hidden).astype(np.float32)
        k = np.random.randn(seq, hidden).astype(np.float32)
        v = np.random.randn(seq, hidden).astype(np.float32)
        out = attention(q, k, v, n_heads=n_heads, n_kv_heads=n_heads)
        assert out.shape == (seq, hidden)

    def test_attention_gqa(self):
        """Grouped-query attention: fewer KV heads than query heads."""
        np.random.seed(42)
        seq = 4
        n_heads, n_kv_heads = 4, 2
        head_dim = 16
        q = np.random.randn(seq, n_heads * head_dim).astype(np.float32)
        k = np.random.randn(seq, n_kv_heads * head_dim).astype(np.float32)
        v = np.random.randn(seq, n_kv_heads * head_dim).astype(np.float32)
        out = attention(q, k, v, n_heads=n_heads, n_kv_heads=n_kv_heads)
        assert out.shape == (seq, n_heads * head_dim)


# --- FeedForward ---

class TestFeedForward:
    def test_feed_forward_output_shape(self):
        np.random.seed(42)
        x = np.random.randn(5, 64).astype(np.float32)
        gate_w = np.random.randn(128, 64).astype(np.float32) * 0.02
        up_w = np.random.randn(128, 64).astype(np.float32) * 0.02
        down_w = np.random.randn(64, 128).astype(np.float32) * 0.02
        out = ffn_silu(x, gate_w, up_w, down_w)
        assert out.shape == (5, 64)

    def test_feed_forward_silu_nonlinear(self):
        """FFN should be nonlinear (different from a pure linear transform)."""
        np.random.seed(42)
        gate_w = np.random.randn(64, 32).astype(np.float32) * 0.02
        up_w = np.random.randn(64, 32).astype(np.float32) * 0.02
        down_w = np.random.randn(32, 64).astype(np.float32) * 0.02
        x1 = np.random.randn(32).astype(np.float32)
        x2 = 2.0 * x1
        out1 = ffn_silu(x1[np.newaxis, :], gate_w, up_w, down_w)[0]
        out2 = ffn_silu(x2[np.newaxis, :], gate_w, up_w, down_w)[0]
        assert not np.allclose(out2, 2.0 * out1, atol=1e-6)

    def test_feed_forward_sparse(self):
        """With a sparsity mask, some neurons are zeroed out."""
        np.random.seed(42)
        x = np.random.randn(1, 64).astype(np.float32)
        gate_w = np.random.randn(128, 64).astype(np.float32) * 0.02
        up_w = np.random.randn(128, 64).astype(np.float32) * 0.02
        down_w = np.random.randn(64, 128).astype(np.float32) * 0.02
        mask = np.zeros(128, dtype=bool)
        mask[:32] = True  # only 25% active
        out_full = ffn_silu(x, gate_w, up_w, down_w, sparse_mask=None)
        out_sparse = ffn_silu(x, gate_w, up_w, down_w, sparse_mask=mask)
        # Sparse output should be different (not just scaled version)
        assert not np.allclose(out_full, out_sparse)


# --- forward_token (single layer) ---

class TestForwardToken:
    def test_forward_token_shape(self):
        np.random.seed(42)
        hidden, n_heads, n_kv_heads = 64, 4, 4
        head_dim = hidden // n_heads
        inter_dim = 128

        x = np.random.randn(1, hidden).astype(np.float32) * 0.02
        weights = {}
        for suffix, shape in [
            ("attn_norm.weight", (hidden,)),
            ("ffn_norm.weight", (hidden,)),
            ("attn_q.weight", (hidden, hidden)),
            ("attn_k.weight", (hidden, hidden)),
            ("attn_v.weight", (hidden, hidden)),
            ("attn_output.weight", (hidden, hidden)),
            ("ffn_gate.weight", (inter_dim, hidden)),
            ("ffn_up.weight", (inter_dim, hidden)),
            ("ffn_down.weight", (hidden, inter_dim)),
        ]:
            weights[f"blk.0.{suffix}"] = np.random.randn(*shape).astype(np.float32) * 0.02

        cos, sin = build_rope_cache(head_dim, 32)
        out, kv_k, kv_v = forward_token(
            x, weights, 0, cos, sin, None, None,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
        )
        assert out.shape == (1, hidden)
        assert kv_k.shape == (n_kv_heads, 1, head_dim)
        assert kv_v.shape == (n_kv_heads, 1, head_dim)


# --- Full prefill + decode ---

def _make_tiny_model(vocab=100, hidden=64, n_heads=4, n_layers=1, inter=128):
    """Create a tiny model's weight dict for testing."""
    np.random.seed(42)
    hidden // n_heads
    weights = {}

    # Embeddings
    weights["token_emb.weight"] = np.random.randn(vocab, hidden).astype(np.float32) * 0.02
    weights["output_norm.weight"] = np.ones(hidden, dtype=np.float32)
    weights["output.weight"] = np.random.randn(vocab, hidden).astype(np.float32) * 0.02

    for layer in range(n_layers):
        pfx = f"blk.{layer}"
        weights[f"{pfx}.attn_norm.weight"] = np.ones(hidden, dtype=np.float32)
        weights[f"{pfx}.ffn_norm.weight"] = np.ones(hidden, dtype=np.float32)
        weights[f"{pfx}.attn_q.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        weights[f"{pfx}.attn_k.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        weights[f"{pfx}.attn_v.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        weights[f"{pfx}.attn_output.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        weights[f"{pfx}.ffn_gate.weight"] = np.random.randn(inter, hidden).astype(np.float32) * 0.02
        weights[f"{pfx}.ffn_up.weight"] = np.random.randn(inter, hidden).astype(np.float32) * 0.02
        weights[f"{pfx}.ffn_down.weight"] = np.random.randn(hidden, inter).astype(np.float32) * 0.02

    return weights


class TestForward:
    def test_prefill_logits_shape(self):
        weights = _make_tiny_model()
        cos, sin = build_rope_cache(16, 32)
        token_ids = np.array([1, 5, 10, 20])
        logits, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, cos_cache=cos, sin_cache=sin,
            n_heads=4, n_kv_heads=4,
        )
        assert logits.shape == (4, 100)
        assert len(kv_k) == 1
        assert len(kv_v) == 1

    def test_decode_single_shape(self):
        weights = _make_tiny_model()
        cos, sin = build_rope_cache(16, 32)
        token_ids = np.array([1, 5, 10, 20])

        # Prefill first
        logits, kv_k, kv_v = forward_prefill(
            token_ids, weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, cos_cache=cos, sin_cache=sin,
            n_heads=4, n_kv_heads=4,
        )

        # Decode one token (forward_decode_single returns 4 values: logits, kv_k, kv_v, sparse_stats)
        logits2, kv_k2, kv_v2, _sp = forward_decode_single(
            42, 4, weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, kv_caches_k=kv_k, kv_caches_v=kv_v,
            cos_cache=cos, sin_cache=sin, n_heads=4, n_kv_heads=4,
        )
        assert logits2.shape == (100,)  # (vocab_size,)
        assert kv_k2[0].shape[1] == 5  # 4 prompt + 1 new token

    def test_decode_deterministic(self):
        """Same input → same output (no randomness in forward pass)."""
        weights = _make_tiny_model()
        cos, sin = build_rope_cache(16, 32)

        logits1, _, _ = forward_prefill(
            np.array([1, 3, 7]), weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, cos_cache=cos, sin_cache=sin,
            n_heads=4, n_kv_heads=4,
        )
        logits2, _, _ = forward_prefill(
            np.array([1, 3, 7]), weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, cos_cache=cos, sin_cache=sin,
            n_heads=4, n_kv_heads=4,
        )
        assert np.allclose(logits1, logits2)

    def test_forward_single_token(self):
        weights = _make_tiny_model()
        cos, sin = build_rope_cache(16, 32)
        token_ids = np.array([5])
        logits, _, _ = forward_prefill(
            token_ids, weights["token_emb.weight"],
            weights["output_norm.weight"], weights["output.weight"],
            weights, n_layers=1, cos_cache=cos, sin_cache=sin,
            n_heads=4, n_kv_heads=4,
        )
        assert logits.shape == (1, 100)
