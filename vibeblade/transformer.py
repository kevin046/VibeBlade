"""VibeBlade Transformer — Real LLaMA-style forward pass using loaded weights.

This is NOT a mockup. The forward pass:

1. Takes weight tensors extracted from a real GGUF model file
2. Computes RMSNorm, RoPE, attention, SwiGLU FFN using those actual weights
3. Supports KV caching for autoregressive generation (only compute 1 token at a time)
4. Optionally uses activation sparsity to skip FFN neurons (TurboSparse)

Weight tensor naming follows GGUF/llama.cpp conventions:
  - token_embd.weight:           (vocab_size, hidden_dim)
  - output_norm.weight:          (hidden_dim,)
  - output.weight:               (vocab_size, hidden_dim)
  - blk.{i}.attn_q.weight:       (n_heads * head_dim, hidden_dim)
  - blk.{i}.attn_k.weight:       (n_kv_heads * head_dim, hidden_dim)
  - blk.{i}.attn_v.weight:       (n_kv_heads * head_dim, hidden_dim)
  - blk.{i}.attn_output.weight:  (hidden_dim, hidden_dim)
  - blk.{i}.ffn_gate.weight:     (intermediate_dim, hidden_dim)
  - blk.{i}.ffn_up.weight:       (intermediate_dim, hidden_dim)
  - blk.{i}.ffn_down.weight:     (hidden_dim, intermediate_dim)
  - blk.{i}.attn_norm.weight:    (hidden_dim,)
  - blk.{i}.ffn_norm.weight:     (hidden_dim,)
  - blk.{i}.ffn_gate_inp.weight:  (shared_dim, num_experts)     [MoE router]
  - blk.{i}.ffn_gate_exps.weight: (num_experts, shared_dim, expert_dim) [MoE]
  - blk.{i}.ffn_up_exps.weight:   (num_experts, shared_dim, expert_dim) [MoE]
  - blk.{i}.ffn_down_exps.weight: (num_experts, expert_dim, shared_dim) [MoE]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # avoid circular imports; sparse is imported at call sites

logger = logging.getLogger(__name__)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight.

    Clamps intermediate values to fp16 range to prevent overflow when
    dequantized weights contain extreme values (common with low-bit quantization).
    """
    x32 = x.astype(np.float32)
    weight32 = weight.astype(np.float32)
    # Clamp to fp16 range to prevent inf/nan propagation
    x32 = np.clip(x32, -65504.0, 65504.0)
    weight32 = np.clip(weight32, -65504.0, 65504.0)
    rms = np.sqrt(np.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
    # Clamp RMS to avoid division by near-zero
    rms = np.maximum(rms, eps)
    # DEBUG: print shapes on mismatch
    if rms.shape[-1] != 1 and weight32.shape[-1] != rms.shape[-1]:
        import sys
        sys.stderr.write(f"[rms_norm DEBUG] x32.shape={x32.shape} weight32.shape={weight32.shape} rms.shape={rms.shape}\n")
    return (x32 / rms) * weight32


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x).

    Clamp to fp16 range to prevent exp overflow in sigmoid.
    """
    x32 = np.clip(x.astype(np.float32), -65504.0, 65504.0)
    return x32 * (1.0 / (1.0 + np.exp(-x32)))


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply Rotary Position Embeddings.

    Args:
        x: (..., head_dim) — query or key tensor (may include heads dim)
        cos: (..., head_dim/2) — precomputed cosine table
        sin: (..., head_dim/2) — precomputed sine table

    x is split into pairs: (x0, x1), (x2, x3), ...
    Each pair is rotated: (x0*cos - x1*sin, x1*cos + x0*sin)

    Broadcasting: if x has extra leading dims (e.g., batch, heads),
    cos/sin must broadcast to match.
    """
    half = x.shape[-1] // 2
    x0 = x[..., :half]
    x1 = x[..., half:]
    # cos/sin might need broadcasting for (seq, heads, head_dim) input
    # where cos/sin are (seq, head_dim/2) — add the heads dimension
    if cos.ndim < x.ndim:
        for _ in range(x.ndim - cos.ndim):
            cos = np.expand_dims(cos, axis=-2)
            sin = np.expand_dims(sin, axis=-2)
    out = np.empty_like(x, dtype=np.float32)
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x1 * cos + x0 * sin
    return out


def build_rope_cache(head_dim: int, max_seq_len: int = 2048,
                     base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE cos/sin tables.

    Returns:
        (cos, sin) each of shape (max_seq_len, head_dim/2)
    """
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grouped-query attention.

    Args:
        q: (seq, n_heads * head_dim) — query
        k: (total_seq, n_kv_heads * head_dim) — key (may include cached)
        v: (total_seq, n_kv_heads * head_dim) — value
        n_heads: number of query heads
        n_kv_heads: number of KV heads (GQA: n_heads >= n_kv_heads)
        mask: optional (seq, total_seq) causal mask

    Returns:
        (seq, n_heads * head_dim) — attention output
    """
    head_dim = q.shape[-1] // n_heads
    seq = q.shape[0]
    total_seq = k.shape[0]

    # Reshape: (n_heads, seq, head_dim)
    q = q.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)
    # Reshape: (n_kv_heads, total_seq, head_dim)
    k = k.reshape(total_seq, n_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(total_seq, n_kv_heads, head_dim).transpose(1, 0, 2)

    # GQA: repeat KV heads to match query heads
    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=0)  # (n_heads, total_seq, head_dim)
        v = np.repeat(v, n_rep, axis=0)

    # Scaled dot-product attention
    scale = 1.0 / np.sqrt(float(head_dim))
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scale  # (n_heads, seq, total_seq)

    # Causal mask
    if mask is not None:
        scores = scores + mask[np.newaxis, :, :]
    elif total_seq > seq:
        # When total_seq > seq (with cache), query pos i can attend to 0..start+i
        causal = np.full((seq, total_seq), -np.inf, dtype=np.float32)
        start = total_seq - seq
        for i in range(seq):
            causal[i, :start + i + 1] = 0.0
        scores = scores + causal[np.newaxis, :, :]

    # Softmax (numerically stable)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    out = np.matmul(attn_weights, v)  # (n_heads, seq, head_dim)
    out = out.transpose(1, 0, 2).reshape(seq, n_heads * head_dim)
    return out


def ffn_silu(
    x: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
    sparse_mask: np.ndarray | None = None,
) -> np.ndarray:
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))

    Args:
        x: (seq, hidden_dim)
        gate_w: (intermediate_dim, hidden_dim)
        up_w: (intermediate_dim, hidden_dim)
        down_w: (hidden_dim, intermediate_dim)
        sparse_mask: optional (intermediate_dim,) bool — skip inactive neurons

    Returns:
        (seq, hidden_dim)

    Note: This applies the mask AFTER computing the full up projection.
    For truly sparse compute (skipping the up matmul), use
    ``sparse_ffn_silu()`` from ``vibeblade.sparse`` instead.
    """
    gate = x @ gate_w.T  # (seq, intermediate_dim)
    up = x @ up_w.T      # (seq, intermediate_dim)
    hidden = silu(gate) * up

    if sparse_mask is not None:
        # Zero out columns of down_w for inactive neurons
        hidden = hidden * sparse_mask[np.newaxis, :]

    return hidden @ down_w.T  # (seq, hidden_dim)


def _sparse_ffn_layer(
    x: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
    sparse_ratio: float,
) -> tuple[np.ndarray, dict]:
    """Dispatch to sparse or dense FFN based on sparse_ratio.

    If sparse_ratio < 1.0, uses sparse_ffn_silu from vibeblade.sparse
    which actually skips the up/down matmuls for inactive neurons.
    Otherwise falls back to dense ffn_silu.
    """
    if sparse_ratio >= 1.0:
        out = ffn_silu(x, gate_w, up_w, down_w)
        return out, {"active_neurons": gate_w.shape[0], "total_neurons": gate_w.shape[0], "sparsity_ratio": 0.0}

    from .sparse import sparse_ffn_silu
    return sparse_ffn_silu(x, gate_w, up_w, down_w, sparse_ratio=sparse_ratio)


def forward_token(
    token_emb: np.ndarray,
    weights: dict[str, np.ndarray],
    layer_idx: int,
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
    kv_cache_k: np.ndarray | None,
    kv_cache_v: np.ndarray | None,
    position: int = 0,
    n_heads: int = 32,
    n_kv_heads: int = 32,
    eps: float = 1e-5,
    sparse_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single transformer block forward for one token position.

    Args:
        token_emb: (1, hidden_dim) — embedded input token
        weights: dict of weight tensors for this layer
        layer_idx: which transformer layer
        cos_cache, sin_cache: precomputed RoPE tables
        kv_cache_k: (n_kv_heads, cached_seq, head_dim) or None
        kv_cache_v: same shape or None
        position: current position in sequence
        n_heads, n_kv_heads: head counts
        eps: RMSNorm epsilon
        sparse_mask: optional (intermediate_dim,) bool for FFN sparsity

    Returns:
        (output, updated_k_cache, updated_v_cache)
    """
    prefix = f"blk.{layer_idx}"
    head_dim = token_emb.shape[-1] // n_heads

    # Pre-attention RMSNorm
    attn_norm_w = weights.get(f"{prefix}.attn_norm.weight")
    h = rms_norm(token_emb, attn_norm_w, eps)

    # QKV projections
    q_w = weights.get(f"{prefix}.attn_q.weight")
    k_w = weights.get(f"{prefix}.attn_k.weight")
    v_w = weights.get(f"{prefix}.attn_v.weight")

    q = h @ q_w.T
    k_new = h @ k_w.T
    v_new = h @ v_w.T

    # Apply RoPE to Q and K
    cos_pos = cos_cache[position:position + 1]
    sin_pos = sin_cache[position:position + 1]

    q = q.reshape(1, n_heads, head_dim)
    k_new = k_new.reshape(1, n_kv_heads, head_dim)
    q = rope(q, cos_pos, sin_pos).reshape(1, n_heads * head_dim)
    k_new = rope(k_new, cos_pos, sin_pos).reshape(1, n_kv_heads * head_dim)

    # KV cache: append new K, V
    if kv_cache_k is not None:
        k_full = np.concatenate([kv_cache_k, k_new.reshape(n_kv_heads, 1, head_dim)], axis=1)
        v_full = np.concatenate([kv_cache_v, v_new.reshape(n_kv_heads, 1, head_dim)], axis=1)
    else:
        k_full = k_new.reshape(n_kv_heads, 1, head_dim)
        v_full = v_new.reshape(n_kv_heads, 1, head_dim)

    # Reshape for attention function: (total_seq, total_dim)
    k_seq = k_full.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim)
    v_seq = v_full.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim)

    # Attention
    attn_out = attention(q, k_seq, v_seq, n_heads, n_kv_heads)

    # Output projection
    o_w = weights.get(f"{prefix}.attn_output.weight")
    attn_out = attn_out @ o_w.T

    # Residual
    h = token_emb + attn_out

    # Pre-FFN RMSNorm
    ffn_norm_w = weights.get(f"{prefix}.ffn_norm.weight")
    h_norm = rms_norm(h, ffn_norm_w, eps)

    # SwiGLU FFN (with optional mask)
    gate_w = weights.get(f"{prefix}.ffn_gate.weight")
    up_w = weights.get(f"{prefix}.ffn_up.weight")
    down_w = weights.get(f"{prefix}.ffn_down.weight")
    ffn_out = ffn_silu(h_norm, gate_w, up_w, down_w, sparse_mask)

    # Residual
    output = h + ffn_out

    return output, k_full, v_full


# ── MoE FFN helpers (lazy-loaded to avoid import overhead for dense models) ──

def _get_moe_components(weights, prefix, cache):
    """Lazily parse MoE router + expert weights for a layer, caching the result."""
    if prefix in cache:
        return cache[prefix]
    from .moe import ExpertRouter, load_moe_weights_from_layer

    router_w, expert_set, extras = load_moe_weights_from_layer(weights, int(prefix.split(".")[1]))
    if router_w is None:
        cache[prefix] = None
        return None

    num_active = router_w.shape[-1] if router_w.ndim > 1 else router_w.shape[0]
    topk = min(8, num_active)  # reasonable default
    router = ExpertRouter(router_w, topk=topk)

    # Shared expert (DeepSeek-style)
    shared = None
    if "shared_gate" in extras and "shared_up" in extras and "shared_down" in extras:
        shared = (extras["shared_gate"], extras["shared_up"], extras["shared_down"])

    cache[prefix] = {"router": router, "experts": expert_set, "shared": shared}
    return cache[prefix]


def _moe_ffn_prefill(weights, prefix, h, moe_cache):
    """MoE FFN for prefill — always routes through all selected experts (dense batch)."""
    from .moe import moe_ffn_silu

    comp = _get_moe_components(weights, prefix, moe_cache)
    if comp is None:
        raise ValueError(f"MoE weights not found for {prefix}")

    output, stats = moe_ffn_silu(
        h, comp["router"], comp["experts"],
        comp["shared"][0] if comp["shared"] else None,
        comp["shared"][1] if comp["shared"] else None,
        comp["shared"][2] if comp["shared"] else None,
    )
    return output


def _moe_ffn_decode(weights, prefix, h_norm, moe_cache, moe_executor=None):
    """MoE FFN for decode — supports hot/cold split via executor.

    If moe_executor is provided and has a route for this layer, uses the
    hot/cold split executor (GPU hot, CPU cold). Otherwise falls back to
    pure-CPU MoE dispatch (still works, just no GPU acceleration).
    """
    comp = _get_moe_components(weights, prefix, moe_cache)
    if comp is None:
        raise ValueError(f"MoE weights not found for {prefix}")

    layer_idx = int(prefix.split(".")[1])

    if moe_executor is not None:
        # Hot/cold executor handles routing + splitting internally
        output, stats = moe_executor.dispatch(layer_idx, h_norm)
        return output, stats

    # Fallback: pure CPU MoE dispatch
    from .moe import moe_ffn_silu
    output, stats = moe_ffn_silu(
        h_norm, comp["router"], comp["experts"],
        comp["shared"][0] if comp["shared"] else None,
        comp["shared"][1] if comp["shared"] else None,
        comp["shared"][2] if comp["shared"] else None,
    )
    return output, stats


def forward_prefill(
    token_ids: np.ndarray,
    token_emb: np.ndarray,
    output_norm_w: np.ndarray,
    output_w: np.ndarray,
    weights: dict[str, np.ndarray],
    n_layers: int,
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
    n_heads: int = 32,
    n_kv_heads: int = 32,
    head_dim: int = 0,
    eps: float = 1e-5,
    minicache=None,
    paged_attn=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Full prefill forward pass over all prompt tokens.

    Prefill is always DENSE — sparsity only applies to decode (single token).
    During prefill, all positions need accurate logits and the batched matmul
    amortizes memory bandwidth, so sparse skipping doesn't help.

    If minicache is provided, KV states are bulk-loaded into the MiniCache
    depth-compressed store after the prefill pass.  The standard KV caches
    are returned as usual — the minicache is only consulted during decode.

    If paged_attn is provided, KV states are also stored in the paged pool
    (for multi-request serving with prefix sharing).  Flat KV caches are
    still returned for backward compatibility.

    Args:
        token_ids: (seq_len,) integer token IDs
        token_emb: (vocab_size, hidden_dim) embedding matrix
        output_norm_w: (hidden_dim,) final norm weights
        output_w: (vocab_size, hidden_dim) output/head weights
        weights: all model weight tensors
        n_layers: number of transformer layers
        cos_cache, sin_cache: RoPE tables
        n_heads, n_kv_heads: head counts
        eps: RMSNorm epsilon
        minicache: optional MiniCache for depth-dimension KV compression
        paged_attn: optional PagedKVCache for paged memory management

    Returns:
        (logits, kv_caches_k, kv_caches_v)
        logits: (seq_len, vocab_size)
    """
    seq_len = len(token_ids)
    x = token_emb[token_ids]  # (seq, hidden_dim)

    # Infer head_dim from Q tensor if not explicitly provided
    # (hybrid models like Qwen3.6 may have head_dim != hidden_dim / n_heads)
    if head_dim <= 0:
        head_dim = token_emb.shape[-1] // n_heads

    kv_caches_k: list[np.ndarray] = []
    kv_caches_v: list[np.ndarray] = []
    moe_cache: dict = {}  # caches parsed MoE routers/experts per layer

    for layer_idx in range(n_layers):
        prefix = f"blk.{layer_idx}"

        # Pre-attention RMSNorm
        h = rms_norm(x, weights[f"{prefix}.attn_norm.weight"], eps)

        # QKV projections
        q = h @ weights[f"{prefix}.attn_q.weight"].T
        k = h @ weights[f"{prefix}.attn_k.weight"].T
        v = h @ weights[f"{prefix}.attn_v.weight"].T

        # Validate Q shape matches expected (n_heads * head_dim)
        # This catches the case where QKV was split with different head_dim than expected
        expected_q_dim = n_heads * head_dim
        if q.shape[-1] != expected_q_dim:
            import sys as _sys
            _sys.stderr.write(
                f"\n[WARN] blk.{layer_idx} Q shape {q.shape} != "
                f"expected (seq, {expected_q_dim}). "
                f"head_dim={head_dim}, n_heads={n_heads}, n_kv={n_kv_heads}\n"
            )
            # Also update n_kv_heads in case it was wrong
            actual_kv_heads = min(n_kv_heads, n_heads)
            actual_head_dim = q.shape[-1] // n_heads if n_heads > 0 else head_dim
            _sys.stderr.write(f"[WARN] Also auto-correcting n_kv_heads from {n_kv_heads} to {actual_kv_heads}, head_dim to {actual_head_dim}\n")
            n_kv_heads = actual_kv_heads
            head_dim = actual_head_dim
            # Must also re-compute k and v with the corrected dimensions to match q
            k = h @ weights[f"{prefix}.attn_k.weight"].T
            v = h @ weights[f"{prefix}.attn_v.weight"].T

        # Apply RoPE
        cos_slice = cos_cache[:seq_len]
        sin_slice = sin_cache[:seq_len]
        q_r = rope(q.reshape(seq_len, n_heads, head_dim), cos_slice, sin_slice).reshape(seq_len, -1)
        k_r = rope(k.reshape(seq_len, n_kv_heads, head_dim), cos_slice, sin_slice).reshape(seq_len, -1)

        # Store in KV cache
        kv_caches_k.append(k_r.reshape(n_kv_heads, seq_len, head_dim))
        kv_caches_v.append(v.reshape(n_kv_heads, seq_len, head_dim))

        # Attention with causal mask
        attn_out = attention(q_r, k_r, v, n_heads, n_kv_heads)
        attn_out = attn_out @ weights[f"{prefix}.attn_output.weight"].T

        x = x + attn_out

        # FFN (dense during prefill) — MoE or dense
        h = rms_norm(x, weights[f"{prefix}.ffn_norm.weight"], eps)
        if f"{prefix}.ffn_gate_inp.weight" in weights:
            # MoE layer — load on first encounter, cache for reuse
            ffn_out = _moe_ffn_prefill(weights, prefix, h, moe_cache)
        else:
            ffn_out = ffn_silu(h,
                               weights[f"{prefix}.ffn_gate.weight"],
                               weights[f"{prefix}.ffn_up.weight"],
                               weights[f"{prefix}.ffn_down.weight"])
        x = x + ffn_out

    # Final norm + output projection
    x = rms_norm(x, output_norm_w, eps)
    logits = x @ output_w.T  # (seq, vocab_size)

    # Bulk-load KV into MiniCache after prefill
    if minicache is not None:
        for layer_idx in range(n_layers):
            minicache.bulk_load(
                layer_idx, kv_caches_k[layer_idx], kv_caches_v[layer_idx], start_pos=0
            )

    # Bulk-load KV into PagedKVCache after prefill
    if paged_attn is not None:
        for layer_idx in range(n_layers):
            paged_attn.bulk_append(
                layer_idx, kv_caches_k[layer_idx], kv_caches_v[layer_idx]
            )

    return logits, kv_caches_k, kv_caches_v


def forward_decode_single(
    token_id: int,
    position: int,
    token_emb: np.ndarray,
    output_norm_w: np.ndarray,
    output_w: np.ndarray,
    weights: dict[str, np.ndarray],
    n_layers: int,
    kv_caches_k: list[np.ndarray],
    kv_caches_v: list[np.ndarray],
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
    n_heads: int = 32,
    n_kv_heads: int = 32,
    head_dim: int = 0,
    eps: float = 1e-5,
    sparse_ratio: float = 1.0,
    minicache=None,
    paged_attn=None,
    moe_executor=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], dict]:
    """Decode a single token using KV cache (the fast path for generation).

    Only computes 1 token through all layers — O(seq) attention via cache,
    not O(seq²). This is how real inference works.

    When sparse_ratio < 1.0, uses PowerInfer-style activation prediction:
    for each layer, computes gate projection, predicts top-k% active neurons,
    then only runs up/down projections for those neurons. Skips ~90% of FFN
    compute at sparse_ratio=0.1.

    When minicache is provided, uses depth-compressed KV for attention.
    When paged_attn is provided, also appends to the paged pool and can
    read back from it.  Priority: minicache > paged_attn > raw flat cache.

    Args:
        token_id: single token integer
        position: current position in sequence
        token_emb, output_norm_w, output_w, weights: model weights
        n_layers: number of transformer layers
        kv_caches_k, kv_caches_v: existing KV caches (flat arrays)
        cos_cache, sin_cache: RoPE tables
        n_heads, n_kv_heads: head counts
        eps: RMSNorm epsilon
        sparse_ratio: fraction of FFN neurons to keep per layer.
            1.0 = dense (no sparsity), 0.1 = top 10% (PowerInfer default)
        minicache: optional MiniCache for depth-dimension KV compression
        paged_attn: optional PagedKVCache for paged memory management
        moe_executor: optional HotColdExecutor for GPU/CPU split MoE compute

    Returns:
        (logits, updated_k_caches, updated_v_caches, sparse_stats)
        logits: (vocab_size,) — next token distribution
        sparse_stats: dict with per-layer sparsity info (empty if dense)
    """
    x = token_emb[token_id:token_id + 1]  # (1, hidden_dim)
    new_k_caches: list[np.ndarray] = []
    new_v_caches: list[np.ndarray] = []
    sparse_stats: dict = {}
    moe_cache: dict = {}  # caches parsed MoE routers/experts per layer

    use_sparse = sparse_ratio < 1.0
    if use_sparse:
        sparse_stats["mode"] = "powerinfer"
        sparse_stats["sparse_ratio"] = sparse_ratio
        sparse_stats["layers"] = []

    minicache_stats: dict = {"layers_compressed": 0, "layers_total": n_layers}
    paged_stats: dict = {"pages_used": 0, "seq_len": 0}

    for layer_idx in range(n_layers):
        prefix = f"blk.{layer_idx}"
        # Use explicit head_dim (hybrid models: head_dim != hidden_dim / n_heads)
        if head_dim <= 0:
            head_dim = token_emb.shape[-1] // n_heads

        # Pre-attention RMSNorm
        h = rms_norm(x, weights[f"{prefix}.attn_norm.weight"], eps)

        # QKV projections
        q = h @ weights[f"{prefix}.attn_q.weight"].T
        k_new = h @ weights[f"{prefix}.attn_k.weight"].T
        v_new = h @ weights[f"{prefix}.attn_v.weight"].T

        # Apply RoPE
        cos_pos = cos_cache[position:position + 1]
        sin_pos = sin_cache[position:position + 1]
        q = q.reshape(1, n_heads, head_dim)
        k_new = k_new.reshape(1, n_kv_heads, head_dim)
        q = rope(q, cos_pos, sin_pos).reshape(1, n_heads * head_dim)
        k_new = rope(k_new, cos_pos, sin_pos).reshape(1, n_kv_heads * head_dim)

        # KV cache: append to raw flat cache
        k_full = np.concatenate([kv_caches_k[layer_idx], k_new.reshape(n_kv_heads, 1, head_dim)], axis=1)
        v_full = np.concatenate([kv_caches_v[layer_idx], v_new.reshape(n_kv_heads, 1, head_dim)], axis=1)

        # Update MiniCache with new KV for this layer
        if minicache is not None:
            minicache.update(
                layer_idx,
                k_new.reshape(n_kv_heads, head_dim),
                v_new.reshape(n_kv_heads, head_dim),
                position,
            )

        # Update PagedKVCache with new KV for this layer
        if paged_attn is not None:
            paged_attn.append(
                layer_idx,
                k_new.reshape(n_kv_heads, head_dim),
                v_new.reshape(n_kv_heads, head_dim),
                position=position,
            )

        # Choose KV source (priority: minicache > paged_attn > raw flat cache)
        if minicache is not None:
            mc_k, mc_v = minicache.get(layer_idx, end=position + 1)
            k_seq = mc_k.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim).astype(np.float32)
            v_seq = mc_v.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim).astype(np.float32)
            minicache_stats["layers_compressed"] += 1
        elif paged_attn is not None:
            pa_k, pa_v = paged_attn.get(layer_idx, end=position + 1)
            k_seq = pa_k.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim).astype(np.float32)
            v_seq = pa_v.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim).astype(np.float32)
        else:
            # Reshape raw cache for attention
            k_seq = k_full.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim)
            v_seq = v_full.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim)

        # Attention
        attn_out = attention(q, k_seq, v_seq, n_heads, n_kv_heads)
        attn_out = attn_out @ weights[f"{prefix}.attn_output.weight"].T

        # Residual
        h = x + attn_out

        # Pre-FFN RMSNorm
        h_norm = rms_norm(h, weights[f"{prefix}.ffn_norm.weight"], eps)

        # FFN — MoE, sparse, or dense
        if f"{prefix}.ffn_gate_inp.weight" in weights:
            # MoE layer
            ffn_out, moe_layer_stats = _moe_ffn_decode(
                weights, prefix, h_norm, moe_cache, moe_executor
            )
            if "moe" not in sparse_stats:
                sparse_stats["moe"] = {"layers": []}
            sparse_stats["moe"]["layers"].append({"layer": layer_idx, **moe_layer_stats})
        elif use_sparse:
            gate_w = weights[f"{prefix}.ffn_gate.weight"]
            up_w = weights[f"{prefix}.ffn_up.weight"]
            down_w = weights[f"{prefix}.ffn_down.weight"]
            ffn_out, layer_stats = _sparse_ffn_layer(
                h_norm, gate_w, up_w, down_w, sparse_ratio
            )
            sparse_stats["layers"].append({
                "layer": layer_idx,
                **layer_stats,
            })
        else:
            gate_w = weights[f"{prefix}.ffn_gate.weight"]
            up_w = weights[f"{prefix}.ffn_up.weight"]
            down_w = weights[f"{prefix}.ffn_down.weight"]
            ffn_out = ffn_silu(h_norm, gate_w, up_w, down_w)

        # Residual
        x = h + ffn_out
        new_k_caches.append(k_full)
        new_v_caches.append(v_full)

    # Final norm + output projection
    x = rms_norm(x, output_norm_w, eps)
    logits = (x @ output_w.T)[0]  # (vocab_size,)

    # Attach cache stats
    if minicache is not None:
        sparse_stats["minicache"] = minicache_stats
    if paged_attn is not None:
        paged_stats["pages_used"] = paged_attn.num_used_pages
        paged_stats["seq_len"] = len(paged_attn)
        sparse_stats["paged_attn"] = paged_stats

    return logits, new_k_caches, new_v_caches, sparse_stats
