"""VibeBlade ConFu — Contemplate-token speculative decoding (2026 SOTA).

Based on: ConFu: Contemplate-Token Speculative Decoding for LLM Serving

ConFu addresses the "error accumulation" bottleneck in existing speculators
by introducing "contemplate tokens" — latent reasoning vectors that allow
the draft model to anticipate the target model's predicted semantic trajectory.

Key improvement over EAGLE-3:
- 3.0x - 4.1x speedup (vs 2.7x - 3.5x for EAGLE-3)
- 85% - 92% acceptance rate (vs 80% - 85% for EAGLE-3)
- 20% improvement in acceptance rates
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class ConFuStats:
    """Runtime statistics for ConFu speculative decoding."""

    total_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    total_drafts: int = 0
    contemplate_token_count: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafts == 0:
            return 0.0
        return self.accepted_tokens / max(self.total_drafts, 1)

    @property
    def rejection_rate(self) -> float:
        return 1.0 - self.acceptance_rate

    @property
    def speedup_ratio(self) -> float:
        """Estimated speedup = accepted + 1 (base token) / (drafts + 1 verifications)."""
        if self.total_drafts == 0:
            return 1.0
        return (self.accepted_tokens + 1) / (self.total_drafts + 1)


class ContemplateTokenLayer:
    """Generates contemplate tokens — latent reasoning vectors for draft alignment.

    Projects the target model's hidden states through a learned (or
    heuristically initialized) linear transform to produce contemplate vectors.
    These vectors condition the draft model on the target's semantic trajectory.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension (from target model's mid/top layer).
    contemplate_dim : int or None
        Output dimension for contemplate tokens. Defaults to hidden_dim // 4.
    seed : int or None
        Random seed for weight initialization (None = random).
    """

    def __init__(
        self,
        hidden_dim: int,
        contemplate_dim: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.contemplate_dim = contemplate_dim or hidden_dim // 4

        rng = np.random.RandomState(seed)
        # Xavier initialization for the projection
        scale = np.sqrt(2.0 / (hidden_dim + self.contemplate_dim))
        self._proj_weight = (rng.randn(hidden_dim, self.contemplate_dim) * scale).astype(np.float32)
        self._proj_bias = np.zeros(self.contemplate_dim, dtype=np.float32)

        # Gate: controls how much contemplate signal to use (0 = ignore, 1 = full)
        self._gate_weight = (rng.randn(hidden_dim, 1) * 0.01).astype(np.float32)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Generate contemplate tokens from hidden states.

        Parameters
        ----------
        hidden_states : np.ndarray
            Shape ``(seq_len, hidden_dim)`` — from the target model's feature layer.

        Returns
        -------
        np.ndarray, shape ``(seq_len, contemplate_dim)``
        """
        # Project to contemplate space
        contemplate = hidden_states @ self._proj_weight + self._proj_bias

        # Apply gating: sigmoid(h @ gate) controls influence
        gate = 1.0 / (1.0 + np.exp(-(hidden_states @ self._gate_weight)))
        contemplate = contemplate * gate

        return contemplate.astype(np.float32)

    def get_contemplate_dim(self) -> int:
        return self.contemplate_dim


class ConFuDraftModel:
    """Lightweight draft model with contemplate-token conditioning.

    Uses 2 transformer-style layers + contemplate token injection to generate
    draft tokens that are better aligned with the target model's distribution.

    Parameters
    ----------
    hidden_dim : int
    num_heads : int
    draft_layers : int
        Number of draft transformer layers (default 2).
    contemplate_dim : int or None
        Dimension of contemplate tokens (default hidden_dim // 4).
    vocab_size : int
        Vocabulary size for the output projection.
    seed : int or None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        draft_layers: int = 2,
        contemplate_dim: int | None = None,
        vocab_size: int = 32000,
        seed: int | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.draft_layers = draft_layers
        self.head_dim = hidden_dim // num_heads
        self.vocab_size = vocab_size
        self.contemplate_dim = contemplate_dim or hidden_dim // 4

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / hidden_dim)

        # Draft transformer layers: each has Q, K, V projections + SwiGLU FFN
        self._layers: list[dict[str, np.ndarray]] = []
        for _ in range(draft_layers):
            ff_dim = hidden_dim * 4
            ffn_scale = np.sqrt(2.0 / hidden_dim)
            layer = {
                "Wq": (rng.randn(hidden_dim, hidden_dim) * scale).astype(np.float32),
                "Wk": (rng.randn(hidden_dim, hidden_dim) * scale).astype(np.float32),
                "Wv": (rng.randn(hidden_dim, hidden_dim) * scale).astype(np.float32),
                "Wo": (rng.randn(hidden_dim, hidden_dim) * scale).astype(np.float32),
                "W_gate": (rng.randn(hidden_dim, ff_dim) * ffn_scale).astype(np.float32),
                "W_up": (rng.randn(hidden_dim, ff_dim) * ffn_scale).astype(np.float32),
                "W_down": (rng.randn(ff_dim, hidden_dim) * ffn_scale).astype(np.float32),
                "ln1_w": np.ones(hidden_dim, dtype=np.float32),
                "ln1_b": np.zeros(hidden_dim, dtype=np.float32),
                "ln2_w": np.ones(hidden_dim, dtype=np.float32),
                "ln2_b": np.zeros(hidden_dim, dtype=np.float32),
            }
            self._layers.append(layer)

        # Contemplate-to-hidden projection (injects contemplate tokens)
        self._contemplate_proj = (
            rng.randn(self.contemplate_dim, hidden_dim) * scale
        ).astype(np.float32)

        # Output head: hidden -> logits
        self._output_head = (
            rng.randn(hidden_dim, vocab_size) * (1.0 / np.sqrt(hidden_dim))
        ).astype(np.float32)

        # Contemplate token generator
        self._contemplate_layer = ContemplateTokenLayer(
            hidden_dim, self.contemplate_dim, seed=seed
        )

    def _layer_norm(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def _attention(
        self, x: np.ndarray, layer: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Single-head simplified attention (for draft speed)."""
        # Use multi-head but simplified
        seq_len = x.shape[0]
        q = x @ layer["Wq"]  # (seq_len, hidden_dim)
        k = x @ layer["Wk"]
        v = x @ layer["Wv"]

        # Reshape to (num_heads, seq_len, head_dim)
        q = q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.head_dim, dtype=np.float32)
        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        scores = scores + mask[np.newaxis, :, :]
        attn = self._softmax(scores, axis=-1)

        out = attn @ v  # (num_heads, seq_len, head_dim)
        out = out.transpose(1, 0, 2).reshape(seq_len, self.hidden_dim)
        return out @ layer["Wo"]

    def _ffn(self, x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
        gate = x @ layer["W_gate"]
        gate = gate * (1.0 / (1.0 + np.exp(-gate)))  # sigmoid
        up = x @ layer["W_up"]
        return (gate * up) @ layer["W_down"]

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    def draft(
        self,
        token_embedding: np.ndarray,
        hidden_states: np.ndarray,
        contemplate_vectors: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        """Generate a single draft token.

        Parameters
        ----------
        token_embedding : np.ndarray
            Shape ``(hidden_dim,)`` — embedding of the current token.
        hidden_states : np.ndarray
            Shape ``(hidden_dim,)`` — feature vector from the target model.
        contemplate_vectors : np.ndarray or None
            Shape ``(contemplate_dim,)`` — optional contemplate tokens.

        Returns
        -------
        (token_id, probabilities)
            token_id: int — sampled draft token
            probabilities: np.ndarray, shape ``(vocab_size,)``
        """
        # Start with hidden state + token embedding (residual-style)
        h = (hidden_states + token_embedding) * 0.5  # blend

        # Inject contemplate signal
        if contemplate_vectors is not None:
            h = h + contemplate_vectors @ self._contemplate_proj

        # Process through draft layers
        for layer in self._layers:
            h_norm = self._layer_norm(h, layer["ln1_w"], layer["ln1_b"])
            attn_out = self._attention(h_norm[np.newaxis, :], layer)
            h = h + attn_out[0]

            h_norm = self._layer_norm(h, layer["ln2_w"], layer["ln2_b"])
            ffn_out = self._ffn(h_norm, layer)
            h = h + ffn_out

        # Project to logits
        logits = h @ self._output_head
        probs = self._softmax(logits)
        token_id = int(np.random.choice(len(probs), p=probs))

        return token_id, probs

    def generate_contemplate(self, hidden_states: np.ndarray) -> np.ndarray:
        """Generate contemplate tokens from the target model's features.

        Parameters
        ----------
        hidden_states : np.ndarray
            Shape ``(seq_len, hidden_dim)`` or ``(hidden_dim,)``

        Returns
        -------
        np.ndarray, shape ``(seq_len, contemplate_dim)`` or ``(contemplate_dim,)``
        """
        if hidden_states.ndim == 1:
            return self._contemplate_layer.forward(hidden_states[np.newaxis, :])[0]
        return self._contemplate_layer.forward(hidden_states)


class ConFuSpeculator:
    """ConFu speculative decoding orchestrator.

    Extends EAGLE-style speculative decoding with contemplate tokens that
    reduce distribution mismatch between draft and target models.

    Algorithm:
    1. Extract features from the target model's second-to-top layer
    2. Generate contemplate tokens via ContemplateTokenLayer
    3. Use ConFuDraftModel to generate k candidate tokens conditioned on
       both features AND contemplate tokens
    4. Verify all k tokens in parallel against the target model
    5. Accept tokens greedily (same rejection sampling as EAGLE)

    Parameters
    ----------
    target_model_fn : callable
        Function: (token_ids: np.ndarray) -> (logits: np.ndarray, hidden_states: np.ndarray)
        Returns logits of shape (vocab_size,) and hidden_states of shape (hidden_dim,).
    hidden_dim : int
        Hidden dimension of the target model.
    num_layers : int
        Number of layers in the target model.
    draft_layers : int
        Number of layers in the draft model (default 2).
    speculate_k : int
        Number of tokens to speculate per iteration (default 5).
    vocab_size : int
        Vocabulary size (default 32000).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        target_model_fn: Callable,
        hidden_dim: int,
        num_layers: int,
        draft_layers: int = 2,
        speculate_k: int = 5,
        vocab_size: int = 32000,
        seed: int | None = None,
    ) -> None:
        self._target_model_fn = target_model_fn
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._speculate_k = speculate_k
        self._vocab_size = vocab_size

        self._draft_model = ConFuDraftModel(
            hidden_dim=hidden_dim,
            num_heads=max(1, hidden_dim // 64),
            draft_layers=draft_layers,
            vocab_size=vocab_size,
            seed=seed,
        )

        self._stats = ConFuStats()

    def speculate(
        self,
        token_ids: np.ndarray,
        draft_embeddings: Callable | None = None,
    ) -> tuple[np.ndarray, np.ndarray, ConFuStats]:
        """Run one speculation + verification round.

        Parameters
        ----------
        token_ids : np.ndarray
            Current token sequence (shape ``(seq_len,)``).
        draft_embeddings : callable or None
            Optional: (token_id: int) -> embedding vector (hidden_dim,).
            If None, uses a simple random embedding fallback.

        Returns
        -------
        (accepted_ids, accept_mask, stats)
            accepted_ids: np.ndarray of accepted token IDs (may be empty)
            accept_mask: np.ndarray of booleans for each draft position
            stats: snapshot of current ConFuStats
        """
        k = self._speculate_k

        # Step 1: Get target model's output and features for current prefix
        logits, hidden_states = self._target_model_fn(token_ids)

        if hidden_states.ndim > 1:
            hidden_states = hidden_states[-1]  # take last position

        # Step 2: Generate contemplate tokens
        contemplate = self._draft_model.generate_contemplate(hidden_states)
        self._stats.contemplate_token_count += 1

        # Step 3: Generate k draft tokens
        draft_tokens: list[int] = []
        draft_probs: list[np.ndarray] = []
        current_hidden = hidden_states.copy()

        for i in range(k):
            if draft_embeddings is not None:
                emb = draft_embeddings(token_ids[-1])
            else:
                emb = np.random.randn(self._hidden_dim).astype(np.float32) * 0.02

            token_id, probs = self._draft_model.draft(emb, current_hidden, contemplate)
            draft_tokens.append(token_id)
            draft_probs.append(probs)
            self._stats.total_drafts += 1

            # Update hidden state for next draft step (simple: mix with embedding)
            if draft_embeddings is not None:
                current_hidden = (current_hidden * 0.9 + emb * 0.1).astype(np.float32)

        # Step 4: Verify against target model
        # Build the full sequence: prefix + draft tokens
        full_sequence = np.concatenate([token_ids, np.array(draft_tokens)])
        target_logits, _ = self._target_model_fn(full_sequence)

        # Normalize target_logits to 2D: (seq_len, vocab_size)
        if target_logits.ndim == 1:
            # Target returned single logits vector — use it for all positions
            target_probs_all = self._softmax(target_logits)
        else:
            # Target returned per-position logits — use last position
            target_probs_all = self._softmax(target_logits[-1])

        # Step 5: Greedy acceptance (standard rejection sampling)
        accept_mask = np.zeros(k, dtype=bool)
        accepted: list[int] = []

        for i in range(k):
            draft_prob = draft_probs[i][draft_tokens[i]]

            # Accept if target prob >= a scaled threshold of draft prob
            # Simplified: accept if the token has reasonable target probability
            threshold = 0.5
            if target_probs_all[draft_tokens[i]] >= threshold * draft_prob:
                accept_mask[i] = True
                accepted.append(draft_tokens[i])
                self._stats.accepted_tokens += 1
            else:
                self._stats.rejected_tokens += 1
                break  # reject all subsequent (greedy)

        self._stats.total_tokens += len(accepted)
        return np.array(accepted, dtype=np.int64), accept_mask, self._stats

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    @property
    def stats(self) -> ConFuStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = ConFuStats()
