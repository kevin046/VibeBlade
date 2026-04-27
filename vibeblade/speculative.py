"""VibeBlade EAGLE — Speculative decoding via feature-level drafting.

Based on: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (2401.15077)

Instead of token-level drafting (which has high entropy), EAGLE drafts at
the feature/hidden-state level using the second-to-top layer. This produces
more coherent draft sequences with higher acceptance rates.

Achieves 2.7-3.5× latency speedup with ~2× throughput on 70B models.
"""

from __future__ import annotations

import numpy as np


class EAGLEDraftHead:
    """Lightweight draft head that predicts multiple future tokens from hidden states.

    Trained on the second-to-top layer features of the target model.
    At inference time, it drafts candidate tokens that are verified against
    the target model via rejection sampling.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the input hidden state (second-to-top layer).
    vocab_size : int
        Vocabulary size.
    num_heads : int
        Draft token tree width (number of parallel draft branches).
    max_draft_tokens : int
        Maximum tokens to draft per speculation step (default 5).
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int = 4,
        max_draft_tokens: int = 5,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.max_draft_tokens = max_draft_tokens

        # Lightweight projection heads (much smaller than full model)
        # Each head: hidden_dim -> vocab_size
        self.head_weights = [
            np.random.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02
            for _ in range(num_heads)
        ]

        # Feature transform: extracts draft features from hidden states
        self.feature_proj = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        self.feature_bias = np.zeros(hidden_dim, dtype=np.float32)

        # Shifted token embedding for temporal context
        self.token_shift_proj = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02

    def extract_features(self, hidden_state: np.ndarray, prev_token_emb: np.ndarray | None = None) -> np.ndarray:
        """Extract draft features from the second-to-top layer hidden state.

        Uses a shifted token sequence for temporal context, following EAGLE's
        approach of combining current features with previous token embeddings.

        Parameters
        ----------
        hidden_state : np.ndarray, shape ``(hidden_dim,)``
        prev_token_emb : np.ndarray or None, shape ``(hidden_dim,)``

        Returns
        -------
        np.ndarray, shape ``(hidden_dim,)``
        """
        features = hidden_state @ self.feature_proj + self.feature_bias
        if prev_token_emb is not None:
            features = features + prev_token_emb @ self.token_shift_proj
        return features

    def draft_tokens(self, hidden_state: np.ndarray, prev_token_emb: np.ndarray | None = None) -> list[list[int]]:
        """Generate draft token tree from hidden state.

        Each head produces a sequence of draft tokens. Returns a list of
        candidate sequences, one per head.

        Parameters
        ----------
        hidden_state : np.ndarray, shape ``(hidden_dim,)``
        prev_token_emb : np.ndarray or None

        Returns
        -------
        list of lists, each inner list contains draft token IDs
        """
        features = self.extract_features(hidden_state, prev_token_emb)
        draft_sequences: list[list[int]] = []

        for head_idx in range(self.num_heads):
            seq = []
            current_features = features.copy()
            for _ in range(self.max_draft_tokens):
                logits = current_features @ self.head_weights[head_idx]
                token = int(np.argmax(logits))
                seq.append(token)
                # Use the logits as next-step feature (simplified — real EAGLE
                # uses the draft token embedding projected back)
                current_features = logits @ self.head_weights[head_idx].T / self.hidden_dim
            draft_sequences.append(seq)

        return draft_sequences

    def draft_single(self, hidden_state: np.ndarray, prev_token_emb: np.ndarray | None = None) -> list[int]:
        """Generate a single draft sequence (head 0).

        Returns
        -------
        list of int, draft token IDs
        """
        return self.draft_tokens(hidden_state, prev_token_emb)[0]


class SpeculativeVerifier:
    """Verifies draft tokens against the target model via rejection sampling.

    Implements the standard speculative decoding verification:
    1. Compare draft token probabilities against target model
    2. Accept tokens where target prob / draft prob > uniform random
    3. Reject first failing token, resample from target distribution
    4. All tokens before rejection are accepted (parallel decoding)
    """

    @staticmethod
    def verify(
        draft_tokens: list[int],
        target_logits: list[np.ndarray],
        draft_logits: list[np.ndarray],
        temperature: float = 1.0,
    ) -> tuple[list[int], bool, int]:
        """Verify draft tokens against target model.

        Parameters
        ----------
        draft_tokens : list of int
            Draft token IDs to verify.
        target_logits : list of np.ndarray
            Target model logits for each draft position, shape ``(vocab_size,)`` each.
        draft_logits : list of np.ndarray
            Draft model logits for each draft position.
        temperature : float

        Returns
        -------
        (accepted_tokens, all_accepted, num_accepted)
            accepted_tokens : list of accepted token IDs
            all_accepted : True if all draft tokens were accepted
            num_accepted : count of accepted tokens
        """
        if len(draft_tokens) == 0:
            return [], True, 0

        if temperature == 0:
            # Greedy: accept if target argmax matches draft
            accepted = []
            for i, token in enumerate(draft_tokens):
                if i >= len(target_logits):
                    break
                if int(np.argmax(target_logits[i])) == token:
                    accepted.append(token)
                else:
                    break
            all_accepted = len(accepted) == len(draft_tokens)
            return accepted, all_accepted, len(accepted)

        accepted = []
        for i, token in enumerate(draft_tokens):
            if i >= len(target_logits):
                break

            # Compute probabilities
            t_logits = target_logits[i] / temperature
            d_logits = draft_logits[i] / temperature if i < len(draft_logits) else t_logits

            t_probs = SpeculativeVerifier._softmax(t_logits)
            d_probs = SpeculativeVerifier._softmax(d_logits)

            # Rejection sampling: accept if U < min(1, p_target / p_draft)
            u = np.random.random()
            p_target = t_probs[token]
            p_draft = max(d_probs[token], 1e-10)
            acceptance_prob = min(1.0, p_target / p_draft)

            if u < acceptance_prob:
                accepted.append(token)
            else:
                # Reject: resample from adjusted target distribution
                adjusted = np.maximum(t_probs - d_probs, 0)
                adjusted_sum = adjusted.sum()
                if adjusted_sum > 0:
                    adjusted /= adjusted_sum
                else:
                    adjusted = t_probs
                resampled = int(np.random.choice(len(adjusted), p=adjusted))
                accepted.append(resampled)
                return accepted, False, len(accepted)

        return accepted, True, len(accepted)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-10)


class EAGLEDecoder:
    """High-level speculative decoding loop combining EAGLE draft + verification.

    Parameters
    ----------
    draft_head : EAGLEDraftHead
    temperature : float
    """

    def __init__(self, draft_head: EAGLEDraftHead, temperature: float = 1.0) -> None:
        self.draft_head = draft_head
        self.verifier = SpeculativeVerifier()
        self.temperature = temperature

        # Stats
        self.total_drafted = 0
        self.total_accepted = 0

    def speculate_step(
        self,
        hidden_state: np.ndarray,
        prev_token_emb: np.ndarray | None,
        target_model_fn,
    ) -> tuple[list[int], np.ndarray]:
        """Run one speculation step.

        Parameters
        ----------
        hidden_state : np.ndarray, shape ``(hidden_dim,)``
        prev_token_emb : np.ndarray or None
        target_model_fn : callable(draft_tokens) -> (target_logits, new_hidden, new_emb)

        Returns
        -------
        (accepted_tokens, new_hidden_state)
        """
        # 1. Draft tokens
        draft_tokens = self.draft_head.draft_single(hidden_state, prev_token_emb)
        if not draft_tokens:
            return [], hidden_state

        self.total_drafted += len(draft_tokens)

        # 2. Run target model on draft tokens
        target_logits, new_hidden, new_emb = target_model_fn(draft_tokens)

        # 3. Generate draft logits (simplified — using the draft head)
        draft_logits = []
        features = hidden_state.copy()
        for _ in draft_tokens:
            logits = features @ self.draft_head.head_weights[0]
            draft_logits.append(logits)
            # Project logits back to hidden_dim for next iteration
            features = (logits @ self.draft_head.head_weights[0].T) / self.draft_head.hidden_dim

        # 4. Verify
        accepted, all_accepted, n_accepted = self.verifier.verify(
            draft_tokens, target_logits, draft_logits, self.temperature
        )
        self.total_accepted += n_accepted

        return accepted, new_hidden

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    def __repr__(self) -> str:
        return (
            f"EAGLEDecoder(heads={self.draft_head.num_heads}, "
            f"max_draft={self.draft_head.max_draft_tokens}, "
            f"accept_rate={self.acceptance_rate:.1%})"
        )
