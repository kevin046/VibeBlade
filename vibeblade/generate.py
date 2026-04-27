"""VibeBlade Generate — High-level text generation API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .grammar.constraint import GrammarConstraint


class TextGenerator:
    """Token-by-token text generation with sampling strategies."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        grammar: Optional["GrammarConstraint"] = None,
        vocab: Optional[list] = None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.grammar = grammar
        self.vocab = vocab

    def sample(self, logits: np.ndarray, grammar: Optional["GrammarConstraint"] = None) -> int:
        """Sample next token from logits. logits shape: (vocab_size,)"""
        # Apply grammar mask before any sampling
        if grammar is not None:
            mask = grammar.get_token_mask()
            logits = np.where(mask, logits, -np.inf)

        if self.temperature == 0:
            return int(np.argmax(logits))

        logits = logits / self.temperature

        # Top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, len(logits))
            indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[indices] = logits[indices]
            logits = mask

        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            probs = self._softmax(sorted_logits)
            cumulative_probs = np.cumsum(probs)
            # Remove tokens with cumulative probability above threshold
            cutoff = np.searchsorted(cumulative_probs, self.top_p) + 1
            mask = np.full_like(logits, -np.inf)
            mask[sorted_indices[:cutoff]] = logits[sorted_indices[:cutoff]]
            logits = mask

        probs = self._softmax(logits)
        return int(np.random.choice(len(probs), p=probs))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def generate(
        self,
        model_fn: Callable,
        token_ids: np.ndarray,
        max_tokens: int = 100,
        callback=None,
        on_token=None,
        grammar: Optional["GrammarConstraint"] = None,
        vocab: Optional[list] = None,
    ) -> Tuple[np.ndarray, float]:
        """Generate tokens autoregressively.

        Args:
            model_fn: callable(token_ids) -> logits (seq_len, vocab_size)
            token_ids: starting token IDs (seq_len,)
            max_tokens: max new tokens to generate
            callback: optional callable(token_id, position) called after each token
            grammar: optional GrammarConstraint for constrained decoding
            vocab: optional token vocabulary (list of decoded strings) — if
                   provided along with *grammar*, ``grammar.advance()`` is
                   called automatically after each token.

        Returns:
            (generated, tokens_per_sec): full sequence including prompt and speed
        """
        import time

        _grammar = grammar or self.grammar
        _vocab = vocab or self.vocab

        generated = list(token_ids)
        start_time = time.time()

        _callback = on_token if on_token is not None else callback

        for i in range(max_tokens):
            logits = model_fn(np.array(generated))
            next_token = self.sample(logits[-1], grammar=_grammar)  # use last position logits
            generated.append(next_token)

            # Advance grammar state
            if _grammar is not None and _vocab is not None:
                if 0 <= next_token < len(_vocab):
                    _grammar.advance(_vocab[next_token])

            if _callback:
                _callback(next_token, i)
            if next_token == 2:  # EOS token
                break

            # If grammar is finished and current state is accepting, stop
            if _grammar is not None and _grammar.is_finished():
                # Allow stopping at an accepting state
                # (but don't force it — the sampler may still choose to continue)
                pass

        elapsed = time.time() - start_time
        tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0
        return np.array(generated), tokens_per_sec
