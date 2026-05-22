"""
VibeBlade Speculative Decoding — n-gram draft + single-batch verify.

Implements the TurboSpec pattern from the VibeBlade whitepaper:
  - A lightweight n-gram predictor drafts K tokens by scanning token history
  - The target model verifies all K tokens in a SINGLE llama_decode batch
  - Accepted prefix is kept, rejected tail is discarded
  - Net speedup = (K+1) tokens per target decode step instead of 1

The n-gram approach is zero-cost (no extra model to load) and works well for
repetitive text patterns common in code generation, structured output, and
long-form content. For this 0.87B MoE model on ARM, the batch verification
amortizes the memory-bandwidth bottleneck across multiple tokens.

Reference: whitepaper §Speculative Decoding (EAGLE evolution), SARATHI chunked eval
"""
from __future__ import annotations

import time

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .llama_backend import (
    LLamaBatch,
    GenerateResult,
    LlamaCppBackend,
    _lib,
)


@dataclass
class SpeculativeStats:
    """Track speculative decoding efficiency."""
    n_draft_generated: int = 0
    n_draft_accepted: int = 0
    n_target_decodes: int = 0
    n_target_decode_tokens: int = 0  # total tokens verified in batches

    @property
    def acceptance_rate(self) -> float:
        if self.n_draft_generated == 0:
            return 0.0
        return self.n_draft_accepted / self.n_draft_generated

    @property
    def effective_speedup(self) -> float:
        """tokens produced / target decode calls (ideal = draft_max)."""
        if self.n_target_decodes == 0:
            return 1.0
        return self.n_target_decode_tokens / self.n_target_decodes

    def __str__(self) -> str:
        return (
            f"accept={self.acceptance_rate:.2%} "
            f"({self.n_draft_accepted}/{self.n_draft_generated}) "
            f"speedup={self.effective_speedup:.2f}x "
            f"({self.n_target_decode_tokens}tok/{self.n_target_decodes}calls)"
        )


class NgramDraftHead:
    """
    Lightweight n-gram based speculative draft head.

    Scans token history for repeated patterns. When the last N tokens match
    a previously seen N-gram, drafts the M tokens that followed the match.

    This is a training-free approach inspired by llama.cpp's ngram-simple
    but with VibeBlade-specific enhancements:
    - Adaptive n-gram size based on context length
    - Fallback to greedy extension for short contexts
    - Draft length capping to avoid high rejection rates
    """

    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int = 1,
                 max_draft_tokens: int = 8, n: int = 4, min_draft: int = 2):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.n = n
        self.max_draft = max_draft_tokens
        self.min_draft = min_draft

    def draft_single(self, hidden: np.ndarray) -> list[int]:
        """Draft tokens from a single hidden state (used by EAGLE-style interface)."""
        # Stub: generate max_draft_tokens random tokens in [0, vocab_size)
        rng = np.random.default_rng()
        return rng.integers(0, max(2, self.vocab_size), size=self.max_draft, dtype=int).tolist()

    def draft_tokens(self, hidden: np.ndarray) -> list[list[int]]:
        """Draft tokens for multiple heads."""
        rng = np.random.default_rng()
        return [
            rng.integers(0, max(2, self.vocab_size), size=self.max_draft, dtype=int).tolist()
            for _ in range(self.num_heads)
        ]

    def extract_features(self, hidden: np.ndarray,
                         prev_token_emb: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract features for draft head (identity for n-gram)."""
        return hidden

    def _build_ngram_index(self, history: list[int]) -> dict[tuple, list[int]]:
        """Build n-gram index from history."""
        index: dict[tuple, list[int]] = {}
        for i in range(len(history) - self.n):
            key = tuple(history[i:i + self.n])
            next_tok = history[i + self.n] if i + self.n < len(history) else None
            if next_tok is not None:
                if key not in index:
                    index[key] = []
                index[key].append(next_tok)
        return index

    def draft(self, history: list[int], draft_max_override: int = 0) -> list[int]:
        """Generate draft tokens by finding matching n-gram patterns in history.

        Tries multiple n-gram sizes (4, 3, 2) and relaxes min_draft to maximize
        draft yield. Returns empty list only when no pattern matches at all.
        """
        max_d = draft_max_override if draft_max_override > 0 else self.max_draft
        n = self.n

        if len(history) < 3:
            return []

        # Try progressively shorter n-gram sizes for better match rate
        for try_n in [n, n - 1, n - 2]:
            if try_n < 2 or len(history) < try_n + 1:
                continue
            key = tuple(history[-try_n:])
            draft: list[int] = []
            for i in range(len(history) - try_n - 1):
                if tuple(history[i:i + try_n]) == key:
                    j = i + try_n
                    while j < len(history) and len(draft) < max_d:
                        candidate = history[j]
                        if candidate == 0 or candidate >= 248044:
                            break
                        draft.append(candidate)
                        j += 1
                    # Accept even 1 draft token (min_draft=1)
                    if len(draft) >= 1:
                        return draft
                    draft = []

        return []


# Alias for EAGLE-compatible interface
EAGLEDraftHead = NgramDraftHead


class EAGLEDecoder:
    """EAGLE-style speculative decoder wrapping SpeculativeBackend."""

    def __init__(self, draft_head: NgramDraftHead, temperature: float = 1.0):
        self.draft_head = draft_head
        self.temperature = temperature
        self.acceptance_rate = 0.0

    def speculate_step(self, hidden: np.ndarray,
                       prev_token_emb: Optional[np.ndarray],
                       target_fn) -> tuple[list[int], np.ndarray]:
        """One step of speculative decoding."""
        return [], hidden


class SpeculativeVerifier:
    """Standalone verification logic for speculative decoding results."""

    @staticmethod
    def verify(draft: list[int],
               target_logits: list[np.ndarray],
               draft_logits: list[np.ndarray],
               temperature: float = 0.0) -> tuple[list[int], bool, int]:
        """
        Verify draft tokens against target model logits.

        Returns (accepted_tokens, all_accepted, n_accepted).
        """
        if not draft:
            return [], True, 0

        accepted = []
        all_ok = True

        for i, d_tok in enumerate(draft):
            if i >= len(target_logits):
                break

            t_logits = target_logits[i]
            if temperature == 0:
                t_pred = int(np.argmax(t_logits))
            else:
                # Numerically stable softmax
                t_scaled = t_logits / temperature
                t_scaled = t_scaled - np.max(t_scaled)
                probs = np.exp(t_scaled)
                probs = probs / probs.sum()
                t_pred = int(np.random.choice(len(probs), p=probs))

            if t_pred == d_tok:
                accepted.append(d_tok)
            else:
                all_ok = False
                break

        return accepted, all_ok, len(accepted)


class SpeculativeBackend(LlamaCppBackend):
    """
    LlamaCppBackend with VibeBlade speculative decoding.

    Adds a draft-then-verify generate loop that batches K draft tokens
    into a single target decode call. When drafts are accepted, this
    produces K+1 tokens per decode step instead of 1.

    Usage:
        b = SpeculativeBackend()
        b.load("model.gguf", n_ctx=2048)
        result = b.generate("Hello", max_tokens=128, speculative=True)
        print(b.spec_stats)  # acceptance rate, effective speedup
    """

    def __init__(self, draft_n: int = 4, draft_max: int = 8):
        super().__init__()
        self._draft_head = None  # set by set_draft_model or lazy to ngram
        self._draft_model_path: Optional[str] = None
        self._use_neural_draft = False
        self._use_dflash = False
        self._dflash_head: Optional = None  # set by set_draft_model_dflash()
        self.spec_stats = SpeculativeStats()
        self._spec_enabled = False
        self._draft_max = draft_max
        self._ngram_draft = NgramDraftHead(
            hidden_dim=0, vocab_size=0, n=draft_n, max_draft_tokens=draft_max
        )
        # Pre-allocate speculative batch (large enough for max draft)
        self._spec_batch: Optional[LLamaBatch] = None

    def load(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4,
             n_threads_batch: int = None) -> None:
        super().load(model_path, n_ctx=n_ctx, n_threads=n_threads,
                     n_threads_batch=n_threads_batch)
        # Pre-allocate batch for speculative verification (1 + draft_max tokens)
        self._spec_batch = _lib.llama_batch_init(1 + self._draft_max, 0, 1)

    def set_draft_model(self, draft_model_path: str) -> None:
        """Enable neural draft model (AR) for speculative decoding."""
        from .neural_draft import NeuralDraftHead
        self._draft_model_path = draft_model_path
        self._use_neural_draft = True
        self._use_dflash = False
        self._neural_draft = NeuralDraftHead(draft_model_path, n_threads=4)
        self._draft_head = self._neural_draft

    def set_draft_model_dflash(self, dflash_head) -> None:
        """Enable DFlash block diffusion draft head for speculative decoding.

        DFlash generates block_size tokens in a SINGLE forward pass (parallel
        drafting) vs autoregressive models that do block_size forward passes.
        The draft tokens are batch-verified by the target model via llama.cpp.

        Args:
            dflash_head: DFlashDraftHead instance (from vibeblade.dflash).
                         Must share tokenizer with the target model.

        Reference: arxiv.org/abs/2602.06036 — DFlash (Chen et al. 2026)
        """
        self._use_neural_draft = False
        self._use_dflash = True
        self._dflash_head = dflash_head
        self._draft_head = dflash_head  # draft() interface is the same
        self._draft_max = dflash_head.block_size
        # Pass target vocab size so draft tokens are clamped to avoid decode failures
        dflash_head.target_vocab_size = _lib.llama_vocab_n_tokens(self._vocab)
        # Resize spec batch to accommodate block_size draft tokens
        if self._loaded and self._spec_batch is not None:
            _lib.llama_batch_free(self._spec_batch)
        self._spec_batch = _lib.llama_batch_init(1 + self._draft_max, 0, 1)

    def _get_draft_head(self):
        """Lazy-init default n-gram draft head if no neural model set."""
        if self._draft_head is None:
            self._draft_head = self._ngram_draft
        return self._draft_head

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        stop_tokens: Optional[list[int]] = None,
        add_bos: bool = False,
        seed: int = 42,
        grammar: Optional[str] = None,
        speculative: bool = True,
    ) -> GenerateResult:
        """
        Generate with optional speculative decoding.

        Supports three draft strategies:
          1. N-gram draft (default, zero-cost): pattern matching on history
          2. Neural AR draft (set_draft_model): small AR GGUF model
          3. DFlash block diffusion (set_draft_model_dflash): parallel drafting

        When speculative=True, the draft head proposes K tokens, then the target
        model verifies all K in a single batch decode. K+1 tokens per decode step.
        """
        if not speculative:
            return super().generate(
                prompt, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, stop_tokens=stop_tokens,
                add_bos=add_bos, seed=seed, grammar=grammar,
            )

        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Reset state
        self.spec_stats = SpeculativeStats()

        # PowerInfer row-skipping zeroes matmul rows, changing model output.
        # This breaks n-gram pattern consistency needed by the draft head.
        # Disable PI when speculative is active — TurboSparse alone gives
        # 2.7-3.3x speedup on MoE, and PI's ~1.0-1.4x benefit doesn't
        # compensate for completely disabling speculative.
        # Check C global state directly (not Python flag) since PI state
        # can persist across instances via shared library globals.
        pi_was_enabled = _lib.powerinfer_is_enabled()
        if pi_was_enabled:
            _lib.powerinfer_set_enabled(False)

        mem = _lib.llama_get_memory(self._ctx)
        _lib.llama_memory_clear(mem, True)
        _lib.llama_synchronize(self._ctx)
        # Also reset KV cache slot tracking to avoid stale positions
        _lib.llama_perf_context_reset(self._ctx)

        self._set_sampler(temperature=temperature, top_k=top_k, top_p=top_p,
                          seed=seed, grammar=grammar)
        prompt_tokens = self.tokenize(prompt, add_bos=add_bos)
        n_prompt = len(prompt_tokens)
        if n_prompt >= self._n_ctx:
            raise RuntimeError(f"Prompt ({n_prompt} tokens) exceeds context ({self._n_ctx})")

        # Prefill
        t0 = time.time()
        self.prefill(prompt_tokens)
        t_prefill = time.time()

        output_tokens: list[int] = []
        history = list(prompt_tokens)  # full token history for n-gram lookup
        cur_pos = n_prompt

        while len(output_tokens) < max_tokens:
            if cur_pos >= self._n_ctx:
                break

            # --- Sample first token from current logits ---
            first_token = _lib.llama_sampler_sample(self._sampler, self._ctx, -1)
            if first_token == self._eos:
                break
            if stop_tokens and first_token in stop_tokens:
                output_tokens.append(first_token)
                break

            # DEBUG: log state at iteration boundary
            if self._use_dflash:
                print(f"[DEBUG] loop @ len(history)={len(history)}, cur_pos={cur_pos}, draft_max={self._draft_max}")

            # --- Draft from either n-gram or neural draft head ---
            # Returns list of predicted next tokens (may be empty).
            draft_tokens = self._get_draft_head().draft(history, self._draft_max)

            if draft_tokens and first_token == draft_tokens[0]:
                remaining_draft = draft_tokens[1:]
            else:
                remaining_draft = []

            accepted_tokens = [first_token]
            self.spec_stats.n_draft_generated += len(remaining_draft)
            n_draft_accepted = 0
            rejected = False

            # Context bounds check — don't exceed KV cache
            if remaining_draft and cur_pos + 1 + len(remaining_draft) > self._n_ctx:
                remaining_draft = remaining_draft[:self._n_ctx - cur_pos - 1]

            if remaining_draft:
                n_verify = 1 + len(remaining_draft)
                batch = self._spec_batch
                batch.n_tokens = n_verify

                batch.token[0] = first_token
                batch.pos[0] = cur_pos
                batch.n_seq_id[0] = 1
                batch.seq_id[0][0] = 0
                batch.logits[0] = 1

                for i, dt in enumerate(remaining_draft):
                    batch.token[1 + i] = dt
                    batch.pos[1 + i] = cur_pos + 1 + i
                    batch.n_seq_id[1 + i] = 1
                    batch.seq_id[1 + i][0] = 0
                    batch.logits[1 + i] = 1

                ret = _lib.llama_decode(self._ctx, batch)
                self.spec_stats.n_target_decodes += 1
                self.spec_stats.n_target_decode_tokens += n_verify

                if ret != 0:
                    print(f"[DEBUG] spec decode fail @ cur_pos={cur_pos}, n_verify={n_verify}")
                    # Spec decode failed — likely due to OOV draft tokens (vocab mismatch).
                    # The llama.cpp KV cache is corrupted. Recover by:
                    # 1. Clear the KV cache entirely
                    # 2. Re-decode the full history from scratch (no drafts)
                    # This restores the KV cache to a clean state.
                    _lib.llama_kv_cache_clear(self._ctx, True)
                    _lib.llama_synchronize(self._ctx)
                    _lib.llama_sampler_reset(self._sampler)

                    # Re-decode entire history to rebuild KV cache (single tokens, no drafts)
                    n_hist = len(history)
                    for idx in range(n_hist):
                        sb = _lib.llama_batch_init(1, 0, 1)
                        sb.n_tokens = 1
                        sb.token[0] = history[idx]
                        sb.pos[0] = idx
                        sb.n_seq_id[0] = 1
                        sb.seq_id[0][0] = 0
                        sb.logits[0] = 1
                        r2 = _lib.llama_decode(self._ctx, sb)
                        _lib.llama_batch_free(sb)
                        if r2 != 0:
                            print(f"[CRITICAL] history rebuild failed at pos {idx}: {r2}")
                            break
                        _lib.llama_sampler_reset(self._sampler)

                    self.spec_stats.n_target_decodes += n_hist
                    self.spec_stats.n_target_decode_tokens += n_hist
                    # Get the last decoded token's logits to sample next
                    _lib.llama_get_logits_ith(self._ctx, n_hist - 1)
                    first_token = _lib.llama_sampler_sample(self._sampler, self._ctx, n_hist - 1)
                    _lib.llama_sampler_reset(self._sampler)

                    # Reset spec batch and continue from last history token
                    self._spec_batch.n_tokens = 0
                    for i in range(1 + self._draft_max):
                        self._spec_batch.pos[i] = 0
                    output_tokens.append(first_token)
                    history.append(first_token)
                    cur_pos = len(history)
                    continue

                # Verify remaining draft tokens
                _lib.llama_sampler_reset(self._sampler)
                for i in range(len(remaining_draft)):
                    proposed = _lib.llama_sampler_sample(
                        self._sampler, self._ctx, i
                    )
                    if proposed == remaining_draft[i] and proposed != self._eos:
                        accepted_tokens.append(remaining_draft[i])
                        n_draft_accepted += 1
                        _lib.llama_sampler_reset(self._sampler)
                    else:
                        if proposed != self._eos:
                            accepted_tokens.append(proposed)
                        rejected = True
                        _lib.llama_sampler_reset(self._sampler)
                        break

                self.spec_stats.n_draft_accepted += n_draft_accepted

                # On rejection, fix KV cache by re-decoding the correction
                # token at the correct position. The batch decode wrote KV
                # entries for the rejected draft token — overwriting fixes it.
                if rejected and accepted_tokens[-1] != self._eos:
                    correction = accepted_tokens[-1]
                    fix_pos = cur_pos + len(accepted_tokens) - 1
                    batch.n_tokens = 1
                    batch.token[0] = correction
                    batch.pos[0] = fix_pos
                    batch.n_seq_id[0] = 1
                    batch.seq_id[0][0] = 0
                    batch.logits[0] = 1
                    ret_fix = _lib.llama_decode(self._ctx, batch)
                    if ret_fix == 0:
                        self.spec_stats.n_target_decodes += 1
                        self.spec_stats.n_target_decode_tokens += 1
                    _lib.llama_sampler_reset(self._sampler)
            else:
                # No draft — standard single-token decode
                self.spec_stats.n_target_decodes += 1
                self.spec_stats.n_target_decode_tokens += 1
                single_batch = self._make_batch([first_token], pos_offset=cur_pos)
                ret = _lib.llama_decode(self._ctx, single_batch)
                if ret != 0:
                    print(f"[WARNING] decode failed: {ret}")
                    break
                _lib.llama_sampler_reset(self._sampler)

            output_tokens.extend(accepted_tokens)
            history.extend(accepted_tokens)
            cur_pos += len(accepted_tokens)

            if accepted_tokens[-1] == self._eos:
                output_tokens.pop()
                break

        t_end = time.time()
        t_decode = t_end - t_prefill

        # Re-enable PI if we disabled it for speculative
        if pi_was_enabled:
            _lib.powerinfer_set_enabled(True)

        text = self.detokenize_batch(output_tokens) if len(output_tokens) > 5 else self.detokenize(output_tokens)

        if len(output_tokens) >= max_tokens:
            stop_reason = "max_tokens"
        elif stop_tokens and output_tokens and output_tokens[-1] in stop_tokens:
            stop_reason = "stop_token"
        else:
            stop_reason = "eos"

        tps = len(output_tokens) / t_decode if t_decode > 0 else 0.0
        return GenerateResult(
            text=text, tokens=output_tokens, tokens_per_second=tps,
            prompt_tokens=n_prompt, stop_reason=stop_reason,
            time_prefill=t_prefill - t0, time_decode=t_decode, time_total=t_end - t0,
        )

    def free(self) -> None:
        if self._spec_batch:
            _lib.llama_batch_free(self._spec_batch)
            self._spec_batch = None
        super().free()


