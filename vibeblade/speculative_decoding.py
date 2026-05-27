"""VibeBlade Speculative Decoding Engine — unified draft-then-verify.

Implements the standard speculative decoding loop:
  1. Draft head proposes K tokens
  2. Target model verifies all K in a single batch decode
  3. Accepted prefix is kept, rejected tail is discarded
  4. Net speedup = (K+1) tokens per target decode step instead of 1

Works with ANY combination of:
  - Target backends: sglang, vLLM, llama.cpp, OpenAI HTTP
  - Draft strategies: N-gram, EAGLE, DFlash, NEXTN

Usage::

    from vibeblade.target_backend import create_target_backend
    from vibeblade.draft_heads import create_draft_head
    from vibeblade.speculative_decoding import SpeculativeDecodingEngine

    # Connect to running sglang
    target = create_target_backend("sglang", base_url="http://localhost:8000", model="qwen3.6-27b")

    # Choose draft strategy
    draft = create_draft_head("ngram", max_draft=8)

    # Create engine
    engine = SpeculativeDecodingEngine(target=target, draft_head=draft)

    # Generate with speculative decoding
    result = engine.generate("Hello, how are you?", max_tokens=256)
    print(f"Tokens/sec: {result.tokens_per_second:.1f}")
    print(f"Accept rate: {engine.stats.acceptance_rate:.1%}")
    print(f"Text: {result.text}")

    # Or use as a pass-through (no speculation)
    result = engine.generate("Hello", max_tokens=128, speculative=False)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from .draft_heads import DraftHead, NgramDraftHead
from .target_backend import (
    GenerateResult,
    TargetBackend,
    TargetLogits,
    sample_from_logits,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Statistics
# ════════════════════════════════════════════════════════════════════

@dataclass
class SpeculativeStats:
    """Runtime statistics for speculative decoding."""
    n_draft_generated: int = 0
    n_draft_accepted: int = 0
    n_target_decodes: int = 0
    n_target_decode_tokens: int = 0
    n_spec_steps: int = 0  # total speculative steps attempted
    n_empty_drafts: int = 0  # steps where draft returned empty

    @property
    def acceptance_rate(self) -> float:
        if self.n_draft_generated == 0:
            return 0.0
        return self.n_draft_accepted / self.n_draft_generated

    @property
    def effective_speedup(self) -> float:
        """tokens produced / target decode calls."""
        if self.n_target_decodes == 0:
            return 1.0
        return self.n_target_decode_tokens / self.n_target_decodes

    @property
    def draft_yield_rate(self) -> float:
        """Fraction of steps that produced a non-empty draft."""
        if self.n_spec_steps == 0:
            return 0.0
        return 1.0 - (self.n_empty_drafts / self.n_spec_steps)

    def __str__(self) -> str:
        return (
            f"accept={self.acceptance_rate:.1%} "
            f"({self.n_draft_accepted}/{self.n_draft_generated}) "
            f"speedup={self.effective_speedup:.2f}x "
            f"({self.n_target_decode_tokens}tok/{self.n_target_decodes}calls) "
            f"draft_yield={self.draft_yield_rate:.0%}"
        )


# ════════════════════════════════════════════════════════════════════
#  Speculative Decoding Engine
# ════════════════════════════════════════════════════════════════════

class SpeculativeDecodingEngine:
    """Unified speculative decoding engine.

    Decouples the target model (sglang/vLLM/llama.cpp) from the draft strategy
    (n-gram/EAGLE/DFlash/NEXTN). The engine orchestrates the draft-verify loop.

    Parameters:
        target: TargetBackend for the large verification model.
        draft_head: DraftHead for proposing candidate tokens.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering.
        top_p: Top-p (nucleus) filtering.
    """

    def __init__(
        self,
        target: TargetBackend,
        draft_head: Optional[DraftHead] = None,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ):
        self.target = target
        self.draft_head = draft_head or NgramDraftHead()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stats = SpeculativeStats()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = SpeculativeStats()

    # ── Non-streaming generate ──────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        speculative: bool = True,
        stop_tokens: Optional[list[int]] = None,
    ) -> GenerateResult:
        """Generate text with optional speculative decoding.

        Args:
            prompt: Input text.
            max_tokens: Maximum new tokens to generate.
            temperature: Override sampling temperature.
            top_k: Override top-k.
            top_p: Override top-p.
            speculative: Enable speculative decoding (True) or plain AR (False).
            stop_tokens: Early stopping token IDs.

        Returns:
            GenerateResult with text, tokens, speed, and stats.
        """
        temp = temperature if temperature is not None else self.temperature
        k = top_k if top_k is not None else self.top_k
        p = top_p if top_p is not None else self.top_p

        self.reset_stats()
        self.draft_head.reset()

        # Tokenize prompt
        t0 = time.time()
        prompt_tokens = self.target.tokenize(prompt)
        n_prompt = len(prompt_tokens)

        # Prefill
        self.target.reset()
        last_logits = self.target.prefill(prompt_tokens)
        t_prefill = time.time()

        # Sample first token from prefill logits
        first_token = sample_from_logits(last_logits, temp, k, p)

        output_tokens: list[int] = []
        history = list(prompt_tokens)
        cur_pos = n_prompt

        if first_token == self.target.eos_token_id():
            t_end = time.time()
            return GenerateResult(
                text="", tokens=[], tokens_per_second=0.0,
                prompt_tokens=n_prompt, stop_reason="eos",
                time_prefill=t_prefill - t0, time_decode=t_end - t_prefill,
                time_total=t_end - t0,
            )

        output_tokens.append(first_token)
        history.append(first_token)

        # Decode loop
        while len(output_tokens) < max_tokens:
            cur_pos = len(history)

            if not speculative:
                # Plain autoregressive — no draft
                self.stats.n_target_decodes += 1
                self.stats.n_target_decode_tokens += 1
                result = self.target.decode_batch(
                    [first_token], [cur_pos - 1],
                )
                next_tok = sample_from_logits(
                    result.logits_per_position[0], temp, k, p,
                )
                output_tokens.append(next_tok)
                history.append(next_tok)
                first_token = next_tok

                if next_tok == self.target.eos_token_id():
                    break
                if stop_tokens and next_tok in stop_tokens:
                    break
                continue

            # ── Speculative decode ───────────────────────────────
            self.stats.n_spec_steps += 1

            # Step 1: Get draft tokens
            draft_tokens = self.draft_head.draft(
                history, max_tokens=self.draft_head.max_draft_tokens,
            )

            if not draft_tokens:
                self.stats.n_empty_drafts += 1

            # Step 2: Check if first draft matches our sampled token
            accepted_tokens = [first_token]
            remaining_draft: list[int] = []

            if draft_tokens and first_token == draft_tokens[0]:
                remaining_draft = draft_tokens[1:]
            elif draft_tokens:
                # First draft doesn't match — use the first_token we already sampled
                # and discard the entire draft
                remaining_draft = []

            self.stats.n_draft_generated += len(remaining_draft)

            # Step 3: Batch verify remaining draft tokens
            if remaining_draft:
                verify_tokens = [first_token] + remaining_draft
                verify_positions = [cur_pos - 1] + [cur_pos + i for i in range(len(remaining_draft))]

                try:
                    verify_result = self.target.decode_batch(verify_tokens, verify_positions)
                except Exception as e:
                    logger.warning(f"Batch verify failed: {e}, falling back to single decode")
                    verify_result = self.target.decode_batch([first_token], [cur_pos - 1])
                    remaining_draft = []

                self.stats.n_target_decodes += 1
                self.stats.n_target_decode_tokens += len(verify_tokens)

                # Verify each draft token against target logits
                n_accepted = 0
                rejected = False

                for i, draft_tok in enumerate(remaining_draft):
                    if i + 1 >= len(verify_result.logits_per_position):
                        break

                    target_tok = sample_from_logits(
                        verify_result.logits_per_position[i + 1], temp, k, p,
                    )

                    if target_tok == draft_tok and target_tok != self.target.eos_token_id():
                        accepted_tokens.append(draft_tok)
                        n_accepted += 1
                    else:
                        # Rejection: use target's prediction instead
                        if target_tok != self.target.eos_token_id():
                            accepted_tokens.append(target_tok)
                        rejected = True
                        break

                self.stats.n_draft_accepted += n_accepted

                # On rejection, we need to re-decode from the correction token
                # to fix the KV cache. But for HTTP backends this is stateless,
                # so no recovery needed.
                first_token = accepted_tokens[-1] if accepted_tokens else first_token
            else:
                # No draft or first didn't match — single token decode
                self.stats.n_target_decodes += 1
                self.stats.n_target_decode_tokens += 1
                result = self.target.decode_batch([first_token], [cur_pos - 1])
                first_token = sample_from_logits(
                    result.logits_per_position[0], temp, k, p,
                )
                accepted_tokens.append(first_token)

            # Update state
            output_tokens.extend(accepted_tokens)
            history.extend(accepted_tokens)

            if first_token == self.target.eos_token_id():
                break
            if stop_tokens and first_token in stop_tokens:
                break

        t_end = time.time()
        t_decode = t_end - t_prefill

        text = self.target.detokenize(output_tokens)

        if len(output_tokens) >= max_tokens:
            stop_reason = "max_tokens"
        elif output_tokens and output_tokens[-1] == self.target.eos_token_id():
            stop_reason = "eos"
        elif stop_tokens and output_tokens and output_tokens[-1] in stop_tokens:
            stop_reason = "stop_token"
        else:
            stop_reason = "max_tokens"

        tps = len(output_tokens) / max(t_decode, 1e-6)

        return GenerateResult(
            text=text,
            tokens=output_tokens,
            tokens_per_second=tps,
            prompt_tokens=n_prompt,
            stop_reason=stop_reason,
            time_prefill=t_prefill - t0,
            time_decode=t_decode,
            time_total=t_end - t0,
        )

    # ── Streaming generate ────────────────────────────────────────

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        speculative: bool = True,
        stop_tokens: Optional[list[int]] = None,
    ) -> Generator[tuple[int, str], None, GenerateResult]:
        """Streaming generate. Yields (token_id, text_so_far) tuples.

        The final yield is a GenerateResult with full stats.
        """
        temp = temperature if temperature is not None else self.temperature
        k = top_k if top_k is not None else self.top_k
        p = top_p if top_p is not None else self.top_p

        self.reset_stats()
        self.draft_head.reset()

        t0 = time.time()
        prompt_tokens = self.target.tokenize(prompt)
        n_prompt = len(prompt_tokens)

        self.target.reset()
        last_logits = self.target.prefill(prompt_tokens)
        t_prefill = time.time()

        first_token = sample_from_logits(last_logits, temp, k, p)
        output_tokens: list[int] = []
        history = list(prompt_tokens)

        if first_token == self.target.eos_token_id():
            yield 0, ""
            t_end = time.time()
            yield GenerateResult(
                text="", tokens=[], tokens_per_second=0.0,
                prompt_tokens=n_prompt, stop_reason="eos",
                time_prefill=t_prefill - t0, time_decode=t_end - t_prefill,
                time_total=t_end - t0,
            )
            return

        output_tokens.append(first_token)
        history.append(first_token)
        yield first_token, self.target.detokenize(output_tokens)

        while len(output_tokens) < max_tokens:
            cur_pos = len(history)

            if not speculative:
                self.stats.n_target_decodes += 1
                self.stats.n_target_decode_tokens += 1
                result = self.target.decode_batch([first_token], [cur_pos - 1])
                first_token = sample_from_logits(result.logits_per_position[0], temp, k, p)
                output_tokens.append(first_token)
                history.append(first_token)
                yield first_token, self.target.detokenize(output_tokens)
                if first_token == self.target.eos_token_id():
                    break
                if stop_tokens and first_token in stop_tokens:
                    break
                continue

            # Speculative
            self.stats.n_spec_steps += 1

            draft_tokens = self.draft_head.draft(
                history, max_tokens=self.draft_head.max_draft_tokens,
            )

            if not draft_tokens:
                self.stats.n_empty_drafts += 1

            accepted_tokens = [first_token]
            remaining_draft: list[int] = []

            if draft_tokens and first_token == draft_tokens[0]:
                remaining_draft = draft_tokens[1:]

            self.stats.n_draft_generated += len(remaining_draft)

            if remaining_draft:
                verify_tokens = [first_token] + remaining_draft
                verify_positions = [cur_pos - 1] + [cur_pos + i for i in range(len(remaining_draft))]

                try:
                    verify_result = self.target.decode_batch(verify_tokens, verify_positions)
                except Exception:
                    verify_result = self.target.decode_batch([first_token], [cur_pos - 1])
                    remaining_draft = []

                self.stats.n_target_decodes += 1
                self.stats.n_target_decode_tokens += len(verify_tokens)

                n_accepted = 0
                for i, draft_tok in enumerate(remaining_draft):
                    if i + 1 >= len(verify_result.logits_per_position):
                        break
                    target_tok = sample_from_logits(
                        verify_result.logits_per_position[i + 1], temp, k, p,
                    )
                    if target_tok == draft_tok and target_tok != self.target.eos_token_id():
                        accepted_tokens.append(draft_tok)
                        n_accepted += 1
                    else:
                        if target_tok != self.target.eos_token_id():
                            accepted_tokens.append(target_tok)
                        break

                self.stats.n_draft_accepted += n_accepted
                first_token = accepted_tokens[-1] if accepted_tokens else first_token
            else:
                self.stats.n_target_decodes += 1
                self.stats.n_target_decode_tokens += 1
                result = self.target.decode_batch([first_token], [cur_pos - 1])
                first_token = sample_from_logits(result.logits_per_position[0], temp, k, p)
                accepted_tokens.append(first_token)

            output_tokens.extend(accepted_tokens)
            history.extend(accepted_tokens)

            # Yield each accepted token
            text_so_far = self.target.detokenize(output_tokens)
            for tok in accepted_tokens:
                yield tok, text_so_far

            if first_token == self.target.eos_token_id():
                break
            if stop_tokens and first_token in stop_tokens:
                break

        t_end = time.time()
        t_decode = t_end - t_prefill
        tps = len(output_tokens) / max(t_decode, 1e-6)

        if output_tokens and output_tokens[-1] == self.target.eos_token_id():
            stop_reason = "eos"
        elif stop_tokens and output_tokens and output_tokens[-1] in stop_tokens:
            stop_reason = "stop_token"
        else:
            stop_reason = "max_tokens"

        yield GenerateResult(
            text=self.target.detokenize(output_tokens),
            tokens=output_tokens,
            tokens_per_second=tps,
            prompt_tokens=n_prompt,
            stop_reason=stop_reason,
            time_prefill=t_prefill - t0,
            time_decode=t_decode,
            time_total=t_end - t0,
        )

    # ── Convenience ───────────────────────────────────────────────

    def benchmark(
        self,
        prompt: str,
        max_tokens: int = 256,
        n_runs: int = 3,
        speculative: bool = True,
    ) -> dict:
        """Run multiple generations and return aggregated stats."""
        results = []
        for i in range(n_runs):
            result = self.generate(
                prompt, max_tokens=max_tokens, speculative=speculative,
            )
            results.append(result)
            logger.info(
                f"Run {i+1}/{n_runs}: {result.tokens_per_second:.1f} tok/s, "
                f"accept={self.stats.acceptance_rate:.1%}"
            )

        avg_tps = sum(r.tokens_per_second for r in results) / len(results)
        avg_accept = self.stats.acceptance_rate

        return {
            "avg_tokens_per_second": avg_tps,
            "acceptance_rate": avg_accept,
            "runs": [
                {
                    "tps": r.tokens_per_second,
                    "tokens": len(r.tokens),
                    "time_decode": r.time_decode,
                }
                for r in results
            ],
            "speculative": speculative,
            "draft_strategy": self.draft_head.name(),
            "target_backend": self.target.name(),
        }
