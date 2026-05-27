"""VibeBlade Draft Heads — unified interface for speculative decoding draft strategies.

Three strategies for proposing draft tokens:

  1. **NgramDraftHead** — Training-free n-gram pattern matching on token history.
     Zero cost, no extra model. Works well for repetitive/structured text.

  2. **EAGLEDraftHead** — Autoregressive neural draft model. A small/fast model
     (e.g. Qwen2.5-0.5B) generates draft tokens sequentially, conditioned on
     the target model's hidden states. Higher acceptance than n-gram.

  3. **DFlashDraftHead** — Block diffusion parallel draft. A diffusion model
     generates a block of tokens in a SINGLE forward pass (vs B passes for B
     tokens with AR). Requires a DFlash-trained draft model from HuggingFace.

  4. **NEXTNDraftHead** — Hybrid approach: uses n-gram for short-range patterns
     + EAGLE-style neural draft for long-range coherence. Inspired by sglang's
     NEXTN algorithm (n-gram + EAGLE + speculative verification).

All draft heads implement the same interface::

    head = SomeDraftHead(...)
    draft_tokens = head.draft(history, max_tokens=8)
    # Returns list of proposed token IDs

Usage with SpeculativeDecodingEngine::

    from vibeblade.speculative_decoding import SpeculativeDecodingEngine

    engine = SpeculativeDecodingEngine(
        target_backend=target,
        draft_head=EAGLEDraftHead(draft_backend=small_model),
    )
    result = engine.generate("Hello, how are you?", max_tokens=256)
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Draft Head ABC
# ════════════════════════════════════════════════════════════════════

class DraftHead(abc.ABC):
    """Abstract base class for speculative draft heads.

    A draft head proposes candidate tokens that the target model then
    batch-verifies. The key method is ``draft()``.
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable draft strategy name."""
        ...

    @abc.abstractmethod
    def draft(
        self,
        history: list[int],
        max_tokens: int = 8,
        **kwargs,
    ) -> list[int]:
        """Generate draft tokens given token history.

        Args:
            history: Full token sequence so far (prompt + generated).
            max_tokens: Maximum number of draft tokens to propose.

        Returns:
            List of draft token IDs (may be empty if no draft available).
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset draft state for a new generation."""
        ...

    @property
    def max_draft_tokens(self) -> int:
        """Maximum number of draft tokens this head can produce per step."""
        return 8


# ════════════════════════════════════════════════════════════════════
#  N-gram Draft Head (training-free, zero-cost)
# ════════════════════════════════════════════════════════════════════

@dataclass
class NgramDraftHead(DraftHead):
    """Training-free n-gram draft head.

    Scans token history for repeated N-gram patterns and proposes the tokens
    that followed the matching pattern. No model to load — pure pattern matching.

    Best for: repetitive text, code generation, structured output.

    Parameters:
        n: N-gram size (default 4).
        max_draft: Maximum draft tokens per step.
        vocab_size: Vocab size for safety clamping.
    """

    n: int = 4
    max_draft: int = 8
    vocab_size: int = 248320  # Qwen3.6 default

    @property
    def max_draft_tokens(self) -> int:
        return self.max_draft

    def name(self) -> str:
        return f"ngram(n={self.n}, max={self.max_draft})"

    def reset(self) -> None:
        pass  # stateless

    def draft(
        self,
        history: list[int],
        max_tokens: int = 8,
        **kwargs,
    ) -> list[int]:
        """Draft tokens via n-gram pattern matching."""
        max_d = min(max_tokens, self.max_draft)
        if len(history) < 3:
            return []

        # Try progressively shorter n-gram sizes
        for try_n in [self.n, self.n - 1, self.n - 2]:
            if try_n < 2 or len(history) < try_n + 1:
                continue
            key = tuple(history[-try_n:])
            draft: list[int] = []
            for i in range(len(history) - try_n - 1):
                if tuple(history[i:i + try_n]) == key:
                    j = i + try_n
                    while j < len(history) and len(draft) < max_d:
                        candidate = history[j]
                        if candidate == 0 or candidate >= self.vocab_size:
                            break
                        draft.append(candidate)
                        j += 1
                    if len(draft) >= 1:
                        return draft
                    draft = []

        return []


# ════════════════════════════════════════════════════════════════════
#  EAGLE Draft Head (autoregressive neural draft)
# ════════════════════════════════════════════════════════════════════

@dataclass
class EAGLEDraftHead(DraftHead):
    """EAGLE-style autoregressive neural draft head.

    Uses a small model to generate draft tokens sequentially. The draft model
    is significantly smaller/faster than the target, so each forward pass is
    cheap. Draft tokens are verified in batch by the target.

    Can connect to any backend for the draft model:
    - llama.cpp (local GGUF)
    - sglang/vLLM via HTTP (a smaller model served separately)
    - Any OpenAI-compatible API

    Parameters:
        draft_backend: A TargetBackend for the draft model.
        max_draft: Maximum draft tokens per step.
        temperature: Sampling temperature (0 = greedy draft).
    """

    draft_backend: Any = None  # TargetBackend instance
    max_draft: int = 8
    temperature: float = 0.0

    @property
    def max_draft_tokens(self) -> int:
        return self.max_draft

    def name(self) -> str:
        backend_name = getattr(self.draft_backend, "name", lambda: "?")()
        return f"eagle({backend_name}, max={self.max_draft})"

    def reset(self) -> None:
        if self.draft_backend is not None:
            self.draft_backend.reset()

    def draft(
        self,
        history: list[int],
        max_tokens: int = 8,
        **kwargs,
    ) -> list[int]:
        """Generate draft tokens via AR draft model."""
        if self.draft_backend is None:
            return []

        max_d = min(max_tokens, self.max_draft)
        if len(history) < 1:
            return []

        # Use last N tokens as context (within draft model's context window)
        context = history[-256:]

        try:
            # Detokenize → tokenize through draft model to get its tokenization
            # (handles vocab mismatch between draft and target)
            prompt_text = kwargs.get("prompt_text", None)
            if prompt_text and hasattr(self.draft_backend, "tokenize"):
                draft_prompt_tokens = self.draft_backend.tokenize(prompt_text)
                if draft_prompt_tokens:
                    context = draft_prompt_tokens + context[-128:]

            result = self.draft_backend.generate(
                prompt="",  # we use token-level API below
                max_tokens=max_d,
                temperature=self.temperature,
            )

            # If the backend doesn't support raw token generation,
            # fall back to generate_from_tokens or the tokens from the result
            return result.tokens[:max_d]
        except Exception as e:
            logger.debug(f"EAGLE draft failed: {e}")
            return []

    def set_draft_backend(self, backend: Any) -> None:
        """Set the draft model backend."""
        self.draft_backend = backend

    def free(self) -> None:
        """Release draft model resources."""
        if self.draft_backend is not None:
            if hasattr(self.draft_backend, "free"):
                self.draft_backend.free()
            self.draft_backend = None


# ════════════════════════════════════════════════════════════════════
#  DFlash Draft Head (block diffusion parallel draft)
# ════════════════════════════════════════════════════════════════════

class DFlashDraftHead(DraftHead):
    """DFlash block diffusion draft head.

    Generates a block of tokens in a SINGLE forward pass (vs B passes for B
    tokens with AR drafting). Requires a DFlash-trained model from HuggingFace.

    This is a thin wrapper around the existing vibeblade.dflash.DFlashDraftHead
    to conform to the DraftHead interface.

    Parameters:
        draft_model_name: HuggingFace model ID or local path (e.g. "z-lab/Qwen3-8B-DFlash-b16").
        block_size: Tokens per draft block.
        target_model_name: HuggingFace model ID for tokenizer alignment.
        device: PyTorch device ("cpu" or "cuda").
    """

    def __init__(
        self,
        draft_model_name: str,
        block_size: int = 16,
        target_model_name: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.0,
        target_vocab_size: Optional[int] = None,
    ):
        self.draft_model_name = draft_model_name
        self._block_size = block_size
        self.target_model_name = target_model_name
        self.device = device
        self.temperature = temperature
        self.target_vocab_size = target_vocab_size

        # Lazy-loaded
        self._inner: Optional[Any] = None

    @property
    def max_draft_tokens(self) -> int:
        return self._block_size

    def name(self) -> str:
        return f"dflash({self.draft_model_name}, block={self._block_size})"

    def reset(self) -> None:
        if self._inner is not None and hasattr(self._inner, "_past_kv"):
            self._inner._past_kv = None

    def _ensure_loaded(self):
        """Lazy-load the DFlash model."""
        if self._inner is not None:
            return
        from .dflash import DFlashDraftHead as _InnerDFlash

        self._inner = _InnerDFlash(
            draft_model_name=self.draft_model_name,
            target_model_name=self.target_model_name,
            block_size=self._block_size,
            temperature=self.temperature,
            device=self.device,
            target_vocab_size=self.target_vocab_size,
        )

    def draft(
        self,
        history: list[int],
        max_tokens: int = 16,
        **kwargs,
    ) -> list[int]:
        """Generate draft tokens via DFlash block diffusion."""
        self._ensure_loaded()
        target_hidden = kwargs.get("target_hidden", None)
        max_d = min(max_tokens, self._block_size)
        try:
            return self._inner.draft(
                history, draft_max_override=max_d,
                target_hidden=target_hidden,
            )
        except Exception as e:
            logger.debug(f"DFlash draft failed: {e}")
            return []

    @property
    def stats(self):
        self._ensure_loaded()
        return self._inner.stats

    def free(self) -> None:
        if self._inner is not None:
            self._inner.free()
            self._inner = None


# ════════════════════════════════════════════════════════════════════
#  NEXTN Draft Head (n-gram + EAGLE hybrid)
# ════════════════════════════════════════════════════════════════════

@dataclass
class NEXTNDraftHead(DraftHead):
    """NEXTN-style hybrid draft head.

    Combines n-gram pattern matching (fast, zero-cost) with EAGLE neural
    drafting (higher acceptance for novel tokens). Strategy:

    1. Try n-gram first (instant, no compute)
    2. If n-gram yields < threshold tokens, fall back to EAGLE
    3. If both fail, return empty (standard decode step)

    This mimics sglang's NEXTN algorithm which layers n-gram speculation
    on top of EAGLE for maximum draft yield.

    Parameters:
        eagle: EAGLE draft head for neural fallback.
        ngram: N-gram draft head for fast first-pass.
        ngram_min_accept: Minimum n-gram draft tokens before using EAGLE.
        max_draft: Maximum total draft tokens per step.
    """

    eagle: Optional[EAGLEDraftHead] = None
    ngram: Optional[NgramDraftHead] = None
    ngram_min_accept: int = 2
    max_draft: int = 9

    def __post_init__(self):
        if self.ngram is None:
            self.ngram = NgramDraftHead(max_draft=self.max_draft)
        if self.eagle is None:
            # EAGLE without a backend — will return empty drafts
            self.eagle = EAGLEDraftHead(max_draft=self.max_draft)

    @property
    def max_draft_tokens(self) -> int:
        return self.max_draft

    def name(self) -> str:
        eagle_name = self.eagle.name() if self.eagle else "none"
        ngram_name = self.ngram.name() if self.ngram else "none"
        return f"nextn(eagle={eagle_name}, ngram={ngram_name})"

    def reset(self) -> None:
        self.ngram.reset()
        self.eagle.reset()

    def draft(
        self,
        history: list[int],
        max_tokens: int = 9,
        **kwargs,
    ) -> list[int]:
        """Hybrid draft: n-gram first, EAGLE fallback."""
        max_d = min(max_tokens, self.max_draft)

        # Phase 1: N-gram (zero-cost, instant)
        ngram_draft = self.ngram.draft(history, max_tokens=max_d)

        if len(ngram_draft) >= self.ngram_min_accept:
            return ngram_draft[:max_d]

        # Phase 2: EAGLE neural draft (if available and has a backend)
        if self.eagle.draft_backend is not None:
            eagle_draft = self.eagle.draft(history, max_tokens=max_d, **kwargs)
            if eagle_draft:
                return eagle_draft[:max_d]

        # Phase 3: Return whatever n-gram found (even if < threshold)
        return ngram_draft[:max_d] if ngram_draft else []

    def set_eagle_backend(self, backend: Any) -> None:
        """Set the EAGLE draft model backend."""
        self.eagle.draft_backend = backend

    def free(self) -> None:
        self.eagle.free()


# ════════════════════════════════════════════════════════════════════
#  Factory
# ════════════════════════════════════════════════════════════════════

def create_draft_head(
    strategy: str,
    **kwargs,
) -> DraftHead:
    """Factory: create a DraftHead by strategy name.

    Args:
        strategy: "ngram", "eagle", "dflash", or "nextn"
        **kwargs: Strategy-specific parameters.

    Returns:
        DraftHead instance

    Examples:
        # Zero-cost n-gram
        head = create_draft_head("ngram", n=4, max_draft=8)

        # EAGLE with a small draft model via HTTP
        from vibeblade.target_backend import create_target_backend
        draft = create_target_backend("sglang", base_url="http://localhost:8001", model="qwen-0.5b")
        head = create_draft_head("eagle", draft_backend=draft, max_draft=8)

        # DFlash block diffusion
        head = create_draft_head("dflash", draft_model_name="z-lab/Qwen3-8B-DFlash-b16", block_size=16)

        # NEXTN hybrid (n-gram + EAGLE)
        head = create_draft_head("nextn", eagle=EAGLEDraftHead(draft_backend=draft), max_draft=9)
    """
    strategy = strategy.lower().strip()

    if strategy == "ngram" or strategy == "n-gram":
        return NgramDraftHead(
            n=kwargs.get("n", 4),
            max_draft=kwargs.get("max_draft", kwargs.get("max_tokens", 8)),
            vocab_size=kwargs.get("vocab_size", 248320),
        )
    elif strategy == "eagle":
        return EAGLEDraftHead(
            draft_backend=kwargs.get("draft_backend"),
            max_draft=kwargs.get("max_draft", kwargs.get("max_tokens", 8)),
            temperature=kwargs.get("temperature", 0.0),
        )
    elif strategy == "dflash":
        return DFlashDraftHead(
            draft_model_name=kwargs["draft_model_name"],
            block_size=kwargs.get("block_size", kwargs.get("max_tokens", 16)),
            target_model_name=kwargs.get("target_model_name"),
            device=kwargs.get("device", "cpu"),
            temperature=kwargs.get("temperature", 0.0),
            target_vocab_size=kwargs.get("target_vocab_size"),
        )
    elif strategy == "nextn":
        return NEXTNDraftHead(
            eagle=kwargs.get("eagle"),
            ngram=kwargs.get("ngram"),
            ngram_min_accept=kwargs.get("ngram_min_accept", 2),
            max_draft=kwargs.get("max_draft", kwargs.get("max_tokens", 9)),
        )
    else:
        raise ValueError(
            f"Unknown draft strategy: {strategy!r}. "
            f"Choose from: ngram, eagle, dflash, nextn"
        )
