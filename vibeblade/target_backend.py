"""VibeBlade Target Backend Abstraction.

Pluggable backends so VibeBlade's speculative decoding layer works with
ANY serving backend — sglang, vLLM, llama.cpp, or any OpenAI-compatible
HTTP API.

Architecture::

  ┌─────────────────────────────────────────────┐
  │           VibeBlade Speculative Layer         │
  │  (draft head → batch verify → accept/reject)  │
  └──────────────────┬──────────────────────────┘
                     │ TargetBackend ABC
        ┌────────────┼────────────┬──────────┐
        ▼            ▼            ▼          ▼
   SglangBackend  VllmBackend  LlamaCpp   OpenAIHttp
   (HTTP API)     (HTTP API)  (ctypes)   (generic)

Usage::

    from vibeblade.target_backend import create_target_backend

    # sglang
    backend = create_target_backend("sglang", base_url="http://localhost:8000", model="qwen3.6-27b")

    # vLLM
    backend = create_target_backend("vllm", base_url="http://localhost:8000", model="model")

    # llama.cpp (local GGUF)
    backend = create_target_backend("llama_cpp", model_path="model.gguf")

    # Any OpenAI-compatible API
    backend = create_target_backend("openai", base_url="http://localhost:8000", model="model")
"""

from __future__ import annotations

import abc
import json
import logging
import random
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ── Result Types ──

@dataclass
class TargetLogits:
    """Logits from the target model after a decode step.

    For batch verification of K draft tokens, this contains K+1 sets of
    logits (one for each position in the batch).
    """
    logits_per_position: list[list[float]]
    sampled_tokens: list[int]
    raw_response: Any = None


@dataclass
class GenerateResult:
    """Full generation result (non-streaming)."""
    text: str
    tokens: list[int]
    tokens_per_second: float
    prompt_tokens: int
    stop_reason: str  # "eos" | "stop_token" | "max_tokens"
    time_prefill: float = 0.0
    time_decode: float = 0.0
    time_total: float = 0.0


# ── Sampling Utility ──

def sample_from_logits(
    logits: list[float],
    temperature: float = 0.0,
    top_k: int = 40,
    top_p: float = 0.95,
) -> int:
    """Sample a token from logits (CPU, no torch/numpy dependency)."""
    if temperature == 0.0:
        return int(max(range(len(logits)), key=lambda i: logits[i]))

    # Apply temperature
    scaled = [l / temperature for l in logits]

    # Top-k filtering
    if 0 < top_k < len(scaled):
        sorted_indices = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
        for idx in sorted_indices[top_k:]:
            scaled[idx] = float("-inf")

    # Softmax
    max_val = max(scaled)
    if max_val == float("-inf"):
        return 0
    exps = [2.718281828 ** (s - max_val) if s != float("-inf") else 0.0 for s in scaled]
    total = sum(exps)
    if total == 0:
        return 0
    probs = [e / total for e in exps]

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_pairs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        cumsum = 0.0
        cutoff_idx = len(sorted_pairs)
        for i, (idx, p) in enumerate(sorted_pairs):
            cumsum += p
            if cumsum >= top_p:
                cutoff_idx = i + 1
                break
        nucleus_set = {idx for idx, _ in sorted_pairs[:cutoff_idx]}
        probs = [p if i in nucleus_set else 0.0 for i, p in enumerate(probs)]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

    # Sample
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1


# ── HTTP Helpers ──

def _http_post(url: str, body: dict, timeout: float = 120.0) -> dict:
    """POST JSON and return parsed response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from {url}: {body_text[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot connect to {url}: {e.reason}") from e


def _http_get(url: str, timeout: float = 10.0) -> dict:
    """GET JSON and return parsed response."""
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Abstract Backend ──

class TargetBackend(abc.ABC):
    """Abstract base class for target model backends.

    VibeBlade's speculative layer calls these methods to:
    1. Prefill the prompt
    2. Get first token + logits
    3. Batch-verify draft tokens in a single forward pass
    4. Sample from logits
    5. Tokenize/detokenize
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    @abc.abstractmethod
    def detokenize(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    @abc.abstractmethod
    def prefill(self, tokens: list[int]) -> list[float]:
        """Prefill prompt tokens. Returns logits for the last position."""
        ...

    @abc.abstractmethod
    def decode_batch(
        self,
        tokens: list[int],
        positions: list[int],
    ) -> TargetLogits:
        """Decode a batch of tokens at given positions.

        This is the core of batch verification: feed K+1 tokens
        (first_token + K draft tokens) to the target model and
        get logits for every position.

        Args:
            tokens: token IDs to decode (first is the sampled token,
                    rest are draft tokens)
            positions: position IDs for each token

        Returns:
            TargetLogits with logits and sampled tokens for each position.
        """
        ...

    @abc.abstractmethod
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        ...

    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset KV cache for a new generation."""
        ...

    @abc.abstractmethod
    def health(self) -> bool:
        """Check if the backend is healthy/ready."""
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        stop_tokens: Optional[list[int]] = None,
    ) -> GenerateResult:
        """Baseline autoregressive generate (no speculative)."""
        tokens = self.tokenize(prompt)
        t0 = time.time()
        self.reset()
        last_logits = self.prefill(tokens)
        t_prefill = time.time()

        output: list[int] = []
        pos = len(tokens)

        first_tok = sample_from_logits(last_logits, temperature, top_k, top_p)
        output.append(first_tok)

        for _ in range(max_tokens - 1):
            if first_tok == self.eos_token_id():
                break
            if stop_tokens and first_tok in stop_tokens:
                break

            result = self.decode_batch([first_tok], [pos])
            first_tok = result.sampled_tokens[0]
            pos += 1

            if first_tok == self.eos_token_id():
                break
            if stop_tokens and first_tok in stop_tokens:
                output.append(first_tok)
                break
            output.append(first_tok)

        t_end = time.time()
        tps = len(output) / max(t_end - t_prefill, 1e-6)
        text = self.detokenize(output)
        reason = "eos" if output and output[-1] == self.eos_token_id() else "max_tokens"
        return GenerateResult(
            text=text, tokens=output, tokens_per_second=tps,
            prompt_tokens=len(tokens), stop_reason=reason,
            time_prefill=t_prefill - t0, time_decode=t_end - t_prefill,
            time_total=t_end - t0,
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        stop_tokens: Optional[list[int]] = None,
    ) -> Generator[tuple[int, str], None, GenerateResult]:
        """Streaming generate. Yields (token_id, text_piece) tuples."""
        tokens = self.tokenize(prompt)
        t0 = time.time()
        self.reset()
        last_logits = self.prefill(tokens)
        t_prefill = time.time()

        output: list[int] = []
        pos = len(tokens)

        first_tok = sample_from_logits(last_logits, temperature, top_k, top_p)
        output.append(first_tok)
        text_so_far = self.detokenize(output)
        yield first_tok, text_so_far

        for _ in range(max_tokens - 1):
            if first_tok == self.eos_token_id():
                break
            if stop_tokens and first_tok in stop_tokens:
                break

            result = self.decode_batch([first_tok], [pos])
            first_tok = result.sampled_tokens[0]
            pos += 1

            if first_tok == self.eos_token_id():
                break
            if stop_tokens and first_tok in stop_tokens:
                output.append(first_tok)
                yield first_tok, self.detokenize(output)
                break

            output.append(first_tok)
            yield first_tok, self.detokenize(output)

        t_end = time.time()
        tps = len(output) / max(t_end - t_prefill, 1e-6)
        reason = "eos" if output and output[-1] == self.eos_token_id() else "max_tokens"
        yield GenerateResult(
            text=self.detokenize(output), tokens=output, tokens_per_second=tps,
            prompt_tokens=len(tokens), stop_reason=reason,
            time_prefill=t_prefill - t0, time_decode=t_end - t_prefill,
            time_total=t_end - t0,
        )


# ── Factory ──

def create_target_backend(
    backend_type: str,
    *,
    model_path: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    n_ctx: int = 2048,
    n_threads: int = 4,
    **kwargs,
) -> TargetBackend:
    """Factory: create a TargetBackend by type name.

    Args:
        backend_type: "sglang", "vllm", "llama_cpp", or "openai"
        base_url: HTTP base URL (for sglang/vllm/openai)
        model: model name (for sglang/vllm/openai served model name)
        model_path: path to GGUF file (for llama_cpp)
        api_key: optional API key for HTTP backends
        n_ctx: context window size (llama_cpp)
        n_threads: CPU threads (llama_cpp)

    Returns:
        TargetBackend instance
    """
    backend_type = backend_type.lower().replace("-", "_")

    if backend_type == "sglang":
        from .backends.sglang_backend import SglangTargetBackend
        return SglangTargetBackend(
            base_url=base_url or "http://localhost:8000",
            model=model or "model",
            api_key=api_key,
        )
    elif backend_type == "vllm":
        from .backends.vllm_backend import VllmTargetBackend
        return VllmTargetBackend(
            base_url=base_url or "http://localhost:8000",
            model=model or "model",
            api_key=api_key,
        )
    elif backend_type in ("llama_cpp", "llama-cpp", "llamacpp"):
        # llama.cpp server exposes OpenAI-compatible HTTP API
        from .backends.openai_http_backend import OpenAIHttpTargetBackend
        return OpenAIHttpTargetBackend(
            base_url=base_url or "http://localhost:8080",
            model=model or "model",
            api_key=api_key,
            eos_token_id=2,  # llama.cpp default EOS
            vocab_size=kwargs.get("vocab_size", 32000),
        )
    elif backend_type == "openai":
        from .backends.openai_http_backend import OpenAIHttpTargetBackend
        return OpenAIHttpTargetBackend(
            base_url=base_url or "http://localhost:8000",
            model=model or "model",
            api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            f"Choose from: sglang, vllm, llama_cpp, openai"
        )
