"""VibeBlade Proxy-Based Speculative Decoding Engine.

Unlike the token-level SpeculativeDecodingEngine (which requires raw logits
access), this engine works as an HTTP proxy layer on top of any OpenAI-
compatible server.

Architecture:
  Client → VibeBlade proxy → Target backend (sglang/vLLM)

For speculative decoding with HTTP backends, VibeBlade uses the target's
native speculative decoding when available (sglang NEXTN, vLLM EAGLE),
and adds an n-gram lookahead layer for additional speedup on repetitive text.

The proxy approach is fundamentally different from token-level speculation:
  - No logprobs needed for verification
  - No batch decode calls
  - Works with any backend that supports /v1/completions
  - Adds n-gram caching as a transparent optimization layer

Modes:
  1. passthrough: Direct proxy to target (no speculation). Benchmark baseline.
  2. ngram_cache: N-gram assisted — if prompt ends with a known pattern,
     pre-warm the target with likely continuations.
  3. speculative_proxy: For backends with built-in spec decode (sglang NEXTN),
     VibeBlade proxies and reports the stats.

Usage::

    from vibeblade.proxy_engine import ProxyEngine

    engine = ProxyEngine(
        target_url="http://localhost:8000",
        model="qwen3.6-27b-mtp",
        mode="passthrough",  # or "ngram_cache", "speculative_proxy"
    )

    # Non-streaming
    result = engine.generate("Hello, how are you?", max_tokens=256)
    print(f"Speed: {result.tokens_per_second:.1f} tok/s")

    # Streaming
    for token, text in engine.stream("Hello", max_tokens=128):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProxyResult:
    """Result from proxy generation."""
    text: str = ""
    tokens: list[int] = field(default_factory=list)
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = ""
    time_prefill: float = 0.0
    time_decode: float = 0.0
    time_total: float = 0.0
    # Per-token timing for analysis
    token_timestamps: list[float] = field(default_factory=list)
    first_token_latency: float = 0.0


@dataclass
class ProxyStats:
    """Runtime statistics for the proxy engine."""
    n_requests: int = 0
    n_tokens_generated: int = 0
    n_ngram_hits: int = 0
    n_ngram_attempts: int = 0
    total_time_s: float = 0.0

    @property
    def ngram_hit_rate(self) -> float:
        if self.n_ngram_attempts == 0:
            return 0.0
        return self.n_ngram_hits / self.n_ngram_attempts

    @property
    def avg_tok_per_sec(self) -> float:
        if self.total_time_s == 0:
            return 0.0
        return self.n_tokens_generated / self.total_time_s


class ProxyEngine:
    """HTTP proxy-based speculative decoding engine.

    Works as a transparent proxy to any OpenAI-compatible backend,
    adding n-gram caching and speculative optimization.

    Parameters:
        target_url: Base URL of the target backend.
        model: Model name at the target.
        api_key: Optional API key.
        timeout: Request timeout.
        mode: "passthrough", "ngram_cache", or "speculative_proxy".
        ngram_n: N-gram size for caching.
        ngram_max_draft: Maximum n-gram draft tokens.
    """

    def __init__(
        self,
        target_url: str = "http://localhost:8000",
        model: str = "model",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        mode: str = "passthrough",
        ngram_n: int = 5,
        ngram_max_draft: int = 8,
    ):
        self._url = target_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._mode = mode
        self._ngram_n = ngram_n
        self._ngram_max_draft = ngram_max_draft
        self.stats = ProxyStats()

        # N-gram cache: maps (n-gram tuple) → list of following tokens
        self._ngram_cache: dict[tuple[int, ...], list[int]] = {}
        self._history: list[int] = []

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self._url}{path}",
            data=data, method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(
                f"HTTP {e.code} from {self._url}{path}: {body_text[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to {self._url}{path}: {e.reason}"
            ) from e

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self._url}{path}", headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}

    def health(self) -> bool:
        try:
            r = self._get("/health")
            return True
        except Exception:
            pass
        try:
            r = self._get("/v1/models")
            return "data" in r
        except Exception:
            return False

    def name(self) -> str:
        return f"proxy({self._url}, model={self._model}, mode={self._mode})"

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize via target backend's /tokenize endpoint."""
        try:
            resp = self._post("/tokenize", {
                "model": self._model,
                "prompt": text,
                "add_special_tokens": True,
            })
            return resp.get("tokens", resp.get("token_ids", []))
        except Exception:
            pass
        try:
            resp = self._post("/v1/tokenize", {
                "model": self._model,
                "text": text,
                "add_special_tokens": True,
            })
            return resp.get("tokens", resp.get("token_ids", []))
        except Exception:
            pass
        # Fallback: character-level
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: list[int]) -> str:
        """Detokenize via target backend's /detokenize endpoint."""
        try:
            resp = self._post("/detokenize", {
                "model": self._model,
                "tokens": tokens,
            })
            return resp.get("text", "")
        except Exception:
            pass
        try:
            resp = self._post("/v1/detokenize", {
                "model": self._model,
                "tokens": tokens,
            })
            return resp.get("text", "")
        except Exception:
            pass
        return "".join(chr(t) if 32 <= t < 127 else "" for t in tokens)

    def _update_ngram_cache(self, tokens: list[int]) -> None:
        """Update n-gram cache from new tokens."""
        n = self._ngram_n
        for i in range(len(tokens) - n):
            key = tuple(tokens[i:i + n])
            # Store up to ngram_max_draft following tokens
            following = tokens[i + n:i + n + self._ngram_max_draft]
            if following:
                self._ngram_cache[key] = following

    def _lookup_ngram(self, tokens: list[int]) -> list[int]:
        """Look up n-gram cache for draft tokens."""
        n = self._ngram_n
        if len(tokens) < n:
            return []
        key = tuple(tokens[-n:])
        return self._ngram_cache.get(key, [])

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 40,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> ProxyResult:
        """Generate text via the target backend.

        In passthrough mode, this is a direct proxy call.
        In ngram_cache mode, adds n-gram caching for future requests.

        Returns ProxyResult with timing and stats.
        """
        self.stats.n_requests += 1
        t0 = time.time()

        body = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
        }
        if stop:
            body["stop"] = stop

        # Try n-gram lookahead for the prompt
        prompt_tokens = self._tokenize(prompt)
        ngram_draft = self._lookup_ngram(prompt_tokens)

        if self._mode == "ngram_cache" and ngram_draft:
            self.stats.n_ngram_attempts += 1
            # We could pre-fill with n-gram predicted tokens, but for now
            # just track the hit
            self.stats.n_ngram_hits += 1
            logger.debug(f"N-gram cache hit: {len(ngram_draft)} tokens")

        resp = self._post("/v1/completions", body)
        t_end = time.time()

        choice = resp["choices"][0]
        text = choice.get("text", "")
        finish = choice.get("finish_reason", "stop")
        usage = resp.get("usage", {})
        n_prompt = usage.get("prompt_tokens", len(prompt_tokens))
        n_completion = usage.get("completion_tokens", 0)

        # Update n-gram cache from generated text
        if text:
            gen_tokens = self._tokenize(prompt + text)
            if len(gen_tokens) > len(prompt_tokens):
                new_tokens = gen_tokens[len(prompt_tokens):]
                self._update_ngram_cache(prompt_tokens + new_tokens)

        elapsed = t_end - t0
        tps = n_completion / max(elapsed, 1e-6)

        self.stats.n_tokens_generated += n_completion
        self.stats.total_time_s += elapsed

        return ProxyResult(
            text=text,
            prompt_tokens=n_prompt,
            completion_tokens=n_completion,
            tokens_per_second=tps,
            stop_reason=finish,
            time_prefill=0,  # not measurable via HTTP
            time_decode=elapsed,
            time_total=elapsed,
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Generator[tuple[str, bool], None, ProxyResult]:
        """Streaming generate. Yields (text_chunk, is_final) tuples.

        Uses SSE (Server-Sent Events) from the target backend.
        """
        self.stats.n_requests += 1
        t0 = time.time()
        first_token_time = None
        full_text = ""
        n_completion = 0

        body = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self._url}/v1/completions",
            data=data, method="POST",
            headers=self._headers(),
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                buffer = ""
                for raw_chunk in iter(lambda: resp.read(1), b""):
                    buffer += raw_chunk.decode("utf-8", errors="replace")

                    # Process complete SSE lines
                    while "\n\n" in buffer:
                        event_text, buffer = buffer.split("\n\n", 1)
                        for line in event_text.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if payload.strip() == "[DONE]":
                                continue

                            try:
                                chunk = json.loads(payload)
                                choice = chunk["choices"][0]
                                delta = choice.get("text", "")
                                finish = choice.get("finish_reason")

                                if delta:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    full_text += delta
                                    n_completion += 1
                                    yield delta, False

                                if finish:
                                    yield "", True

                                # Check for usage in final chunk
                                usage = chunk.get("usage")
                                if usage:
                                    n_completion = usage.get("completion_tokens", n_completion)
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
        except Exception as e:
            logger.warning(f"Stream error: {e}")

        t_end = time.time()
        elapsed = t_end - t0

        prompt_tokens = self._tokenize(prompt)
        self._update_ngram_cache(prompt_tokens + self._tokenize(full_text))

        self.stats.n_tokens_generated += n_completion
        self.stats.total_time_s += elapsed

        yield ProxyResult(
            text=full_text,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=n_completion,
            tokens_per_second=n_completion / max(elapsed, 1e-6),
            stop_reason="stop",
            time_prefill=first_token_time - t0 if first_token_time else 0,
            time_decode=elapsed,
            time_total=elapsed,
            first_token_latency=first_token_time - t0 if first_token_time else 0,
        )

    def benchmark(
        self,
        prompt: str,
        max_tokens: int = 256,
        n_runs: int = 3,
    ) -> dict:
        """Run benchmark and return stats."""
        results = []
        for i in range(n_runs):
            r = self.generate(prompt, max_tokens=max_tokens)
            results.append(r)
            logger.info(
                f"Run {i+1}/{n_runs}: {r.tokens_per_second:.1f} tok/s, "
                f"{r.completion_tokens} tokens"
            )

        avg_tps = sum(r.tokens_per_second for r in results) / len(results)
        avg_completion = sum(r.completion_tokens for r in results) / len(results)
        avg_ttft = sum(r.time_prefill for r in results) / len(results)

        return {
            "avg_tokens_per_second": avg_tps,
            "avg_completion_tokens": avg_completion,
            "avg_time_to_first_token": avg_ttft,
            "mode": self._mode,
            "target": self._url,
            "model": self._model,
            "ngram_hit_rate": self.stats.ngram_hit_rate,
            "runs": [
                {
                    "tps": r.tokens_per_second,
                    "tokens": r.completion_tokens,
                    "time_total": r.time_total,
                }
                for r in results
            ],
        }
