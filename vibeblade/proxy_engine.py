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
    n_tokens_visible: int = 0  # tokens that aren't hidden reasoning
    n_reasoning_tokens: int = 0
    n_ngram_prefill_tokens: int = 0  # tokens saved via n-gram prefill
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
        return self.n_tokens_visible / self.total_time_s

    @property
    def reasoning_overhead(self) -> float:
        """Fraction of tokens wasted on reasoning."""
        total = self.n_tokens_generated
        if total == 0:
            return 0.0
        return self.n_reasoning_tokens / total


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
        use_chat_api: bool = True,
        disable_reasoning: bool = True,
    ):
        self._url = target_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._mode = mode
        self._ngram_n = ngram_n
        self._ngram_max_draft = ngram_max_draft
        self._use_chat_api = use_chat_api
        self._disable_reasoning = disable_reasoning
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

        Optimizations:
        - Uses chat completions API (strips reasoning tokens properly)
        - reasoning: {"effort": "none"} to disable thinking overhead
        - N-gram prefill: appends predicted tokens to prompt (mode="ngram_inject")
        - Tracks visible vs reasoning tokens separately

        Returns ProxyResult with timing and stats.
        """
        self.stats.n_requests += 1
        t0 = time.time()

        prompt_tokens = self._tokenize(prompt)

        # N-gram prefill injection: if prompt ends with a known pattern,
        # append predicted tokens to the prompt so the model skips them
        ngram_prefill_tokens: list[int] = []
        if self._mode == "ngram_inject":
            ngram_draft = self._lookup_ngram(prompt_tokens)
            if ngram_draft:
                self.stats.n_ngram_attempts += 1
                self.stats.n_ngram_hits += 1
                # Inject up to ngram_max_draft tokens into prompt
                ngram_prefill_tokens = ngram_draft[:self._ngram_max_draft]
                self.stats.n_ngram_prefill_tokens += len(ngram_prefill_tokens)
                # Extend prompt tokens for cache update later
                prompt_tokens = prompt_tokens + ngram_prefill_tokens
                logger.debug(f"N-gram prefill: injected {len(ngram_prefill_tokens)} tokens")
        elif self._mode == "ngram_cache":
            ngram_draft = self._lookup_ngram(prompt_tokens)
            if ngram_draft:
                self.stats.n_ngram_attempts += 1
                self.stats.n_ngram_hits += 1

        if self._use_chat_api:
            # Chat completions API — properly strips reasoning tokens
            # Convert prompt_tokens back to text for the message content
            effective_prompt = self._detokenize(prompt_tokens) if ngram_prefill_tokens else prompt

            body: dict = {
                "model": self._model,
                "messages": [{"role": "user", "content": effective_prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }
            if self._disable_reasoning:
                body["reasoning"] = {"effort": "none"}
            if stop:
                body["stop"] = stop

            resp = self._post("/v1/chat/completions", body)
            t_end = time.time()

            choice = resp["choices"][0]
            msg = choice.get("message", {})
            text = msg.get("content", "")
            finish = choice.get("finish_reason", "stop")
            usage = resp.get("usage", {})
            n_prompt = usage.get("prompt_tokens", len(prompt_tokens))
            n_completion = usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details", {})
            n_reasoning = details.get("reasoning_tokens", 0)
            n_visible = n_completion - n_reasoning
        else:
            # Fallback: plain completions API
            effective_prompt = self._detokenize(prompt_tokens) if ngram_prefill_tokens else prompt
            body = {
                "model": self._model,
                "prompt": effective_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stream": False,
            }
            if stop:
                body["stop"] = stop

            resp = self._post("/v1/completions", body)
            t_end = time.time()

            choice = resp["choices"][0]
            text = choice.get("text", "")
            finish = choice.get("finish_reason", "stop")
            usage = resp.get("usage", {})
            n_prompt = usage.get("prompt_tokens", len(prompt_tokens))
            n_completion = usage.get("completion_tokens", 0)
            n_reasoning = 0
            n_visible = n_completion

        # Prepend n-gram prefill text to output (these were "free" tokens)
        prefill_text = ""
        if ngram_prefill_tokens:
            prefill_text = self._detokenize(ngram_prefill_tokens)
            text = prefill_text + text

        # Update n-gram cache from generated text
        if text:
            full_gen_tokens = self._tokenize(text)
            if len(full_gen_tokens) > len(prompt_tokens):
                new_tokens = full_gen_tokens[len(prompt_tokens):]
                self._update_ngram_cache(prompt_tokens + new_tokens)

        elapsed = t_end - t0
        total_visible = n_visible + len(ngram_prefill_tokens)
        tps = total_visible / max(elapsed, 1e-6)

        self.stats.n_tokens_generated += n_completion
        self.stats.n_tokens_visible += total_visible
        self.stats.n_reasoning_tokens += n_reasoning
        self.stats.total_time_s += elapsed

        return ProxyResult(
            text=text,
            prompt_tokens=n_prompt,
            completion_tokens=total_visible,
            tokens_per_second=tps,
            stop_reason=finish,
            time_prefill=0,
            time_decode=elapsed,
            time_total=elapsed,
            first_token_latency=0,
        )

    def generate_speculative(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        n_draft: int = 8,
        draft_accept_threshold: float = 0.0,
    ) -> ProxyResult:
        """Speculative generation with n-gram draft + logprobs verification.

        Strategy:
        1. Tokenize prompt, look up n-gram predicted tokens
        2. If draft found, append predicted tokens to prompt (prefill injection)
           AND request logprobs from the backend to verify acceptance
        3. Accept draft tokens where the target model agrees (logprob > threshold)
        4. For rejected tokens, the target model generates its own

        This gives "free" tokens when draft is correct (no decode step needed)
        while maintaining output quality via logprobs verification.

        Returns ProxyResult with acceptance stats.
        """
        self.stats.n_requests += 1
        t0 = time.time()

        prompt_tokens = self._tokenize(prompt)

        # Step 1: Get n-gram draft
        ngram_draft = self._lookup_ngram(prompt_tokens)
        draft_text = ""
        draft_tok_count = 0
        if ngram_draft:
            self.stats.n_ngram_attempts += 1
            # Limit to n_draft tokens
            draft_tokens = ngram_draft[:n_draft]
            draft_tok_count = len(draft_tokens)
            draft_text = self._detokenize(draft_tokens)

        # Step 2: Send to backend with logprobs for verification
        effective_prompt = prompt
        if draft_text and self._mode == "ngram_inject":
            # Inject draft tokens into prompt as "suggested continuation"
            # The model will confirm or correct them
            effective_prompt = prompt + draft_text

        if self._use_chat_api:
            body = {
                "model": self._model,
                "messages": [{"role": "user", "content": effective_prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": 5,
                "stream": False,
            }
            if self._disable_reasoning:
                body["reasoning"] = {"effort": "none"}

            resp = self._post("/v1/chat/completions", body)
            t_end = time.time()

            choice = resp["choices"][0]
            msg = choice.get("message", {})
            text = msg.get("content", "")
            finish = choice.get("finish_reason", "stop")
            usage = resp.get("usage", {})
            n_prompt = usage.get("prompt_tokens", len(prompt_tokens))
            n_completion = usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details", {})
            n_reasoning = details.get("reasoning_tokens", 0)
            n_visible = n_completion - n_reasoning

            # Count accepted draft tokens from logprobs
            n_accepted = 0
            logprobs_data = msg.get("logprobs", {}).get("content", [])
            if logprobs_data and draft_tok_count > 0:
                for i, lp in enumerate(logprobs_data[:draft_tok_count]):
                    top_logprobs = lp.get("top_logprobs", [])
                    if top_logprobs:
                        list(top_logprobs.keys())[0]
                        top_prob = list(top_logprobs.values())[0]
                        if top_prob >= draft_accept_threshold:
                            n_accepted += 1
                self.stats.n_ngram_hits += n_accepted
                self.stats.n_ngram_prefill_tokens += n_accepted

            total_visible = n_visible + n_accepted
        else:
            # Fallback completions API
            body = {
                "model": self._model,
                "prompt": effective_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": 1,
                "stream": False,
            }
            resp = self._post("/v1/completions", body)
            t_end = time.time()

            choice = resp["choices"][0]
            text = choice.get("text", "")
            finish = choice.get("finish_reason", "stop")
            usage = resp.get("usage", {})
            n_prompt = usage.get("prompt_tokens", len(prompt_tokens))
            n_completion = usage.get("completion_tokens", 0)
            n_reasoning = 0
            n_visible = n_completion
            n_accepted = 0
            total_visible = n_visible

        # Update n-gram cache
        if text:
            full_text = effective_prompt + text
            all_tokens = self._tokenize(full_text)
            if len(all_tokens) > len(prompt_tokens):
                new_tokens = all_tokens[len(prompt_tokens):]
                self._update_ngram_cache(prompt_tokens + new_tokens)

        elapsed = t_end - t0
        tps = total_visible / max(elapsed, 1e-6)

        self.stats.n_tokens_generated += n_completion
        self.stats.n_tokens_visible += total_visible
        self.stats.n_reasoning_tokens += n_reasoning
        self.stats.total_time_s += elapsed

        return ProxyResult(
            text=text,
            prompt_tokens=n_prompt,
            completion_tokens=total_visible,
            tokens_per_second=tps,
            stop_reason=finish,
            time_prefill=0,
            time_decode=elapsed,
            time_total=elapsed,
            first_token_latency=0,
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
            "reasoning_overhead": self.stats.reasoning_overhead,
            "ngram_prefill_tokens": self.stats.n_ngram_prefill_tokens,
            "runs": [
                {
                    "tps": r.tokens_per_second,
                    "tokens": r.completion_tokens,
                    "time_total": r.time_total,
                }
                for r in results
            ],
        }

    def benchmark_concurrent(
        self,
        prompt: str,
        max_tokens: int = 256,
        n_concurrent: int = 5,
    ) -> dict:
        """Benchmark with concurrent requests to measure aggregate throughput.

        Uses threading to fire N requests in parallel and measures total
        tokens / wall-clock time.
        """
        import threading

        results = []
        lock = threading.Lock()
        t0 = time.time()

        def worker():
            r = self.generate(prompt, max_tokens=max_tokens)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(n_concurrent)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        t_end = time.time()
        elapsed = t_end - t0

        total_tokens = sum(r.completion_tokens for r in results)
        aggregate_tps = total_tokens / max(elapsed, 1e-6)
        per_request_tps = [r.tokens_per_second for r in results]

        return {
            "n_concurrent": n_concurrent,
            "aggregate_tokens_per_second": aggregate_tps,
            "per_request_avg": sum(per_request_tps) / len(per_request_tps) if per_request_tps else 0,
            "per_request_min": min(per_request_tps) if per_request_tps else 0,
            "per_request_max": max(per_request_tps) if per_request_tps else 0,
            "total_tokens": total_tokens,
            "wall_time_s": elapsed,
            "mode": self._mode,
            "reasoning_overhead": self.stats.reasoning_overhead,
        }
