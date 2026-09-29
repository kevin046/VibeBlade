"""VibeBlade OpenAI-Compatible Server.

Implements the OpenAI Chat Completions and Completions API so any
OpenAI-compatible client (curl, python-openai, LangChain, etc.) can
talk to a VibeBlade speculative decoding layer on top of any backend.

Supports:
  - Target backends: sglang, vLLM, llama.cpp, any OpenAI-compatible HTTP server
  - Draft strategies: ngram, eagle, dflash, nextn

Endpoints:
  POST /v1/chat/completions      — Chat (system/user/assistant messages)
  POST /v1/completions            — Legacy text completions
  GET  /v1/models                 — List loaded models
  GET  /health                    — Liveness check
  GET  /v1/me                     — API info (whoami-like)

Usage:
    vibeblade serve --backend sglang --backend-url http://localhost:8000 \\
                    --model qwen3.6-27b --draft ngram --port 8080

    vibeblade serve --backend openai --backend-url http://localhost:8000 \\
                    --model my-model --draft eagle --draft-model draft.gguf
"""

from __future__ import annotations

import asyncio
import json
import threading
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# ── Constants ──

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
API_VERSION = "2.0.0"

# ── Request/Response Models (stdlib only, no pydantic) ──


@dataclass
class ChatMessage:
    role: str = "user"
    content: str = ""


@dataclass
class ChatCompletionRequest:
    model: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stream: bool = False
    stop: list[str] = field(default_factory=list)
    # Extra fields for speculative decoding control
    speculative: bool = True
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    user: str = ""


@dataclass
class CompletionRequest:
    model: str = ""
    prompt: str = ""
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stream: bool = False
    stop: list[str] = field(default_factory=list)
    echo: bool = False
    speculative: bool = True
    logprobs: Optional[int] = None


# ── Engine Registry ──


class EngineRegistry:
    """Holds SpeculativeDecodingEngine instances keyed by model ID."""

    def __init__(self):
        self._engines: dict[str, object] = {}

    def register(self, model_id: str, engine) -> None:
        self._engines[model_id] = engine
        logger.info("Registered engine '%s' (draft=%s)", model_id, type(engine).__name__)

    def get(self, model_id: str):
        engine = self._engines.get(model_id)
        if engine is None:
            raise KeyError(f"Engine '{model_id}' not loaded. Available: {list(self._engines.keys())}")
        return engine

    def remove(self, model_id: str) -> bool:
        """Unload an engine. Returns True if it was loaded, False otherwise."""
        if model_id in self._engines:
            del self._engines[model_id]
            logger.info("Unloaded engine '%s'", model_id)
            return True
        return False

    def list_models(self) -> list[dict]:
        return [
            {"id": mid, "object": "model", "owned_by": "vibeblade"}
            for mid in self._engines
        ]


_registry = EngineRegistry()
# Legacy alias: the test suite and docs refer to the model-facing registry by this name.
ModelRegistry = EngineRegistry


# ── Prompt formatting ──


def _format_chat_prompt(messages: list[ChatMessage]) -> str:
    """Convert OpenAI-style messages to a single prompt string."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


def _apply_stop(generated_text: str, stop: list[str]) -> str:
    if not stop:
        return generated_text
    earliest = len(generated_text)
    for s in stop:
        idx = generated_text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return generated_text[:earliest]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate when no tokenizer stats are available (~4 chars/token).

    Always returns at least 1 — the OpenAI API never reports 0 tokens for a request.
    """
    return max(1, len(text or "") // 4)


def _finish_reason(text: str, max_tokens: int) -> str:
    """OpenAI finish_reason: 'length' when the output filled the budget, else 'stop'."""
    if max_tokens is not None and _estimate_tokens(text) >= max_tokens:
        return "length"
    return "stop"


# ── Generation (runs in thread pool) ──


async def _generate(
    engine, prompt: str, max_tokens: int, temperature: float,
    top_p: float, top_k: int, stop: list[str], speculative: bool,
) -> tuple[str, int, str, dict]:
    """Run engine.generate in thread pool.

    Returns (text, n_completion_tokens, finish_reason, stats_dict).
    """
    loop = asyncio.get_event_loop()

    def _run():
        temp = temperature if temperature > 0 else 0.0
        result = engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            speculative=speculative,
        )

        text = _apply_stop(result.text or "", stop)

        if result.stop_reason == "max_tokens":
            finish = "length"
        else:
            finish = _finish_reason(text, max_tokens)

        stats = {
            "tokens_per_second": result.tokens_per_second,
            "time_prefill": result.time_prefill,
            "time_decode": result.time_decode,
            "time_total": result.time_total,
            "prompt_tokens": result.prompt_tokens,
        }

        # Add speculative stats if available
        if hasattr(engine, 'stats'):
            stats["spec_acceptance_rate"] = engine.stats.acceptance_rate
            stats["spec_speedup"] = engine.stats.effective_speedup
            stats["spec_draft_yield"] = engine.stats.draft_yield_rate
            stats["spec_draft_accepted"] = engine.stats.n_draft_accepted
            stats["spec_draft_generated"] = engine.stats.n_draft_generated

        return text, len(result.tokens), finish, stats

    return await loop.run_in_executor(None, _run)




# ── Prometheus-style Metrics (stdlib, no dependency) ─────────────────────────


class MetricsRegistry:
    """Minimal Prometheus text-format metrics: counters + latency histograms.

    Thread-safe; intended to be exposed at GET /metrics.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._latency_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                                 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")]
        self._latency: dict[str, list[int]] = {}   # route -> bucket counts
        self._latency_sum: dict[str, float] = {}   # route -> total seconds
        self._latency_count: dict[str, int] = {}

    def inc(self, name: str, value: float = 1.0, labels: str = "") -> None:
        key = name if not labels else f"{name}{{{labels}}}"
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe_latency(self, route: str, seconds: float) -> None:
        with self._lock:
            if route not in self._latency:
                self._latency[route] = [0] * len(self._latency_buckets)
                self._latency_sum[route] = 0.0
                self._latency_count[route] = 0
            for i, upper in enumerate(self._latency_buckets):
                if seconds <= upper:
                    self._latency[route][i] += 1
            self._latency_sum[route] += seconds
            self._latency_count[route] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for key, value in sorted(self._counters.items()):
                lines.append(f"{key} {value}")
            for route in sorted(self._latency):
                safe = route.strip("/").replace("/", "_") or "root"
                cum = 0
                for i, upper in enumerate(self._latency_buckets):
                    cum += self._latency[route][i]
                    bound = "" if upper == float("inf") else f"{upper}"
                    lines.append(
                        f'http_request_duration_seconds_bucket{{route="{safe}",le="{bound}"}} {cum}')
                lines.append(f'http_request_duration_seconds_sum{{route="{safe}"}} '
                             f'{self._latency_sum[route]:.6f}')
                lines.append(f'http_request_duration_seconds_count{{route="{safe}"}} '
                             f'{self._latency_count[route]}')
        lines.append("# TYPE vibeblade_uptime_seconds gauge")
        lines.append(f"vibeblade_uptime_seconds {time.time() - _START_TIME:.3f}")
        return "\n".join(lines) + "\n"


_START_TIME = time.time()
_metrics = MetricsRegistry()


# ── App Factory ──


def create_app(
    engine=None,
    model_id: str = "vibeblade",
    registry: Optional[EngineRegistry] = None,
    api_keys: Optional[set] = None,
    enable_metrics: bool = True,
):
    """Create the FastAPI application wired to a SpeculativeDecodingEngine.

    Args:
        api_keys: Set of valid caller Bearer keys. None/empty = auth disabled
            (bind to localhost only in that case!).
        enable_metrics: Expose GET /metrics in Prometheus text format.
    """
    import secrets as _secrets
    import time as _time

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    app = FastAPI(
        title="VibeBlade",
        description="Universal speculative decoding layer — OpenAI-compatible API",
        version=API_VERSION,
        docs_url="/docs",
        redoc_url=None,
    )

    reg = registry or _registry
    # Routes that never require auth.
    _PUBLIC_PATHS = {"/health", "/metrics"}

    class _VibeBladeMiddleware(BaseHTTPMiddleware):
        """Incoming API-key auth + request metrics + structured access log."""

        async def dispatch(self, request: Request, call_next):
            request_id = _secrets.token_hex(8)
            started = _time.perf_counter()
            auth_error = None
            if api_keys and request.url.path not in _PUBLIC_PATHS:
                header = request.headers.get("authorization", "")
                token = header[7:].strip() if header.lower().startswith("bearer ") else ""
                if not token:
                    auth_error = "missing"
                elif not any(_secrets.compare_digest(token, k) for k in api_keys):
                    auth_error = "invalid"
            if auth_error:
                _metrics.inc("vibeblade_auth_failures_total", labels=f'reason="{auth_error}"')
                logger.warning("req_id=%s auth=%s path=%s", request_id, auth_error, request.url.path)
                return JSONResponse(
                    {"error": {"message": f"Invalid API key ({auth_error})", "type": "auth_error",
                               "code": "invalid_api_key"}},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer", "X-Request-Id": request_id},
                )
            try:
                response = await call_next(request)
            except Exception:
                _metrics.inc("vibeblade_requests_total",
                             labels=f'route="{request.url.path}",status="500"')
                logger.exception("req_id=%s unhandled_error path=%s", request_id, request.url.path)
                raise
            elapsed = _time.perf_counter() - started
            response.headers["X-Request-Id"] = request_id
            _metrics.inc("vibeblade_requests_total",
                         labels=f'route="{request.url.path}",status="{response.status_code}"')
            _metrics.observe_latency(request.url.path, elapsed)
            logger.info("req_id=%s %s %s -> %s %.1fms", request_id, request.method,
                        request.url.path, response.status_code, elapsed * 1000)
            return response

    app.add_middleware(_VibeBladeMiddleware)

    @app.on_event("startup")
    async def _startup():
        if engine is not None:
            reg.register(model_id, engine)
            logger.info("Auto-registered engine '%s'", model_id)

    # ── Health ──

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": API_VERSION, "model": model_id}

    # ── Metrics (Prometheus text format) ──

    @app.get("/metrics")
    async def metrics():
        if not enable_metrics:
            raise HTTPException(404, "metrics disabled")
        return PlainTextResponse(_metrics.render(), media_type="text/plain; version=0.0.4")

    # ── API Info ──

    @app.get("/v1/me")
    async def api_info():
        return {
            "name": "VibeBlade",
            "version": API_VERSION,
            "models": [m["id"] for m in reg.list_models()],
        }

    # ── List Models ──

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": reg.list_models()}

    # ── Chat Completions ──

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(400, "messages is required")

        mid = req.model or model_id
        try:
            eng = reg.get(mid)
        except KeyError as e:
            raise HTTPException(404, str(e))

        prompt = _format_chat_prompt(req.messages)

        if req.stream:
            return StreamingResponse(
                _stream_chat(eng, prompt, req),
                media_type="text/event-stream",
            )

        text, n_gen, finish, stats = await _generate(
            eng, prompt, req.max_tokens, req.temperature,
            req.top_p, req.top_k, req.stop, req.speculative,
        )

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": mid,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": n_gen,
                "total_tokens": stats.get("prompt_tokens", 0) + n_gen,
            },
            "vibeblade_stats": stats,
        })

    # ── Text Completions (legacy) ──

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if not req.prompt:
            raise HTTPException(400, "prompt is required")

        mid = req.model or model_id
        try:
            eng = reg.get(mid)
        except KeyError as e:
            raise HTTPException(404, str(e))

        if req.stream:
            return StreamingResponse(
                _stream_completion(eng, req),
                media_type="text/event-stream",
            )

        text, n_gen, finish, stats = await _generate(
            eng, req.prompt, req.max_tokens, req.temperature,
            req.top_p, req.top_k, req.stop, req.speculative,
        )

        result_text = (req.prompt + text) if req.echo else text

        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": mid,
            "choices": [{
                "index": 0,
                "text": result_text,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": n_gen,
                "total_tokens": stats.get("prompt_tokens", 0) + n_gen,
            },
            "vibeblade_stats": stats,
        })

    # ── Speculative decoding stats endpoint ──

    @app.get("/v1/stats")
    async def get_stats():
        """Return speculative decoding statistics."""
        mid = model_id
        try:
            eng = reg.get(mid)
        except KeyError:
            return {"error": f"Engine '{mid}' not found"}

        if hasattr(eng, 'stats'):
            return {
                "engine": mid,
                "draft_strategy": eng.draft_head.name(),
                "target_backend": eng.target.name(),
                "stats": str(eng.stats),
                "acceptance_rate": eng.stats.acceptance_rate,
                "effective_speedup": eng.stats.effective_speedup,
                "draft_yield_rate": eng.stats.draft_yield_rate,
            }
        return {"engine": mid, "stats": "not available"}

    # ── Streaming helpers ──

    async def _stream_chat(eng, prompt: str, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        mid = req.model or model_id
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        full_text, n_gen, finish, stats = await _generate(
            eng, prompt, req.max_tokens, req.temperature,
            req.top_p, req.top_k, req.stop, req.speculative,
        )

        # Stream word-by-word
        words = full_text.split(" ")
        for i, word in enumerate(words):
            token_text = word if i == 0 else " " + word
            chunk = {
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": mid,
                "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

        chunk = {
            "id": req_id, "object": "chat.completion.chunk",
            "created": created, "model": mid,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_completion(eng, req: CompletionRequest) -> AsyncGenerator[str, None]:
        mid = req.model or model_id
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        full_text, n_gen, finish, stats = await _generate(
            eng, req.prompt, req.max_tokens, req.temperature,
            req.top_p, req.top_k, req.stop, req.speculative,
        )

        if req.echo:
            chunk = {
                "id": req_id, "object": "text_completion",
                "created": created, "model": mid,
                "choices": [{"index": 0, "text": req.prompt, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        words = full_text.split(" ")
        for i, word in enumerate(words):
            token_text = word if i == 0 else " " + word
            chunk = {
                "id": req_id, "object": "text_completion",
                "created": created, "model": mid,
                "choices": [{"index": 0, "text": token_text, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

        chunk = {
            "id": req_id, "object": "text_completion",
            "created": created, "model": mid,
            "choices": [{"index": 0, "text": "", "finish_reason": finish}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return app


# ── CLI Entry Point ──


def main(argv: list[str] | None = None):
    """Start the VibeBlade speculative decoding server."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="VibeBlade — Universal Speculative Decoding Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # N-gram draft with sglang target
  vibeblade serve --backend sglang --backend-url http://localhost:8000 \\
                  --model qwen3.6-27b --draft ngram --max-draft 8

  # EAGLE draft with vLLM target
  vibeblade serve --backend vllm --backend-url http://localhost:8000 \\
                  --model qwen3.6-27b --draft eagle --draft-model draft.gguf

  # DFlash draft (requires target with hidden state access)
  vibeblade serve --backend sglang --backend-url http://localhost:8000 \\
                  --model qwen3.6-27b --draft dflash

  # NEXTN draft (n-gram + neural hybrid)
  vibeblade serve --backend sglang --backend-url http://localhost:8000 \\
                  --model qwen3.6-27b --draft nextn --max-draft 6
        """,
    )

    # Backend args
    parser.add_argument("--backend", default="openai",
                        choices=["sglang", "vllm", "llama_cpp", "openai"],
                        help="Target model backend type (default: openai)")
    parser.add_argument("--backend-url",
                        default=os.environ.get("VIBEBLADE_BACKEND_URL",
                                               "http://localhost:8000"),
                        help="Target backend URL (default: http://localhost:8000 "
                             "or VIBEBLADE_BACKEND_URL)")
    parser.add_argument("--model", default=os.environ.get("VIBEBLADE_MODEL"),
                        help="Model name at the target backend (or VIBEBLADE_MODEL)")
    parser.add_argument("--api-key", default=None,
                        help="API key for target backend (if required)")

    # Draft args
    parser.add_argument("--draft", default="ngram",
                        choices=["ngram", "eagle", "dflash", "nextn", "none"],
                        help="Draft strategy (default: ngram)")
    parser.add_argument("--max-draft", type=int, default=8,
                        help="Maximum draft tokens per step (default: 8)")
    parser.add_argument("--draft-model", default=None,
                        help="Path to draft model (for EAGLE/DFlash)")
    parser.add_argument("--draft-ngram-size", type=int, default=5,
                        help="N-gram context size (default: 5)")

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k filtering (default: 40)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p (nucleus) filtering (default: 0.95)")

    # Server args (env overrides let Docker/compose configure without argv changes)
    parser.add_argument("--host", default=os.environ.get("VIBEBLADE_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("VIBEBLADE_PORT", DEFAULT_PORT)))
    parser.add_argument("--reload", action="store_true", help="Hot reload")
    _env_key = os.environ.get("VIBEBLADE_API_KEY")
    parser.add_argument("--require-api-key", default=_env_key, metavar="KEY",
                        help="Require this Bearer key from API callers (server-side auth). "
                             "Use 'env:VARNAME' to read from an environment variable. "
                             "Falls back to VIBEBLADE_API_KEY. "
                             "Auto-generates one when binding to a non-localhost host without a key.")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Disable the GET /metrics endpoint")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Build target backend ──
    if args.backend == "sglang":
        from .backends.sglang_backend import SglangTargetBackend
        target = SglangTargetBackend(
            base_url=args.backend_url,
            model=args.model,
            api_key=args.api_key,
        )
    elif args.backend == "vllm":
        from .backends.vllm_backend import VllmTargetBackend
        target = VllmTargetBackend(
            base_url=args.backend_url,
            model=args.model,
            api_key=args.api_key,
        )
    elif args.backend == "llama_cpp":
        from .backends.openai_http_backend import OpenAIHttpTargetBackend
        target = OpenAIHttpTargetBackend(
            base_url=args.backend_url,
            model=args.model,
            api_key=args.api_key,
            eos_token_id=2,  # llama.cpp default
            vocab_size=32000,
        )
    else:
        from .backends.openai_http_backend import OpenAIHttpTargetBackend
        target = OpenAIHttpTargetBackend(
            base_url=args.backend_url,
            model=args.model,
            api_key=args.api_key,
        )

    # ── Build draft head ──
    if args.draft == "none":
        from .draft_heads import NgramDraftHead
        draft = NgramDraftHead(max_draft=0)  # disabled
    elif args.draft == "ngram":
        from .draft_heads import NgramDraftHead
        draft = NgramDraftHead(
            n=args.draft_ngram_size,
            max_draft=args.max_draft,
        )
    elif args.draft == "eagle":
        from .draft_heads import EAGLEDraftHead
        draft = EAGLEDraftHead(
            max_draft=args.max_draft,
            draft_backend=None,  # set via --draft-model in future
        )
    elif args.draft == "dflash":
        from .draft_heads import DFlashDraftHead
        if not args.draft_model:
            print("  ⚠️  --draft-model is required for DFlash. Using n-gram fallback.")
            from .draft_heads import NgramDraftHead
            draft = NgramDraftHead(max_draft=args.max_draft)
        else:
            draft = DFlashDraftHead(
                draft_model_name=args.draft_model,
                block_size=args.max_draft,
            )
    elif args.draft == "nextn":
        from .draft_heads import NEXTNDraftHead
        draft = NEXTNDraftHead(
            max_draft=args.max_draft,
            ngram=NgramDraftHead(
                n=args.draft_ngram_size,
                max_draft=args.max_draft,
            ),
        )
    else:
        raise ValueError(f"Unknown draft strategy: {args.draft}")

    # ── Build engine ──
    from .speculative_decoding import SpeculativeDecodingEngine
    engine = SpeculativeDecodingEngine(
        target=target,
        draft_head=draft,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # ── Health check ──
    print(f"\n  ⚡ VibeBlade Speculative Decoding Server v{API_VERSION}")
    print(f"  Target:  {args.backend} @ {args.backend_url} (model={args.model})")
    print(f"  Draft:   {args.draft} (max_draft={args.max_draft})")
    print(f"  API:     http://localhost:{args.port}/v1")
    print(f"  Docs:    http://localhost:{args.port}/docs\n")

    if target.health():
        print("  ✅ Target backend is healthy\n")
    else:
        print(f"  ⚠️  Target backend not responding at {args.backend_url}")
        print("     Will retry on first request.\n")

    # ── Server-side auth ──
    caller_keys: set = set()
    if args.require_api_key:
        if args.require_api_key.startswith("env:"):
            import os
            key = os.environ.get(args.require_api_key[4:], "")
            if not key:
                parser.error(f"environment variable {args.require_api_key[4:]} is not set")
        else:
            key = args.require_api_key
        caller_keys = {key}
    elif args.host not in ("127.0.0.1", "localhost", "::1"):
        import secrets as _secrets_mod
        generated = f"vb-{_secrets_mod.token_hex(24)}"
        caller_keys = {generated}
        print(f"  🔑 Auto-generated API key (no key given for non-localhost bind): {generated}")

    app = create_app(engine=engine, model_id=args.model,
                     api_keys=caller_keys or None,
                     enable_metrics=not args.no_metrics)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
