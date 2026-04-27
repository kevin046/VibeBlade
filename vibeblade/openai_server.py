"""VibeBlade OpenAI-Compatible Server.

Implements the OpenAI Chat Completions and Completions API so any
OpenAI-compatible client (curl, python-openai, LangChain, etc.) can
talk to a local VibeBlade model.

Endpoints:
  POST /v1/chat/completions      — Chat (system/user/assistant messages)
  POST /v1/completions            — Legacy text completions
  GET  /v1/models                 — List loaded models
  GET  /health                    — Liveness check
  GET  /v1/me                     — API info (whoami-like)

All inference runs in a thread pool so the async event loop stays free.

Usage:
    vibeblade serve --model /path/to/model.gguf --port 8080
    # or
    python -m vibeblade.openai_server --model /path/to/model.gguf
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
API_VERSION = "1.0.0"

# ── Pydantic-free request/response models (stdlib only) ──


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
    stream: bool = False
    stop: list[str] = field(default_factory=list)
    # OpenAI compat fields we accept but may not fully honor
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
    stream: bool = False
    stop: list[str] = field(default_factory=list)
    echo: bool = False


@dataclass
class LogProbs:
    pass  # placeholder for future token logprobs


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ── Model Registry (holds loaded VibeBladeModel instances) ──


class ModelRegistry:
    """Thread-safe singleton holding loaded model instances."""

    def __init__(self):
        self._models: dict[str, object] = {}
        self._lock = asyncio.Lock()

    async def load(self, model_id: str, model_path: str, **kwargs) -> None:
        """Load a VibeBladeModel in a background thread."""
        from . import VibeBladeModel

        def _load():
            return VibeBladeModel(model_path, **kwargs)

        model = await asyncio.get_event_loop().run_in_executor(None, _load)
        self._models[model_id] = model
        logger.info("Loaded model %s from %s", model_id, model_path)

    def get(self, model_id: str):
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(f"Model '{model_id}' not loaded. Available: {list(self._models.keys())}")
        return model

    def list_models(self) -> list[dict]:
        return [
            {"id": mid, "object": "model", "owned_by": "vibeblade"}
            for mid in self._models
        ]

    def remove(self, model_id: str) -> bool:
        return self._models.pop(model_id, None) is not None


# Global registry
_registry = ModelRegistry()


# ── Prompt formatting ──


def _format_chat_prompt(messages: list[ChatMessage]) -> str:
    """Convert OpenAI-style messages to a single prompt string.

    Uses LLaMA-style chat template:
      <|im_start|>system
      {system}<|im_end|>
      <|im_start|>user
      {user}<|im_end|>
      <|im_start|>assistant
    """
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


def _apply_stop(generated_text: str, stop: list[str]) -> str:
    """Truncate text at the first occurrence of any stop sequence."""
    if not stop:
        return generated_text
    earliest = len(generated_text)
    for s in stop:
        idx = generated_text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return generated_text[:earliest]


# ── Tokenizer helpers (works without a real tokenizer) ──


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English."""
    return max(1, len(text) // 4)


# ── Chat completion generation (runs in thread pool) ──


async def _generate_chat(
    model, prompt: str, max_tokens: int, temperature: float,
    top_p: float, stop: list[str],
) -> tuple[str, int, dict]:
    """Run VibeBladeModel.generate in a thread and return (text, n_tokens, stats).

    This uses a simple character-level fallback if no tokenizer is loaded,
    or the real tokenizer if available.
    """
    loop = asyncio.get_event_loop()

    def _run():
        # Try to use the model's tokenizer if available
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(prompt)
        else:
            # Fallback: character-level encoding (works for any text)
            token_ids = np.array([ord(c) % 32000 for c in prompt], dtype=np.int32)

        # Run generation
        result_ids, tps, stats = model.generate(
            token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Decode output
        n_prompt = stats.get("n_prompt", len(token_ids))
        n_generated = stats.get("n_generated", len(result_ids) - n_prompt)
        gen_ids = result_ids[n_prompt:]

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            text = tokenizer.decode(gen_ids)
        else:
            # Character-level decode: map back to chars
            text = "".join(chr(t % 256) for t in gen_ids)

        text = _apply_stop(text, stop)
        return text, n_generated, stats

    return await loop.run_in_executor(None, _run)


# ── App Factory ──


def create_app(model_id: str = "vibeblade", model_path: str = "",
               registry: ModelRegistry | None = None):
    """Create the FastAPI application.

    Args:
        model_id: Name to expose the model as in /v1/models
        model_path: Path to GGUF file (auto-loads on startup)
        registry: Inject a custom ModelRegistry (for testing)
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(
        title="VibeBlade",
        description="OpenAI-compatible API for local LLM inference",
        version=API_VERSION,
        docs_url="/docs",
        redoc_url=None,
    )

    reg = registry or _registry

    # Auto-load model on startup if path provided
    @app.on_event("startup")
    async def _startup():
        if model_path:
            try:
                await reg.load(model_id, model_path)
                logger.info("Auto-loaded model '%s' from %s", model_id, model_path)
            except Exception as e:
                logger.error("Failed to load model '%s': %s", model_id, e)

    # ── Health ──

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": API_VERSION}

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
            model = reg.get(mid)
        except KeyError as e:
            raise HTTPException(404, str(e))

        prompt = _format_chat_prompt(req.messages)

        if req.stream:
            return StreamingResponse(
                _stream_chat(model, prompt, req),
                media_type="text/event-stream",
            )

        # Non-streaming
        text, n_gen, stats = await _generate_chat(
            model, prompt, req.max_tokens, req.temperature,
            req.top_p, req.stop,
        )
        n_prompt_tokens = _estimate_tokens(prompt)

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": mid,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": _finish_reason(text, req.max_tokens),
            }],
            "usage": {
                "prompt_tokens": n_prompt_tokens,
                "completion_tokens": n_gen,
                "total_tokens": n_prompt_tokens + n_gen,
            },
        })

    # ── Text Completions (legacy) ──

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if not req.prompt:
            raise HTTPException(400, "prompt is required")

        mid = req.model or model_id
        try:
            model = reg.get(mid)
        except KeyError as e:
            raise HTTPException(404, str(e))

        if req.stream:
            return StreamingResponse(
                _stream_completion(model, req),
                media_type="text/event-stream",
            )

        text, n_gen, stats = await _generate_chat(
            model, req.prompt, req.max_tokens, req.temperature,
            req.top_p, req.stop,
        )

        result_text = (req.prompt + text) if req.echo else text
        n_prompt_tokens = _estimate_tokens(req.prompt)

        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": mid,
            "choices": [{
                "index": 0,
                "text": result_text,
                "finish_reason": _finish_reason(text, req.max_tokens),
            }],
            "usage": {
                "prompt_tokens": n_prompt_tokens,
                "completion_tokens": n_gen,
                "total_tokens": n_prompt_tokens + n_gen,
            },
        })

    # ── Streaming helpers ──

    async def _stream_chat(model, prompt: str, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks as SSE.

        Note: VibeBlade generates synchronously (numpy), so we generate the
        full response then stream it word-by-word. True token-by-token streaming
        would require an async-native generation loop (future work).
        """
        mid = req.model or model_id
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        full_text, n_gen, stats = await _generate_chat(
            model, prompt, req.max_tokens, req.temperature,
            req.top_p, req.stop,
        )

        # Stream word-by-word for SSE effect
        words = full_text.split(" ")
        for i, word in enumerate(words):
            token_text = word if i == 0 else " " + word
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": mid,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

        # Final chunk with finish_reason
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": mid,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": _finish_reason(full_text, req.max_tokens),
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_completion(model, req: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream text completion chunks as SSE."""
        mid = req.model or model_id
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        full_text, n_gen, stats = await _generate_chat(
            model, req.prompt, req.max_tokens, req.temperature,
            req.top_p, req.stop,
        )

        # If echo, emit prompt as first chunk
        if req.echo:
            chunk = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": mid,
                "choices": [{
                    "index": 0,
                    "text": req.prompt,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        words = full_text.split(" ")
        for i, word in enumerate(words):
            token_text = word if i == 0 else " " + word
            chunk = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": mid,
                "choices": [{
                    "index": 0,
                    "text": token_text,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

        chunk = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": mid,
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": _finish_reason(full_text, req.max_tokens),
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return app


def _finish_reason(text: str, max_tokens: int) -> str:
    """Determine why generation stopped."""
    if len(text) >= max_tokens:
        return "length"
    return "stop"


# ── CLI Entry Point ──


def main():
    """Start the OpenAI-compatible VibeBlade server."""
    import argparse

    parser = argparse.ArgumentParser(description="VibeBlade OpenAI-compatible Server")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--model-id", default="vibeblade", help="Model name in API")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--sparse", action="store_true", help="Enable sparse activation")
    parser.add_argument("--minicache", action="store_true", help="Enable MiniCache KV compression")
    parser.add_argument("--paged-attn", action="store_true", help="Enable PagedAttention")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n  ⚡ VibeBlade OpenAI Server")
    print(f"  Model: {args.model_id} ({args.model})")
    print(f"  API:   http://localhost:{args.port}/v1")
    print(f"  Docs:  http://localhost:{args.port}/docs\n")

    app = create_app(model_id=args.model_id, model_path=args.model)

    # Apply optimizations if requested
    @app.on_event("startup")
    async def _apply_opts():
        try:
            model = _registry.get(args.model_id)
            if args.sparse:
                model.enable_sparse()
                logger.info("Sparse activation enabled")
            if args.minicache:
                model.enable_minicache()
                logger.info("MiniCache enabled")
            if args.paged_attn:
                model.enable_paged_attn()
                logger.info("PagedAttention enabled")
        except Exception as e:
            logger.error("Failed to apply model options: %s", e)

    import uvicorn
    uvicorn.run(
        "vibeblade.openai_server:create_app",
        factory=False,  # we've already created the app
        app=app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
