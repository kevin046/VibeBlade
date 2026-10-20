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
import re
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
    tool_calls: object = None   # assistant history: prior tool calls
    tool_call_id: str | None = None  # tool-role messages: which call this answers


@dataclass
class ToolDef:
    """OpenAI-style function tool definition."""
    type: str = "function"
    function: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ToolDef":
        fn = d.get("function", d)
        return cls(type=d.get("type", "function"), function=dict(fn) if isinstance(fn, dict) else {})


@dataclass
class ChatCompletionRequest:
    model: str = ""
    messages: list = field(default_factory=list)
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
    # ── Tool calling (OpenAI-compatible) ──
    tools: list = field(default_factory=list)
    tool_choice: object = None  # "auto" | "none" | "required" | {"type":"function",...}

    def __post_init__(self):
        # Normalize raw JSON dicts into ToolDef / ChatMessage
        self.tools = [t if isinstance(t, ToolDef) else ToolDef.from_dict(t) for t in (self.tools or [])]
        msgs = []
        for m in (self.messages or []):
            if isinstance(m, ChatMessage):
                msgs.append(m)
                continue
            d = dict(m)
            msgs.append(ChatMessage(
                role=d.get("role", "user"),
                content=d.get("content") or "",
                tool_calls=d.get("tool_calls"),
                tool_call_id=d.get("tool_call_id"),
            ))
        self.messages = msgs


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


def _format_chat_prompt(messages: list[ChatMessage], tools: list[ToolDef] | None = None) -> str:
    """Convert OpenAI-style messages (+ tools) to a Qwen3 ChatML prompt string.

    Tool definitions are rendered into a leading system block in the
    <tools>...</tools> format Qwen3 models are trained on; tool results come
    back as <tool_response> user messages; prior assistant tool calls are
    replayed as <tool_call> blocks.
    """
    parts = []
    if tools:
        entries = []
        for t in tools:
            fn = t.function or {}
            entries.append(json.dumps({
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                },
            }, ensure_ascii=False))
        tools_block = "<tools>\n" + ",\n".join(entries) + "\n</tools>"
        parts.append(
            "<|im_start|>system\n" + tools_block
            + "\nYou may call one or more functions to assist with the user query.<|im_end|>"
        )
    for msg in messages:
        if msg.role == "tool":
            parts.append(
                "<|im_start|>user\n<tool_response>\n" + (msg.content or "")
                + "\n</tool_response><|im_end|>"
            )
        elif msg.role == "assistant" and msg.tool_calls:
            blocks = []
            for tc in (msg.tool_calls or []):
                fn = (tc or {}).get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                blocks.append(
                    "<tool_call>\n"
                    + json.dumps({"name": fn.get("name", ""), "arguments": args},
                                 ensure_ascii=False)
                    + "\n</tool_call>"
                )
            parts.append("<|im_start|>assistant\n" + "".join(blocks) + "<|im_end|>")
        else:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract Qwen3 <tool_call>{...}</tool_call> blocks from generated text
    into OpenAI-format tool_calls."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or ""
        if not name:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args if isinstance(args, dict) else {"value": args},
                                         ensure_ascii=False),
            },
        })
    return calls


def _strip_tool_blocks(text: str) -> str:
    """Remove <tool_call> blocks (and stray think tags) from displayed content."""
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S)
    return cleaned.strip()


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

        tools = req.tools if (req.tool_choice != "none") else []
        prompt = _format_chat_prompt(req.messages, tools=tools)

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

        # Tool-call extraction (Qwen3 <tool_call> blocks → OpenAI tool_calls)
        tool_calls = _parse_tool_calls(text) if tools else []
        finish = "tool_calls" if tool_calls else _finish_reason(text, req.max_tokens)
        content = _strip_tool_blocks(text) if tool_calls else text

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": mid,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content if content else None,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": finish,
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

    # ── True token-by-token streaming helper ──

    async def _stream_tokens_async(
        model,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncGenerator[tuple[str, float], None]:
        """True token-by-token streaming via on_token callback.
        
        Uses run_in_executor with a generator that yields tokens as they're
        generated. Each yielded tuple is (cumulative_text, tps).
        Yields None as last item to signal completion.
        """
        import queue
        
        loop = asyncio.get_event_loop()
        token_queue: queue.Queue[tuple[str, float] | None] = queue.Queue()
        
        def on_token_callback(token_id: int, pos: int) -> None:
            """Called in the generation thread. Decodes and queues each token."""
            try:
                tokenizer = getattr(model, "tokenizer", None)
                if tokenizer is not None and hasattr(tokenizer, "decode"):
                    text = tokenizer.decode([token_id], stop_at_eos=False)
                else:
                    try:
                        text = chr(token_id) if 32 <= token_id < 127 else ""
                    except (ValueError, OverflowError):
                        text = ""
                token_queue.put_nowait((text, 0.0))  # tps computed later
            except queue.Full:
                pass  # Skip if queue is full (backpressure)
        
        def run_generation():
            """Runs in thread pool. Uses on_token to stream tokens."""
            try:
                tokenizer = getattr(model, "tokenizer", None)
                if tokenizer is not None and hasattr(tokenizer, "encode"):
                    token_ids = tokenizer.encode(prompt)
                else:
                    token_ids = np.array([ord(c) % 32000 for c in prompt], dtype=np.int32)
                
                model.generate(
                    token_ids=token_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,  # We handle streaming via on_token
                    on_token=on_token_callback,
                )
            except Exception as e:
                logger.error("Generation error: %s", e)
            finally:
                # Signal completion with None sentinel
                try:
                    token_queue.put_nowait(None)
                except queue.Full:
                    pass
        
        # Start generation in thread pool
        future = loop.run_in_executor(None, run_generation)
        
        # Yield tokens as they arrive
        accumulated_text = ""
        while True:
            item = await loop.run_in_executor(None, token_queue.get)
            if item is None:
                break
            text_piece, _ = item
            accumulated_text += text_piece
            yield accumulated_text, 0.0  # tps can be computed on client if needed
        
        # Wait for generation to complete
        await future


    async def _stream_chat(model, prompt: str, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks as SSE with true token-by-token streaming.

        Uses run_in_executor with an on_token callback generator to yield
        tokens immediately as they're generated (not after full generation).
        """
        mid = req.model or model_id
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        # Yield chunks as tokens arrive
        accumulated = ""
        async for text, _ in _stream_tokens_async(
            model, prompt, req.max_tokens, req.temperature, req.top_p
        ):
            # Send diff (new content since last yield)
            if text != accumulated:
                new_content = text[len(accumulated):]
                if new_content:
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": mid,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_content},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                accumulated = text
        
        # Apply stop sequences (may truncate)
        accumulated = _apply_stop(accumulated, req.stop)
        
        # Final chunk with finish_reason
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": mid,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": _finish_reason(accumulated, req.max_tokens),
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


    async def _stream_completion(model, req: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream text completion chunks as SSE with true token-by-token streaming."""
        mid = req.model or model_id
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
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
        
        # Yield chunks as tokens arrive
        accumulated = ""
        async for text, _ in _stream_tokens_async(
            model, req.prompt, req.max_tokens, req.temperature, req.top_p
        ):
            if text != accumulated:
                new_content = text[len(accumulated):]
                if new_content:
                    chunk = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": mid,
                        "choices": [{
                            "index": 0,
                            "text": new_content,
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                accumulated = text
        
        # Apply stop sequences
        accumulated = _apply_stop(accumulated, req.stop)
        
        chunk = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": mid,
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": _finish_reason(accumulated, req.max_tokens),
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
