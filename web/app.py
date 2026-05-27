"""
VibeBlade Chat — FastAPI backend for ChatGPT-like web UI.

Proxies chat requests to sglang backend with streaming SSE,
manages conversation persistence in JSON storage.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_MODEL = "qwen3.6-27b-mtp"
DATA_DIR = Path(__file__).parent / "data"

# ── Data models ──────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)

    def __init__(self, **kwargs):
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = time.time()
        super().__init__(**kwargs)

class Conversation(BaseModel):
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "New Chat"
    messages: list[Message] = []
    model: str = DEFAULT_MODEL
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 40
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    model: Optional[str] = None

class NewConversationRequest(BaseModel):
    title: Optional[str] = None

class RenameConversationRequest(BaseModel):
    title: str

# ── Dataclass fix for field() in Pydantic ────────────────────────────────
# Pydantic v2 doesn't support dataclass field() in BaseModel. Use defaults.

def _msg_default_tokens() -> int:
    return 0

def _msg_default_ts() -> float:
    return time.time()

def _conv_default_id() -> str:
    return str(uuid.uuid4())[:8]

def _conv_default_msgs() -> list:
    return []

def _conv_default_created() -> float:
    return time.time()

def _conv_default_updated() -> float:
    return time.time()


# ── Storage ───────────────────────────────────────────────────────────────

class ConversationStore:
    """JSON-file-backed conversation storage."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conv_id: str) -> Path:
        return self.data_dir / f"{conv_id}.json"

    def list_conversations(self) -> list[dict]:
        convs = []
        for p in sorted(self.data_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
                convs.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
        return convs

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        p = self._path(conv_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def save_conversation(self, conv: dict) -> str:
        conv_id = conv.get("id", _conv_default_id())
        conv["updated_at"] = time.time()
        p = self._path(conv_id)
        p.write_text(json.dumps(conv, indent=2, ensure_ascii=False))
        return conv_id

    def delete_conversation(self, conv_id: str) -> bool:
        p = self._path(conv_id)
        if p.exists():
            p.unlink()
            return True
        return False

    def rename_conversation(self, conv_id: str, title: str) -> bool:
        conv = self.get_conversation(conv_id)
        if not conv:
            return False
        conv["title"] = title
        self.save_conversation(conv)
        return True


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="VibeBlade Chat", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ConversationStore()

# ── Static files ─────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/style.css")
async def style():
    return FileResponse(STATIC_DIR / "style.css")

@app.get("/app.js")
async def js():
    return FileResponse(STATIC_DIR / "app.js")

@app.get("/icon.svg")
async def icon():
    return FileResponse(STATIC_DIR / "icon.svg")

# ── API routes ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/models")
async def list_models():
    """List available models from backend."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DEFAULT_BACKEND_URL}/v1/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                return [m["id"] for m in models]
    except Exception:
        pass
    return [DEFAULT_MODEL]


@app.get("/api/conversations")
async def list_conversations():
    return store.list_conversations()


@app.post("/api/conversations")
async def create_conversation(req: Optional[NewConversationRequest] = None):
    conv = {
        "id": _conv_default_id(),
        "title": req.title if req and req.title else "New Chat",
        "messages": [],
        "model": DEFAULT_MODEL,
        "system_prompt": "",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.9,
        "top_k": 40,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    conv_id = store.save_conversation(conv)
    conv["id"] = conv_id
    return conv


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if not store.delete_conversation(conv_id):
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}


@app.patch("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, req: RenameConversationRequest):
    if not store.rename_conversation(conv_id, req.title):
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream chat completion from sglang backend via SSE."""

    # Load or create conversation
    conv = None
    if req.conversation_id:
        conv = store.get_conversation(req.conversation_id)
    if not conv:
        conv = {
            "id": _conv_default_id(),
            "title": "New Chat",
            "messages": [],
            "model": req.model or DEFAULT_MODEL,
            "system_prompt": req.system_prompt or "",
            "temperature": req.temperature or 0.7,
            "max_tokens": req.max_tokens or 4096,
            "top_p": req.top_p or 0.9,
            "top_k": req.top_k or 40,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    # Apply request overrides
    model = req.model or conv.get("model", DEFAULT_MODEL)
    temperature = req.temperature if req.temperature is not None else conv.get("temperature", 0.7)
    max_tokens = req.max_tokens or conv.get("max_tokens", 4096)
    top_p = req.top_p if req.top_p is not None else conv.get("top_p", 0.9)
    top_k = req.top_k if req.top_k is not None else conv.get("top_k", 40)
    system_prompt = req.system_prompt if req.system_prompt is not None else conv.get("system_prompt", "")

    # Build messages for sglang
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Load history
    for msg in conv.get("messages", []):
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    # Add current user message
    messages.append({"role": "user", "content": req.message})

    # Save user message
    user_msg = {
        "role": "user",
        "content": req.message,
        "tokens": 0,
        "timestamp": time.time(),
    }
    conv["messages"].append(user_msg)

    # Auto-title: use first message as title
    if len([m for m in conv["messages"] if m["role"] == "user"]) == 1:
        conv["title"] = req.message[:60] + ("..." if len(req.message) > 60 else "")

    # Stream from sglang
    async def event_stream():
        full_content = ""
        token_count = 0
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{DEFAULT_BACKEND_URL}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "top_k": top_k,
                        "stream": True,
                        "reasoning": {"effort": "none"},
                    },
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        yield f"data: {json.dumps({'error': f'Backend error {resp.status_code}: {error_text.decode()}'})}\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                token_count += 1
                                yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to inference backend. Is sglang running?'})}\n\n"
            return
        except httpx.ReadTimeout:
            yield f"data: {json.dumps({'error': 'Request timed out. Try shorter output.'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Save assistant message
        assistant_msg = {
            "role": "assistant",
            "content": full_content,
            "tokens": token_count,
            "timestamp": time.time(),
        }
        conv["messages"].append(assistant_msg)
        store.save_conversation(conv)

        # Send final event with conv_id for client to track
        yield f"data: {json.dumps({'done': True, 'tokens': token_count, 'conversation_id': conv['id']})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
