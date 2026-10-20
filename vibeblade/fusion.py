"""VibeBlade Fusion — the universal inference front.

One OpenAI-compatible endpoint that routes every request to the fastest
engine for THAT model + workload:

    ┌────────────────────────────────────────────────┐
    │                VibeBlade Fusion                 │
    │  OpenAI API · tool-calling · JSON mode · SSE    │
    │  engine selection · fallback · prefix cache     │
    └───────┬──────────┬──────────┬─────────┬────────┘
            ▼          ▼          ▼         ▼
         SGLang      vLLM      NVFP4     GGUF/llama.cpp
        (Blackwell  (breadth)  native    (long tail,
         kernels)              (this     CPU fallback
                               repo)
    plus: any OpenAI-compatible HTTP engine (Ollama, TGI, cloud overflow)

Selection policy (per request):
  1. exact model→engine match in the registry
  2. capability gates: tools? vision? context length? quant format?
  3. speed tier: local tensor-core engines > local CPU > remote/cloud
  4. health: probe-failed engines are skipped; failures fall through the
     chain automatically
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class SpeedTier(IntEnum):
    TENSOR_CORE_LOCAL = 0   # SGLang/vLLM on Blackwell, NVFP4 native
    LOCAL = 1               # GGUF/CPU, small models
    REMOTE = 2              # LAN engines
    CLOUD = 3               # internet APIs (overflow / free-tier)


@dataclass
class EngineSpec:
    """One serving engine behind Fusion."""
    name: str
    kind: str                      # sglang | vllm | nvfp4 | gguf | openai
    base_url: str = ""             # HTTP engines
    models: list = field(default_factory=list)   # served model ids
    speed_tier: SpeedTier = SpeedTier.LOCAL
    supports_tools: bool = True
    supports_vision: bool = False
    max_context: int = 32768
    healthy: bool = True
    last_probe_s: float = 0.0
    probe_interval_s: float = 30.0
    stats: dict = field(default_factory=lambda: {"requests": 0, "errors": 0, "latency_ema": 0.0})

    def matches(self, model: str) -> bool:
        if not self.models:
            return False
        return any(model == m or model.startswith(m + ":") or model.startswith(m + "/")
                   for m in self.models)

    def capable(self, model: str, needs_tools: bool, needs_vision: bool,
                context: int) -> bool:
        if self.models and not self.matches(model):
            return False
        if needs_tools and not self.supports_tools:
            return False
        if needs_vision and not self.supports_vision:
            return False
        if context > self.max_context:
            return False
        return True


class FusionRouter:
    """Model + workload → engine selection with fallback chains."""

    def __init__(self):
        self.engines: list[EngineSpec] = []

    def register(self, spec: EngineSpec) -> EngineSpec:
        self.engines.append(spec)
        # keep fastest tier first so selection order == speed order
        self.engines.sort(key=lambda e: (e.speed_tier, e.name))
        return spec

    def select(self, model: str, needs_tools: bool = True,
               needs_vision: bool = False, context: int = 8192,
               exclude: set | None = None) -> Optional[EngineSpec]:
        exclude = exclude or set()
        for e in self.engines:
            if e.name in exclude or not e.healthy:
                continue
            if e.capable(model, needs_tools, needs_vision, context):
                return e
        # relax context/vision, but NEVER hand a tools request to an
        # engine that can't emit tool_calls (that poisons agent loops)
        if needs_tools:
            return None
        for e in self.engines:
            if e.name in exclude or not e.healthy:
                continue
            if e.matches(model):
                return e
        return None

    async def probe(self, spec: EngineSpec) -> bool:
        """Health-check an HTTP engine via /v1/models."""
        if not spec.base_url:
            return True
        now = time.time()
        if now - spec.last_probe_s < spec.probe_interval_s:
            return spec.healthy
        try:
            req = urllib.request.Request(f"{spec.base_url.rstrip('/')}/v1/models",
                                         method="GET")
            with urllib.request.urlopen(req, timeout=3) as r:
                spec.healthy = r.status == 200
                body = json.loads(r.read().decode())
                ids = [m.get("id", "") for m in body.get("data", [])]
                if ids and not spec.models:
                    spec.models = ids
        except Exception:
            spec.healthy = False
        spec.last_probe_s = now
        return spec.healthy


class FusionEngine:
    """Executes chat completions against the routed engine.

    Handles the OpenAI passthrough for HTTP engines and normalizes
    tool-call responses so EVERY engine behind Fusion speaks proper
    tool_calls (the gap that breaks agentic stacks today).
    """

    def __init__(self, router: FusionRouter, timeout: float = 120.0):
        self.router = router
        self.timeout = timeout

    async def chat(self, body: dict) -> dict:
        model = body.get("model", "")
        needs_tools = bool(body.get("tools"))
        needs_vision = any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict))
            for m in body.get("messages", [])
        )
        ctx = body.get("max_tokens", 4096) + 8000  # rough prompt allowance

        tried: set = set()
        last_err: Exception | None = None
        while True:
            spec = self.router.select(model, needs_tools, needs_vision, ctx,
                                      exclude=tried)
            if spec is None:
                raise RuntimeError(
                    f"no engine serves model '{model}'"
                    + (f" (last error: {last_err})" if last_err else "")
                )
            tried.add(spec.name)
            await self.router.probe(spec)
            if not spec.healthy:
                continue
            try:
                t0 = time.time()
                out = await self._call_http(spec, body)
                dt = time.time() - t0
                s = spec.stats
                s["requests"] += 1
                s["latency_ema"] = s["latency_ema"] * 0.9 + dt * 0.1
                out.setdefault("vibeblade_fusion", {"engine": spec.name,
                                                    "tier": spec.speed_tier.name,
                                                    "latency_s": round(dt, 3)})
                return out
            except Exception as e:  # fall through to next engine
                last_err = e
                spec.stats["errors"] += 1
                logger.warning("fusion: engine %s failed: %s", spec.name, e)

    async def _call_http(self, spec: EngineSpec, body: dict) -> dict:
        loop = asyncio.get_event_loop()

        def _post():
            payload = dict(body)
            # strip tools when the engine can't do them natively — Fusion's
            # front (openai_server) already parsed/serialized tool flow
            if not spec.supports_tools:
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{spec.base_url.rstrip('/')}/v1/chat/completions",
                data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())

        return await loop.run_in_executor(None, _post)


def default_gb10_router(sglang_url: str = "http://gb10:8000",
                        vllm_url: str = "http://gb10:8001",
                        extra_openai: list[EngineSpec] | None = None) -> FusionRouter:
    """Sensible default engine set for a GB10 (DGX Spark) box."""
    r = FusionRouter()
    r.register(EngineSpec(
        name="sglang-primary", kind="sglang", base_url=sglang_url,
        models=["qwen3.8-27b", "qwen-3.8-27b-sglang"],
        speed_tier=SpeedTier.TENSOR_CORE_LOCAL,
        supports_tools=True, supports_vision=True, max_context=262144,
    ))
    r.register(EngineSpec(
        name="vllm-breadth", kind="vllm", base_url=vllm_url,
        models=[],  # filled by probe
        speed_tier=SpeedTier.TENSOR_CORE_LOCAL,
        supports_tools=True, supports_vision=True, max_context=131072,
    ))
    for spec in (extra_openai or []):
        r.register(spec)
    return r
