# VibeBlade

Universal speculative decoding layer. Run any LLM faster on your own hardware — no cloud, no subscription.

[![Build Status](https://github.com/kevin046/VibeBlade/actions/workflows/build.yml/badge.svg)](https://github.com/kevin046/VibeBlade/actions)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

---

## What it does

VibeBlade is a **speculative decoding engine** that sits between your application and any inference backend (sglang, vLLM, llama.cpp, or OpenAI-compatible HTTP servers). It generates draft tokens using one of four strategies, then verifies them against the target model — accepting correct tokens in batches for 1.5–3x throughput improvement.

**Draft strategies:**

| Strategy | How it works | Best for |
|---|---|---|
| **N-gram** | Predicts next tokens from recent history patterns | Repetitive/code text, zero overhead |
| **EAGLE** | Lightweight neural draft model | General-purpose, high acceptance |
| **DFlash** | Block diffusion with target hidden-state conditioning | Qwen3 models, parallel draft |
| **NEXTN** | N-gram + neural hybrid (built into sglang) | Qwen3.6 hybrid MoE+SSM models |

**Also includes:**
- **ChatGPT-like web UI** — dark theme, streaming SSE, conversation history, Markdown + syntax highlighting, settings panel
- **Native C++ inference engine** — mmap'd GGUF weights, SIMD-optimized (AVX-512/NEON), optional CUDA backend
- **Research modules** — TurboSparse activation sparsity, RotateKV quantization, SARATHI chunked prefill, SageSched scheduling

---

## Quick start

### Install

```bash
git clone https://github.com/kevin046/VibeBlade && cd VibeBlade
pip install -e ".[all]"    # Python deps + dev tools
```

### Web UI (chat interface)

```bash
vibeblade chat --backend-url http://localhost:8000 --port 8080
```

Opens a ChatGPT-like dark-themed UI at `http://localhost:8080`. Requires an inference backend running (sglang, vLLM, etc.).

### Speculative decoding API server

```bash
vibeblade serve --backend sglang --backend-url http://localhost:8000 \
                --model qwen3.6-27b-mtp --draft ngram --max-draft 8
```

Starts an OpenAI-compatible API at `/v1/chat/completions` and `/v1/completions` with speculative decoding.

### Benchmark

```bash
vibeblade bench --backend-url http://localhost:8000 --concurrent 8 --max-tokens 512
```

Reports per-request and aggregate tok/s across multiple rounds.

---

## CLI

```
vibeblade serve   Start speculative decoding API server
vibeblade chat    Launch web UI
vibeblade bench   Run throughput benchmarks
```

Each subcommand has its own `--help`:

```bash
vibeblade serve --help
vibeblade chat --help
vibeblade bench --help
```

---

## Web UI

The `vibeblade chat` command launches a full-featured ChatGPT-like interface:

- **Sidebar** — conversation list with search, create, delete, rename
- **Streaming** — real-time token delivery via SSE with cursor animation
- **Markdown** — full GFM rendering with syntax-highlighted code blocks (highlight.js)
- **Settings** — temperature, max tokens, top-p, top-k, system prompt
- **Persistence** — conversations saved to JSON, settings to localStorage
- **Copy** — one-click copy on messages and code blocks
- **Mobile** — responsive layout with collapsible sidebar
- **Keyboard** — Ctrl+N (new chat), Enter (send), Shift+Enter (newline), Esc (close panels)

Architecture: FastAPI backend proxies to the inference server with `reasoning: {"effort": "none"}` to eliminate hidden thinking token overhead. Frontend is a single-page app (HTML/CSS/JS, no framework).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Client (curl, openai SDK, web UI, LangChain, etc.)  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  VibeBlade API Server (OpenAI-compatible)            │
│                                                      │
│  ┌─────────────┐    ┌────────────────────────────┐   │
│  │ Draft Head  │───▶│ Speculative Decoding Engine │   │
│  │ (n-gram /   │    │  draft → verify → accept    │   │
│  │  eagle /    │    └──────────┬─────────────────┘   │
│  │  dflash /   │               │                     │
│  │  nextn)     │               ▼                     │
│  └─────────────┘    ┌──────────────────────┐        │
│                      │ Target Backend      │        │
│                      │ (sglang / vLLM /     │        │
│                      │  llama.cpp / HTTP)   │        │
│                      └──────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

### Target backends

| Backend | Class | Protocol |
|---|---|---|
| sglang | `SglangTargetBackend` | HTTP (OpenAI-compatible) |
| vLLM | `VllmTargetBackend` | HTTP (OpenAI-compatible) |
| llama.cpp | `OpenAIHttpTargetBackend` | HTTP |
| Any OpenAI server | `OpenAIHttpTargetBackend` | HTTP |

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/completions` | POST | Legacy text completions |
| `/v1/models` | GET | List available models |
| `/health` | GET | Liveness check |
| `/v1/me` | GET | API info |

### Web UI API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Send message (SSE stream) |
| `/api/conversations` | GET/POST | List / create conversations |
| `/api/conversations/{id}` | GET/DELETE/PATCH | Read / delete / rename |
| `/api/models` | GET | List backend models |
| `/api/health` | GET | Health check |

---

## Python API

### Speculative decoding server

```python
from vibeblade.openai_server import main

# Launch with defaults: OpenAI backend, n-gram draft, port 8080
main()

# Or with custom args
main([
    "--backend", "sglang",
    "--backend-url", "http://localhost:8000",
    "--model", "qwen3.6-27b-mtp",
    "--draft", "ngram",
    "--max-draft", 8,
    "--port", 8080,
])
```

### Proxy engine (HTTP-based, no local model needed)

```python
from vibeblade.proxy_engine import ProxyEngine

engine = ProxyEngine(
    backend_url="http://localhost:8000",
    model="qwen3.6-27b-mtp",
    mode="ngram_inject",  # passthrough | ngram_cache | ngram_inject
)

# Single request
result = engine.generate("Write a Python sort function", max_tokens=256)
print(result.text)
print(f"{result.stats.tokens_per_second:.1f} tok/s")

# Concurrent benchmark
results = engine.benchmark(n_concurrent=8, max_tokens=256)
```

### Draft heads

```python
from vibeblade.draft_heads import create_draft_head

# N-gram draft (zero overhead, good for code)
draft = create_draft_head("ngram", n=5, max_draft=8)

# EAGLE neural draft (higher quality predictions)
draft = create_draft_head("eagle", max_draft=8)

# DFlash block diffusion (requires draft model)
draft = create_draft_head("dflash", draft_model_name="z-lab/Qwen3.6-27B-DFlash")

# NEXTN hybrid (n-gram + neural)
draft = create_draft_head("nextn", max_draft=8, ngram=NgramDraftHead(n=5))
```

### C++ native inference (GGUF files)

```python
from vibeblade import VibeBladeModel

model = VibeBladeModel("model.gguf")
print(model.generate("Hello world", max_tokens=128))
```

Auto-detects and uses the native C++ engine for GGUF files — the entire generate pipeline runs in C++ with zero Python in the decode loop. Supports all architectures: dense transformers, MoE (Mistral, Qwen, DeepSeek), and hybrid attention+SSM.

```bash
# Build C++ engine (optional — only needed for local GGUF inference)
python cpp/build_cpp.py
```

SIMD auto-detection at build time: AVX-512, AVX2, NEON, or scalar fallback. Optional CUDA backend for NVIDIA GPUs (sm_121 Blackwell).

---

## Research modules

VibeBlade includes several research-backed inference optimization components:

| Module | Description | Source |
|---|---|---|
| `sparse.py` | TurboSparse — EMA neuron prediction + dReLU gating (~90% FFN skip) | PowerInfer |
| `confu.py` | ConFu — contemplate-token speculative decoding (85-92% acceptance) | Original |
| `rotatekv.py` | RotateKV — Hadamard rotation + 2-bit KV quantization (~8x reduction) | RotateKV |
| `dflash.py` | DFlash — block diffusion speculative decoding with hidden-state conditioning | z-lab |
| `sarathi.py` | SARATHI — chunked prefill scheduling | SARATHI |
| `sagesched.py` | SageSched — Shannon entropy-based uncertainty-aware scheduling | Original |
| `paged_attn.py` | Paged attention for KV cache management | vLLM |
| `grammar.py` | Constrained decoding (regex, JSON schema, EBNF) | llama.cpp |

```python
from vibeblade import (
    EMANeuronPredictor, drelu_gate,           # TurboSparse
    ConFuSpeculator, ContemplateTokenLayer,    # ConFu
    RotateKVCache, rotate_kv,                  # RotateKV
    DFlashDraftHead, dflash_generate,          # DFlash
    SarathiScheduler, SageSched,               # Scheduling
)
```

---

## Project structure

```
vibeblade/
  cli.py              # Unified CLI (serve / chat / bench)
  openai_server.py    # OpenAI-compatible API server
  proxy_engine.py     # HTTP proxy with n-gram cache/inject modes
  speculative_decoding.py  # Draft-then-verify engine
  draft_heads.py      # Draft head ABC + 4 implementations
  target_backend.py   # Target backend ABC + factory
  dflash.py           # DFlash draft model (PyTorch)
  backends/
    sglang_backend.py     # sglang target
    vllm_backend.py       # vLLM target
    openai_http_backend.py  # Generic HTTP target
  web_app.py          # uvicorn-reloadable web app factory
  fast_backend.py     # C++ engine wrapper
  sparse.py           # TurboSparse activation sparsity
  confu.py            # ConFu speculative decoding
  rotatekv.py         # RotateKV quantization
  sarathi.py / sagesched.py  # Scheduling
  moe*.py             # MoE routing + tiered memory
  grammar.py          # Constrained decoding

web/
  app.py              # FastAPI backend for Chat UI
  static/
    index.html        # Single-page app
    style.css         # Dark theme (Linear/Vercel aesthetic)
    app.js            # Client-side logic
    icon.svg          # Favicon

cpp/                  # Native C++ inference engine
  build_cpp.py        # Cross-platform build script
  include/            # Headers (gguf, dequant, CUDA kernels, SIMD)
  src/                # Implementation + pybind11 bindings

tests/                # 794 tests
```

---

## Benchmarks

### GB10 (NVIDIA) — Qwen3.6-27B-FP8 with NEXTN speculative decoding

| Config | Single request | 5 concurrent | 8 concurrent |
|---|---:|---:|---:|
| sglang baseline | 15.6 tok/s | 63.0 tok/s aggregate | — |
| Optimized (no thinking) | 24.5 tok/s | 102.5 tok/s aggregate | 152.5 tok/s aggregate |

**Key optimization:** Using `/v1/chat/completions` with `reasoning: {"effort": "none"}` eliminates 30-50% hidden thinking token overhead, giving a 1.6x improvement on visible output speed.

### ARM NEON — GGUF Q4_K_M models

**3-run validation, 256 ctx, temp=0.0**

Best results per model:

| Model | Type | Best Config | Speedup |
|---|---|---|---:|
| TinyLlama-1.1B | Dense 1.1B | PowerInfer | 2.54x |
| DeepSeek-Coder-V2-Lite | MoE 16B (2.4B active) | PowerInfer | 1.97x |
| Llama-3.2-3B | Dense 3.2B | Spec+TS | 1.57x |
| Granite-3.0-3B-A800M | MoE 3B | PI+TS | 1.52x |
| Qwen2.5-MoE | MoE 3B | TurboSparse | 1.41x |
| Qwen3.6-35B-A3B | Hybrid MoE+SSM | PI+TS | 1.32x |
| Llama-3.1-8B | Dense 8B | PI+TS | 1.25x |

Full benchmark results are in [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md).

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check vibeblade/ tests/

# Build C++ engine
python cpp/build_cpp.py

# Build with CUDA (requires CUDA Toolkit 13.0+)
VIBEBLADE_CUDA=ON python cpp/build_cpp.py
```

---

## Powered by

GGUF format · [sglang](https://github.com/sgl-project/sglang) · [vLLM](https://github.com/vllm-project/vllm) · [llama.cpp](https://github.com/ggerganov/llama.cpp) · [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) · [EAGLE](https://arxiv.org/abs/2401.15077) · [SARATHI](https://arxiv.org/abs/2403.07219) · [RotateKV](https://arxiv.org/abs/2408.00784) · [DFlash](https://github.com/z-lab/Qwen3.6-27B-DFlash) · [highlight.js](https://highlightjs.org) · [marked.js](https://marked.js.org)

---

## License

BSL 1.1 — free for personal, educational, and non-commercial use. Automatically converts to Apache 2.0 on May 1, 2028. See [LICENSE](LICENSE) for details.

For commercial licensing, contact [kevin.lin@vibedrift.com](mailto:kevin.lin@vibedrift.com).
