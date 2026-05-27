# VibeBlade

LLM inference acceleration platform. Speed up any model on your own hardware using a toolkit of complementary optimization strategies — speculative decoding, activation sparsity, KV quantization, chunked scheduling, and more. No cloud, no subscription.

[![Build Status](https://github.com/kevin046/VibeBlade/actions/workflows/build.yml/badge.svg)](https://github.com/kevin046/VibeBlade/actions)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

---

## Install

### Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **C++ build tools** — required for the native engine (optional but recommended):
  - **Ubuntu/Debian:** `sudo apt install build-essential cmake`
  - **macOS:** `xcode-select --install && brew install cmake`
  - **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ workload) + [CMake](https://cmake.org/download/)
- **CUDA Toolkit 13.0+** (optional) — for GPU acceleration on NVIDIA GPUs

**Linux / macOS**
```bash
git clone https://github.com/kevin046/VibeBlade && cd VibeBlade
pip install -e ".[all]"            # Python deps + all optional extras
python cpp/build_cpp.py            # Build native C++ engine (recommended)
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/kevin046/VibeBlade; cd VibeBlade
pip install -e ".[all]"
python cpp/build_cpp.py
```

**Web UI only** (no C++ build needed — just needs an inference backend like sglang or vLLM running):
```bash
git clone https://github.com/kevin046/VibeBlade && cd VibeBlade
pip install -e .
vibeblade chat --backend-url http://localhost:8000 --port 8080
```

> **Tip:** If you only plan to use VibeBlade as a speculative decoding proxy or web UI over an existing sglang/vLLM server, you don't need the C++ engine at all — just `pip install -e .` and you're good.

### Build C++ engine (optional)

The native engine gives you local GGUF inference with SIMD acceleration. Not required for the proxy/web UI modes.

```bash
# Linux / macOS
python cpp/build_cpp.py

# With CUDA support (NVIDIA GPUs)
VIBEBLADE_CUDA=ON python cpp/build_cpp.py

# Windows (from Developer PowerShell for VS)
python cpp/build_cpp.py
```

SIMD auto-detected at build time: AVX-512, AVX2, NEON (Apple Silicon / ARM), or scalar fallback.

---

## What it does

VibeBlade is an **inference acceleration platform** that combines multiple complementary optimization strategies into a unified pipeline. Each technique targets a different bottleneck — compute, memory, scheduling, or token generation — so you can stack them for maximum throughput.

**Optimization strategies:**

| Strategy | Target bottleneck | Technique | Typical speedup |
|---|---|---|---:|
| **Speculative decoding** | Token-by-token serial generation | Draft model proposes, target verifies in batch | 1.3–3x |
| **Activation sparsity** (TurboSparse) | Dense FFN compute (90%+ wasted) | EMA neuron predictor + dReLU gating skips inactive neurons | 1.3–2.5x |
| **KV quantization** (RotateKV) | KV cache memory bandwidth | Hadamard rotation + 2-bit quantization | ~8x memory reduction |
| **Chunked prefill** (SARATHI) | Head-of-line blocking | Interleave prefill chunks with decode iterations | Higher batch throughput |
| **Entropy scheduling** (SageSched) | Unfair resource allocation | Shannon entropy-based priority for uncertain requests | Better QoS |
| **MoE tiered memory** | Expert weight transfer over PCIe | 3-tier VRAM/RAM/SSD with adaptive eviction | Near-native latency |
| **Grammar constraints** | Wasteful re-sampling | Constrained decoding (regex, JSON schema, EBNF) | 2–10x on structured output |
| **Native C++ engine** | Python interpreter overhead | mmap GGUF, SIMD-optimized (AVX-512/NEON), CUDA optional | Up to 5x over pure Python |

These compose — activation sparsity reduces compute per token, speculative decoding amortizes verification cost, KV quantization fits more sequences in cache, and chunked scheduling keeps the GPU fed.

---

## Quick start

### Web UI (chat interface)

```bash
vibeblade chat --backend-url http://localhost:8000 --port 8080
```

ChatGPT-like dark-themed UI with streaming, Markdown, code highlighting, conversation history, and a settings panel. Just point it at any running inference backend.

### Speculative decoding server

```bash
vibeblade serve --backend sglang --backend-url http://localhost:8000 \
                --model qwen3.6-27b-mtp --draft ngram --max-draft 8
```

Wraps any inference backend with speculative decoding. Exposes an OpenAI-compatible API that any client can consume.

### Benchmark

```bash
vibeblade bench --backend-url http://localhost:8000 --concurrent 8 --max-tokens 512
```

---

## CLI

```
vibeblade serve   Start optimized inference API server (speculative decoding)
vibeblade chat    Launch web UI
vibeblade bench   Run throughput benchmarks
```

```bash
vibeblade serve --help    # Backend, draft strategy, sampling params
vibeblade chat --help     # Backend URL, host, port
vibeblade bench --help    # Concurrency, token limits, rounds
```

---

## Speculative decoding

The core of VibeBlade's acceleration. A lightweight draft model proposes multiple candidate tokens; the target model verifies them in a single forward pass. Accepted tokens are emitted at batch speed; rejected tokens fall back to autoregressive.

**Draft strategies:**

| Strategy | How it works | Overhead | Best for |
|---|---|---|---|
| **N-gram** | Predicts from recent token history | Zero (no model) | Code, repetitive text |
| **EAGLE** | Lightweight neural draft head | Small | General-purpose, high acceptance |
| **DFlash** | Block diffusion conditioned on target hidden states | Medium | Qwen3 models, parallel block gen |
| **NEXTN** | N-gram + neural hybrid (sglang built-in) | Built-in | Qwen3.6 hybrid MoE+SSM |

**Target backends:** sglang, vLLM, llama.cpp, any OpenAI-compatible HTTP server.

```python
from vibeblade.proxy_engine import ProxyEngine

engine = ProxyEngine(
    backend_url="http://localhost:8000",
    model="qwen3.6-27b-mtp",
    mode="ngram_inject",  # passthrough | ngram_cache | ngram_inject
)

result = engine.generate("Write a sort function", max_tokens=256)
print(f"{result.stats.tokens_per_second:.1f} tok/s")

# Concurrent benchmark
engine.benchmark(n_concurrent=8, max_tokens=256)
```

---

## Web UI

The `vibeblade chat` command launches a full-featured interface:

- Streaming SSE with real-time token delivery
- Sidebar with conversation list (search, create, delete, rename)
- Markdown rendering with syntax-highlighted code blocks (highlight.js)
- Settings panel (temperature, max tokens, top-p, top-k, system prompt)
- Conversation persistence (server-side JSON) and settings (localStorage)
- Copy buttons on messages and code blocks
- Mobile responsive, keyboard shortcuts (Ctrl+N, Enter, Shift+Enter, Esc)

Architecture: FastAPI backend proxies to the inference server with `reasoning: {"effort": "none"}` to strip hidden thinking token overhead. Frontend is a single-page app — no framework, no build step.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Clients: curl, openai SDK, web UI, LangChain, browsers  │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  VibeBlade Inference Acceleration Platform               │
│                                                          │
│  ┌───────────────┐  ┌────────────┐  ┌────────────────┐  │
│  │ Speculative   │  │ TurboSparse│  │ Chunked Prefill│  │
│  │ Decoding      │  │ Activation │  │ Scheduling     │  │
│  │ Engine        │  │ Sparsity   │  │ (SARATHI)      │  │
│  │               │  │            │  │                │  │
│  │ n-gram        │  │ EMA Neuron │  │ Dynamic chunk  │  │
│  │ EAGLE         │  │ Predictor  │  │ interleaving   │  │
│  │ DFlash        │  │ + dReLU    │  │                │  │
│  │ NEXTN         │  │ gating     │  │                │  │
│  └───────┬───────┘  └──────┬─────┘  └───────┬────────┘  │
│          │                 │                 │           │
│          ▼                 ▼                 ▼           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Unified Inference Pipeline                          │ │
│  │                                                     │ │
│  │  RotateKV    SageSched    Grammar    MoE Tiered     │ │
│  │  (2-bit KV) (entropy     (regex/    (VRAM/RAM/     │ │
│  │  quantiz.    scheduling)  JSON/EBNF) SSD memory)     │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────────┐│
│  │  Target Backend                                      ││
│  │  sglang / vLLM / llama.cpp / OpenAI HTTP / C++ GGUF ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

Each optimization is a composable module — use one, or stack them all. The speculative decoding engine runs over HTTP and works with any backend. The native C++ engine runs locally for GGUF models with all optimizations compiled in.

---

## API

### Inference server (`vibeblade serve`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List models |
| `/health` | GET | Liveness check |

### Web UI (`vibeblade chat`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Send message (SSE stream) |
| `/api/conversations` | GET/POST | List / create conversations |
| `/api/conversations/{id}` | GET/DELETE/PATCH | Read / delete / rename |
| `/api/models` | GET | List backend models |

---

## Python API

### Speculative decoding

```python
from vibeblade.draft_heads import create_draft_head
from vibeblade.proxy_engine import ProxyEngine

draft = create_draft_head("ngram", n=5, max_draft=8)
engine = ProxyEngine("http://localhost:8000", "qwen3.6-27b-mtp")
result = engine.generate("Hello world", max_tokens=256)
```

### Activation sparsity (TurboSparse)

```python
from vibeblade import EMANeuronPredictor, drelu_gate

predictor = EMANeuronPredictor(hidden_dim=4096, n_layers=32)
mask = predictor.predict(layer_idx, gate_activations)  # ~10% active
predictor.update(layer_idx, actual_activations)
```

### KV quantization (RotateKV)

```python
from vibeblade import RotateKVCache, rotate_kv

kv_cache = RotateKVCache(num_layers=32, num_heads=32, head_dim=128)
kv_cache.compress(layer_idx)  # Hadamard rotate + 2-bit quantize
```

### Constrained decoding

```python
from vibeblade import Grammar

grammar = Grammar.from_json_schema(my_schema)
# Forces model output to match schema — no re-sampling needed
```

### Native C++ inference (GGUF)

```python
from vibeblade import VibeBladeModel

model = VibeBladeModel("model.gguf")
print(model.generate("Hello world", max_tokens=128))
```

Entire pipeline runs in C++ — tokenization, forward pass, sampling, detokenization. SIMD auto-detected (AVX-512/AVX2/NEON). Optional CUDA backend.

```bash
python cpp/build_cpp.py              # Build (Linux/macOS/Windows)
VIBEBLADE_CUDA=ON python cpp/build_cpp.py  # With CUDA
```

---

## Benchmarks

### GB10 (NVIDIA) — Qwen3.6-27B-FP8

Optimized via sglang NEXTN + reasoning elimination + n-gram prefill injection:

| Config | Single request | 5 concurrent | 8 concurrent |
|---|---:|---:|---:|
| Baseline (no optimizations) | 15.6 tok/s | 63.0 tok/s | — |
| Full optimization stack | 24.5 tok/s | 102.5 tok/s | 152.5 tok/s |
| **Speedup** | **1.6x** | **1.6x** | — |

### ARM NEON — GGUF Q4_K_M models

3-run validation, 256 ctx, temp=0.0, best config per model:

| Model | Type | Best config | Speedup |
|---|---|---|---:|
| TinyLlama-1.1B | Dense 1.1B | PowerInfer | 2.54x |
| DeepSeek-Coder-V2-Lite | MoE 16B (2.4B active) | PowerInfer | 1.97x |
| Llama-3.2-3B | Dense 3.2B | Spec+TS | 1.57x |
| Granite-3.0-3B-A800M | MoE 3B | PI+TS | 1.52x |
| Qwen3.6-35B-A3B | Hybrid MoE+SSM | PI+TS | 1.32x |
| Llama-3.1-8B | Dense 8B | PI+TS | 1.25x |

Full results: [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md)

---

## Project structure

```
vibeblade/
  cli.py                    # Unified CLI (serve / chat / bench)
  openai_server.py          # OpenAI-compatible API server
  proxy_engine.py           # HTTP proxy with n-gram cache/inject
  speculative_decoding.py   # Draft-then-verify engine
  draft_heads.py            # Draft head ABC + 4 implementations
  target_backend.py         # Target backend ABC + factory
  dflash.py                 # DFlash block diffusion (PyTorch)
  sparse.py                 # TurboSparse activation sparsity
  confu.py                  # ConFu speculative decoding
  rotatekv.py               # RotateKV Hadamard + 2-bit quant
  sarathi.py                # Chunked prefill scheduling
  sagesched.py              # Entropy-based scheduling
  moe.py / moe_executor.py  # MoE routing + tiered memory
  grammar.py                # Constrained decoding
  paged_attn.py             # Paged attention
  fast_backend.py           # C++ native engine wrapper
  backends/                 # sglang, vLLM, HTTP target backends

web/
  app.py                    # FastAPI backend for Chat UI
  static/                   # index.html, style.css, app.js

cpp/                        # Native C++ engine (GGUF + SIMD + CUDA)
tests/                      # 794 tests
```

---

## Development

```bash
pip install -e ".[dev]"           # Install with dev dependencies (ruff, pytest)
pytest                             # Run tests (794 tests)
ruff check vibeblade/ tests/       # Lint
python cpp/build_cpp.py            # Build native C++ engine
```

**Platform-specific notes:**
- **Linux:** Requires `build-essential` and `cmake` for C++ build
- **macOS:** Requires Xcode CLI tools (`xcode-select --install`) and cmake (`brew install cmake`). Apple Silicon (M1–M4) uses NEON SIMD automatically
- **Windows:** Requires Visual Studio Build Tools (C++ workload) and CMake. Run build commands from "Developer PowerShell for VS"

---

## Powered by

[sglang](https://github.com/sgl-project/sglang) · [vLLM](https://github.com/vllm-project/vllm) · [llama.cpp](https://github.com/ggerganov/llama.cpp) · [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) · [EAGLE](https://arxiv.org/abs/2401.15077) · [SARATHI](https://arxiv.org/abs/2403.07219) · [RotateKV](https://arxiv.org/abs/2408.00784) · [DFlash](https://github.com/z-lab/Qwen3.6-27B-DFlash) · [highlight.js](https://highlightjs.org) · [marked.js](https://marked.js.org)

---

## License

BSL 1.1 — free for personal, educational, and non-commercial use. Automatically converts to Apache 2.0 on May 1, 2028. See [LICENSE](LICENSE) for details.

For commercial licensing, contact [kevin.lin@vibedrift.com](mailto:kevin.lin@vibedrift.com).
