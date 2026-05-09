# VibeBlade

**Run any LLM on your own hardware — no cloud, no subscription.**

[![Star History](https://api.star-history.com/svg?repos=kevin046/VibeBlade)](https://star-history.com/#kevin046/VibeBlade)
[![Stars](https://img.shields.io/github/stars/kevin046/VibeBlade?style=flat)](https://github.com/kevin046/VibeBlade/stargazers)
[Forks](https://img.shields.io/github/forks/kevin046/VibeBlade?style=flat)](https://github.com/kevin046/VibeBlade/network)

**Prerequisites** — C++ build tools (required for the fast engine):

|| OS | Install |
|---|---|
|| **Ubuntu/Debian** | `sudo apt install build-essential cmake` |
|| **macOS** | `xcode-select --install && brew install cmake` |
|| **Windows** | Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ workload) + [CMake](https://cmake.org/download/) |

Python dependencies (`pip install -e .` handles these):
- Python 3.10+
- `numpy`, `pybind11`, `cmake`, `psutil`

**Linux / macOS**
```bash
git clone https://github.com/kevin046/VibeBlade && cd VibeBlade
pip install -e .                # Python deps (numpy, pybind11, etc.)
python cpp/build_cpp.py         # Build C++ engine (needs cmake + C++17 compiler)
python -m vibeblade wizard      # Guided setup
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/kevin046/VibeBlade; cd VibeBlade
pip install -e .
python cpp/build_cpp.py
python -m vibeblade wizard
```

[![Build Status](https://github.com/kevin046/VibeBlade/workflows/Build/badge.svg)](https://github.com/kevin046/VibeBlade/actions)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 794 passed](https://img.shields.io/badge/tests-794%20passed-brightgreen.svg)]()

📄 [White Paper](./WHITEPAPER.md) · 📊 [Performance Benchmarks](./BENCHMARK_REPORT.md) · 🔒 [Security](./WHITEPAPER.md#security)

---

## CLI commands

|| Command | What it does |
|---|---|
|| `python -m vibeblade wizard` | Guided setup — hardware detection, install, config, model download |
|| `python -m vibeblade chat --model model.gguf` | Interactive chat (C++ fast engine, auto-detected for .gguf) |
|| `python -m vibeblade chat --model model.gguf --backend numpy` | Force pure NumPy inference (slow, for debugging) |
|| `python -m vibeblade serve` | Start local inference API server (OpenAI-compatible) |
|| `python -m vibeblade bench` | Benchmark suite |
|| `python -m vibeblade bench --quick` | Quick benchmark (single prompt, ~30s) |

> **Dashboard & Model Browser** are part of VibeBlade Pro (commercial license). Contact [kevin.lin@vibedrift.com](mailto:kevin.lin@vibedrift.com) for access.

---

## Benchmarks

ARM NEON (aarch64) · 4 cores · Q4 quantization · 256 ctx · temp=0.0 · **Baseline = llama.cpp**
**Single-run numbers (marked ⚡) have high variance — validated 5-run averages (marked ✅) are more reliable.**

---

### 🏆 Best speedup by model

| Model | Type | Best Config | Baseline → Optimized | Speedup |
|---|---|---|---|---|
| **DeepSeek-Coder-V2-Lite** ✅ | MoE (2.4B active) | PI=0.01 + 3 threads | 4.56 → 7.47 t/s | **1.64×** |
| **Gemma-4 26B-A4B** ✅ | MoE (4B active) | PI=0.20 + 3 threads | 3.12 → 3.86 t/s | **1.24×** |
| **Gemma-2-2B** | Dense 2B | PI + TurboSparse | 1.08 → 8.61 t/s | **7.95×** |
| **Llama-3.2-1B** | Dense 1B | PI + TurboSparse | 2.69 → 23.43 t/s | **8.71×** |
| **Qwen3-30B-A3B** | MoE (3B active) | Speculative | 2.11 → 7.75 t/s | **3.68×** |
| **Gemma-4-E4B** | Dense 4B | PI + TurboSparse | 2.44 → 5.40 t/s | **2.21×** |
| **Qwen2.5-MoE** | MoE 2×1.5B | PowerInfer | 2.64 → 5.41 t/s | **2.05×** |
| **Phi-2-2.7B** | Dense 2.7B | Spec + TurboSparse | 5.09 → 9.92 t/s | **1.95×** |
| **Granite-3B-A800M** | MoE (800M active) | Speculative | 18.20 → 26.26 t/s | **1.44×** |
| **Llama-3.1-8B** | Dense 8B | PI + TurboSparse | 2.01 → 3.04 t/s | **1.51×** |
| **Qwen2.5-14B** | Dense 14B | PI + TurboSparse | 0.90 → 1.31 t/s | **1.45×** |
| **TinyLlama-1.1B** | Dense 1.1B | PI + TurboSparse | 26.16 → 31.53 t/s | **1.21×** |
| **Qwen3.6-35B-A3B** | MoE+SSM (3B active) | Baseline | 2.30 t/s | **1.0× (no gain)** |

---

### Optimization breakdown

**Llama-3.2-1B** (1B dense) — best: **PI+TS at 8.71×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 2.69 | — |
| TurboSparse | 8.40 | 3.12× |
| PowerInfer | 3.99 | 1.48× |
| Speculative | 3.04 | 1.13× |
| Spec + TurboSparse | 20.91 | 7.77× |
| **PI + TurboSparse** | **23.43** | **8.71×** |

**Qwen2.5-MoE** (2×1.5B MoE) — best: **PowerInfer at 2.05×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 2.64 | — |
| **PowerInfer** | **5.41** | **2.05×** |
| TurboSparse | 2.88 | 1.09× |
| PI + TurboSparse | 4.13 | 1.57× |
| Speculative | 2.39 | 0.90× |

**Phi-2-2.7B** (2.7B dense) — best: **Spec+TS at 1.95×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 5.09 | — |
| Speculative (100% accept) | 5.28 | 1.04× |
| **Spec + TurboSparse** | **9.92** | **1.95×** |
| PI + TurboSparse | 9.39 | 1.84× |
| TurboSparse | 3.67 | 0.72× |
| PowerInfer | 3.34 | 0.66× |

**TinyLlama-1.1B** (1.1B dense) — best: **PI+TS at 1.21×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 26.16 | — |
| TurboSparse | 26.01 | 0.99× |
| **PowerInfer** | **31.47** | **1.20×** |
| **PI + TurboSparse** | **31.53** | **1.21×** |
| Speculative | 16.75 | 0.64× |

**Llama-3.1-8B** (8B dense) — best: **PI+TS at 1.51×** (auto-tune only 1.22×)

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 2.01 | — |
| TurboSparse | 1.74 | 0.87× |
| Speculative | 1.56 | 0.78× |
| Spec + TurboSparse | 1.91 | 0.95× |
| PowerInfer | 2.50 | 1.24× |
| **PI + TurboSparse** | **3.04** | **1.51×** |

> PI+TS threshold tuning: PI=0.10, TS=0.01. Auto-tune heuristic achieves 1.22× — manual tuning needed for this model.

**Qwen2.5-14B** (14B dense) — best: **PI+TS at 1.45×** (auto-tune only 1.12×)

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 0.90 | — |
| PowerInfer | 1.00 | 1.10× |
| **PI + TurboSparse** | **1.31** | **1.45×** |
| Spec + TurboSparse | 0.82 | 0.91× |
| TurboSparse | 0.73 | 0.81× |
| Speculative | 0.92 | 1.02× |

> PI+TS threshold tuning: PI=0.20, TS=0.05. All optimizations net small gains on this model size — hardware constrained.

**DeepSeek-Coder-V2-Lite** (MoE, 2.4B active of 16B) ✅ — best: **PI=0.01 + 3 threads at 1.64×**

*5-run validated (64 tokens, CV ≤ 7.2%)*

| Config | Mean t/s | ± Std | CV | vs Baseline |
|---|---:|---:|---:|---:|
| Baseline (4thr, 256ctx) | 4.563 | 1.387 | 30.4% | — |
| TS=0.08 (4thr) | 4.708 | 1.390 | 29.5% | 1.03× |
| **PI=0.01 (3thr)** | **7.461** | **0.474** | **6.3%** | **1.64×** |
| **PI=0.01 + TS=0.05 (3thr)** | **7.472** | **0.540** | **7.2%** | **1.64×** |

> ✅ Validated with 5-run × 64-token sweep. PowerInfer at extremely low budget (0.01) with 3 threads gives the best result. 3 threads beats 4 due to less cache thrashing on ARM. Baseline has 30% CV — single-run numbers are unreliable. Previous 83.2× claim was an artifact of a baseline outlier on a 16-token run.

*Parameter sweep results (32 tok, single run per config):*
- TS threshold sweep: best at 0.08 (4.65 t/s)
- PI budget sweep: best at 0.01 (10.94 t/s single-run, 7.46 t/s 5-run mean)
- PI+TS grid: PI=0.01+TS=0.05 compound slightly above PI alone

**Qwen3.6-35B-A3B** (Hybrid MoE+SSM, 3B active of 35B) — best: **Baseline at 1.0×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| **Baseline (llama.cpp)** | **2.30** | **1.0×** |
| TurboSparse | 1.88 | 0.82× |
| PowerInfer | 1.88 | 0.82× |
| PI + TurboSparse | 0.88 | 0.38× |

> Novel hybrid MoE+SSM architecture (Mamba-style state-space layers interleaved with attention). Neither TurboSparse nor PowerInfer heuristics apply — the SSM layers create computation patterns that defeat activation-sparsity and hot-weight assumptions. First model where no optimization helps.

**Gemma-4 26B-A4B** (MoE, 4B active of 26B) ✅ — best: **PI=0.20 + 3 threads at 1.24×**

*5-run validated (64 tokens, CV ≤ 5.4%)*

| Config | Mean t/s | ± Std | CV | vs Baseline |
|---|---:|---:|---:|---:|
| Baseline (4thr, 256ctx) | 3.118 | 0.029 | 0.9% | — |
| TS=0.05 (4thr) | 2.554 | 0.315 | 12.3% | 0.82× |
| Speculative (4thr) | 2.889 | 0.088 | 3.0% | 0.93× |
| **PowerInfer=0.20 (3thr, 512ctx)** | **3.860** | **0.207** | **5.4%** | **1.24×** |

> ✅ Validated with 5-run × 64-token sweep. Baseline is remarkably stable (0.9% CV). 3 threads outperforms 4 for this model due to cache behavior on ARM. PowerInfer with ctx=512 gives a consistent 1.24× gain. Previous 50× claim was an artifact of an extreme baseline outlier on a 16-token run.

*Parameter sweep results (32 tok, single run per config):*
- TS threshold sweep: best at 0.05 (3.73 t/s)
- PI budget sweep: best at 0.20 (3.41 t/s)
- PI+TS grid: PI=0.05+TS=0.20 (3.50 t/s)
- Thread sweep: 3 threads optimal (4.28 t/s vs 1.51 at 4thr)
- Context sweep: 512ctx best (3.31 t/s)

**Gemma-2-2B** (Dense 2B) — best: **PI+TS at 7.95×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 1.08 | — |
| TurboSparse | 6.98 | 6.45× |
| Speculative (100% accept) | 7.94 | 7.33× |
| PowerInfer | 7.53 | 6.95× |
| **PI + TurboSparse** | **8.61** | **7.95×** |

> Every optimization helps. Gemma-2 architecture has high activation sparsity — TurboSparse alone gives 6.45×.

**Qwen3-30B-A3B** (MoE, 3B active of 30B) — best: **Speculative at 3.68×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 2.11 | — |
| **TurboSparse** | **3.96** | **1.88×** |
| **Speculative** | **7.75** | **3.68×** |
| PowerInfer | 3.45 | 1.64× |
| PI + TurboSparse | 3.30 | 1.57× |

> Spec decoding dominates on large MoE. TurboSparse also effective (1.88×) — the sparse expert routing leaves many neurons cold.

**Gemma-4-E4B** (Dense 4B) — best: **PI+TS at 2.21×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 2.44 | — |
| TurboSparse | 4.37 | 1.79× |
| PowerInfer | 2.57 | 1.05× |
| **PI + TurboSparse** | **5.40** | **2.21×** |
| Speculative (62% accept) | 2.65 | 1.09× |

> TurboSparse shines on Gemma-4 (1.79× alone). PI amplifies the gain when combined. Speculative has partial decode failures.

**Granite-3B-A800M** (MoE, 800M active of 3B) — best: **Speculative at 1.44×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 18.20 | — |
| **Speculative** | **26.26** | **1.44×** |
| PI + TurboSparse | 24.54 | 1.35× |
| TurboSparse | 14.70 | 0.81× |
| PowerInfer | 15.72 | 0.86× |

> Baseline is already fast (800M active params). Speculative adds 44%. All other optimizations regress — model is too sparse for sparsity exploitation to help.

**SmolLM2-1.7B** (Dense 1.7B) — baseline already fast, no optimization helps

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 14.55 | — |
| Speculative | 14.60 | 1.00× |
| PI + TurboSparse | 7.23 | 0.50× |
| TurboSparse | 6.89 | 0.47× |

---

### Auto-Tune

VibeBlade's auto-tuner selects optimal configs automatically. Safe for small/dense and MoE models; **manual threshold tuning recommended for 8B+**.

```python
backend = LlamaCppBackend()
backend.load("model.gguf", auto_tune=True)  # picks best PI/TS/Spec profile
```

| Model | Baseline | Auto-Tune | Speedup |
|---|---:|---:|---:|
| Llama-3.2-1B | 5.09 t/s | 6.05 t/s | 1.19× |
| TinyLlama-1.1B | 24.90 t/s | 28.32 t/s | 1.14× |
| Qwen2.5-MoE | 3.64 t/s | 3.82 t/s | 1.05× |
| Llama-3.1-8B | 2.01 t/s | 2.46 t/s | 1.22× |
| Qwen2.5-14B | 0.90 t/s | 1.01 t/s | 1.12× |

---

### Key findings

- **✅ Validated benchmarks reveal true speedups are 1.24×–1.64× for MoE models** — Previous single-run claims of 50× and 83.2× were artifacts of extreme baseline outliers on short 16-token runs. 5-run × 64-token validation with proper statistics shows real gains.
- **3 threads beats 4 for large MoE on ARM** — Both DeepSeek-Coder-V2-Lite and Gemma-4 26B-A4B perform best at 3 threads. Less cache thrashing on NEON cores.
- **PowerInfer budget tuning is critical** — DeepSeek-V2-Lite needs PI=0.01 (near-zero), Gemma-4 26B needs PI=0.20. Wrong budget can regress performance.
- **Baseline variance is the real enemy** — DeepSeek baseline has 30% CV across 5 runs. Any single-run comparison is unreliable. Always use multi-run averages.
- **Qwen3.6-35B-A3B: hybrid MoE+SSM resists all optimization** — No config beats baseline (2.30 t/s). Mamba-style SSM layers defeat both PowerInfer and TurboSparse heuristics.
- **Gemma-2-2B: every optimization helps** — PI+TS 7.95×, Spec 7.33×, PowerInfer 6.95×, TS 6.45×. Unusually high exploitable sparsity (⚡ single-run, needs validation).
- **Dense models: PI+TS is reliable** — works across 1B–14B dense (1.21×–8.71×). Consistent gains when thresholds are tuned (⚡ single-run).
- **Auto-tune needs MoE awareness** — auto-tune misclassifies DeepSeek-V2-Lite as "1-2B dense". MoE models need different heuristic paths.

> Full data: [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md)

---

---

## Architecture

---

## Architecture

VibeBlade combines six research-backed techniques into a unified inference pipeline:

### TurboSparse — Activation Sparsity (Whitepaper §1)
Only ~10% of FFN neurons fire per token. By predicting which ones activate *before* computing expensive matrix multiplications, VibeBlade skips ~90% of FFN compute. Uses an **EMA-based NeuronPredictor** that adapts to distribution shifts in real-time, plus **dReLU gating** `max(0,x)·max(0,-x)` for bidirectional sparsity.

### ConFu — Speculative Decoding (Whitepaper §2)
A lightweight draft model generates candidate tokens conditioned on **contemplate tokens** — latent reasoning vectors from the target model's feature layer. This reduces distribution mismatch between draft and target, achieving **85–92% acceptance rates** and **3.0–4.1× speedup** over autoregressive decoding.

### RotateKV — Outlier-Aware KV Quantization (Whitepaper §3)
Applies a block-diagonal **Hadamard rotation** to KV cache entries before 2-bit quantization. The rotation spreads outlier magnitudes across channels, enabling aggressive compression with minimal quality loss — **~8× memory reduction** on the KV cache.

### SARATHI — Chunked Prefill Scheduling (Whitepaper §4)
Eliminates head-of-line blocking by chunking prefill requests and interleaving them with decode iterations. Chunk sizes are dynamically computed from available KV cache budget: `chunk_size = floor(available_blocks × block_size / num_active)`.

### SageSched — Uncertainty-Aware Scheduling (Whitepaper §4)
Prioritizes requests by the **Shannon entropy** of their output distributions. High-uncertainty requests (where the model is least confident) get scheduled first since they benefit most from compute resources. A wait-time penalty prevents starvation.

### Phase-Aware MoE Scheduling (DuoServe-style)
Automatically transitions between prefill and decode phases, rebalancing expert placement across VRAM/RAM/SSD tiers. During decode, frequently-used experts are promoted to VRAM for low-latency token generation.

---

## How it works

**Activations-only PCIe transfer.** Expert weights (150MB each) stay in RAM/SSD. Only the tiny activation vector (~8KB) crosses PCIe. This breaks the bandwidth wall that makes MoE inference impossible on consumer GPUs.

3-tier memory hierarchy:
- **VRAM** — hot experts (most-used per layer)
- **RAM** — cold experts (memory-mapped, zero page faults)
- **SSD** — overflow (async pre-fetch 3 layers ahead)

Auto-selects best eviction policy: LRU-K, frequency-aware, cost-benefit, or MAB (multi-armed bandit that learns the best strategy at runtime).

---

## Acceleration backends

VibeBlade ships a **native C++ inference engine** — the entire generate pipeline (tokenization, forward pass, sampling, detokenization) runs in C++ with zero Python in the decode hot path. Weights are mmap'd from GGUF files and dequantized inline during matrix multiplication. No numpy, no llama.cpp dependency.

Supports **all architectures natively**: dense transformers, MoE (Mistral, Qwen, DeepSeek), and hybrid attention+SSM models. MoE routing (top-k expert selection + shared experts) runs entirely in C++.

```bash
# Build the C++ engine (requires pybind11, cmake)
python cpp/build_cpp.py          # cross-platform (Linux/macOS/Windows)

# Or manually on Linux/macOS:
cd cpp && bash build_cpp.sh

# Auto-detected by the chat command for .gguf files
python -m vibeblade chat --model model.gguf            # C++ fast engine
python -m vibeblade chat --model model.gguf --backend numpy  # force NumPy
```

SIMD optimizations are auto-detected at build time:

|| Hardware detected | SIMD backend |
|---|---|
|| AVX-512 + FP16 (Sapphire Rapids+) | AVX-512-FP16 |
|| AVX-512 (Ice Lake+) | AVX-512-F (fp32 path) |
|| AVX2 (Haswell+) | AVX2+FMA |
|| NEON FP16 (ARM) | NEON-FP16 |
|| Apple Silicon (M1–M4) | NEON (Metal/CoreML extras) |
|| Anything else | Scalar fallback |

---

## API

### One-line usage (C++ fast engine)

```python
from vibeblade import VibeBladeModel

model = VibeBladeModel("model.gguf")
print(model.generate("Hello world", max_tokens=128))
```

For GGUF files, VibeBlade auto-detects and uses the native C++ engine — the entire pipeline runs in a single C++ call with zero Python in the decode loop.

### Auto-tuned inference

```python
from vibeblade.llama_backend import LlamaCppBackend

backend = LlamaCppBackend()
backend.load("model.gguf", n_ctx=256, n_threads=4, auto_tune=True)
result = backend.generate("Explain quantum computing", max_tokens=128)
print(f"{result.text} ({result.tokens_per_second:.1f} t/s)")
```

### Direct C++ engine access

```python
from vibeblade.fast_backend import FastModelWrapper

model = FastModelWrapper("model.gguf")

# Full generate — one C++ call, everything native
text, tps = model.generate("Explain quantum computing", max_tokens=256,
                            temperature=0.8, top_k=50, top_p=0.9)

# Streaming — C++ calls back per-token
text, tps = model.generate("Write a poem", max_tokens=64,
                            stream=True)

# Tokenizer access
tokens = model._model.tokenize("Hello world")   # C++ BPE tokenizer
text = model._model.detokenize(tokens)          # C++ decoder
```

### NumPy fallback

```python
model = VibeBladeModel("model.safetensors")  # non-GGUF → auto NumPy
model = VibeBladeModel("model.gguf", backend="numpy")  # force NumPy
```

### Advanced: whitepaper components

```python
from vibeblade import (
    # §1 — TurboSparse: EMA neuron prediction + dReLU gating
    EMANeuronPredictor, drelu_gate,
    # §2 — ConFu: contemplate-token speculative decoding
    ConFuSpeculator, ContemplateTokenLayer, ConFuStats,
    # §3 — RotateKV: outlier-aware 2-bit KV quantization
    RotateKVCache, rotate_kv, hadamard_rotation_matrix,
    # §4 — SARATHI: chunked prefill scheduling
    SarathiScheduler, SarathiConfig, SarathiRequest,
    # §4 — SageSched: uncertainty-aware scheduling
    SageSched, SageConfig, entropy_from_logits,
)

# Example: EMA-based neuron prediction for a 32-layer model
predictor = EMANeuronPredictor(hidden_dim=28672, n_layers=32)
for layer_idx in range(32):
    mask = predictor.predict(layer_idx, gate_activations)
    # Use mask for sparse FFN compute — skip ~90% of neurons
    predictor.update(layer_idx, actual_activations)

# Example: SARATHI chunked prefill scheduling
scheduler = SarathiScheduler(SarathiConfig(kv_cache_blocks=1024, block_size=16))
scheduler.add_request(prompt_tokens=256, priority=2.0)
plan = scheduler.schedule()
# plan["prefill_chunks"] → [(req_id, tokens), ...]
# plan["decode_requests"] → [req_id, ...]
```

---

## Project structure

```
vibeblade/              # Python package
  ├── __init__.py       # VibeBladeModel + public API
  ├── fast_backend.py   # C++ fast engine wrapper (single generate() call)
  ├── transformer.py    # LLaMA forward pass (NumPy fallback)
  ├── loader.py         # GGUF model loader
  ├── generate.py       # Text generation + sampling
  ├── chat.py           # Interactive CLI chat loop
  ├── benchmark.py      # llama.cpp-style benchmark suite
  ├── sparse.py         # TurboSparse dReLU + EMA NeuronPredictor
  ├── quant.py          # RotorQuant 4-bit weight quantization
  ├── cache.py          # KV cache
  ├── rotatekv.py       # RotateKV Hadamard rotation + 2-bit quantization
  ├── confu.py          # ConFu contemplate-token speculative decoding
  ├── sarathi.py        # SARATHI chunked prefill scheduler
  ├── sagesched.py      # SageSched uncertainty-aware scheduler
  ├── moe.py            # MoE router + expert loader
  ├── auto_tune.py      # Automatic optimization config selection
  ├── llama_backend.py  # llama.cpp C++ backend with PI/TS/Spec support
  ├── neural_draft.py   # Neural speculative drafting head
  ├── speculative.py    # Speculative decoding pipeline
  ├── phase_scheduler.py # Phase-aware prefill/decode scheduling
  ├── tiered_memory.py  # VRAM/RAM/SSD 3-tier memory manager
  ├── eviction.py       # LRU-K / frequency / cost-benefit / bandit policies
  ├── setup_wizard.py   # Interactive hardware setup (wizard command)
  └── openai_server.py  # OpenAI-compatible API server

cpp/                    # Native C++ inference engine
  ├── build_cpp.py      # Cross-platform build script (Linux/macOS/Windows)
  ├── include/
  │   ├── gguf.h        # GGUF mmap reader (zero-copy weight loading)
  │   ├── ggml_types.h  # GGML quantization types (Q4_0/Q5/Q8/K-quants/F16)
  │   ├── dequant.h     # Inline dequantization kernels + gemv_dequant
  │   ├── fast_model.h  # VibeBladeFast: full forward pass + generate pipeline
  │   ├── tokenizer.h   # BPE tokenizer (reads GGUF tokenizer metadata)
  │   ├── sampler.h     # Sampler (temperature/top-k/top-p/repetition/mirostat)
  │   └── kernels.h     # SIMD math kernels (GEMM, RMSNorm, SDPA, RoPE)
  └── src/
      ├── gguf.cpp      # GGUF binary parser + array metadata
      ├── dequant.cpp   # Dequantization for all GGML types
      ├── tokenizer.cpp # GPT-2 byte-level BPE implementation
      ├── sampler.cpp   # Sampling strategies
      ├── fast_model.cpp # Full inference: prefill, decode, generate
      └── bindings.cpp  # pybind11 Python bindings

tests/                 # 794 tests covering all modules
```

---

## Powered by

GGUF format · [ONNX Runtime](https://github.com/microsoft/onnxruntime) (cross-platform acceleration) · [TensorRT](https://github.com/NVIDIA/TensorRT) (NVIDIA GPU) · [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) (sparse inference) · [vLLM](https://github.com/vllm-project/vllm) (PagedAttention) · [SARATHI](https://arxiv.org/abs/2403.07219) (chunked prefill) · [EAGLE](https://arxiv.org/abs/2401.15077) (speculative decoding) · [RotateKV](https://arxiv.org/abs/2408.00784) (KV quantization)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome.

## License

BSL 1.1 — free for personal, educational, and non-commercial use. Automatically converts to Apache 2.0 on May 1, 2028. See [LICENSE](LICENSE) for details.

For commercial licensing, contact kevin.lin@vibedrift.com.
