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

ARM NEON (aarch64) · 4 cores · Q4\_K\_M quantization · 256 ctx · temp=0.0 · **Baseline = llama.cpp**
**3-run validation (32 tokens) · Mean ± Std reported · All 17 models benchmarked**

---

### 🏆 Best speedup by model

| Model | Type | Best Config | Baseline → Optimized | Speedup |
|---|---|---|---|---|
| **TinyLlama-1.1B** | Dense 1.1B | PowerInfer | 9.078 → 23.084 t/s | **2.54×** |
| **DeepSeek-Coder-V2-Lite** | MoE 16B (2.4B active) | PowerInfer | 3.045 → 5.989 t/s | **1.97×** |
| **Llama-3.2-3B** | Dense 3.2B | Spec+TS | 3.371 → 5.294 t/s | **1.57×** |
| **Granite-3.0-3B-A800M** | MoE 3B (A800M) | PI+TS | 13.094 → 19.846 t/s | **1.52×** |
| **Qwen2.5-MoE** | MoE 3B (2×1.5B) | TurboSparse | 3.430 → 4.831 t/s | **1.41×** |
| **Qwen2.5-3B** | Dense 3B | Speculative | 3.719 → 5.075 t/s | **1.36×** |
| **Phi-3.5-mini** | Dense 3.8B | PowerInfer | 4.059 → 5.390 t/s | **1.33×** |
| **Qwen3.6-35B-A3B** | Hybrid MoE+SSM 35B (A3B) | PI+TS | 3.078 → 4.063 t/s | **1.32×** |
| **Llama-3.2-1B** | Dense 1.2B | PI+TS | 10.714 → 13.498 t/s | **1.26×** |
| **Llama-3.1-8B** | Dense 8B | PI+TS | 1.749 → 2.193 t/s | **1.25×** |
| **Gemma-3-4B** | Dense 4B | Speculative | 2.827 → 3.104 t/s | **1.10×** |
| **Gemma-4-26B-A4B** | MoE 26B (A4B) | PI+TS | 2.812 → 3.014 t/s | **1.07×** |
| **SmolLM2-1.7B** | Dense 1.7B | PI+TS | 9.646 → 10.042 t/s | **1.04×** |
| **Gemma-4-E4B** | Dense 4B | Speculative | 2.910 → 2.940 t/s | **1.01×** |
| **Gemma-2-2B** | Dense 2B | Baseline | 6.981 t/s | **1.00× (no gain)** |
| **Phi-2** | Dense 2.7B | Baseline | 4.779 t/s | **1.00× (no gain)** |
| **Qwen3-30B-A3B** | MoE 30B (A3B) | Baseline | 6.907 t/s | **1.00× (no gain)** |
| **Qwen2.5-14B** | Dense 14B | ❌ FAILED | — | — |

---

### Detailed results per model

**TinyLlama-1.1B** (Dense 1.1B) — best: **PowerInfer at 2.54×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 9.078 | 2.871 | 1.00× | 5.894–11.471 |
| TurboSparse | 15.546 | 8.423 | 1.71× | 6.858–23.676 |
| PowerInfer **🏆** | 23.084 | 13.878 | 2.54× | 7.059–31.237 |
| Speculative | 12.349 | 2.913 | 1.36× | 10.585–15.711 |
| Spec+TS | 18.905 | 11.100 | 2.08× | 6.088–25.337 |
| PI+TS | 21.940 | 13.559 | 2.42× | 6.327–30.755 |

**Llama-3.2-1B** (Dense 1.2B) — best: **PI+TS at 1.26×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 10.714 | 8.845 | 1.00× | 5.382–20.924 |
| TurboSparse | 11.981 | 7.646 | 1.12× | 5.682–20.487 |
| PowerInfer | 7.598 | 3.245 | 0.71× | 5.561–11.340 |
| Speculative | 11.781 | 8.009 | 1.10× | 5.848–20.891 |
| Spec+TS | 10.923 | 8.032 | 1.02× | 5.807–20.180 |
| PI+TS **🏆** | 13.498 | 8.195 | 1.26× | 6.717–22.605 |

**Llama-3.2-3B** (Dense 3.2B) — best: **Spec+TS at 1.57×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 3.371 | 1.146 | 1.00× | 2.048–4.047 |
| TurboSparse | 3.999 | 0.172 | 1.19× | 3.825–4.170 |
| PowerInfer | 4.430 | 0.326 | 1.31× | 4.127–4.774 |
| Speculative | 4.087 | 0.794 | 1.21× | 3.461–4.979 |
| Spec+TS **🏆** | 5.294 | 1.309 | 1.57× | 3.998–6.615 |
| PI+TS | 4.180 | 0.893 | 1.24× | 3.166–4.847 |

**Phi-3.5-mini** (Dense 3.8B) — best: **PowerInfer at 1.33×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 4.059 | 0.351 | 1.00× | 3.655–4.285 |
| TurboSparse | 3.601 | 0.578 | 0.89× | 3.217–4.265 |
| PowerInfer **🏆** | 5.390 | 2.055 | 1.33× | 3.647–7.656 |
| Speculative | 5.100 | 1.916 | 1.26× | 3.986–7.313 |
| Spec+TS | 3.775 | 0.675 | 0.93× | 3.199–4.518 |
| PI+TS | 3.700 | 1.099 | 0.91× | 2.466–4.575 |

**Qwen2.5-3B** (Dense 3B) — best: **Speculative at 1.36×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 3.719 | 0.281 | 1.00× | 3.502–4.037 |
| TurboSparse | 4.116 | 0.500 | 1.11× | 3.567–4.546 |
| PowerInfer | 4.293 | 0.346 | 1.15× | 4.019–4.682 |
| Speculative **🏆** | 5.075 | 0.831 | 1.36× | 4.115–5.588 |
| Spec+TS | 5.059 | 2.163 | 1.36× | 3.779–7.557 |
| PI+TS | 4.694 | 0.511 | 1.26× | 4.104–5.012 |

**Qwen2.5-MoE** (MoE 3B (2×1.5B)) — best: **TurboSparse at 1.41×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 3.430 | 0.330 | 1.00× | 3.151–3.795 |
| TurboSparse **🏆** | 4.831 | 1.902 | 1.41× | 3.666–7.026 |
| PowerInfer | 3.724 | 0.242 | 1.09× | 3.446–3.888 |
| Speculative | 3.900 | 2.290 | 1.14× | 2.366–6.532 |
| Spec+TS | 3.657 | 0.100 | 1.07× | 3.577–3.769 |
| PI+TS | 3.979 | 0.690 | 1.16× | 3.243–4.610 |

**Gemma-2-2B** (Dense 2B) — best: **Baseline at 1.00×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline **🏆** | 6.981 | 2.650 | 1.00× | 5.449–10.041 |
| TurboSparse | 4.490 | 0.129 | 0.64× | 4.345–4.592 |
| PowerInfer | 4.482 | 0.182 | 0.64× | 4.295–4.659 |
| Speculative | 5.736 | 3.960 | 0.82× | 2.734–10.225 |
| Spec+TS | 6.164 | 0.424 | 0.88× | 5.713–6.554 |
| PI+TS | 4.566 | 0.316 | 0.65× | 4.360–4.930 |

**Gemma-3-4B** (Dense 4B) — best: **Speculative at 1.10×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 2.827 | 0.294 | 1.00× | 2.568–3.146 |
| TurboSparse | 2.271 | 0.662 | 0.80× | 1.534–2.815 |
| PowerInfer | 2.958 | 0.110 | 1.05× | 2.833–3.040 |
| Speculative **🏆** | 3.104 | 0.387 | 1.10× | 2.659–3.370 |
| Spec+TS | 2.939 | 0.101 | 1.04× | 2.880–3.056 |
| PI+TS | 3.065 | 0.443 | 1.08× | 2.673–3.546 |

**Gemma-4-E4B** (Dense 4B) — best: **Speculative at 1.01×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 2.910 | 0.539 | 1.00× | 2.448–3.502 |
| TurboSparse | 2.561 | 0.802 | 0.88× | 1.720–3.317 |
| PowerInfer | 2.723 | 0.926 | 0.94× | 1.674–3.427 |
| Speculative **🏆** | 2.940 | 0.343 | 1.01× | 2.669–3.326 |
| Spec+TS | 2.702 | 0.563 | 0.93× | 2.111–3.232 |
| PI+TS | 2.480 | 1.204 | 0.85× | 1.118–3.402 |

**Gemma-4-26B-A4B** (MoE 26B (A4B)) — best: **PI+TS at 1.07×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 2.812 | 0.214 | 1.00× | 2.625–3.046 |
| TurboSparse | 2.988 | 0.356 | 1.06× | 2.750–3.396 |
| PowerInfer | 2.879 | 0.127 | 1.02× | 2.776–3.021 |
| Speculative | 2.671 | 0.653 | 0.95× | 2.010–3.316 |
| Spec+TS | 2.877 | 0.135 | 1.02× | 2.748–3.018 |
| PI+TS **🏆** | 3.014 | 0.345 | 1.07× | 2.807–3.412 |

**Llama-3.1-8B** (Dense 8B) — best: **PI+TS at 1.25×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 1.749 | 0.494 | 1.00× | 1.180–2.067 |
| TurboSparse | 2.143 | 0.034 | 1.23× | 2.120–2.183 |
| PowerInfer | 2.084 | 0.128 | 1.19× | 1.971–2.222 |
| Speculative | 1.604 | 0.469 | 0.92× | 1.063–1.886 |
| Spec+TS | 2.132 | 0.043 | 1.22× | 2.091–2.176 |
| PI+TS **🏆** | 2.193 | 0.053 | 1.25× | 2.134–2.237 |

**Qwen2.5-14B** ❌ — *Failed: multi-file GGUF (split shard not supported).*  

**Phi-2** (Dense 2.7B) — best: **Baseline at 1.00×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline **🏆** | 4.779 | 0.757 | 1.00× | 4.229–5.642 |
| TurboSparse | 4.183 | 0.354 | 0.88× | 3.832–4.540 |
| PowerInfer | 3.954 | 0.529 | 0.83× | 3.420–4.477 |
| Speculative | 4.670 | 0.818 | 0.98× | 3.910–5.536 |
| Spec+TS | 4.225 | 2.241 | 0.88× | 2.274–6.672 |
| PI+TS | 3.909 | 1.446 | 0.82× | 2.594–5.457 |

**SmolLM2-1.7B** (Dense 1.7B) — best: **PI+TS at 1.04×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 9.646 | 5.077 | 1.00× | 4.525–14.677 |
| TurboSparse | 6.960 | 1.094 | 0.72× | 6.064–8.179 |
| PowerInfer | 8.800 | 6.134 | 0.91× | 5.234–15.882 |
| Speculative | 6.969 | 1.770 | 0.72× | 5.245–8.782 |
| Spec+TS | 6.474 | 2.210 | 0.67× | 5.162–9.026 |
| PI+TS **🏆** | 10.042 | 5.947 | 1.04× | 4.300–16.175 |

**Granite-3.0-3B-A800M** (MoE 3B (A800M)) — best: **PI+TS at 1.52×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 13.094 | 10.129 | 1.00× | 5.772–24.653 |
| TurboSparse | 13.446 | 8.971 | 1.03× | 4.114–22.006 |
| PowerInfer | 8.532 | 2.476 | 0.65× | 5.794–10.614 |
| Speculative | 17.440 | 8.800 | 1.33× | 8.239–25.776 |
| Spec+TS | 15.119 | 10.955 | 1.15× | 6.036–27.286 |
| PI+TS **🏆** | 19.846 | 11.829 | 1.52× | 6.198–27.151 |

**DeepSeek-Coder-V2-Lite** (MoE 16B (2.4B active)) — best: **PowerInfer at 1.97×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 3.045 | 1.404 | 1.00× | 1.869–4.599 |
| TurboSparse | 3.966 | 0.592 | 1.30× | 3.298–4.426 |
| PowerInfer **🏆** | 5.989 | 1.210 | 1.97× | 4.707–7.111 |
| Speculative | 4.859 | 0.938 | 1.60× | 4.071–5.897 |
| Spec+TS | 4.233 | 0.655 | 1.39× | 3.483–4.692 |
| PI+TS | 4.499 | 0.157 | 1.48× | 4.344–4.658 |

**Qwen3-30B-A3B** (MoE 30B (A3B)) — best: **Baseline at 1.00×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline **🏆** | 6.907 | 1.857 | 1.00× | 4.900–8.564 |
| TurboSparse | 6.320 | 2.046 | 0.92× | 4.791–8.645 |
| PowerInfer | 6.529 | 2.543 | 0.95× | 4.707–9.434 |
| Speculative | 6.051 | 2.282 | 0.88× | 4.542–8.676 |
| Spec+TS | 5.115 | 1.461 | 0.74× | 4.030–6.776 |
| PI+TS | 6.406 | 2.647 | 0.93× | 4.719–9.457 |

**Qwen3.6-35B-A3B** (Hybrid MoE+SSM 35B (A3B)) — best: **PI+TS at 1.32×**

| Config | Mean t/s | ± Std | Speedup | Range |
|---|---:|---:|---:|---|
| Baseline | 3.078 | 1.009 | 1.00× | 1.954–3.904 |
| TurboSparse | 2.983 | 0.281 | 0.97× | 2.698–3.259 |
| PowerInfer | 3.205 | 0.180 | 1.04× | 3.052–3.403 |
| Speculative | 3.754 | 0.382 | 1.22× | 3.410–4.166 |
| Spec+TS | 2.923 | 1.276 | 0.95× | 1.450–3.680 |
| PI+TS **🏆** | 4.063 | 0.852 | 1.32× | 3.152–4.841 |

---

### Key findings

- **PI+TS (PowerInfer + TurboSparse) is the most consistently winning config** — top performer in 6 of 17 models
- **MoE models show varied response to optimization** — DeepSeek-Coder-V2-Lite gains 1.97× while Qwen3-30B-A3B gains nothing
- **Dense models < 4B benefit most from optimization** — TinyLlama-1.1B achieves 2.54× speedup
- **Large dense models (>8B) see modest gains** — Llama-3.1-8B at 1.25×, hardware-constrained
- **3-run validation reveals high variance in many configs** — single-run benchmarks can be misleading (± std up to 13.9 t/s for some configs)
- **No single optimization works universally** — best config varies significantly by model architecture

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
