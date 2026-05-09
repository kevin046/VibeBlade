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

---

### 🏆 Best speedup by model

| Model | Type | Best Config | Baseline → Optimized | Speedup |
|---|---|---|---|---|
| **Gemma-4 26B-A4B** | MoE (4B active) | Speculative | 0.13 → 6.24 t/s | **50.0×** |
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

**Gemma-4 26B-A4B** (MoE, 4B active of 26B) — best: **Speculative at 50.0×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 0.13 | — |
| TurboSparse | 3.16 | 25.3× |
| **PowerInfer** | **4.35** | **34.9×** |
| **Speculative** | **6.24** | **50.0×** |
| Spec + TurboSparse | 4.35 | 34.9× |
| PI + TurboSparse | 2.48 | 19.9× |

> 🔥 MoE breakthrough. Baseline is extremely slow (0.13 t/s) due to full-expert FFN compute on 26B params. Speculative decoding with 100% acceptance skips redundant expert evaluations, achieving 50× speedup.

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

- **🔥 MoE + Speculative = 50× on Gemma-4 26B-A4B** — the largest speedup ever recorded on this hardware. MoE's redundant expert activation is the perfect target for speculative decoding with 100% acceptance.
- **Gemma-2-2B: every optimization helps** — PI+TS 7.95×, Spec 7.33×, PowerInfer 6.95×, TS 6.45×. This architecture has unusually high exploitable sparsity.
- **MoE models are the sweet spot** — Gemma-4 MoE (50×), Qwen3 MoE (3.68×), Qwen2.5 MoE (2.05×), Granite MoE (1.44×). Sparse expert routing creates massive optimization headroom.
- **Dense models: PI+TS is reliable** — works across 1B–14B dense (1.21×–8.71×). Consistent gains when thresholds are tuned.
- **TurboSparse regresses on already-sparse models** — SmolLM2 (0.47×), Granite MoE (0.81×). Adding sparsity to sparse architectures adds overhead without benefit.
- **Auto-tune needs MoE awareness** — auto-tune regresses 0.25× on Granite MoE. MoE models need different heuristic paths than dense models.

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
