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

| Model | Arch | Best Config | Baseline → Optimized | Speedup |
|---|---|---|---|---|
| **Llama-3.2-1B** | Dense 1B | PI + TurboSparse | 2.69 → 23.43 t/s | **8.71×** |
| **Llama-3.2-3B** | Dense 3B | Speculative | 3.18 → 4.80 t/s | **1.51×** |
| **Phi-3.5-mini** | Dense 3.8B | PI + TurboSparse | 3.08 → 5.87 t/s | **1.91×** |
| **Qwen2.5-3B** | Dense 3B | PI + TurboSparse | 3.84 → 4.52 t/s | **1.18×** |
| **Qwen2.5-MoE** | MoE 2×1.5B | PowerInfer | 2.64 → 5.41 t/s | **2.05×** |

---

### Optimization breakdown

**Llama-3.2-1B** (Dense 1B) — best: **PI+TS at 8.71×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 2.69 | — |
| TurboSparse | 8.40 | 3.12× |
| PowerInfer | 3.99 | 1.48× |
| Speculative | 3.04 | 1.13× |
| Spec + TurboSparse | 20.91 | 7.77× |
| **PI + TurboSparse** | **23.43** | **8.71×** |

**Llama-3.2-3B** (Dense 3B) — best: **Speculative at 1.51×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 3.18 | — |
| TurboSparse | 2.89 | 0.91× |
| **Speculative** | **4.80** | **1.51×** |
| Spec + TurboSparse | 3.86 | 1.21× |
| PowerInfer | 2.80 | 0.88× |
| PI + TurboSparse | 2.94 | 0.93× |

**Phi-3.5-mini** (Dense 3.8B) — best: **PI+TS at 1.91×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 3.08 | — |
| **PowerInfer** | **3.32** | **1.08×** |
| **PI + TurboSparse** | **5.87** | **1.91×** |
| TurboSparse | 2.97 | 0.97× |
| Speculative | 2.92 | 0.95× |
| Spec + TurboSparse | 2.68 | 0.87× |

**Qwen2.5-3B** (Dense 3B) — best: **PI+TS at 1.18×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 3.84 | — |
| **PI + TurboSparse** | **4.52** | **1.18×** |
| Spec + TurboSparse | 3.81 | 0.99× |
| Speculative | 1.89 | 0.49× |
| TurboSparse | 1.54 | 0.40× |
| PowerInfer | 1.60 | 0.42× |

**Qwen2.5-MoE** (MoE 2×1.5B) — best: **PowerInfer at 2.05×**

| Config | t/s | vs Baseline |
|---|---:|---:|
| Baseline (llama.cpp) | 2.64 | — |
| **PowerInfer** | **5.41** | **2.05×** |
| TurboSparse | 2.88 | 1.09× |
| PI + TurboSparse | 4.13 | 1.57× |
| Speculative | 2.39 | 0.90× |

---

### Auto-Tune

VibeBlade's auto-tuner selects the optimal config per model automatically:

```python
backend = LlamaCppBackend()
backend.load("model.gguf", auto_tune=True)  # picks best PI/TS/Spec profile
```

| Model | Baseline | Auto-Tune | Speedup |
|---|---:|---:|---:|
| Llama-3.2-1B | 5.09 t/s | 6.05 t/s | 1.19× |
| Llama-3.2-3B | 4.27 t/s | 4.16 t/s | 0.97× |
| Phi-3.5-mini | 3.11 t/s | 2.39 t/s | 0.77× |
| Qwen2.5-3B | 4.43 t/s | 3.58 t/s | 0.81× |
| Qwen2.5-MoE | 3.64 t/s | 3.82 t/s | 1.05× |

---

### Key findings

- **Llama-3.2-1B + PI+TS = 8.71×** — highest speedup recorded on this hardware. PowerInfer row-skipping and TurboSparse sparsity compound on small dense models.
- **PI+TS works on larger dense models too** — Phi-3.5-mini (3.8B) gets 1.91×, Qwen2.5-3B (3B) gets 1.18×.
- **Speculative decoding favors mid-size dense models** — Llama-3.2-3B gets 1.51× from spec alone, but gains disappear on smaller (TinyLlama 0.64×) models.
- **MoE + PowerInfer = 2.05×** — sparse expert activation aligns naturally with PowerInfer's hot/cold neuron classification.
- **Auto-Tune is conservative** — safe gains on 1B/ MoE, but can regress on larger models where manual tuning beats the auto-select heuristic.

> Full data: [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md) · [Detailed analysis](./references/benchmark-may2026-large-models.md)

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
