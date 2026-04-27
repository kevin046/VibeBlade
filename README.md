# VibeBlade

**Run any LLM on your own hardware — no cloud, no subscription.**

[![Star History](https://api.star-history.com/svg?repos=kevin046/VibeBlade)](https://star-history.com/#kevin046/VibeBlade)
[![Stars](https://img.shields.io/github/stars/kevin046/VibeBlade?style=flat)](https://github.com/kevin046/VibeBlade/stargazers)
[![Forks](https://img.shields.io/github/forks/kevin046/VibeBlade?style=flat)](https://github.com/kevin046/VibeBlade/network)

**Linux / macOS**
```bash
git clone https://github.com/kevin046/VibeBlade && cd VibeBlade
pip install -e . && python -m vibeblade wizard
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/kevin046/VibeBlade; cd VibeBlade
pip install -e .; python -m vibeblade wizard
```

The wizard detects your hardware, installs any missing tools, configures memory offload, and downloads a model. You're ready to run in under 5 minutes.

[![Build Status](https://github.com/kevin046/VibeBlade/workflows/Build/badge.svg)](https://github.com/kevin046/VibeBlade/actions)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 776 passed](https://img.shields.io/badge/tests-776%20passed-brightgreen.svg)]()

📄 [White Paper](./WHITEPAPER.md) · 📊 [Performance Benchmarks](./WHITEPAPER.md#performance) · 🔒 [Security](./WHITEPAPER.md#security)

---

## CLI commands

| Command | What it does |
|---|---|
| `python -m vibeblade wizard` | Guided setup — hardware detection, install, config, model download |
| `python -m vibeblade serve` | Start local inference API server (OpenAI-compatible) |
| `python -m vibeblade bench` | Benchmark suite |
| `python -m vibeblade run <model>` | Run a model directly from terminal |

> **Dashboard & Model Browser** are part of VibeBlade Pro (commercial license). Contact [kevin.lin@vibedrift.com](mailto:kevin.lin@vibedrift.com) for access.

---

## What it runs

*From the [white paper](./WHITEPAPER.md) — Table 5: Model Feasibility on Consumer Hardware*

| System | RAM / VRAM | Baseline Max | VibeBlade Max | Gain |
|---|---|---|---|---|
| Budget Laptop | 16GB / 0GB | 8B Q4 (~15 t/s) | **13B Q4 (~35 t/s)** | 2× capacity |
| Standard Desktop | 32GB / 12GB | 13B Q4 (~8 t/s) | **70B Q4 (~12 t/s)** | 5× capacity |
| Pro Workstation | 64GB / 24GB | 32B Q4 (~10 t/s) | **236B MoE (~15 t/s)** | 7× capacity |
| Unified Mac | 128GB / 128GB | 70B Q4 (~15 t/s) | **1T MoE (~8 t/s)** | 14× capacity |

VibeBlade achieves this by running 3–6× larger models at the same speed through activation sparsity, adaptive memory tiering, and speculative decoding. A 70B model on a $500 desktop — previously requiring a $4,000 GPU — now runs at conversational speed.

### Peak decode throughput (7B Q4, whitepaper Table 4)

| Hardware | VibeBlade | Baseline | Scaling |
|---|---|---|---|
| RTX 5090 (32GB) | 62.5 t/s | 18.2 t/s | 3.4× |
| M4 Ultra (128GB) | 114.0 t/s | 15.0 t/s | 7.6× |
| RTX 4090 (24GB) | 18.4 t/s | 3.1 t/s | 5.9× |
| Strix Halo (128GB) | 22.0 t/s | 4.2 t/s | 5.2× |

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

Auto-detected during `python -m vibeblade wizard` — no manual setup needed.

| Hardware detected | Backend installed |
|---|---|
| NVIDIA GPU (nvidia-smi) | CUDA extras |
| Apple Silicon (M1–M4) | Metal / CoreML |
| AMD GPU (rocm-smi) | ROCm / Vulkan |
| Intel/AMD CPU with AVX-512 | AVX-512 optimized |
| Intel/AMD CPU with AVX2 | AVX2 optimized |
| Apple CPU | NEON optimized |
| Anything else | Base (NumPy, universal) |

Manual install also supported:
```bash
pip install -e .                        # Base (works everywhere)
pip install -e ".[gpu-metal]"           # Apple Silicon (CoreML)
pip install -e ".[gpu-vulkan]"          # AMD Vulkan
pip install -e ".[grammar]"             # Structured output support
```

---

## API

### One-line usage

```python
from vibeblade import VibeBladeModel

model = VibeBladeModel("model.gguf")
print(model.generate("Hello world", max_tokens=128))
```

All acceleration layers auto-enable based on detected hardware. No configuration needed.

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
  ├── transformer.py    # LLaMA forward pass (RMSNorm, RoPE, SwiGLU)
  ├── loader.py         # GGUF model loader
  ├── generate.py       # Text generation + sampling
  ├── benchmark.py      # llama.cpp-style benchmark suite
  ├── sparse.py         # TurboSparse dReLU + EMA NeuronPredictor
  ├── quant.py          # RotorQuant 4-bit weight quantization
  ├── cache.py          # KV cache
  ├── rotatekv.py       # RotateKV Hadamard rotation + 2-bit quantization
  ├── confu.py          # ConFu contemplate-token speculative decoding
  ├── sarathi.py        # SARATHI chunked prefill scheduler
  ├── sagesched.py      # SageSched uncertainty-aware scheduler
  ├── moe.py            # MoE router + expert loader
  ├── phase_scheduler.py # Phase-aware prefill/decode scheduling
  ├── tiered_memory.py  # VRAM/RAM/SSD 3-tier memory manager
  ├── eviction.py       # LRU-K / frequency / cost-benefit / bandit policies
  ├── setup_wizard.py   # Interactive hardware setup (wizard command)
  └── openai_server.py  # OpenAI-compatible API server

tests/                 # 776 tests covering all modules
cpp/                   # Optional C++ AVX-512/NEON kernels
```

---

## Powered by

[llama.cpp](https://github.com/ggml-org/llama.cpp) (GGUF format) · [ONNX Runtime](https://github.com/microsoft/onnxruntime) (cross-platform acceleration) · [TensorRT](https://github.com/NVIDIA/TensorRT) (NVIDIA GPU) · [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) (sparse inference) · [vLLM](https://github.com/vllm-project/vllm) (PagedAttention) · [SARATHI](https://arxiv.org/abs/2403.07219) (chunked prefill) · [EAGLE](https://arxiv.org/abs/2401.15077) (speculative decoding) · [RotateKV](https://arxiv.org/abs/2408.00784) (KV quantization)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome.

## License

BSL 1.1 — free for personal, educational, and non-commercial use. Automatically converts to Apache 2.0 on May 1, 2028. See [LICENSE](LICENSE) for details.

For commercial licensing, contact kevin.lin@vibedrift.com.
