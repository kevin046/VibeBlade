# VibeBlade

**Run any LLM on your own hardware — no cloud, no subscription.**

[![Star History](https://api.star-history.com/svg?repos=kevin046/VibeBlade)](https://star-history.com/#kevin046/VibeBlade)
[![Stars](https://img.shields.io/github/stars/kevin046/VibeBlade?style=flat)](https://github.com/kevin046/VibeBlade/stargazers)
[![Forks](https://img.shields.io/github/forks/kevin046/VibeBlade?style=flat)](https://github.com/kevin046/VibeBlade/network)

```bash
git clone https://github.com/kevin046/VibeBlade && cd VibeBlade
pip install -e . && python -m vibeblade wizard
```

Wizard auto-detects hardware, installs remaining prerequisites, configures memory offload, and downloads a model.

[![Build](https://github.com/kevin046/VibeBlade/actions/workflows/build.yml/badge.svg?branch=main&style=flat-square)](https://github.com/kevin046/VibeBlade/actions)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 665 passed](https://img.shields.io/badge/tests-665%20passed-brightgreen.svg)]()

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

| Model | Hardware | Speed |
|---|---|---|
| Llama-3 8B | 16GB RAM | ~15 t/s |
| Mistral 7B | 16GB RAM + 4GB GPU | ~18 t/s |
| Llama-3 70B | 64GB RAM + 8GB GPU | ~7 t/s |
| **MiniMax M2.7** (230B MoE) | 16GB VRAM + 256GB RAM | **~8–14 t/s** |
| **MiniMax M2.7** (230B MoE) | 16GB VRAM + 32GB RAM + NVMe | **~2–4 t/s** |
| **Mixtral 8×7B** | 16GB VRAM + 32GB RAM | **~12–16 t/s** |

*Estimated. MiniMax M2.7 is a 230B MoE model (~115GB at 4-bit). VibeBlade keeps hot experts in VRAM and cold experts in RAM/SSD — only activations cross PCIe.*

---

## Quick start

Pick your platform and run these two commands:

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

## API (one line)

```python
from vibeblade import VibeBladeModel

model = VibeBladeModel("model.gguf")
print(model.generate("Hello world", max_tokens=128))
```

All acceleration layers auto-enable based on detected hardware. No configuration needed.

---

## Project structure

```
vibeblade/          # Python package
  ├── __init__.py    # VibeBladeModel + enable_*() API
  ├── transformer.py # LLaMA forward pass (RMSNorm, RoPE, SwiGLU)
  ├── loader.py      # GGUF model loader
  ├── generate.py    # Text generation + sampling
  ├── benchmark.py   # llama.cpp-style benchmark suite
  ├── sparse.py      # TurboSparse dReLU activation sparsity
  ├── quant.py       # RotorQuant 4-bit weight quantization
  ├── cache.py       # KV cache
  ├── moe.py         # MoE router + expert loader
  ├── tiered_memory.py # VRAM/RAM/SSD 3-tier memory manager
  ├── setup_wizard.py # Interactive hardware setup (wizard command)
  └── openai_server.py # OpenAI-compatible API server

tests/               # 578+ tests covering all modules
cpp/                 # Optional C++ AVX-512/NEON kernels
```

---

## Powered by

[llama.cpp](https://github.com/ggml-org/llama.cpp) (GGUF format) · [ONNX Runtime](https://github.com/microsoft/onnxruntime) (cross-platform acceleration) · [TensorRT](https://github.com/NVIDIA/TensorRT) (NVIDIA GPU) · [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) (sparse inference) · [vLLM](https://github.com/vllm-project/vllm) (PagedAttention)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome.

## License

BSL 1.1 — free for personal, educational, and non-commercial use. Automatically converts to Apache 2.0 on May 1, 2028. See [LICENSE](LICENSE) for details.

For commercial licensing, contact kevin.lin@vibedrift.com.