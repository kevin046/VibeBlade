# Contributing to VibeBlade

First off, thanks for taking the time to contribute! VibeBlade is a CPU/RAM
sparse inference framework, and community contributions make it better for
everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Architecture Overview](#architecture-overview)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold this standard.

## Getting Started

1. **Fork** the repository and clone it locally
2. **Create a branch** for your work: `git checkout -b feature/your-feature`
3. **Install dependencies** (see Development Setup below)
4. **Make your changes** with tests
5. **Submit a PR** following the process below

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/VibeBlade.git
cd VibeBlade

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: install grammar support
pip install -e ".[grammar]"

# Optional: install GPU backends (requires platform-specific SDK)
pip install -e ".[gpu-metal]"   # macOS + Metal SDK
pip install -e ".[gpu-vulkan]"  # Linux/macOS + Vulkan SDK
```

### Verify Setup

```bash
# Run tests
python -m pytest tests/

# Run linting
ruff check vibeblade/ tests/

# Run benchmarks
python -m vibeblade bench --quick
```

## Making Changes

### What to Contribute

- **Bug fixes** — always welcome, even for minor issues
- **New operations** — sparse attention variants, quantization schemes, kernel optimizations
- **Backends** — new hardware acceleration targets (ROCm, OpenCL, WebGPU, etc.)
- **Benchmarks** — new benchmark cases, platform-specific tuning
- **Documentation** — guides, examples, API docs
- **Tests** — coverage improvements, edge cases

### Before You Start

1. **Check existing issues** — someone may already be working on it
2. **Open an issue** for significant changes so we can discuss the approach
3. **Keep it focused** — one PR per concern makes review easier

## Pull Request Process

1. **Update tests** — all existing tests must pass (`python -m pytest tests/`)
2. **Add tests** — new features need test coverage; bug fixes need a regression test
3. **Run linting** — `ruff check vibeblade/ tests/` must produce no errors
4. **Keep it clean** — remove debug prints, commented-out code, and unnecessary imports
5. **Commit messages** — use conventional commits format:
   - `feat: add rotary positional embedding`
   - `fix: correct GQA attention dimension mismatch`
   - `docs: update installation guide`
   - `test: add edge case for empty KV cache`
   - `refactor: simplify memory pool allocation`
   - `perf: optimize matmul kernel for ARM NEON`

## Coding Standards

- **Python 3.10+** — use type hints where practical
- **Ruff** for linting and formatting (configured in `pyproject.toml`)
- **Docstrings** — Google-style for public functions and classes
- **Imports** — absolute imports from `vibeblade.*`
- **No secrets** — never commit API keys, tokens, or credentials
- **Platform-aware** — always provide a CPU/NumPy fallback; don't hardcode GPU assumptions

### Example Contribution

```python
"""My new operation."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def my_operation(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """One-line description of what this does.

    Args:
        x: Input array of shape (batch, dim).

    Returns:
        Output array of shape (batch, dim).
    """
    result = np.zeros_like(x)
    # ... your implementation ...
    return result
```

## Testing

- Tests live in `tests/` directory
- Test files follow the pattern `test_<module>.py`
- Use `pytest` fixtures for common setup
- Mock external dependencies (GPU SDKs, ONNX Runtime) when not available
- Use `pytest.importorskip("optional_dep")` for optional dependency tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_grammar.py -v

# Run with coverage
python -m pytest tests/ --cov=vibeblade
```

## Architecture Overview

```
vibeblade/
├── __init__.py          # Package entry, lazy imports
├── __main__.py          # CLI (bench, version)
├── transformer.py       # Model loading & forward pass
├── generate.py          # Token sampling & grammar-constrained decoding
├── sparse.py            # Sparse attention & KV cache
├── gpu.py               # GPU backend auto-selection (Metal/Vulkan/NumPy)
├── onnx_export.py       # ONNX model export
├── onnx_backend.py      # ONNX Runtime per-op backend with NumPy fallback
├── ort_backend.py       # ORT provider auto-detection (CUDA/TRT/ROCm/CoreML/CPU)
├── tensorrt_backend.py  # TensorRT conversion utility
├── accelerated.py       # Auto-router to fastest available backend
├── grammar/             # Grammar-constrained decoding
│   ├── fsm.py           # DFA/NFA finite state machines
│   ├── regex_grammar.py # Regex-to-grammar compiler
│   ├── json_schema.py   # JSON Schema subset to grammar converter
│   ├── ebnf.py          # EBNF parser
│   └── constraint.py    # Integration with sampling loop
├── bench/               # Benchmark suite
│   ├── runner.py        # Benchmark runner with timing & stats
│   └── cases.py         # 13 benchmark groups
src/
├── kernels.cpp          # C++ compute kernels
├── kernels.metal        # Metal shaders (macOS)
├── kernels_vulkan.glsl  # GLSL shaders (cross-platform)
├── metal_bridge.mm      # ObjC++ Metal bridge
└── vulkan_backend.cpp   # Vulkan runtime
```

### Key Design Principles

1. **Fallback-first** — every operation must work on CPU with just NumPy
2. **Per-op acceleration** — individual ops get GPU sessions, not monolithic graphs
3. **Lazy imports** — optional deps (onnxruntime, torch) import only when used
4. **Session caching** — compiled GPU/ORT sessions are cached after first use

## Questions?

- Open a [GitHub Discussion](https://github.com/kevin046/VibeBlade/discussions) for questions
- Open a [GitHub Issue](https://github.com/kevin046/VibeBlade/issues) for bugs or feature requests
