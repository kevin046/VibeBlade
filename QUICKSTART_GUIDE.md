# VibeBlade — Step-by-Step Usage Guide

This guide covers how to install and use VibeBlade on **Windows**, **Linux**, and **macOS**.

---

## Prerequisites

| OS | Python | Required |
|---|--------|----------|
| **Linux** | Python 3.10+ | `pip`, `git` |
| **macOS** | Python 3.10+ | `pip`, `git` (via XCode Command Line Tools) |
| **Windows** | Python 3.10+ | `pip`, `git` (via Git for Windows) |

**Recommended hardware:**
- Minimum: 16 GB RAM (for small models)
- Recommended: 32+ GB RAM, 16 GB VRAM (for MoE models)
- SSD: NVMe recommended for HYBRID_SSD mode

---

## Step 1: Install Python (if not already installed)

### Linux (Ubuntu/Debian)
```bash
# Check if Python is installed
python3 --version

# If not, install it
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

### macOS
```bash
# Install via Homebrew (recommended)
brew install python@3.11

# OR download from python.org
# https://www.python.org/downloads/
```

### Windows
```powershell
# Download from python.org
# https://www.python.org/downloads/
# Check "Add Python to PATH" during installation

# OR via winget
winget install Python.Python.3.11
```

---

## Step 2: Clone the Repository

```bash
# Clone VibeBlade from GitHub
git clone https://github.com/kevin046/VibeBlade.git
cd VibeBlade
```

---

## Step 3: Create a Virtual Environment

### Linux / macOS / Windows (PowerShell)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# You should see (venv) at the start of your prompt
```

---

## Step 4: Install VibeBlade

```bash
# Upgrade pip
pip install --upgrade pip

# Install VibeBlade (editable mode)
pip install -e .

# Install with ONNX Runtime acceleration (optional, faster)
pip install -e ".[onnx]"

# Install C++ acceleration (optional, fastest on x86)
pip install -e ".[cpp]"

# Install all extras
pip install -e ".[onnx,testing]"
```

---

## Step 5: Verify Installation

```bash
# Run tests
pytest tests/test_sparse.py -v

# Run benchmark
python -m vibeblade bench --quick
```

**Expected output:**
```
============================= test session starts ==============================
...
tests/test_sparse.py::TestDreluActivation::test_drelu_basic PASSED       [  5%]
...
============================== 20 passed in 0.18s ==============================
```

---

## Step 6: Use VibeBlade

### 6A. Quick Benchmark

```bash
python -m vibeblade bench --quick
```

### 6B. Run Inference (with MoE memory tiering)

```bash
# Basic inference (requires a GGUF model file)
python -m vibeblade run path/to/model.gguf --prompt "Your prompt here"
```

**With memory options:**
```bash
# Run with 16GB VRAM, 128GB RAM
python -m vibeblade run model.gguf --vram 16 --ram 128 --prompt "Hello"

# Enable SSD tier for systems with only 32GB RAM
python -m vibeblade run model.gguf --vram 4 --ram 32 --allow-ssd --prompt "Hello"
```

### 6C. Start OpenAI-Compatible API Server

```bash
# Start server with a GGUF model
python -m vibeblade serve --model path/to/model.gguf --port 8000

# With optimizations enabled
python -m vibeblade serve --model model.gguf --sparse --paged-attn --minicache
```

**API endpoint:** `http://localhost:8000/v1/chat/completions`

**cURL example:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

### 6D. Python API Usage

```python
from vibeblade import VibeBladeModel

# Load a GGUF model
model = VibeBladeModel("path/to/model.gguf")

# Enable stack layers
model.enable_paged_attention(num_pages=256)
model.enable_kivi(max_seq_len=4096)
model.enable_eagle(max_draft_tokens=5)
model.enable_batching(max_batch_size=32)

# Generate text
text = model.generate("The future of AI is", max_tokens=128)
print(text)
```

---

## MoE-Specific Commands

### Profile Expert Activations

```bash
# Run profiling to generate hot/cold map
python -m vibeblade run minimax-m2.7.gguf --profile-experts --prompts calibration.txt
```

### Run with Memory Tiering

```bash
# RAM_ONLY mode (128GB+ RAM recommended)
python -m vibeblade run model.gguf --vram 4 --ram 128

# HYBRID_SSD mode (32GB RAM + NVMe SSD)
python -m vibeblade run model.gguf --vram 4 --ram 32 --allow-ssd
```

### Custom Configuration

```bash
# Use a config file
python -m vibeblade run model.gguf --config vibeblade.yaml
```

---

## Troubleshooting

### Issue: "externally-managed-environment" error

**Fix:** Use a virtual environment (Step 3)

### Issue: ModuleNotFoundError: No module named 'vibeblade'

**Fix:** 
```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
pip install -e .
```

### Issue: Tests fail

**Fix:** Install test dependencies
```bash
pip install -e ".[testing]"
pytest tests/ -v
```

### Issue: Slow inference

**Fix:** Install ONNX Runtime
```bash
pip install -e ".[onnx]"
```

---

## Performance Tips

| Hardware | Recommended Mode | Expected Throughput |
|----------|-----------------|---------------------|
| 16GB VRAM + 256GB RAM | RAM_ONLY | 8–14 t/s |
| 16GB VRAM + 128GB RAM | RAM_ONLY | 6–10 t/s |
| 16GB VRAM + 32GB RAM + NVMe | HYBRID_SSD | 2–4 t/s |

---

## Files Generated

- `venv/` — Virtual environment
- `vibeblade/` — Source code
- `tests/` — Test suite
- `cpp/` — C++ backend (optional)

---

## Uninstall

```bash
# Deactivate and remove venv
deactivate
rm -rf venv

# Or on Windows (PowerShell)
rm -Recurse -Force venv
