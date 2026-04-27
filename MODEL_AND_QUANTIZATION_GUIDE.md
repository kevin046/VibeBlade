# VibeBlade — Model Selection, Quantization & HuggingFace Integration

---

## 1. How to Select Which LLM

VibeBlade supports two model sources:

### A. HuggingFace Hub Models (auto-download)

```bash
# Any HuggingFace model ID
python -m vibeblade run meta-llama/Llama-3.1-8B-Instruct --prompt "Hello"
python -m vibeblade run mistralai/Mistral-7B-Instruct-v0.3 --prompt "Hello"
python -m vibeblade run Qwen/Qwen2.5-14B-Instruct --prompt "Hello"
python -m vibeblade run deepseek-ai/DeepSeek-V2.5 --prompt "Hello"
```

VibeBlade will auto-detect MoE models (Mixtral, DeepSeek, etc.) and enable hot/cold expert splitting.

### B. Local GGUF Files (your own)

```bash
# Any GGUF file from llama.cpp, Ollama, etc.
python -m vibeblade run /path/to/model-q4_0.gguf --prompt "Hello"
```

### C. With Config File

```bash
python -m vibeblade run meta-llama/Llama-3.1-8B-Instruct --config vibeblade.yaml
```

`vibeblade.yaml` example:
```yaml
model: meta-llama/Llama-3.1-8B-Instruct
vram_gb: 16
ram_gb: 128
hot_threshold: 0.15
offload_mode: RAM_ONLY
cpu_threads: 16
```

---

## 2. Quantization Methods (rotorquant / GGUF)

VibeBlade natively supports **GGUF format** — the standard quantization format from llama.cpp.

### Quantization Levels Explained

| Format  | Size vs FP16 | Quality | Use Case |
|---------|-------------|---------|----------|
| Q8_0    | ~60%        | ★★★★★  | Near-lossless, when RAM is plenty |
| Q6_K    | ~50%        | ★★★★☆  | Good balance for mid-range |
| Q5_K_M  | ~43%        | ★★★★☆  | Popular choice |
| Q4_0    | ~29%        | ★★★☆☆  | Good quality, small |
| Q4_K_M  | ~38%        | ★★★★☆  | **Most popular** — best quality/size |
| Q3_K_M  | ~27%        | ★★★☆☆  | Smaller, slight quality loss |
| Q2_K    | ~20%        | ★★☆☆☆  | Very small, notable quality loss |
| IQ4_XS  | ~24%        | ★★★☆☆  | Improved 4-bit (better than Q4_K_M per bit) |
| IQ2_XXS | ~13%        | ★☆☆☆☆  | Ultra-small, significant loss |

### How to Quantize with llama.cpp (rotorquant equivalent)

```bash
# Step 1: Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. && cmake --build . --config Release

# Step 2: Download base model (FP16 / BF16)
# From HuggingFace:
huggingface-cli download meta-llama/Llama-3.1-8B-Flash下水   --local-dir ./models/llama-3.1-8b-fp16   --local-dir-use-symlinks False

# Step 3: Quantize to Q4_K_M (most recommended)
./build/bin/llama-quantize   ./models/llama-3.1-8b-fp16/consolidated.00.pt   ./models/llama-3.1-8b-q4_k_m.gguf   Q4_K_M

# Step 4: Use with VibeBlade
python -m vibeblade run ./models/llama-3.1-8b-q4_k_m.gguf --prompt "Hello"
```

### VibeBlade's Built-in RotorQuant (`rotor_unpack`)

VibeBlade has a C++ backend (`rotor_unpack`) that does fast dequantization of 4-bit weights:

```python
from vibeblade import rotor_unpack

# Fast 4-bit dequantization (C++ backend)
# block: (scale: f16, nibbles: uint8[16]) → float32[32]
dequantized = rotor_unpack(block_4bit)
```

This bypasses slow Python numpy for GGUF weight dequantization — **critical for inference speed**.

---

## 3. HuggingFace Quantized Models (Ready-to-Use)

Many HF repos already host GGUF-quantized models — no quantization step needed.

### Popular Pre-Quantized Models on HuggingFace

```bash
# TheBloke / Disketron repos — GGUF format, ready to download
# Just append ?multiple to list files, then download the .gguf file

# Llama 3.1 8B
huggingface-cli download TheBloke/Llama-3.1-8B-Instruct-GGUF   llama-3.1-8b-instruct-q4_k_m.gguf

# Mistral 7B
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.3-GGUF   mistral-7b-instruct-v0.3-q4_k_m.gguf

# Mixtral 8x7B MoE
huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF   mixtral-8x7b-instruct-v0.1-q4_k_m.gguf

# Qwen 2.5 14B
huggingface-cli download TheBloke/Qwen2.5-14B-Instruct-GGUF   qwen2.5-14b-instruct-q4_k_m.gguf

# DeepSeek V2.5 (MoE)
huggingface-cli download cf Mello/DeepSeek-V2.5-GGUF   deepseek-v2.5-q4_k_m.gguf
```

Then run directly:
```bash
python -m vibeblade run ./mixtral-8x7b-instruct-q4_k_m.gguf   --vram 8 --ram 64 --prompt "Explain quantum entanglement"
```

### VibeBlade + HuggingFace Token

If accessing gated models (Llama, Mistral):

```python
# Option A: Environment variable
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Option B: Pass token programmatically
from huggingface_hub import HfApi
api = HfApi()
api.login(token="hf_xxxxxxxxxxxxxxxxxxxx")
```

---

## 4. Recommended Models by Hardware

| Hardware              | Recommended Model              | Format | Size  |
|-----------------------|-------------------------------|--------|-------|
| 16GB RAM, no GPU      | Mistral-7B-Instruct           | Q4_K_M | ~4GB  |
| 32GB RAM, no GPU      | Llama-3.1-8B-Instruct        | Q4_K_M | ~5GB  |
| 64GB RAM, no GPU      | Qwen2.5-14B-Instruct         | Q4_K_M | ~9GB  |
| 32GB RAM + 8GB VRAM  | Mixtral-8x7B-Instruct        | Q4_K_M | ~26GB |
| 128GB RAM + 16GB VRAM | DeepSeek-V2.5 or Qwen2.5-32B | Q4_K_M | ~52GB |
| 256GB RAM + 16GB VRAM | DeepSeek-V2.5 + full MoE     | Q6_K   | ~80GB |

---

## 5. Quick Start with HuggingFace Model

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Run any model directly (VibeBlade auto-downloads)
python -m vibeblade run   "mistralai/Mistral-7B-Instruct-v0.3"   --prompt "What is retrieval-augmented generation?"

# 3. With memory tiering (recommended for MoE)
python -m vibeblade run   "mistralai/Mixtral-8x7B-Instruct-v0.1"   --vram 8 --ram 64 --allow-ssd   --prompt "Write a Python function to reverse a linked list"

# 4. Start API server
python -m vibeblade serve   --model "meta-llama/Llama-3.1-8B-Instruct"   --port 8000
```

---

## 6. VibeBlade Config Generator

```python
# generate_config.py — run once to generate your vibeblade.yaml
import yaml, os

ram_gb  = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024**3)
config  = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "vram_gb": 0,
    "ram_gb": ram_gb,
    "hot_threshold": 0.15,
    "offload_mode": "RAM_ONLY",
    "cpu_threads": os.cpu_count() or 8,
}

with open("vibeblade.yaml", "w") as f:
    yaml.dump(config, f)
print("vibeblade.yaml created")
```
