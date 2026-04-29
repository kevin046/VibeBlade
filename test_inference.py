#!/usr/bin/env python3
"""End-to-end inference test for VibeBlade with MoE model."""
import sys
import time
import numpy as np

MODEL_PATH = "models/qwen3.5-moe-0.87B-d0.8B.Q4_K_S.gguf"

# ── Step 1: Load model ──
print("=" * 60)
print("Step 1: Loading GGUF model...")
print("=" * 60)
t0 = time.time()

from vibeblade.loader import load_model

result = load_model(MODEL_PATH, lazy=True)
meta = result["metadata"]
weights = result["tensors"]
config = result["config"]

t1 = time.time()
print(f"Loaded in {t1 - t0:.1f}s")

# ── Step 2: Inspect model architecture ──
print("\n" + "=" * 60)
print("Step 2: Model architecture")
print("=" * 60)
arch = meta.get("general.architecture", "unknown")
n_layers = config.get("n_layer", config.get("block_count", "?"))
hidden_dim = config.get("hidden_dim", config.get("embedding_length", "?"))
n_heads = config.get("n_heads", config.get("attention.head_count", "?"))
vocab = config.get("vocab_size", "?")
print(f"Architecture: {arch}")
print(f"Layers: {n_layers}, Hidden: {hidden_dim}, Heads: {n_heads}, Vocab: {vocab}")

# Check tensor names (lazy loader)
try:
    if hasattr(weights, '_loader'):
        ti = weights._loader.tensor_infos
        tensor_names = [t.name for t in ti] if isinstance(ti, list) else list(ti.keys())
    elif isinstance(weights, dict):
        tensor_names = list(weights.keys())
    else:
        tensor_names = []
except Exception:
    tensor_names = []

print(f"\nTotal tensors: {len(tensor_names)}")

# Show MoE-specific tensors
moe_found = [k for k in tensor_names if 'gate_inp' in k or 'gate_exps' in k or 'up_exps' in k or 'down_exps' in k]
print(f"MoE tensors found: {len(moe_found)}")
for k in moe_found[:6]:
    print(f"  {k}")

# ── Step 3: Initialize VibeBladeModel ──
print("\n" + "=" * 60)
print("Step 3: Initialize VibeBladeModel")
print("=" * 60)

from vibeblade import VibeBladeModel

model = VibeBladeModel(MODEL_PATH)
print(f"Model initialized: is_moe={model.is_moe}")
if hasattr(model, 'config'):
    print(f"Config: {model.config}")

# ── Step 4: Run inference ──
print("\n" + "=" * 60)
print("Step 4: Running inference...")
print("=" * 60)

prompt = "Hello, my name is"
print(f'Prompt: "{prompt}"')

t2 = time.time()
try:
    response, tps = model.generate(
        prompt=prompt,
        max_tokens=30,
        temperature=0.7,
        stream=False,
    )
    t3 = time.time()
    print(f'\nResponse: "{response}"')
    print(f"Time: {t3 - t2:.2f}s, Speed: {tps:.2f} t/s")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# ── Step 5: Second prompt (tests KV cache) ──
print("\n" + "=" * 60)
print("Step 5: Second prompt (KV cache test)")
print("=" * 60)

prompt2 = "What is 2+2?"
print(f'Prompt: "{prompt2}"')

t4 = time.time()
try:
    response2, tps2 = model.generate(
        prompt=prompt2,
        max_tokens=30,
        temperature=0.7,
        stream=False,
    )
    t5 = time.time()
    print(f'\nResponse: "{response2}"')
    print(f"Time: {t5 - t4:.2f}s, Speed: {tps2:.2f} t/s")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("INFERENCE TEST COMPLETE")
print("=" * 60)
