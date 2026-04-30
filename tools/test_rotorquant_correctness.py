#!/usr/bin/env python3
"""Test: RotorQuant correctness — rotated-weights model with runtime H4 
should produce identical output to baseline model.

Model A (baseline): qwen2.5-0.5b-instruct-Q4_K_S.gguf, no RotorQuant
Model B (RQ):       qwen2.5-0.5b-instruct-Q4_K_S-rq.gguf, RotorQuant ON

Both should produce identical text since (W·H4/2)·(H4/2·x) = W·x in exact arithmetic.
Quantization introduces rounding, so we expect NEAR-identical (not bit-exact).
"""

import sys, os
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

from vibeblade.llama_backend import LlamaCppBackend

PROMPT = "The capital of France is"
N_GEN = 32
MODEL_DIR = "/home/ubuntu/VibeBlade/models"

print("=" * 60)
print("TEST: RotorQuant Correctness")
print("=" * 60)

# Baseline: standard Q4_K, no RotorQuant
print("\n[1] Baseline: standard Q4_K_S, RotorQuant OFF")
backend_a = LlamaCppBackend()
backend_a.load(os.path.join(MODEL_DIR, "qwen2.5-0.5b-instruct-Q4_K_S.gguf"))
backend_a.set_turbosparse(False)
backend_a.set_rotorquant(False)
backend_a.set_powerinfer(False)
out_a = backend_a.generate(PROMPT, max_tokens=N_GEN, temperature=0.0)
print(f"  Text: {out_a.text!r}")
print(f"  Speed: {out_a.tokens_per_second:.3f} t/s")
print(f"  Tokens: {out_a.tokens}")
del backend_a

# RQ: rotated Q4_K, RotorQuant ON (H4/2 on activations at runtime)
print("\n[2] RQ: rotated Q4_K_S, RotorQuant ON")
backend_b = LlamaCppBackend()
backend_b.load(os.path.join(MODEL_DIR, "qwen2.5-0.5b-instruct-Q4_K_S-rq.gguf"))
backend_b.set_turbosparse(False)
backend_b.set_rotorquant(True)
backend_b.set_powerinfer(False)
out_b = backend_b.generate(PROMPT, max_tokens=N_GEN, temperature=0.0)
print(f"  Text: {out_b.text!r}")
print(f"  Speed: {out_b.tokens_per_second:.3f} t/s")
print(f"  Tokens: {out_b.tokens}")
del backend_b

# Also test: rotated weights WITHOUT RotorQuant (should be DIFFERENT — wrong output)
print("\n[3] Control: rotated Q4_K_S, RotorQuant OFF (should be garbage)")
backend_c = LlamaCppBackend()
backend_c.load(os.path.join(MODEL_DIR, "qwen2.5-0.5b-instruct-Q4_K_S-rq.gguf"))
backend_c.set_turbosparse(False)
backend_c.set_rotorquant(False)
backend_c.set_powerinfer(False)
out_c = backend_c.generate(PROMPT, max_tokens=N_GEN, temperature=0.0)
print(f"  Text: {out_c.text!r}")
print(f"  Speed: {out_c.tokens_per_second:.3f} t/s")
print(f"  Tokens: {out_c.tokens}")
del backend_c

# Compare
print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
match_ab = out_a.text == out_b.text
print(f"  Baseline vs RQ match: {'✓ IDENTICAL' if match_ab else '✗ DIFFERENT'}")
match_ac = out_a.text == out_c.text
print(f"  Baseline vs Control (no RQ): {'✓ IDENTICAL' if match_ac else '✗ DIFFERENT (expected)'}")

if match_ab and not match_ac:
    print("\n  🎉 RotorQuant works correctly!")
    print("  - Rotated weights + runtime H4 = same as baseline")
    print("  - Rotated weights without runtime H4 = different (as expected)")
elif match_ab and match_ac:
    print("\n  ⚠️  Both match baseline — rotation may not be significant at this scale")
else:
    print("\n  ❌ RotorQuant produces different output than baseline!")
    print(f"     Baseline: {out_a.text!r}")
    print(f"     RQ:       {out_b.text!r}")
