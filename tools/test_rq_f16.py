#!/usr/bin/env python3
"""Isolate the RotorQuant bug: test F16 models directly (no quantization)."""

import sys, os
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

from vibeblade.llama_backend import LlamaCppBackend

PROMPT = "The capital of France is"
N_GEN = 32
MODEL_DIR = "/home/ubuntu/VibeBlade/models"

configs = [
    ("F16 baseline",        "qwen2.5-0.5b-instruct-fp16.gguf",      False),
    ("F16 rotated, RQ OFF", "qwen2.5-0.5b-instruct-fp16-rq.gguf",    False),
    ("F16 rotated, RQ ON",  "qwen2.5-0.5b-instruct-fp16-rq.gguf",    True),
]

for label, model, rq in configs:
    print(f"\n[{label}]")
    b = LlamaCppBackend()
    b.load(os.path.join(MODEL_DIR, model))
    b.set_turbosparse(False)
    b.set_rotorquant(rq)
    b.set_powerinfer(False)
    out = b.generate(PROMPT, max_tokens=N_GEN, temperature=0.0)
    print(f"  Text: {out.text!r}")
    print(f"  Speed: {out.tokens_per_second:.3f} t/s")
    del b
