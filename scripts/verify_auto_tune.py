#!/usr/bin/env python3
"""Verify auto_tune picks the right profile and applies it on each model."""
import sys, os, ctypes, gc
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

from vibeblade.llama_backend import _lib, _helper, LlamaCppBackend
from vibeblade.auto_tune import apply_auto_tune, get_model_info

_helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_helper.override_model_params(1, 0, 0)

_lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool]

MODELS = {
    "TinyLlama-1.1B": "models/tinyllama-1.1b-q4km.gguf",
    "Llama-3.2-1B":   "models/llama-3.2-1b-instruct-Q4_K_S.gguf",
    "Phi-2-2.7B":     "models/phi-2.Q4_K_M.gguf",
}

PROMPT = "The quick brown fox jumps over the lazy dog. In the year 2025, artificial intelligence"

for name, path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Load without optimizations for baseline
    # Disable PI+TS for baseline
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
    gc.collect()

    b = LlamaCppBackend()
    b.load(path, n_ctx=256, n_threads=4)
    b._set_sampler(temperature=0.0)
    out_base = b.generate(PROMPT, max_tokens=32, temperature=0.0)
    base_tps = out_base.tokens_per_second
    b.free()
    gc.collect()

    # Load and auto-tune
    b2 = LlamaCppBackend()
    b2.load(path, n_ctx=256, n_threads=4)
    
    info = get_model_info(b2)
    print(f"  Model: {info['name']}")
    print(f"  Arch: {info['arch']} (class: {info['arch_class']})")
    print(f"  n_embd={info['n_embd']}, n_layer={info['n_layer']}, ~{info['est_params_B']}B params")
    print(f"  Bucket: {info['bucket']}")

    profile = apply_auto_tune(b2)
    print(f"  Auto-tune: PI={profile.pi_budget}, TS={profile.ts_threshold}")
    print(f"  Expected: {profile.expected_speedup:.2f}x ({profile.notes})")

    b2._set_sampler(temperature=0.0)
    out_tuned = b2.generate(PROMPT, max_tokens=32, temperature=0.0)
    tuned_tps = out_tuned.tokens_per_second
    b2.free()
    gc.collect()

    actual_speedup = tuned_tps / base_tps
    print(f"\n  Baseline: {base_tps:.2f} t/s")
    print(f"  Auto-tuned: {tuned_tps:.2f} t/s")
    print(f"  Actual speedup: {actual_speedup:.2f}x")

print("\nDone.")
