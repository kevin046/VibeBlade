#!/usr/bin/env python3
"""Benchmark: PI + TS + Speculative (and variants)."""
import sys, os, time, gc, ctypes
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

# Reset C globals BEFORE importing backend (shared lib gets loaded on import)
_lib = ctypes.CDLL('cpp/build/libllama.so')
_lib.turbosparse_set_enabled(False)
_lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
_lib.turbosparse_set_threshold(0.01)
_lib.powerinfer_set_enabled(False)
_lib.rotorquant_set_enabled(False)

from vibeblade.speculative import SpeculativeBackend
from vibeblade.llama_backend import LlamaCppBackend, _helper

MODEL = "/home/ubuntu/VibeBlade/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
PROMPT = "The capital of France is"
N_TOKENS = 32

def run(label, cls, ts, pi, ts_th, speculative=False, draft_max=4):
    gc.collect()
    # Reset globals for each run
    _lib.turbosparse_set_enabled(False)
    _lib.turbosparse_set_threshold(ts_th)
    _lib.powerinfer_set_enabled(False)
    _lib.rotorquant_set_enabled(False)

    _helper.override_model_params(1, 0, 0)
    if cls == SpeculativeBackend:
        b = cls(draft_max=draft_max)
    else:
        b = cls()
    b.load(MODEL, n_ctx=512, n_threads=4)
    b.set_turbosparse(ts, threshold=ts_th)
    b.set_powerinfer(pi, hot_budget=0.1)

    print(f"[{label}] warmup...", flush=True)
    _ = b.generate(PROMPT, max_tokens=2, temperature=0.0,
                   speculative=(speculative and cls == SpeculativeBackend))
    print(f"[{label}] running...", flush=True)

    t0 = time.time()
    out = b.generate(PROMPT, max_tokens=N_TOKENS, temperature=0.0,
                     speculative=(speculative and cls == SpeculativeBackend))
    elapsed = time.time() - t0

    n_gen = len(out.tokens)
    tps = n_gen / elapsed if elapsed > 0 else 0
    extras = ""
    if cls == SpeculativeBackend and speculative:
        extras = f" accept={b.spec_stats.acceptance_rate:.0%} eff_speedup={b.spec_stats.effective_speedup:.2f}x"
    print(f"[{label}] tok={n_gen} t/s={tps:.3f} dec={out.time_decode:.1f}s{extras}", flush=True)
    del b

run("1-Baseline",         LlamaCppBackend, False, False, 0.01)
run("2-TS-0.01",          LlamaCppBackend, True,  False, 0.01)
run("3-SpecOnly-d4",      SpeculativeBackend, False, False, 0.01, speculative=True, draft_max=4)
run("4-TS+Spec-d4",       SpeculativeBackend, True,  False, 0.01, speculative=True, draft_max=4)
run("5-TS+Spec-d8",       SpeculativeBackend, True,  False, 0.01, speculative=True, draft_max=8)
run("6-PI+TS+Spec-d4",    SpeculativeBackend, True,  True,  0.01, speculative=True, draft_max=4)
