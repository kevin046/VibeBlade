#!/usr/bin/env python3
"""PI+TS threshold optimizer: grid search for best PI hot_budget × TS threshold combo."""
import sys, os, ctypes, gc, time, statistics, itertools

sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

_lib = ctypes.CDLL('/home/ubuntu/VibeBlade/cpp/build/libllama.so')
_lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
_lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
_lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool, ctypes.c_float]
_lib.rotorquant_set_enabled.argtypes = [ctypes.c_bool]

from vibeblade.llama_backend import _helper, LlamaCppBackend
_helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_helper.override_model_params(1, 0, 0)

MODELS = {
    "TinyLlama-1.1B": "/home/ubuntu/VibeBlade/models/tinyllama-1.1b-q4km.gguf",
    "Llama-3.2-1B":   "/home/ubuntu/VibeBlade/models/llama-3.2-1b-instruct-Q4_K_S.gguf",
    "Phi-2-2.7B":     "/home/ubuntu/VibeBlade/models/phi-2.Q4_K_M.gguf",
}

PROMPT = "The quick brown fox jumps over the lazy dog. In the year 2025, artificial intelligence"
MAX_TOKENS = 32  # more tokens = PI EMA warms up better
N_THREADS = 4

PI_BUDGETS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
TS_THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

def disable_all():
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False, 0.1)
    _lib.rotorquant_set_enabled(False)

def bench_baseline(model_path):
    disable_all(); gc.collect()
    b = LlamaCppBackend()
    b.load(model_path, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    tps = out.tokens_per_second
    b.free(); gc.collect()
    return tps

def bench_pi_ts(model_path, pi_budget, ts_threshold):
    disable_all(); gc.collect()
    _lib.powerinfer_set_enabled(True, ctypes.c_float(pi_budget))
    _lib.turbosparse_set_enabled(True)
    _lib.turbosparse_set_threshold(ctypes.c_float(ts_threshold))
    b = LlamaCppBackend()
    b.load(model_path, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    tps = out.tokens_per_second
    b.free(); gc.collect()
    return tps

for model_name, model_path in MODELS.items():
    print(f"\n{'='*70}")
    print(f"  {model_name} — PI+TS Grid Search")
    print(f"{'='*70}")
    
    # Baseline
    base_tps = bench_baseline(model_path)
    print(f"  Baseline: {base_tps:.2f} t/s")
    print(f"\n  {'PI Budget':<10} {'TS Thresh':<10} {'t/s':>8} {'Speedup':>8}")
    print(f"  {'-'*40}")
    
    results = []
    for pi_b in PI_BUDGETS:
        for ts_t in TS_THRESHOLDS:
            tps = bench_pi_ts(model_path, pi_b, ts_t)
            speedup = tps / base_tps
            results.append((pi_b, ts_t, tps, speedup))
            marker = " ★" if speedup > 1.5 else (" ✓" if speedup > 1.0 else "")
            print(f"  {pi_b:<10.2f} {ts_t:<10.2f} {tps:>8.2f} {speedup:>7.2f}x{marker}")
    
    # Find best
    best = max(results, key=lambda x: x[3])
    print(f"\n  BEST: PI={best[0]:.2f}, TS={best[1]:.2f} → {best[2]:.2f} t/s ({best[3]:.2f}x)")
    
    # Also show top 5
    top5 = sorted(results, key=lambda x: x[3], reverse=True)[:5]
    print(f"\n  Top 5:")
    for i, (pi_b, ts_t, tps, sp) in enumerate(top5, 1):
        print(f"    {i}. PI={pi_b:.2f}, TS={ts_t:.2f} → {tps:.2f} t/s ({sp:.2f}x)")

print("\n\nDone.")
