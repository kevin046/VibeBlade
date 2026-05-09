#!/usr/bin/env python3
"""MoE model benchmark: 6-config sweep + grid search + auto-tune verification.

Runs one config at a time with full model reload for clean isolation.
"""
import sys, os, ctypes, gc, time

sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

_lib = ctypes.CDLL('/home/ubuntu/VibeBlade/cpp/build/libllama.so')
_lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
_lib.turbosparse_is_enabled.restype = ctypes.c_bool
_lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
_lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool]
_lib.powerinfer_set_hot_budget.argtypes = [ctypes.c_float]
_lib.powerinfer_is_enabled.restype = ctypes.c_bool
_lib.rotorquant_set_enabled.argtypes = [ctypes.c_bool]

from vibeblade.llama_backend import _helper, LlamaCppBackend
from vibeblade.speculative import SpeculativeBackend

_helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_helper.override_model_params(1, 0, 0)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/VibeBlade/models/qwen25-moe-2x1.5b-q4km.gguf"
PROMPT = "The quick brown fox jumps over the lazy dog. In the year 2025, artificial intelligence"
MAX_TOKENS = 32
N_THREADS = 4

def disable_all():
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
    _lib.rotorquant_set_enabled(False)

def bench_config(name, setup_fn):
    """Run one config: disable all, setup, load, generate, cleanup."""
    disable_all()
    gc.collect()
    setup_fn()
    b = LlamaCppBackend()
    b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    t0 = time.time()
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    elapsed = time.time() - t0
    tps = len(out.tokens) / elapsed if elapsed > 0 else 0
    b.free()
    gc.collect()
    return len(out.tokens), tps

def bench_spec(name, ts=False):
    disable_all()
    gc.collect()
    spec = SpeculativeBackend(draft_n=4, draft_max=5)
    spec.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    if ts:
        spec.set_turbosparse(True, threshold=0.05)
    out = spec.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
    st = spec.spec_stats
    tps = out.tokens_per_second
    spec.free()
    gc.collect()
    return len(out.tokens), tps, st

# ── Part 1: 6-Config Baseline ──────────────────────────────────────────
print("=" * 70)
print(f"MoE Benchmark: {os.path.basename(MODEL)}")
print(f"  ctx=256, threads={N_THREADS}, max_tokens={MAX_TOKENS}")
print("=" * 70)

print("\n[1/3] 6-Config Baseline Sweep")
print("-" * 70)

results = {}

# Baseline
tok, tps = bench_config("Baseline", lambda: None)
results['Baseline'] = (tok, tps)
print(f"  Baseline       {tok:>4} tok  {tps:>8.2f} t/s")

# TurboSparse
tok, tps = bench_config("TurboSparse", lambda: (
    _lib.turbosparse_set_enabled(True),
    _lib.turbosparse_set_threshold(ctypes.c_float(0.05))
))
results['TurboSparse'] = (tok, tps)
print(f"  TurboSparse    {tok:>4} tok  {tps:>8.2f} t/s  ({tps/results['Baseline'][1]:.2f}x)")

# Speculative
tok, tps, st = bench_spec("Speculative", ts=False)
results['Speculative'] = (tok, tps)
print(f"  Speculative    {tok:>4} tok  {tps:>8.2f} t/s  ({tps/results['Baseline'][1]:.2f}x)  accept={st.acceptance_rate:.0%}")

# Spec+TS
tok, tps, st = bench_spec("Spec+TS", ts=True)
results['Spec+TS'] = (tok, tps)
print(f"  Spec+TS        {tok:>4} tok  {tps:>8.2f} t/s  ({tps/results['Baseline'][1]:.2f}x)  accept={st.acceptance_rate:.0%}")

# PowerInfer
tok, tps = bench_config("PowerInfer", lambda: (
    _lib.powerinfer_set_enabled(True),
    _lib.powerinfer_set_hot_budget(ctypes.c_float(0.1))
))
results['PowerInfer'] = (tok, tps)
print(f"  PowerInfer     {tok:>4} tok  {tps:>8.2f} t/s  ({tps/results['Baseline'][1]:.2f}x)")

# PI+TS
tok, tps = bench_config("PI+TS", lambda: (
    _lib.powerinfer_set_enabled(True),
    _lib.powerinfer_set_hot_budget(ctypes.c_float(0.1)),
    _lib.turbosparse_set_enabled(True),
    _lib.turbosparse_set_threshold(ctypes.c_float(0.05))
))
results['PI+TS'] = (tok, tps)
print(f"  PI+TS          {tok:>4} tok  {tps:>8.2f} t/s  ({tps/results['Baseline'][1]:.2f}x)")

base_tps = results['Baseline'][1]

# ── Part 2: PI×TS Grid Search ─────────────────────────────────────────
print(f"\n[2/3] PI×TS Grid Search (PI budget × TS threshold)")
print("-" * 70)

PI_GRID = [0.03, 0.05, 0.10, 0.15, 0.20, 0.25]
TS_GRID = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

best_tps = 0
best_pi = 0
best_ts = 0
grid_results = []

for pi in PI_GRID:
    row = []
    for ts in TS_GRID:
        disable_all()
        gc.collect()
        _lib.powerinfer_set_enabled(True)
        _lib.powerinfer_set_hot_budget(ctypes.c_float(pi))
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(ctypes.c_float(ts))
        b = LlamaCppBackend()
        b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
        b._set_sampler(temperature=0.0)
        out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
        tps = out.tokens_per_second
        b.free()
        gc.collect()
        row.append(tps)
        if tps > best_tps:
            best_tps = tps
            best_pi = pi
            best_ts = ts
    grid_results.append((pi, row))
    bar = " ".join(f"{t:>7.2f}" for t in row)
    print(f"  PI={pi:.2f}: {bar}")

print(f"\n  Best: PI={best_pi:.2f}, TS={best_ts:.2f} → {best_tps:.2f} t/s ({best_tps/base_tps:.2f}x)")

# ── Part 3: Auto-Tune Verification ────────────────────────────────────
print(f"\n[3/3] Auto-Tune Verification")
print("-" * 70)

from vibeblade.auto_tune import auto_tune, estimate_params_from_file, get_profile

n_est = estimate_params_from_file(MODEL)
profile = get_profile(n_est)
print(f"  Estimated params: {n_est:.2f}B")
print(f"  Profile selected: PI={profile.pi_budget}, TS={profile.ts_threshold} ({profile.notes})")

# Baseline (no optimizations)
disable_all()
gc.collect()
b = LlamaCppBackend()
b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
b._set_sampler(temperature=0.0)
out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
base2_tps = out.tokens_per_second
b.free()
gc.collect()

# Auto-tuned
p = auto_tune(MODEL)
b = LlamaCppBackend()
b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
b._set_sampler(temperature=0.0)
out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
auto_tps = out.tokens_per_second
b.free()
gc.collect()

print(f"  Baseline:   {base2_tps:.2f} t/s")
print(f"  Auto-tuned: {auto_tps:.2f} t/s ({auto_tps/base2_tps:.2f}x)")

# ── Summary ────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  Model: {os.path.basename(MODEL)} (~{n_est:.2f}B estimated, qwen2moe arch)")
print(f"  Best 6-config:  {max(r[1] for r in results.values()):.2f} t/s")
print(f"  Best grid:      PI={best_pi}, TS={best_ts} → {best_tps:.2f} t/s ({best_tps/base_tps:.2f}x)")
print(f"  Auto-tuned:     {auto_tps:.2f} t/s ({auto_tps/base2_tps:.2f}x)")
print(f"{'=' * 70}")
