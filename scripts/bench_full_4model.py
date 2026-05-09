#!/usr/bin/env python3
"""Full 4-model benchmark: 6-config sweep + auto-tune.

Runs configs sequentially within each model. Uses os.execv for process
isolation between models (cleanest C library state reset).
"""
import sys, os, time, json, ctypes

sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

_lib = ctypes.CDLL('/home/ubuntu/VibeBlade/cpp/build/libllama.so')
_lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
_lib.turbosparse_is_enabled.restype = ctypes.c_bool
_lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
_lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool]
_lib.powerinfer_set_hot_budget.argtypes = [ctypes.c_float]
_lib.powerinfer_is_enabled.restype = ctypes.c_bool

from vibeblade.llama_backend import _helper, LlamaCppBackend
from vibeblade.speculative import SpeculativeBackend
from vibeblade.auto_tune import auto_tune, disable_all

_helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

CTX = 256
THREADS = 4
MAX_TOKENS = 32
PROMPT = "The quick brown fox jumps over the lazy dog. In 2025, AI"
BENCH_RUNS = 3

def reset_all():
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
    _lib.powerinfer_set_hot_budget(ctypes.c_float(0.1))
    _helper.override_model_params(1, 0, 0)
    disable_all()

def bench(model_path, setup_fn, gen_fn, runs=BENCH_RUNS):
    """Run setup+gen N times, return median t/s."""
    results = []
    for i in range(runs):
        try:
            setup_fn()
            out = gen_fn()
            results.append(out.tokens_per_second)
        except Exception as e:
            print(f"    ERROR run {i}: {e}")
            results.append(0)
    if not results or all(r == 0 for r in results):
        return None
    results.sort()
    return results[len(results)//2]

def run_model(model_name, model_path):
    print(f"\n{'='*60}")
    print(f"  {model_name} ({os.path.basename(model_path)})")
    print(f"{'='*60}")
    
    results = {}
    baseline_tps = None
    
    # Config 1: Baseline
    def setup():
        reset_all()
        setup.b = LlamaCppBackend()
        setup.b.load(model_path, n_ctx=CTX, n_threads=THREADS)
    def gen():
        return setup.b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    tps = bench(model_path, setup, gen)
    if tps:
        baseline_tps = tps
        results['Baseline'] = tps
        print(f"  Baseline       : {tps:8.2f} t/s  (1.00×)")
    
    # Config 2: TurboSparse
    def setup():
        reset_all()
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(ctypes.c_float(0.05))
        setup.b = LlamaCppBackend()
        setup.b.load(model_path, n_ctx=CTX, n_threads=THREADS)
    tps = bench(model_path, setup, gen)
    if tps and baseline_tps:
        results['TurboSparse'] = tps
        print(f"  TurboSparse    : {tps:8.2f} t/s  ({tps/baseline_tps:.2f}×)")
    
    # Config 3: Speculative
    def setup():
        reset_all()
        setup.s = SpeculativeBackend(draft_n=4, draft_max=5)
        setup.s.load(model_path, n_ctx=CTX, n_threads=THREADS)
    def gen():
        return setup.s.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
    tps = bench(model_path, setup, gen)
    if tps and baseline_tps:
        results['Speculative'] = tps
        print(f"  Speculative    : {tps:8.2f} t/s  ({tps/baseline_tps:.2f}×)")
    
    # Config 4: PowerInfer
    def setup():
        reset_all()
        _lib.powerinfer_set_enabled(True)
        _lib.powerinfer_set_hot_budget(ctypes.c_float(0.15))
        setup.b = LlamaCppBackend()
        setup.b.load(model_path, n_ctx=CTX, n_threads=THREADS)
    def gen():
        return setup.b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    tps = bench(model_path, setup, gen)
    if tps and baseline_tps:
        results['PowerInfer'] = tps
        print(f"  PowerInfer     : {tps:8.2f} t/s  ({tps/baseline_tps:.2f}×)")
    
    # Config 5: PI+TS
    def setup():
        reset_all()
        _lib.powerinfer_set_enabled(True)
        _lib.powerinfer_set_hot_budget(ctypes.c_float(0.15))
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(ctypes.c_float(0.05))
        setup.b = LlamaCppBackend()
        setup.b.load(model_path, n_ctx=CTX, n_threads=THREADS)
    tps = bench(model_path, setup, gen)
    if tps and baseline_tps:
        results['PI+TS'] = tps
        print(f"  PI+TS          : {tps:8.2f} t/s  ({tps/baseline_tps:.2f}×)")
    
    # Config 6: Auto-Tune
    def setup():
        reset_all()
        setup.b = LlamaCppBackend()
        setup.b.load(model_path, n_ctx=CTX, n_threads=THREADS, auto_tune=True)
    def gen():
        return setup.b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    tps = bench(model_path, setup, gen)
    if tps and baseline_tps:
        results['Auto-Tune'] = tps
        print(f"  Auto-Tune      : {tps:8.2f} t/s  ({tps/baseline_tps:.2f}×)")
    
    return results

# ── Main ──
MODELS = [
    ('TinyLlama-1.1B', 'models/tinyllama-1.1b-q4km.gguf'),
    ('Llama-3.2-1B',   'models/llama-3.2-1b-q4ks.gguf'),
    ('Phi-2-2.7B',     'models/phi-2-2.7b-q4km.gguf'),
    ('Qwen2.5-MoE-2x1.5B', 'models/qwen25-moe-2x1.5b-q4km.gguf'),
]

print("VibeBlade Full 4-Model Benchmark")
print(f"Config: {CTX} ctx, {THREADS} threads, {MAX_TOKENS} tokens, median of {BENCH_RUNS} runs")
print(f"Prompt: \"{PROMPT[:50]}...\"")

all_results = {}
for name, path in MODELS:
    full_path = os.path.join('/home/ubuntu/VibeBlade', path)
    if os.path.exists(full_path):
        all_results[name] = run_model(name, full_path)
    else:
        print(f"\n⚠️  {name}: not found at {path}, skipping")

# Summary table
print(f"\n{'='*75}")
print(f"  SUMMARY (speedup vs baseline)")
print(f"{'='*75}")
print(f"{'Model':<22} {'Base t/s':>8} {'TS':>6} {'Spec':>6} {'PI':>6} {'PI+TS':>6} {'AutoT':>6}")
print(f"{'-'*75}")
for name, data in all_results.items():
    base = data.get('Baseline', 0)
    def sx(k): 
        v = data.get(k, 0)
        return f"{v/base:.2f}×" if base and v else "  -  "
    print(f"{name:<22} {base:>7.1f}  {sx('TurboSparse'):>6} {sx('Speculative'):>6} {sx('PowerInfer'):>6} {sx('PI+TS'):>6} {sx('Auto-Tune'):>6}")
print(f"\n{'-'*75}")
print("Done.")
