#!/usr/bin/env python3
"""Full 4-model benchmark: 6-config sweep + auto-tune verification.

Runs sequentially (one model at a time) to avoid cross-contamination.
Each model gets its own subprocess for clean C library state.
"""
import subprocess, sys, os, json, time

sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

MODELS = {
    'TinyLlama-1.1B': 'models/tinyllama-1.1b-q4km.gguf',
    'Llama-3.2-1B':   'models/llama-3.2-1b-q4ks.gguf',
    'Phi-2-2.7B':     'models/phi-2-2.7b-q4km.gguf',
    'Qwen2.5-MoE-2x1.5B': 'models/qwen25-moe-2x1.5b-q4km.gguf',
}

BENCH_SCRIPT = '''
import sys, os, ctypes, time, gc
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

import ctypes
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

MODEL = sys.argv[1]
CTX = 256
THREADS = 4
MAX_TOKENS = 32
PROMPT = "The quick brown fox jumps over the lazy dog. In 2025, AI"
WARMUP_RUNS = 1
BENCH_RUNS = 3

_helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

def reset_all():
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
    _lib.powerinfer_set_hot_budget(ctypes.c_float(0.1))
    _helper.override_model_params(1, 0, 0)

def bench_config(name, setup_fn, gen_fn):
    """Run a config with warmup + N bench runs, return median t/s."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        setup_fn()
        try:
            gen_fn()
        except Exception as e:
            print(f"  {name}: WARMUP ERROR: {e}")
            return None
    
    results = []
    for i in range(BENCH_RUNS):
        setup_fn()
        try:
            out = gen_fn()
            results.append(out.tokens_per_second)
        except Exception as e:
            print(f"  {name}: RUN {i} ERROR: {e}")
            return None
    
    results.sort()
    median = results[len(results)//2]
    return median

def make_bench():
    configs = []
    
    # 1. Baseline
    def setup_base():
        reset_all()
        b = LlamaCppBackend()
        b.load(MODEL, n_ctx=CTX, n_threads=THREADS)
        setup_base.backend = b
    def gen_base():
        return setup_base.backend.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    configs.append(("Baseline", setup_base, gen_base))
    
    # 2. TurboSparse
    def setup_ts():
        reset_all()
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(ctypes.c_float(0.05))
        b = LlamaCppBackend()
        b.load(MODEL, n_ctx=CTX, n_threads=THREADS)
        setup_ts.backend = b
    def gen_ts():
        return setup_ts.backend.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    configs.append(("TurboSparse", setup_ts, gen_ts))
    
    # 3. Speculative (n-gram)
    def setup_spec():
        reset_all()
        s = SpeculativeBackend(draft_n=4, draft_max=5)
        s.load(MODEL, n_ctx=CTX, n_threads=THREADS)
        setup_spec.backend = s
    def gen_spec():
        return setup_spec.backend.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
    configs.append(("Speculative", setup_spec, gen_spec))
    
    # 4. PowerInfer
    def setup_pi():
        reset_all()
        _lib.powerinfer_set_enabled(True)
        _lib.powerinfer_set_hot_budget(ctypes.c_float(0.15))
        b = LlamaCppBackend()
        b.load(MODEL, n_ctx=CTX, n_threads=THREADS)
        setup_pi.backend = b
    def gen_pi():
        return setup_pi.backend.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    configs.append(("PowerInfer", setup_pi, gen_pi))
    
    # 5. PI+TS (optimized)
    def setup_pits():
        reset_all()
        _lib.powerinfer_set_enabled(True)
        _lib.powerinfer_set_hot_budget(ctypes.c_float(0.15))
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(ctypes.c_float(0.05))
        b = LlamaCppBackend()
        b.load(MODEL, n_ctx=CTX, n_threads=THREADS)
        setup_pits.backend = b
    def gen_pits():
        return setup_pits.backend.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    configs.append(("PI+TS", setup_pits, gen_pits))
    
    # 6. Auto-Tune
    def setup_at():
        reset_all()
        auto_tune(MODEL)
        b = LlamaCppBackend()
        b.load(MODEL, n_ctx=CTX, n_threads=THREADS, auto_tune=True)
        setup_at.backend = b
    def gen_at():
        return setup_at.backend.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    configs.append(("Auto-Tune", setup_at, gen_at))
    
    return configs

print(f"\\n{'='*60}")
print(f"  {os.path.basename(MODEL)}")
print(f"{'='*60}")

configs = make_bench()
baseline_tps = None
results = {}

for name, setup_fn, gen_fn in configs:
    t0 = time.time()
    tps = bench_config(name, setup_fn, gen_fn)
    elapsed = time.time() - t0
    
    if tps is not None:
        if baseline_tps is None:
            baseline_tps = tps
            speedup = 1.0
        else:
            speedup = tps / baseline_tps
        results[name] = {'tps': tps, 'speedup': speedup}
        print(f"  {name:15s}: {tps:8.2f} t/s  ({speedup:.2f}×)  [{elapsed:.1f}s]")
    else:
        results[name] = {'tps': 0, 'speedup': 0}
        print(f"  {name:15s}: FAILED  [{elapsed:.1f}s]")

# Output JSON for aggregation
print(f"\\n__JSON__{json.dumps(results)}")
'''

# Run each model in its own subprocess
all_results = {}

for model_name, model_path in MODELS.items():
    full_path = os.path.join('/home/ubuntu/VibeBlade', model_path)
    if not os.path.exists(full_path):
        print(f"\n⚠️  {model_name}: model not found at {full_path}, skipping")
        continue
    
    print(f"\n{'#'*60}")
    print(f"# {model_name}")
    print(f"{'#'*60}")
    
    result = subprocess.run(
        [sys.executable, '-c', BENCH_SCRIPT, full_path],
        capture_output=True, text=True, timeout=300,
        cwd='/home/ubuntu/VibeBlade',
        env={**os.environ, 'LD_LIBRARY_PATH': '/home/ubuntu/VibeBlade/cpp/build'}
    )
    
    # Print stdout
    for line in result.stdout.splitlines():
        if line.startswith('__JSON__'):
            json_str = line[8:]
            all_results[model_name] = json.loads(json_str)
        else:
            print(line)
    
    if result.stderr:
        for line in result.stderr.splitlines():
            if 'warning' in line.lower() or 'error' in line.lower():
                print(f"  ⚠ {line}")
    
    if result.returncode != 0:
        print(f"  ❌ Process exited with code {result.returncode}")

# Final summary table
print(f"\n{'='*80}")
print(f"  FULL BENCHMARK SUMMARY")
print(f"{'='*80}")
print(f"{'Model':<22} {'Base':>8} {'TS':>8} {'Spec':>8} {'PI':>8} {'PI+TS':>8} {'AutoTune':>8}")
print(f"{'-'*80}")

for model_name, data in all_results.items():
    base = data.get('Baseline', {}).get('tps', 0)
    ts = data.get('TurboSparse', {}).get('speedup', 0)
    spec = data.get('Speculative', {}).get('speedup', 0)
    pi = data.get('PowerInfer', {}).get('speedup', 0)
    pits = data.get('PI+TS', {}).get('speedup', 0)
    at = data.get('Auto-Tune', {}).get('speedup', 0)
    print(f"{model_name:<22} {base:>7.1f}t/s {ts:>7.2f}× {spec:>7.2f}× {pi:>7.2f}× {pits:>7.2f}× {at:>7.2f}×")

print(f"\nConfig: 256 ctx, 4 threads, 32 tokens, median of 3 runs")
print(f"Hardware: Oracle A1 ARM64 (4 cores, NEON SIMD)")
