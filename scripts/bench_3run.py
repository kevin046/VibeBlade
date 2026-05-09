#!/usr/bin/env python3
"""Run 6-config benchmark 3 times per model, report mean ± std."""
import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'build')

import ctypes
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'build', 'libllama.so'))
lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool]
lib.powerinfer_set_hot_budget.argtypes = [ctypes.c_float]

from vibeblade.auto_tune import disable_all
from vibeblade.llama_backend import LlamaCppBackend

PROMPT = "The quick brown fox jumps over the lazy dog. In 2025, AI"
MAX_TOKENS = 32
N_CTX = 256
N_THREADS = 4
N_RUNS = 3

CONFIGS = [
    ("Baseline",       False, 0,    False, 0),
    ("TurboSparse",    True,  0.05, False, 0),
    ("PowerInfer",     False, 0,    True,  0.20),
    ("Speculative",    False, 0,    False, 0),
    ("Spec+TS",        True,  0.05, False, 0),
    ("PI+TS",          True,  0.05, True,  0.20),
]


def bench_one(model_path, ts_on, ts_thr, pi_on, pi_budget, spec=False):
    disable_all()
    lib.turbosparse_set_enabled(ts_on)
    if ts_on:
        lib.turbosparse_set_threshold(ts_thr)
    lib.powerinfer_set_enabled(pi_on)
    if pi_on:
        lib.powerinfer_set_hot_budget(pi_budget)

    b = LlamaCppBackend()
    b.load(model_path, n_ctx=N_CTX, n_threads=N_THREADS)
    r = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    return r.tokens_per_second, len(r.tokens)


def main():
    if len(sys.argv) < 2:
        print("Usage: bench_3run.py <model.gguf>")
        sys.exit(1)

    model = sys.argv[1]
    model_name = os.path.basename(model).replace('.gguf', '')
    print(f"\n{'='*70}")
    print(f"  {model_name} — 3-run benchmark")
    print(f"  {N_RUNS} runs × {len(CONFIGS)} configs × {MAX_TOKENS} tokens")
    print(f"{'='*70}\n", flush=True)

    all_results = {}

    for cfg_name, ts_on, ts_thr, pi_on, pi_budget in CONFIGS:
        tps_list = []
        toks_list = []

        for run in range(N_RUNS):
            tps, ntok = bench_one(model, ts_on, ts_thr, pi_on, pi_budget)
            tps_list.append(tps)
            toks_list.append(ntok)
            print(f"  {cfg_name:15s} run {run+1}: {tps:8.3f} t/s  ({ntok} tok)", flush=True)

        mean_tps = statistics.mean(tps_list)
        std_tps = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
        min_tps = min(tps_list)
        max_tps = max(tps_list)
        mean_toks = statistics.mean(toks_list)

        all_results[cfg_name] = {
            'mean': mean_tps, 'std': std_tps,
            'min': min_tps, 'max': max_tps,
            'runs': tps_list, 'tokens': mean_toks,
        }
        print(f"  {cfg_name:15s} MEAN: {mean_tps:8.3f} ± {std_tps:.3f}  (range {min_tps:.3f}–{max_tps:.3f})", flush=True)
        print()

    # Summary table
    base = all_results['Baseline']['mean']
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"  {'Config':15s} {'Mean t/s':>10s} {'± Std':>8s} {'Speedup':>8s} {'Range':>20s}")
    print(f"  {'-'*65}")
    for cfg_name in [c[0] for c in CONFIGS]:
        r = all_results[cfg_name]
        sp = r['mean'] / base if base > 0 else 0
        rng = f"{r['min']:.3f}–{r['max']:.3f}"
        print(f"  {cfg_name:15s} {r['mean']:10.3f} {r['std']:8.3f} {sp:7.2f}× {rng:>20s}")

    # Best config
    best_cfg = max(all_results, key=lambda k: all_results[k]['mean'])
    best = all_results[best_cfg]
    print(f"\n  🏆 Best: {best_cfg} — {best['mean']:.3f} t/s ({best['mean']/base:.2f}× vs baseline)")
    print(f"  Coefficient of variation:", flush=True)
    for cfg_name in [c[0] for c in CONFIGS]:
        r = all_results[cfg_name]
        cv = (r['std'] / r['mean'] * 100) if r['mean'] > 0 else 0
        print(f"    {cfg_name:15s} CV = {cv:.1f}%")

    print(f"\nDone.", flush=True)


if __name__ == '__main__':
    main()
