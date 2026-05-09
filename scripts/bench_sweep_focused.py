#!/usr/bin/env python3
"""Focused sweep — find optimal PI budget and TS threshold for a model."""
import sys, os, time, statistics, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ctypes
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'build', 'libllama.so')
lib = ctypes.CDLL(LIB_PATH)
lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool]
lib.powerinfer_set_hot_budget.argtypes = [ctypes.c_float]

from vibeblade.auto_tune import disable_all
from vibeblade.llama_backend import LlamaCppBackend

PROMPT = "The quick brown fox jumps over the lazy dog. In 2025, AI"


def bench(model, max_tokens=32, n_ctx=256, n_threads=4, ts_on=False, ts_thr=0.05, pi_on=False, pi_budget=0.20):
    disable_all()
    lib.turbosparse_set_enabled(ts_on)
    if ts_on: lib.turbosparse_set_threshold(ts_thr)
    lib.powerinfer_set_enabled(pi_on)
    if pi_on: lib.powerinfer_set_hot_budget(pi_budget)
    b = LlamaCppBackend()
    b.load(model, n_ctx=n_ctx, n_threads=n_threads)
    r = b.generate(PROMPT, max_tokens=max_tokens, temperature=0.0)
    return r.tokens_per_second


def main():
    if len(sys.argv) < 2:
        print("Usage: bench_sweep_focused.py <model.gguf> [--phase 1|2|3|4|5]")
        sys.exit(1)

    model = sys.argv[1]
    model_name = os.path.basename(model).replace('.gguf', '')
    phase = 0
    if '--phase' in sys.argv:
        phase = int(sys.argv[sys.argv.index('--phase') + 1])

    # ── Phase 1: TS threshold sweep ──
    if phase == 0 or phase == 1:
        print(f"\n── Phase 1: TS threshold sweep ({model_name}) ──", flush=True)
        best_ts_thr, best_ts_tps = 0, 0
        for thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
            tps = bench(model, ts_on=True, ts_thr=thr, max_tokens=32)
            print(f"  TS {thr:5.3f}: {tps:8.3f} t/s", flush=True)
            if tps > best_ts_tps:
                best_ts_tps = tps
                best_ts_thr = thr
        print(f"  → Best TS: {best_ts_thr} → {best_ts_tps:.3f} t/s", flush=True)

    # ── Phase 2: PI budget sweep ──
    if phase == 0 or phase == 2:
        print(f"\n── Phase 2: PI budget sweep ({model_name}) ──", flush=True)
        best_pi_bud, best_pi_tps = 0, 0
        for bud in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
            tps = bench(model, pi_on=True, pi_budget=bud, max_tokens=32)
            print(f"  PI {bud:4.2f}: {tps:8.3f} t/s", flush=True)
            if tps > best_pi_tps:
                best_pi_tps = tps
                best_pi_bud = bud
        print(f"  → Best PI: {best_pi_bud} → {best_pi_tps:.3f} t/s", flush=True)

    # ── Phase 3: PI+TS combined top candidates ──
    if phase == 0 or phase == 3:
        print(f"\n── Phase 3: PI+TS combined grid ({model_name}) ──", flush=True)
        # Read previous bests if available
        best_combo_tps = 0
        best_combo_cfg = ""
        for bud in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            for thr in [0.001, 0.01, 0.05, 0.10, 0.20]:
                tps = bench(model, ts_on=True, ts_thr=thr, pi_on=True, pi_budget=bud, max_tokens=32)
                cfg = f"PI={bud:.2f}+TS={thr:.3f}"
                print(f"  {cfg:25s}: {tps:8.3f} t/s", flush=True)
                if tps > best_combo_tps:
                    best_combo_tps = tps
                    best_combo_cfg = cfg
        print(f"  → Best PI+TS: {best_combo_cfg} → {best_combo_tps:.3f} t/s", flush=True)

    # ── Phase 4: Thread + ctx sweep with best config ──
    if phase == 0 or phase == 4:
        print(f"\n── Phase 4: Thread sweep ──", flush=True)
        for nt in [1, 2, 3, 4]:
            tps = bench(model, n_threads=nt, max_tokens=32)
            print(f"  {nt} threads: {tps:8.3f} t/s", flush=True)

        print(f"\n── Phase 4b: Context sweep ──", flush=True)
        for ctx in [64, 128, 256, 512]:
            tps = bench(model, n_ctx=ctx, max_tokens=32)
            print(f"  ctx {ctx:4d}: {tps:8.3f} t/s", flush=True)

    # ── Phase 5: Validate best configs with 3 runs ──
    if phase == 0 or phase == 5:
        print(f"\n── Phase 5: 3-run validation ──", flush=True)
        
        configs = [
            ("Baseline", False, 0, False, 0),
            ("Best-TS", True, best_ts_thr, False, 0),
            ("Best-PI", False, 0, True, best_pi_bud),
            ("Best-PI+TS", True, best_ts_thr, True, best_pi_bud),
        ]
        for name, ts_on, ts_thr, pi_on, pi_bud in configs:
            runs = []
            for i in range(3):
                tps = bench(model, ts_on=ts_on, ts_thr=ts_thr, pi_on=pi_on, pi_budget=pi_bud, max_tokens=64)
                runs.append(tps)
            m = statistics.mean(runs)
            s = statistics.stdev(runs) if len(runs) > 1 else 0
            print(f"  {name:15s}: {m:8.3f} ± {s:.3f} t/s  {['%.3f'%r for r in runs]}", flush=True)

    print(f"\nDone.", flush=True)


if __name__ == '__main__':
    main()
