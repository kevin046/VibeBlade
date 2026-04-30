#!/usr/bin/env python3
"""Benchmark: Baseline vs VibeBlade optimizations (one model at a time)."""

import sys, os, time, gc
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

from vibeblade.llama_backend import LlamaCppBackend, _helper

MODEL = "/home/ubuntu/VibeBlade/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
PROMPT = "The capital of France is"
N_TOKENS = 64

configs = [
    # (label, mlock, turbosparse, powerinfer, ts_threshold)
    ("1. Baseline (mmap, no opts)",       False, False, False, 0.0),
    ("2. mlock (no opts)",                True,  False, False, 0.0),
    ("3. mlock + TurboSparse 0.01",       True,  True,  False, 0.01),
    ("4. mlock + TurboSparse 0.05",       True,  True,  False, 0.05),
    ("5. mlock + PowerInfer",             True,  False, True,  0.0),
    ("6. mlock + PI + TS 0.01",           True,  True,  True,  0.01),
    ("7. mlock + PI + TS 0.05",           True,  True,  True,  0.05),
]

results = []

for label, mlock, ts, pi, ts_th in configs:
    gc.collect()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    # mlock must be set BEFORE load() since helper is called inside load()
    _helper.override_model_params(1, int(mlock), 0)  # mmap=1, mlock, nogpu
    
    b = LlamaCppBackend()
    b.load(MODEL, n_ctx=512, n_threads=4)
    b.set_turbosparse(ts, threshold=ts_th)
    b.set_powerinfer(pi, hot_budget=0.1)
    
    # Warmup
    try:
        _ = b.generate(PROMPT, max_tokens=4, temperature=0.0)
    except Exception as e:
        print(f"  Warmup error: {e}")
        results.append((label, "ERR", "ERR", "ERR"))
        del b
        continue
    
    # Timed run
    t0 = time.time()
    out = b.generate(PROMPT, max_tokens=N_TOKENS, temperature=0.0)
    t1 = time.time()
    
    elapsed = t1 - t0
    n_gen = len(out.tokens)
    tps = n_gen / elapsed if elapsed > 0 else 0
    
    print(f"  Tokens:  {n_gen}")
    print(f"  Time:    {elapsed:.2f}s")
    print(f"  Speed:   {tps:.3f} t/s")
    print(f"  Prefill: {out.time_prefill:.3f}s")
    print(f"  Decode:  {out.time_decode:.3f}s")
    print(f"  Text:    {out.text[:80]}...")
    
    results.append((label, n_gen, f"{tps:.3f}", f"{elapsed:.2f}"))
    del b

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  {'Config':<40s} {'Toks':>5s} {'t/s':>7s} {'Time':>6s}")
print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*6}")
for label, n_tok, tps, elapsed in results:
    print(f"  {label:<40s} {str(n_tok):>5s} {tps:>7s} {elapsed:>6s}")
