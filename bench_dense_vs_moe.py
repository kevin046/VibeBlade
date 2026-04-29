#!/usr/bin/env python3
"""VibeBlade Benchmark: Dense vs MoE — Clean run with proper error handling."""
import sys, os, time, traceback

devnull = os.open(os.devnull, os.O_WRONLY)
stderr_save = os.dup(2)

def quiet():
    os.dup2(devnull, 2)
def loud():
    os.dup2(stderr_save, 2)

quiet()
from vibeblade.speculative import SpeculativeBackend
from vibeblade.llama_backend import LlamaCppBackend
loud()

models = [
    ("Qwen2.5-0.5B (Dense)",        "models/qwen2.5-0.5b-instruct-Q4_K_S.gguf"),
    ("Qwen3.5-MoE-0.87B (MoE 2/8)", "models/qwen3.5-moe-0.87B-d0.8B.Q4_K_S.gguf"),
    ("Llama-3.2-1B (Dense)",         "models/llama-3.2-1b-instruct-Q4_K_S.gguf"),
]

prompt = "The quick brown fox jumps over the lazy dog. " * 8
gen_tokens = 64
tests = [
    ("Baseline",     False, 0.0),
    ("TurboSparse",  False, 0.05),
    ("Speculative",  True,  0.0),
    ("Spec+TS",      True,  0.05),
]

W = 78
print("=" * W)
print("  VIBEBLADE BENCHMARK: Dense vs MoE Inference Optimization")
print("=" * W)
print(f"  Hardware: ARM NEON (aarch64), {os.cpu_count()} cores")
print(f"  Quant: Q4_K_S | Prompt: ~50 tok | Gen: {gen_tokens} tok | 4 threads")
print(f"  Temperature: 0.0 (greedy)")
print("=" * W)

all_results = {}

for model_name, model_path in models:
    fsize = os.path.getsize(model_path) // 1024 // 1024
    print(f"\n--- {model_name} ({fsize}MB) ---")
    print(f"  {'Config':<16s}  {'t/s':>7s}  {'Prefill':>9s}  {'Decode':>9s}  {'Tok':>5s}  {'Accept':>7s}  {'vs Base':>7s}")
    print(f"  {'-'*16}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*5}  {'-'*7}  {'-'*7}")

    model_results = {}
    baseline_tps = None

    for config_name, use_spec, ts_th in tests:
        quiet()
        try:
            b = SpeculativeBackend() if use_spec else LlamaCppBackend()
            b.load(model_path, n_threads=4)
            if ts_th > 0:
                b.turbosparse_enable(ts_th)

            if use_spec:
                out = b.generate(prompt, max_tokens=gen_tokens, temperature=0.0, speculative=True)
                acc = f"{b.spec_stats.acceptance_rate*100:.0f}%"
                n_gen = len(out.tokens)
            else:
                out = b.generate(prompt, max_tokens=gen_tokens, temperature=0.0)
                acc = "-"
                n_gen = len(out.tokens)

            loud()
            tps = out.tokens_per_second

            # Validate: if n_gen < 5 tokens, something went wrong (early EOS or crash)
            if n_gen < 5:
                print(f"  {config_name:<16s}  {tps:>7.1f}  {out.time_prefill*1000:>8.0f}ms  {out.time_decode*1000:>8.0f}ms  {n_gen:>5d}  {acc:>7s}  (early stop)")
                model_results[config_name] = 0  # don't count invalid runs
            else:
                model_results[config_name] = tps
                if config_name == "Baseline":
                    baseline_tps = tps
                vs_base = f"{tps/baseline_tps:.2f}x" if baseline_tps else "-"
                print(f"  {config_name:<16s}  {tps:>7.1f}  {out.time_prefill*1000:>8.0f}ms  {out.time_decode*1000:>8.0f}ms  {n_gen:>5d}  {acc:>7s}  {vs_base:>7s}")

        except SystemExit:
            loud()
            print(f"  {config_name:<16s}  CRASH (segfault)")
            model_results[config_name] = 0
        except Exception as e:
            loud()
            lines = traceback.format_exc().strip().split('\n')
            print(f"  {config_name:<16s}  ERROR: {lines[-1][:50]}")
            model_results[config_name] = 0

    all_results[model_name] = model_results

# Summary
print(f"\n{'=' * W}")
print(f"  SUMMARY")
print(f"{'=' * W}")
print(f"  {'Model':<30s}  {'Baseline':>9s}  {'Best Config':>14s}  {'Best t/s':>8s}  {'Gain':>7s}")
print(f"  {'-'*30}  {'-'*9}  {'-'*14}  {'-'*8}  {'-'*7}")

for model_name in all_results:
    res = all_results[model_name]
    bl = res.get("Baseline", 0)
    best_cfg = max(res, key=res.get)
    best_tps = res[best_cfg]
    if best_tps > 0 and bl > 0:
        gain = f"+{((best_tps/bl)-1)*100:.0f}%"
    else:
        gain = "-"
    print(f"  {model_name:<30s}  {bl:>9.1f}  {best_cfg:>14s}  {best_tps:>8.1f}  {gain:>7s}")

# Dense vs MoE
print(f"\n{'=' * W}")
print(f"  DENSE vs MoE ANALYSIS")
print(f"{'=' * W}")
keys = list(all_results.keys())
if len(keys) >= 2:
    dense = all_results[keys[0]]
    moe = all_results[keys[1]]
    db = dense.get("Baseline", 0)
    mb = moe.get("Baseline", 0)
    dbest = max(dense.values()) if dense else 0
    mbest = max(moe.values()) if moe else 0
    print(f"  Dense baseline (Qwen2.5-0.5B):    {db:.1f} t/s")
    print(f"  MoE baseline (Qwen3.5-MoE):        {mb:.1f} t/s")
    print(f"  Dense best VibeBlade:              {dbest:.1f} t/s")
    print(f"  MoE best VibeBlade:                {mbest:.1f} t/s")
    if db > 0 and mb > 0:
        print(f"  MoE vs Dense baseline:            {mb/db:.2f}x {'(MoE slower)' if mb < db else '(MoE faster)'}")
    if dbest > 0 and mbest > 0:
        print(f"  MoE+VB vs Dense+VB:              {mbest/dbest:.2f}x")

    # Llama comparison
    llama = all_results.get(keys[2], {})
    lb = llama.get("Baseline", 0)
    lbest = max(llama.values()) if llama else 0
    print(f"\n  Llama-3.2-1B baseline:            {lb:.1f} t/s")
    print(f"  Llama-3.2-1B best VibeBlade:      {lbest:.1f} t/s")
    if lb > 0:
        print(f"  Llama vs Qwen2.5 baseline:       {lb/db:.2f}x")
    if dbest > 0 and lbest > 0:
        print(f"  Llama+VB vs Qwen+VB:             {lbest/dbest:.2f}x")

print(f"\n{'=' * W}")
print(f"  Benchmark complete.")
print(f"{'=' * W}")
os.close(devnull)
