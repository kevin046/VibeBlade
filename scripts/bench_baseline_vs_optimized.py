#!/usr/bin/env python3
"""Benchmark: Baseline vs VibeBlade Optimized — C++ Native Engine.

Tests GGUF models across multiple configurations via the C++ fast engine:
  1. Baseline — C++ autoregressive decode (temperature=0, greedy)
  2. Top-k/p sampling — C++ with sampling (realistic inference)
  3. Speculative (4 draft) — C++ speculative decoding, 4 draft tokens
  4. Speculative (8 draft) — C++ speculative decoding, 8 draft tokens

SR&ED: Systematic evaluation of speculative decoding and sampling
strategies across model architectures to quantify throughput improvement
and identify acceptance-rate sensitivity to model depth and quantization.
"""
import sys
import os
import time
import gc
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPT = "The capital of France is"
MAX_TOKENS = 64
N_RUNS = 3

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODELS = [
    "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf",
]


def bench_generate(model_path, prompt, max_tokens, temperature=0.0,
                   top_k=50, top_p=0.9, n_spec=0, label=""):
    """Benchmark via C++ FastModelWrapper."""
    from vibeblade.fast_backend import FastModelWrapper

    model = FastModelWrapper(model_path)

    # Warmup
    try:
        model.generate(prompt, max_tokens=4, temperature=0.0, stream=False)
    except Exception as e:
        print(f"    warmup error: {e}", flush=True)

    tps_list = []
    n_tok_list = []

    for run in range(N_RUNS):
        t0 = time.perf_counter()
        if n_spec > 0:
            result = model._model.speculative_decode(
                prompt, max_tokens, temperature, top_k, top_p,
                1.0, -1, n_spec
            )
            tps = result.tokens_per_second
            n_tok = len(result.token_ids)
        else:
            text, tps = model.generate(
                prompt, max_tokens=max_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                stream=False
            )
            n_tok = max_tokens
        t1 = time.perf_counter()
        elapsed = t1 - t0
        actual_tps = n_tok / elapsed if elapsed > 0 else tps

        tps_list.append(actual_tps)
        n_tok_list.append(n_tok)
        print(f"    run {run+1}: {actual_tps:8.3f} t/s ({n_tok} tok, {elapsed:.2f}s)", flush=True)

    del model
    gc.collect()
    return tps_list, n_tok_list


CONFIGS = [
    ("1. Baseline (greedy)", {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "n_spec": 0}),
    ("2. Sampling (T=0.8)", {"temperature": 0.8, "top_k": 50, "top_p": 0.9, "n_spec": 0}),
    ("3. Speculative (4 draft)", {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "n_spec": 4}),
    ("4. Speculative (8 draft)", {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "n_spec": 8}),
]


def main():
    print("=" * 76)
    print("  VibeBlade Benchmark — Baseline vs Optimized (C++ Native Engine)")
    print(f"  {N_RUNS} runs × {len(CONFIGS)} configs × {MAX_TOKENS} tokens")
    print(f"  Prompt: \"{PROMPT}\"")
    print("=" * 76, flush=True)

    all_results = {}

    for model_name in MODELS:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            print(f"\n  ⚠ Skipping {model_name} — not found", flush=True)
            continue

        short_name = model_name.replace("-Q4_K_M.gguf", "").replace("-Instruct", "")
        sz_mb = os.path.getsize(model_path) / 1e6
        print(f"\n{'─' * 76}", flush=True)
        print(f"  📦 {short_name} ({sz_mb:.0f} MB)", flush=True)
        print(f"{'─' * 76}", flush=True)

        for cfg_name, cfg in CONFIGS:
            print(f"\n  {cfg_name}", flush=True)
            try:
                tps_list, n_tok_list = bench_generate(
                    model_path, PROMPT, MAX_TOKENS, **cfg
                )
                mean_tps = statistics.mean(tps_list)
                std_tps = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0

                all_results[(short_name, cfg_name)] = {
                    "mean": mean_tps, "std": std_tps,
                    "min": min(tps_list), "max": max(tps_list),
                    "mean_tok": statistics.mean(n_tok_list),
                }
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_results[(short_name, cfg_name)] = {
                    "mean": 0, "std": 0, "min": 0, "max": 0,
                    "mean_tok": 0, "error": str(e)
                }

    # ─── Summary table ───
    print(f"\n{'=' * 76}")
    print("  RESULTS SUMMARY — Mean tokens/second (± std)")
    print(f"{'=' * 76}")

    models_seen = sorted(set(k[0] for k in all_results.keys()))
    cfg_labels = [c[0] for c in CONFIGS]
    short_cfgs = ["Greedy", "Sample(T=.8)", "Spec(4d)", "Spec(8d)"]

    header = f"  {'Model':<24s}"
    for sc in short_cfgs:
        header += f" {sc:>14s}"
    header += f" {'Best↑':>8s}"
    print(header)
    print(f"  {'─' * 24}" + ("─" * 15) * len(short_cfgs) + "─" * 9)

    for model in models_seen:
        row = f"  {model:<24s}"
        baseline_mean = None
        best_mean = 0
        for i, cfg_name in enumerate(cfg_labels):
            key = (model, cfg_name)
            r = all_results.get(key, {})
            mean = r.get("mean", 0)
            std = r.get("std", 0)
            if "error" in r:
                row += f" {'ERR':>14s}"
            else:
                row += f" {mean:>7.2f}±{std:<5.2f}"
                if i == 0:
                    baseline_mean = mean
                best_mean = max(best_mean, mean)

        if baseline_mean and baseline_mean > 0:
            speedup = best_mean / baseline_mean
            row += f" {speedup:>6.2f}×"

        print(row)

    # ─── Per-config speedup vs baseline ───
    print(f"\n  Speedup vs Baseline (greedy decode):")
    print(f"  {'Model':<24s} {'Sample':>10s} {'Spec(4d)':>10s} {'Spec(8d)':>10s}")
    print(f"  {'─' * 24}{'─' * 10}{'─' * 10}{'─' * 10}")

    for model in models_seen:
        baseline = all_results.get((model, cfg_labels[0]), {}).get("mean", 0)
        if baseline <= 0:
            continue
        row = f"  {model:<24s}"
        for i in range(1, len(cfg_labels)):
            val = all_results.get((model, cfg_labels[i]), {}).get("mean", 0)
            if val > 0:
                row += f" {val/baseline:>8.2f}×  "
            else:
                row += f" {'N/A':>10s}"
        print(row)

    print(f"\n  SR&ED: Performance evaluation quantifies speculative decoding")
    print(f"  speedup across model architectures. Draft token count sensitivity")
    print(f"  analysis informs optimal n_spec_tokens selection per model depth.")


if __name__ == "__main__":
    main()
