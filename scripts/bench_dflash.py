#!/usr/bin/env python3
"""
DFlash speculative decoding benchmark for VibeBlade.

Compares three configurations:
  1. Baseline — standard llama.cpp autoregressive decode
  2. DFlash — block diffusion parallel drafting + batch verification
  3. N-gram (existing) — pattern-matching draft for comparison

Reference: arxiv.org/abs/2602.06036 — DFlash (Chen et al. 2026)
github.com/z-lab/dflash
"""

import gc
import os
import sys
import time
import ctypes
import warnings

# VibeBlade setup
VB_DIR = "/home/ubuntu/VibeBlade"
sys.path.insert(0, VB_DIR)
os.environ["LD_LIBRARY_PATH"] = f"{VB_DIR}/cpp/build"

from vibeblade.speculative import SpeculativeBackend
from vibeblade.dflash import DFlashDraftHead, DFlashStats

# ── Config ────────────────────────────────────────────────────────────
MODEL = os.environ.get("MODEL", "models/tinyllama-1.1b-q4km.gguf")
DFLASH_MODEL = os.environ.get("DFLASH_MODEL", "z-lab/Qwen3-8B-DFlash-b16")
TARGET_MODEL = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B")
N_RUNS = int(os.environ.get("N_RUNS", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
N_CTX = int(os.environ.get("N_CTX", "512"))
N_THREADS = int(os.environ.get("N_THREADS", "4"))
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. In 2025, AI",
    "Write a Python function to compute factorial recursively.",
    "Explain the concept of speculative decoding in machine learning.",
    "What are the main advantages of block diffusion models?",
    "List three techniques for accelerating LLM inference.",
]


def bench_baseline(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run baseline (no speculative decoding)."""
    import ctypes
    sys.path.insert(0, VB_DIR)
    from vibeblade.llama_backend import _lib, _helper

    _helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _helper.override_model_params(1, 0, 0)
    _helper.override_context_params.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32]
    _helper.override_context_params(N_CTX, N_THREADS, N_THREADS)

    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
    _lib.rotorquant_set_enabled(False)

    b = SpeculativeBackend()
    b.load(model_path, n_ctx=N_CTX, n_threads=N_THREADS)
    t0 = time.time()
    r = b.generate(prompt, max_tokens=max_tokens, speculative=False, temperature=0.0)
    t_total = time.time() - t0
    tps = len(r.tokens) / r.time_decode if r.time_decode > 0 else 0.0
    b.free()
    del b
    gc.collect()
    return {"tps": tps, "tokens": len(r.tokens), "time": t_total, "text": r.text[:80]}


def bench_ngram(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run n-gram speculative decoding (existing VibeBlade)."""
    sys.path.insert(0, VB_DIR)
    from vibeblade.llama_backend import _lib, _helper

    _helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _helper.override_model_params(1, 0, 0)
    _helper.override_context_params.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32]
    _helper.override_context_params(N_CTX, N_THREADS, N_THREADS)

    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
    _lib.rotorquant_set_enabled(False)

    b = SpeculativeBackend(draft_n=4, draft_max=8)
    b.load(model_path, n_ctx=N_CTX, n_threads=N_THREADS)
    t0 = time.time()
    r = b.generate(prompt, max_tokens=max_tokens, speculative=True, temperature=0.0)
    stats = b.spec_stats
    b.free()
    del b
    gc.collect()
    return {
        "tps": r.tokens_per_second,
        "tokens": len(r.tokens),
        "time": r.time_total,
        "text": r.text[:80],
        "accept_rate": stats.acceptance_rate,
        "speedup": stats.effective_speedup,
    }


def bench_dflash(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run DFlash block diffusion speculative decoding.

    This requires:
      1. A matching DFlash draft model on HuggingFace
      2. The target model to share the same tokenizer as the draft
      3. transformers + torch installed (optional dep)
    """
    sys.path.insert(0, VB_DIR)
    from vibeblade.llama_backend import _lib, _helper

    _helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _helper.override_model_params(1, 0, 0)
    _helper.override_context_params.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32]
    _helper.override_context_params(N_CTX, N_THREADS, N_THREADS)

    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
    _lib.rotorquant_set_enabled(False)

    # Load target model
    b = SpeculativeBackend(draft_n=1, draft_max=16)
    b.load(model_path, n_ctx=N_CTX, n_threads=N_THREADS)

    # Check tokenizer match
    target_vocab = _lib.llama_vocab_n_tokens(b._vocab)
    print(f"  Target vocab size: {target_vocab}")

    # Attach DFlash
    try:
        dflash = DFlashDraftHead(
            draft_model_name=DFLASH_MODEL,
            target_model_name=TARGET_MODEL,
            block_size=16,
            temperature=0.0,
            device="cpu",
            torch_dtype="float32",
        )
        draft_vocab = len(dflash.tokenizer)
        print(f"  DFlash vocab size: {draft_vocab}")

        if target_vocab > 0 and abs(draft_vocab - target_vocab) > 100:
            warnings.warn(
                f"Tokenizer mismatch! Target={target_vocab}, DFlash={draft_vocab}. "
                f"DFlash models require the same tokenizer family as the target. "
                f"Acceptance rate will be near 0%. Use: {TARGET_MODEL}"
            )

        b.set_draft_model_dflash(dflash)
    except ImportError as e:
        print(f"  [SKIP] DFlash requires transformers+torch: {e}")
        b.free()
        del b
        gc.collect()
        return {"tps": 0, "tokens": 0, "error": str(e)}

    t0 = time.time()
    try:
        r = b.generate(prompt, max_tokens=max_tokens, speculative=True, temperature=0.0)
        stats = b.spec_stats
        dflash_stats = dflash.stats
        tps = r.tokens_per_second
        text = r.text[:80]
        n_tokens = len(r.tokens)
        accept_rate = stats.acceptance_rate
        speedup = stats.effective_speedup
        dflash_time_ms = dflash_stats.time_draft_ms
    except Exception as e:
        print(f"  [ERROR] DFlash generation failed: {e}")
        b.free()
        del b
        gc.collect()
        return {"tps": 0, "tokens": 0, "error": str(e)}

    b.free()
    del b
    gc.collect()

    return {
        "tps": tps,
        "tokens": n_tokens,
        "text": text,
        "accept_rate": accept_rate,
        "speedup": speedup,
        "dflash_time_ms": dflash_time_ms,
    }


def format_row(name: str, tps: float, tokens: int, extra: str = "") -> str:
    return f"  {name:<12} {tps:>7.2f} t/s  {tokens:>4} tok  {extra}"


def main():
    print(f"DFlash Speculative Decoding Benchmark")
    print(f"=" * 60)
    print(f"Model:       {MODEL}")
    print(f"DFlash:     {DFLASH_MODEL}")
    print(f"Target:     {TARGET_MODEL}")
    print(f"Runs:       {N_RUNS} × 3 configs")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Context:    {N_CTX}, Threads: {N_THREADS}")
    print(f"Prompts:    {len(PROMPTS)}")
    print()

    results = {"baseline": [], "ngram": [], "dflash": []}

    for run_i in range(N_RUNS):
        print(f"\n[Run {run_i + 1}/{N_RUNS}]")

        for i, prompt in enumerate(PROMPTS[:2]):  # 2 prompts for speed
            print(f"  Prompt {i+1}: {prompt[:50]}...")

            # Baseline
            r = bench_baseline(MODEL, prompt, MAX_TOKENS)
            results["baseline"].append(r)
            print(format_row("Baseline", r["tps"], r["tokens"]))

            # N-gram
            r = bench_ngram(MODEL, prompt, MAX_TOKENS)
            results["ngram"].append(r)
            extra = f"accept={r.get('accept_rate', 0):.0%} speedup={r.get('speedup', 0):.2f}×"
            print(format_row("N-gram", r["tps"], r["tokens"], extra))

            # DFlash
            r = bench_dflash(MODEL, prompt, MAX_TOKENS)
            results["dflash"].append(r)
            if "error" in r:
                print(f"  {'DFlash':<12} SKIP — {r['error']}")
            else:
                extra = f"accept={r.get('accept_rate', 0):.0%} speedup={r.get('speedup', 0):.2f}×"
                extra += f" draft={r.get('dflash_time_ms', 0):.0f}ms"
                print(format_row("DFlash", r["tps"], r["tokens"], extra))

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    def avg(lst, key):
        vals = [r[key] for r in lst if key in r and r[key] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def std(lst, key):
        vals = [r[key] for r in lst if key in r and r[key] > 0]
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        import math
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

    baseline_tps = avg(results["baseline"], "tps")
    ngram_tps = avg(results["ngram"], "tps")
    dflash_tps = avg([r for r in results["dflash"] if "error" not in r], "tps")

    baseline_std = std(results["baseline"], "tps")
    ngram_std = std(results["ngram"], "tps")
    dflash_std = std([r for r in results["dflash"] if "error" not in r], "tps")

    def cv(x, s):
        return f"±{s:.2f} ({s/x*100:.1f}% CV)" if x > 0 else "—"

    print(f"\nBaseline  {baseline_tps:>7.2f} t/s  {cv(baseline_tps, baseline_std)}")
    print(f"N-gram    {ngram_tps:>7.2f} t/s  {cv(ngram_tps, ngram_std)}")
    if dflash_tps > 0:
        dflash_speedup = dflash_tps / baseline_tps if baseline_tps > 0 else 0
        print(f"DFlash    {dflash_tps:>7.2f} t/s  {cv(dflash_tps, dflash_std)}")
        print(f"          Speedup vs baseline: {dflash_speedup:.2f}×")
    else:
        print(f"DFlash    SKIP (no successful runs)")

    # ── DFlash stats ────────────────────────────────────────────────────
    dflash_ok = [r for r in results["dflash"] if "error" not in r]
    if dflash_ok:
        avg_accept = sum(r.get("accept_rate", 0) for r in dflash_ok) / len(dflash_ok)
        avg_speedup = sum(r.get("speedup", 0) for r in dflash_ok) / len(dflash_ok)
        avg_dflash_ms = sum(r.get("dflash_time_ms", 0) for r in dflash_ok) / len(dflash_ok)
        print(f"\nDFlash details:")
        print(f"  Acceptance rate:  {avg_accept:.1%}")
        print(f"  Effective speedup: {avg_speedup:.2f}×")
        print(f"  Draft time:       {avg_dflash_ms:.1f} ms avg")

    print("\nNote: DFlash requires tokenizer compatibility with target model.")
    print(f"Available DFlash models: github.com/z-lab/dflash")
    print(f"  z-lab/Qwen3-8B-DFlash-b16       (Qwen3-8B)")
    print(f"  z-lab/Qwen3.5-9B-DFlash          (Qwen3.5-9B)")
    print(f"  z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat")


if __name__ == "__main__":
    main()
