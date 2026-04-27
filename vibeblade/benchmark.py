"""VibeBlade Benchmark — llama.cpp-style performance profiling.

Measures all critical inference kernels across realistic model dimensions.
Usage:
    python -m vibeblade.benchmark              # full suite
    python -m vibeblade.benchmark --quick       # fast smoke test
    python -m vibeblade.benchmark --csv out.csv # CSV output
    python -m vibeblade.benchmark --json        # JSON output
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

# ─────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    label: str
    tokens_or_units: str  # "tokens", "elements", "bytes"
    total_us: float       # microseconds
    n: int                # iterations
    size_desc: str = ""   # human-readable size

    @property
    def per_unit_us(self) -> float:
        return self.total_us / self.n if self.n > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Units per second."""
        if self.per_unit_us == 0:
            return float("inf")
        return 1_000_000.0 / self.per_unit_us

    def __str__(self) -> str:
        if self.n == 0:
            return f"{self.label:<45s} {self.size_desc}"
        t = self.throughput
        if self.tokens_or_units == "tokens":
            if t >= 1000:
                return f"{self.label:<45s} {self.n:>6d} iters  {self.total_us/1000:>10.2f} ms  {t:>10.1f} t/s"
            return f"{self.label:<45s} {self.n:>6d} iters  {self.total_us/1000:>10.2f} ms  {t:>10.1f} t/s"
        elif self.tokens_or_units == "gbps":
            return f"{self.label:<45s} {self.n:>6d} iters  {self.total_us/1000:>10.2f} ms  {t/1e9:>10.2f} GB/s"
        else:
            return f"{self.label:<45s} {self.n:>6d} iters  {self.total_us/1000:>10.2f} ms  {self.per_unit_us:>10.3f} µs/u"


def bench(fn: Callable, n: int, warmup: int = 3) -> tuple[float, int]:
    """Run fn() n times (after warmup), return (total_us, actual_n)."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)  # ns → µs

    # Remove outliers (top/bottom 10%)
    if len(times) > 10:
        times.sort()
        cut = max(1, len(times) // 10)
        times = times[cut:-cut]

    return sum(times), len(times)


def format_size(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f}G"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def format_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n/(1<<30):.1f} GB"
    if n >= 1 << 20:
        return f"{n/(1<<20):.1f} MB"
    if n >= 1 << 10:
        return f"{n/(1<<10):.1f} KB"
    return f"{n} B"


# ─────────────────────────────────────────────
# Model configs to benchmark (real LLaMA scales)
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "llama-0.5B": {
        "hidden_dim": 2048, "num_heads": 32, "num_kv_heads": 8,
        "head_dim": 64, "intermediate_dim": 5632,
        "vocab_size": 32000, "num_layers": 22,
    },
    "llama-1.5B": {
        "hidden_dim": 2048, "num_heads": 32, "num_kv_heads": 8,
        "head_dim": 64, "intermediate_dim": 5632,
        "vocab_size": 32064, "num_layers": 28,
    },
    "llama-7B": {
        "hidden_dim": 4096, "num_heads": 32, "num_kv_heads": 32,
        "head_dim": 128, "intermediate_dim": 11008,
        "vocab_size": 32000, "num_layers": 32,
    },
    "llama-13B": {
        "hidden_dim": 5120, "num_heads": 40, "num_kv_heads": 40,
        "head_dim": 128, "intermediate_dim": 13696,
        "vocab_size": 32000, "num_layers": 40,
    },
    "llama-70B": {
        "hidden_dim": 8192, "num_heads": 64, "num_kv_heads": 8,
        "head_dim": 128, "intermediate_dim": 28672,
        "vocab_size": 32000, "num_layers": 80,
    },
}


# ─────────────────────────────────────────────
# Kernel benchmarks
# ─────────────────────────────────────────────

def bench_rms_norm(results: list, quick: bool):
    from vibeblade.transformer import rms_norm
    sizes = [(2048,), (4096,), (8192,)] if not quick else [(4096,)]
    for dim in sizes:
        x = np.random.randn(*dim).astype(np.float32)
        w = np.random.randn(*dim).astype(np.float32)
        iters = 100 if not quick else 20
        us, n = bench(lambda: rms_norm(x, w), iters)
        results.append(BenchResult(
            "rms_norm", f"rms_norm({format_size(dim[0])}, batch=1)",
            "elements", us, n, f"{dim[0]} elements"
        ))
        # Batch (seq_len × dim)
        batch_dim = (64, dim[0])
        xb = np.random.randn(*batch_dim).astype(np.float32)
        us, n = bench(lambda: rms_norm(xb, w), iters)
        results.append(BenchResult(
            "rms_norm", f"rms_norm({format_size(dim[0])}, batch=64)",
            "elements", us, n, f"{batch_dim[0]}×{dim[0]}"
        ))


def bench_silu(results: list, quick: bool):
    from vibeblade.transformer import silu
    sizes = [(5632,), (11008,), (28672,)] if not quick else [(11008,)]
    for dim in sizes:
        x = np.random.randn(*dim).astype(np.float32)
        iters = 100 if not quick else 20
        us, n = bench(lambda: silu(x), iters)
        results.append(BenchResult(
            "silu", f"silu({format_size(dim[0])})",
            "elements", us, n, f"{dim[0]} elements"
        ))


def bench_rope(results: list, quick: bool):
    from vibeblade.transformer import rope
    configs = [(64, 64), (128, 128)] if not quick else [(128, 128)]
    for head_dim, seq in configs:
        x = np.random.randn(seq, 1, head_dim).astype(np.float32)  # (seq, 1_head, head_dim)
        cos = np.random.randn(seq, head_dim // 2).astype(np.float32)
        sin = np.random.randn(seq, head_dim // 2).astype(np.float32)
        iters = 50 if not quick else 10
        us, n = bench(lambda: rope(x, cos, sin), iters)
        results.append(BenchResult(
            "rope", f"rope(seq={seq}, head_dim={head_dim})",
            "elements", us, n, f"{seq}×{head_dim}"
        ))


def bench_matmul(results: list, quick: bool):
    """Benchmark dense matmul at FFN projection scales."""
    configs = [
        (1, 4096, 11008, "ffn_up 7B single"),
        (1, 8192, 28672, "ffn_up 70B single"),
        (64, 4096, 11008, "ffn_up 7B batch=64"),
        (1, 4096, 4096, "attn_out 7B single"),
        (1, 32000, 4096, "output_proj 7B single"),
    ]
    if quick:
        configs = configs[:2]

    for M, K, N, label in configs:
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        iters = 20 if not quick else 5
        us, n = bench(lambda: a @ b, iters)
        bytes_total = M * K * 2 + K * N * 2 + M * N * 2  # read A + read B + write C
        results.append(BenchResult(
            "matmul", f"matmul {label} ({K}→{N})",
            "gbps", us, n, f"{M}×{K}×{N} ({format_bytes(bytes_total)})"
        ))


def bench_attention(results: list, quick: bool):
    from vibeblade.transformer import attention
    configs = [
        (1, 32, 32, 128, 1, "7B decode (kv=1)"),
        (1, 64, 8, 128, 64, "70B decode (kv=64)"),
        (32, 32, 32, 128, 32, "7B prefill (seq=32)"),
        (128, 32, 32, 128, 128, "7B prefill (seq=128)"),
    ]
    if quick:
        configs = configs[:2]

    for seq, n_h, n_kv, hd, kv_len, label_suffix in configs:
        q = np.random.randn(seq, n_h * hd).astype(np.float32)
        k = np.random.randn(kv_len, n_kv * hd).astype(np.float32)
        v = np.random.randn(kv_len, n_kv * hd).astype(np.float32)
        mask = np.triu(np.full((seq, kv_len), -np.inf, dtype=np.float32), k=1) if seq > 1 else None
        iters = 10 if not quick else 3
        us, n = bench(lambda: attention(q, k, v, n_h, n_kv, mask), iters)
        results.append(BenchResult(
            "attention", f"attention({label_suffix})",
            "tokens", us, n, f"seq={seq} heads={n_h} kv_heads={n_kv} hd={hd}"
        ))


def bench_softmax(results: list, quick: bool):
    sizes = [(32, 128), (512, 128), (2048, 128)] if not quick else [(512, 128)]
    for seq, hd in sizes:
        x = np.random.randn(seq * hd).astype(np.float32).reshape(seq, hd)
        iters = 50 if not quick else 10
        us, n = bench(lambda: (lambda: (np.exp(x - x.max(axis=-1, keepdims=True)) /
                np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)))(), iters)
        results.append(BenchResult(
            "softmax", f"softmax(seq={seq}, dim={hd})",
            "elements", us, n, f"{seq}×{hd}"
        ))


def bench_sparse(results: list, quick: bool):
    from vibeblade.sparse import drelu_activation, compute_sparsity, sparse_matmul, topk_activation_mask

    # dReLU
    for dim in [(5632,), (11008,)] if not quick else [(11008,)]:
        x = np.random.randn(*dim).astype(np.float32)
        us, n = bench(lambda: drelu_activation(x), 100 if not quick else 20)
        results.append(BenchResult(
            "sparse", f"drelu({format_size(dim[0])})",
            "elements", us, n
        ))

    # Sparsity measurement
    x = np.random.randn(11008).astype(np.float32) * 0.5
    us, n = bench(lambda: compute_sparsity(x), 100 if not quick else 20)
    results.append(BenchResult(
        "sparse", "compute_sparsity(11008)",
        "elements", us, n
    ))

    # Sparse matmul (10% sparsity)
    M, K, N = 1, 4096, 11008
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    mask = topk_activation_mask(a, K // 10)  # top 10% = 90% sparse
    iters = 20 if not quick else 5
    us, n = bench(lambda: sparse_matmul(a, b, mask), iters)
    # Compare with dense
    us_dense, _ = bench(lambda: a @ b, iters)
    speedup = us_dense / us if us > 0 else 0
    results.append(BenchResult(
        "sparse", f"sparse_matmul(1×{K}×{N}, 90% sparse)",
        "gbps", us, n, f"vs dense speedup: {speedup:.2f}x"
    ))


def bench_quant(results: list, quick: bool):
    from vibeblade.quant import quantize_4bit, dequantize_4bit

    # Use smaller sizes that won't hang in pure Python
    for n_elements, label in [(4096, "4K"), (11008, "11K")] if not quick else [(4096, "4K")]:
        w = np.random.randn(n_elements).astype(np.float32)  # 1-D
        iters = 20 if not quick else 5
        us, n = bench(lambda: quantize_4bit(w), iters)
        results.append(BenchResult(
            "quant", f"quantize_4bit({label})",
            "gbps", us, n, f"{format_bytes(n_elements * 2)} → {format_bytes(n_elements // 2)}"
        ))

        # Dequantize
        packed, scales, rotors = quantize_4bit(w)
        us, n = bench(lambda: dequantize_4bit(packed, scales, rotors, n_elements), iters)
        results.append(BenchResult(
            "quant", f"dequantize_4bit({label})",
            "gbps", us, n
        ))


def bench_kv_cache(results: list, quick: bool):
    from vibeblade.cache import KVCache
    configs = [
        (32, 32, 128, 2048, "7B"),
        (80, 8, 128, 2048, "70B"),
    ]
    if quick:
        configs = configs[:1]

    for layers, heads, hd, max_seq, label in configs:
        cache = KVCache(layers, heads, hd, max_seq)
        k = np.random.randn(1, hd).astype(np.float16)
        v = np.random.randn(1, hd).astype(np.float16)
        iters = 100 if not quick else 20
        pos = 0

        def _update():
            nonlocal pos
            for layer_idx in range(layers):
                cache.update(layer_idx, k, v, pos)
            pos += 1

        us, n = bench(_update, iters, warmup=1)
        results.append(BenchResult(
            "kv_cache", f"kv_cache.update({label}, {layers} layers)",
            "tokens", us, n, f"mem={format_bytes(cache.memory_usage_bytes())}"
        ))

        # KV cache read (for attention)
        us, n = bench(lambda: cache.get(0, 0, max_seq), iters)
        results.append(BenchResult(
            "kv_cache", f"kv_cache.get({label}, seq={max_seq})",
            "tokens", us, n
        ))


def bench_scheduler(results: list, quick: bool):
    from vibeblade.scheduler import PowerInferScheduler

    configs = [(4096, 32, 0.1, "7B"), (8192, 80, 0.1, "70B")]
    if quick:
        configs = configs[:1]

    for hidden, layers, budget, label in configs:
        sched = PowerInferScheduler(hidden, layers, budget)
        acts = np.random.randn(hidden).astype(np.float32)
        iters = 100 if not quick else 20

        def _schedule():
            for layer_idx in range(layers):
                sched.update(layer_idx, acts)
                sched.schedule_layer(layer_idx, acts)

        us, n = bench(_schedule, iters)
        results.append(BenchResult(
            "scheduler", f"scheduler({label}, {layers} layers)",
            "tokens", us, n
        ))


def bench_forward_single(results: list, quick: bool):
    """Benchmark a single-token forward pass (decode step)."""
    from vibeblade.transformer import forward_token
    from vibeblade.cache import KVCache

    # Use a small config for forward bench (real model weights too large for VM)
    configs = [
        (512, 8, 2, 64, 1536, 4, 256, "tiny"),
        (1024, 16, 4, 64, 3072, 8, 512, "small"),
    ]
    if quick:
        configs = configs[:1]

    for hidden, n_h, n_kv, hd, inter, layers, vocab, label in configs:
        # Create mock weights per layer
        weights = {}
        for i in range(layers):
            weights[f"blk.{i}.attn_norm.weight"] = np.random.randn(hidden).astype(np.float32)
            weights[f"blk.{i}.ffn_norm.weight"] = np.random.randn(hidden).astype(np.float32)
            weights[f"blk.{i}.attn_q.weight"] = np.random.randn(n_h * hd, hidden).astype(np.float32)
            weights[f"blk.{i}.attn_k.weight"] = np.random.randn(n_kv * hd, hidden).astype(np.float32)
            weights[f"blk.{i}.attn_v.weight"] = np.random.randn(n_kv * hd, hidden).astype(np.float32)
            weights[f"blk.{i}.attn_output.weight"] = np.random.randn(hidden, n_h * hd).astype(np.float32)
            weights[f"blk.{i}.ffn_gate.weight"] = np.random.randn(inter, hidden).astype(np.float32)
            weights[f"blk.{i}.ffn_up.weight"] = np.random.randn(inter, hidden).astype(np.float32)
            weights[f"blk.{i}.ffn_down.weight"] = np.random.randn(hidden, inter).astype(np.float32)
        weights["token_embd.weight"] = np.random.randn(vocab, hidden).astype(np.float32)
        weights["output_norm.weight"] = np.random.randn(hidden).astype(np.float32)
        weights["output.weight"] = np.random.randn(vocab, hidden).astype(np.float32)

        cache = KVCache(layers, n_kv, hd, 512)
        iters = 10 if not quick else 3

        # Precompute RoPE caches
        from vibeblade.transformer import build_rope_cache
        rope_cos, rope_sin = build_rope_cache(hd, 512)

        pos = [0]
        def _forward():
            # Embed token
            tok = np.array([pos[0] % vocab], dtype=np.int64)
            x = weights["token_embd.weight"][tok]  # (1, hidden)
            # Loop through all layers (single-token decode)
            for i in range(layers):
                k_c, v_c = cache.get(i)  # may be empty (n_kv, 0, hd)
                x, k_new, v_new = forward_token(
                    x, weights, i,
                    rope_cos, rope_sin,
                    k_c, v_c,
                    position=pos[0],
                    n_heads=n_h, n_kv_heads=n_kv,
                )
                # forward_token returns full concat cache; store just the last slot
                cache.update(i, k_new[:, -1:, :], v_new[:, -1:, :], pos[0])
            pos[0] += 1

        us, n = bench(_forward, iters, warmup=1)
        tps = n / (us / 1_000_000) if us > 0 else 0
        results.append(BenchResult(
            "forward", f"forward_decode({label}, {layers}L, {hidden}d)",
            "tokens", us, n, f"{tps:.1f} t/s"
        ))


def bench_grammar(results: list, quick: bool):
    from vibeblade.grammar import RegexGrammar, JsonSchemaGrammar, GrammarConstraint

    # Regex compilation
    patterns = [
        r'[a-z]+',
        r'[0-9]{3}-[0-9]{4}',
        r'"(?:[^"\\]|\\.)*"',  # JSON string
    ]
    for pat in patterns:
        iters = 50 if not quick else 10
        us, n = bench(lambda: RegexGrammar(pat), iters)
        results.append(BenchResult(
            "grammar", f"regex_compile({pat[:30]})",
            "elements", us, n
        ))

    # Token mask generation
    vocab_sizes = [3200, 12800]  # reduced from 32000/128000 to avoid slow DFA on VM
    for vs in vocab_sizes:
        # Build a simple vocab
        vocab = [chr(i % 256) if i % 256 >= 32 else ' ' for i in range(vs)]
        gc = GrammarConstraint.from_regex(vocab, r'[a-z]+')
        iters = 50 if not quick else 10
        us, n = bench(lambda: gc.get_token_mask(), iters)
        results.append(BenchResult(
            "grammar", f"get_token_mask(vocab={format_size(vs)})",
            "elements", us, n, f"{vs} tokens"
        ))

    # JSON Schema compilation
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    [chr(i % 128) if i % 128 >= 32 else ' ' for i in range(1000)]
    iters = 20 if not quick else 5
    us, n = bench(lambda: JsonSchemaGrammar(schema), iters)
    results.append(BenchResult(
        "grammar", "json_schema_compile(nested object)",
        "elements", us, n
    ))


def bench_sampling(results: list, quick: bool):
    from vibeblade.generate import TextGenerator

    gen = TextGenerator(temperature=0.8, top_k=50, top_p=0.9)
    vocab_sizes = [3200, 32000]
    for vs in vocab_sizes:
        logits = np.random.randn(vs).astype(np.float32)
        iters = 1000 if not quick else 200
        us, n = bench(lambda: gen.sample(logits), iters)
        results.append(BenchResult(
            "sampling", f"sample(vocab={format_size(vs)}, t=0.8, k=50, p=0.9)",
            "tokens", us, n
        ))


def bench_memory(results: list, quick: bool):
    """Benchmark memory allocation patterns using NumPy."""

    # Allocation speed — measure time to allocate and zero arrays
    sizes = [(4096, 4096), (8192, 8192)]
    labels = ["4K×4K (64 MB)", "8K×8K (256 MB)"]
    if quick:
        sizes = sizes[:1]
        labels = labels[:1]

    for (rows, cols), label in zip(sizes, labels):
        iters = 20 if not quick else 5
        us, n = bench(lambda: np.zeros((rows, cols), dtype=np.float32), iters)
        rows * cols * 4
        results.append(BenchResult(
            "memory", f"alloc zeros {label}",
            "gbps", us, n
        ))


def bench_cpp_backend(results: list, quick: bool):
    """Benchmark C++ accelerated kernels if available."""
    try:
        from vibeblade.vibeblade_core import rotor_unpack, drelu, _CPP_BACKEND
    except ImportError:
        return

    if not _CPP_BACKEND:
        return

    # rotor_unpack
    n = 4096
    packed = np.random.randint(0, 256, (n // 2,), dtype=np.uint8)
    rotor = np.random.randn(4).astype(np.float32)
    iters = 50 if not quick else 10
    us, py_n = bench(lambda: rotor_unpack(packed, rotor, n), iters)

    # Compare with numpy
    from vibeblade.quant import unpack_nibbles
    def py_unpack():
        raw = unpack_nibbles(packed, n).astype(np.float32)
        for i in range(0, n, 4):
            g = min(4, n - i)
            raw[i:i+g] *= rotor[:g]
        return raw

    us_py, cpp_n = bench(py_unpack, iters)
    speedup = us_py / us if us > 0 else 0
    results.append(BenchResult(
        "cpp", f"rotor_unpack({format_size(n)}, C++ vs numpy speedup={speedup:.2f}x)",
        "elements", us, py_n, "C++ AVX-512" if _CPP_BACKEND else "numpy"
    ))

    # drelu
    x = np.random.randn(4096).astype(np.float32)
    us, n = bench(lambda: drelu(x), iters)
    results.append(BenchResult(
        "cpp", f"drelu({format_size(4096)})",
        "elements", us, n
    ))


def bench_onnx_backend(results: list, quick: bool):
    """Benchmark ONNX Runtime backend: compile time, layer inference, device info."""
    try:
        from vibeblade.onnx_backend import ONNXBackend, detect_device, _pick_providers
    except ImportError:
        results.append(BenchResult("ort_available", 0, 0, note="onnxruntime not installed"))
        return

    # Device info
    info = detect_device()
    providers = _pick_providers()
    results.append(BenchResult(
        "device_detection", 0, 0,
        note=f"{info['device']} | {info['description']} | providers={providers}",
    ))

    np.random.seed(42)
    sizes = [(32, 4, 2, 2)] if quick else [(32, 4, 2, 2), (64, 8, 4, 4), (128, 8, 4, 4)]
    iters = 20 if quick else 100

    for hidden, n_heads, n_kv_heads, n_layers in sizes:
        head_dim = hidden // n_heads
        inter_dim = hidden * 2
        vocab = hidden * 2
        weights = {
            "token_embd.weight": np.random.randn(vocab, hidden).astype(np.float32) * 0.1,
            "output_norm.weight": np.ones(hidden, dtype=np.float32),
            "output.weight": np.random.randn(vocab, hidden).astype(np.float32) * 0.01,
        }
        for i in range(n_layers):
            pfx = f"blk.{i}"
            weights[f"{pfx}.attn_norm.weight"] = np.ones(hidden, dtype=np.float32)
            weights[f"{pfx}.ffn_norm.weight"] = np.ones(hidden, dtype=np.float32)
            weights[f"{pfx}.attn_q.weight"] = np.random.randn(n_heads * head_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.attn_k.weight"] = np.random.randn(n_kv_heads * head_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.attn_v.weight"] = np.random.randn(n_kv_heads * head_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.attn_output.weight"] = np.random.randn(hidden, n_heads * head_dim).astype(np.float32) * 0.1
            weights[f"{pfx}.ffn_gate.weight"] = np.random.randn(inter_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.ffn_up.weight"] = np.random.randn(inter_dim, hidden).astype(np.float32) * 0.1
            weights[f"{pfx}.ffn_down.weight"] = np.random.randn(hidden, inter_dim).astype(np.float32) * 0.1

        label = f"h{hidden}_l{n_layers}_q{n_heads}_kv{n_kv_heads}"

        # Compile time
        t0 = time.perf_counter()
        backend = ONNXBackend(weights, n_layers, n_heads, n_kv_heads, hidden, head_dim, inter_dim, num_threads=1)
        compile_ms = (time.perf_counter() - t0) * 1000
        results.append(BenchResult(f"ort_compile_{label}", n_layers, compile_ms))

        # Single-token inference (no cache)
        x = np.random.randn(1, hidden).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(iters):
            for layer in range(n_layers):
                backend.forward_token(x, layer, None, None, 0)
        elapsed = (time.perf_counter() - t0) * 1000
        per_tok = elapsed / iters
        results.append(BenchResult(f"ort_decode_{label}_pos0", iters * n_layers, per_tok))

        # Multi-position with KV cache
        kv = {i: (None, None) for i in range(n_layers)}
        t0 = time.perf_counter()
        for _ in range(iters):
            for pos in range(8):
                for layer in range(n_layers):
                    kk, vv = kv[layer]
                    _, k_out, v_out = backend.forward_token(x, layer, kk, vv, pos)
                    kv[layer] = (k_out, v_out)
        elapsed = (time.perf_counter() - t0) * 1000
        per_seq = elapsed / iters
        results.append(BenchResult(f"ort_decode_{label}_8tok", iters * 8 * n_layers, per_seq))

    # TensorRT availability
    try:
        from vibeblade.tensorrt_backend import is_available
        trt = is_available()
        results.append(BenchResult("trt_available", 0, 0, note="yes" if trt else "no (NVIDIA GPU required)"))
    except ImportError:
        results.append(BenchResult("trt_available", 0, 0, note="tensorrt_backend not importable"))


def bench_model_summary(results: list, quick: bool):
    """Print model size estimates (no timing needed)."""
    for name, cfg in MODEL_CONFIGS.items():
        if quick and name not in ("llama-7B",):
            continue
        d = cfg["hidden_dim"]
        num_layers = cfg["num_layers"]
        inter = cfg["intermediate_dim"]
        v = cfg["vocab_size"]
        h = cfg["head_dim"]
        n_h = cfg["num_heads"]
        n_kv = cfg["num_kv_heads"]

        # Params per layer
        attn_q = d * n_h * h
        attn_k = d * n_kv * h
        attn_v = d * n_kv * h
        attn_o = d * d
        ffn = 3 * d * inter  # gate + up + down
        norms = 2 * d
        per_layer = attn_q + attn_k + attn_v + attn_o + ffn + norms

        total = per_layer * num_layers + v * d + d  # layers + embed + output_norm

        # KV cache memory
        kv_cache_bytes = 2 * num_layers * 2 * h * 2 * 2048  # 2 (K+V) * layers * n_kv_heads * head_dim * 2 bytes * max_seq

        results.append(BenchResult(
            "model", f"{name}: {format_size(total)} params, KV cache {format_bytes(kv_cache_bytes)}",
            "elements", 0, 0,
            f"d={d} L={num_layers} inter={inter} vocab={v} n_h={n_h} n_kv={n_kv}"
        ))


# ─────────────────────────────────────────────
# Full benchmark runner
# ─────────────────────────────────────────────

def run_all_benchmarks(quick: bool = False) -> list[BenchResult]:
    results = []

    print("=" * 90)
    print("  VibeBlade Benchmark Suite")
    print("  Developed by VibeDrift Inc. — vibedrift.com")
    print(f"  Python {sys.version.split()[0]} | NumPy {np.__version__} | "
          f"{'C++ AVX-512' if _get_cpp_backend() else 'NumPy only'}")
    print(f"  {'Quick mode' if quick else 'Full benchmark'}")
    print("=" * 90)
    print()

    groups = [
        ("Model Summary", bench_model_summary),
        ("RMSNorm", bench_rms_norm),
        ("SiLU", bench_silu),
        ("RoPE", bench_rope),
        ("MatMul / FFN Projections", bench_matmul),
        ("Attention", bench_attention),
        ("Softmax", bench_softmax),
        ("Sparse (dReLU, PowerInfer)", bench_sparse),
        ("Quantization (4-bit)", bench_quant),
        ("KV Cache", bench_kv_cache),
        ("Scheduler (PowerInfer)", bench_scheduler),
        ("Forward Pass (Decode)", bench_forward_single),
        ("Grammar Decoding", bench_grammar),
        ("Sampling", bench_sampling),
        ("Memory", bench_memory),
        ("C++ Backend", bench_cpp_backend),
        ("ONNX Runtime Backend", bench_onnx_backend),
    ]

    for group_name, bench_fn in groups:
        print(f"── {group_name} {'─' * (90 - 3 - len(group_name))}")
        before = len(results)
        bench_fn(results, quick)
        for r in results[before:]:
            print(r)
        print()

    return results


def _get_cpp_backend() -> bool:
    try:
        from vibeblade.vibeblade_core import _CPP_BACKEND
        return bool(_CPP_BACKEND)
    except ImportError:
        return False


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VibeBlade Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Fast smoke test (fewer sizes/iters)")
    parser.add_argument("--csv", type=str, help="Output results to CSV file")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    results = run_all_benchmarks(quick=args.quick)

    # Summary
    kernel_results = [r for r in results if r.n > 0]
    if kernel_results:
        print("=" * 90)
        print("  SUMMARY — Top 5 slowest kernels (bottleneck analysis)")
        print("=" * 90)
        sorted_by_time = sorted(kernel_results, key=lambda r: r.per_unit_us, reverse=True)
        for r in sorted_by_time[:5]:
            print(f"  {r.per_unit_us:>12.1f} µs  {r.label}")
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "label", "tokens_or_units",
                                                     "total_us", "n", "per_unit_us",
                                                     "throughput", "size_desc"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "name": r.name, "label": r.label,
                    "tokens_or_units": r.tokens_or_units,
                    "total_us": f"{r.total_us:.2f}",
                    "n": r.n,
                    "per_unit_us": f"{r.per_unit_us:.3f}",
                    "throughput": f"{r.throughput:.2f}",
                    "size_desc": r.size_desc,
                })
        print(f"  Results written to {args.csv}")

    if args.json:
        out = []
        for r in results:
            d = asdict(r)
            d["per_unit_us"] = r.per_unit_us
            d["throughput"] = r.throughput
            out.append(d)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
