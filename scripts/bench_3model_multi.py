#!/usr/bin/env python3
"""3-model × 6-config × 3-iteration benchmark. Reports median t/s per config."""
import sys, os, ctypes, gc, time, statistics

sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

_lib = ctypes.CDLL('/home/ubuntu/VibeBlade/cpp/build/libllama.so')
_lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
_lib.turbosparse_is_enabled.restype = ctypes.c_bool
_lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
_lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool, ctypes.c_float]
_lib.powerinfer_is_enabled.restype = ctypes.c_bool
_lib.rotorquant_set_enabled.argtypes = [ctypes.c_bool]

from vibeblade.llama_backend import _helper, LlamaCppBackend
from vibeblade.speculative import SpeculativeBackend

_helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_helper.override_model_params(1, 0, 0)

MODELS = {
    "TinyLlama-1.1B":    "/home/ubuntu/VibeBlade/models/tinyllama-1.1b-q4km.gguf",
    "Llama-3.2-1B":      "/home/ubuntu/VibeBlade/models/llama-3.2-1b-instruct-Q4_K_S.gguf",
    "Phi-2-2.7B":        "/home/ubuntu/VibeBlade/models/phi-2.Q4_K_M.gguf",
}

PROMPT = "The quick brown fox jumps over the lazy dog. In the year 2025, artificial intelligence"
MAX_TOKENS = 16
N_THREADS = 4
N_ITERS = 3

def disable_all():
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False, 0.1)
    _lib.rotorquant_set_enabled(False)

def bench_config(name, model_path):
    """Run one config, return t/s."""
    disable_all()
    gc.collect()

    if name == "Baseline":
        b = LlamaCppBackend()
        b.load(model_path, n_ctx=256, n_threads=N_THREADS)
        b._set_sampler(temperature=0.0)
        out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
        b.free(); gc.collect()
        return out.tokens_per_second

    elif name == "TurboSparse":
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(0.05)
        b = LlamaCppBackend()
        b.load(model_path, n_ctx=256, n_threads=N_THREADS)
        b._set_sampler(temperature=0.0)
        out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
        b.free(); gc.collect()
        return out.tokens_per_second

    elif name == "Speculative":
        spec = SpeculativeBackend(draft_n=4, draft_max=5)
        spec.load(model_path, n_ctx=256, n_threads=N_THREADS)
        out = spec.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
        rate = spec.spec_stats.acceptance_rate
        spec.free(); gc.collect()
        return out.tokens_per_second, rate

    elif name == "Spec+TS":
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(0.05)
        spec = SpeculativeBackend(draft_n=4, draft_max=5)
        spec.load(model_path, n_ctx=256, n_threads=N_THREADS)
        out = spec.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
        rate = spec.spec_stats.acceptance_rate
        spec.free(); gc.collect()
        return out.tokens_per_second, rate

    elif name == "PowerInfer":
        _lib.powerinfer_set_enabled(True, 0.1)
        b = LlamaCppBackend()
        b.load(model_path, n_ctx=256, n_threads=N_THREADS)
        b._set_sampler(temperature=0.0)
        out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
        b.free(); gc.collect()
        return out.tokens_per_second

    elif name == "PI+TS":
        _lib.powerinfer_set_enabled(True, 0.1)
        _lib.turbosparse_set_enabled(True)
        _lib.turbosparse_set_threshold(0.05)
        b = LlamaCppBackend()
        b.load(model_path, n_ctx=256, n_threads=N_THREADS)
        b._set_sampler(temperature=0.0)
        out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
        b.free(); gc.collect()
        return out.tokens_per_second

CONFIGS = ["Baseline", "TurboSparse", "Speculative", "Spec+TS", "PowerInfer", "PI+TS"]

print(f"\n{'Model':<18} {'Config':<14} {'Median t/s':>10} {'Speedup':>8} {'Accept':>8}")
print("=" * 65)

all_results = {}
for model_name, model_path in MODELS.items():
    model_results = {}
    for config in CONFIGS:
        tps_list = []
        accept_list = []
        for i in range(N_ITERS):
            result = bench_config(config, model_path)
            if isinstance(result, tuple):
                tps_list.append(result[0])
                accept_list.append(result[1])
            else:
                tps_list.append(result)
        median_tps = statistics.median(tps_list)
        median_accept = statistics.median(accept_list) if accept_list else None
        model_results[config] = {"tps": median_tps, "accept": median_accept}

    baseline_tps = model_results["Baseline"]["tps"]
    for config in CONFIGS:
        r = model_results[config]
        sp = f"{r['tps']/baseline_tps:.2f}x"
        acc = f"{r['accept']:.0%}" if r['accept'] is not None else "—"
        print(f"{model_name:<18} {config:<14} {r['tps']:>10.2f} {sp:>8} {acc:>8}")

    all_results[model_name] = model_results
    print()

# Summary
print("=" * 65)
print("SPEEDUP SUMMARY (vs Baseline)")
print("=" * 65)
for model_name, model_results in all_results.items():
    baseline_tps = model_results["Baseline"]["tps"]
    print(f"\n{model_name}:")
    for config in CONFIGS[1:]:
        r = model_results[config]
        acc = f" (accept: {r['accept']:.0%})" if r['accept'] is not None else ""
        print(f"  {config:<14}: {r['tps']/baseline_tps:.2f}x ({r['tps']:.1f} t/s){acc}")

print("\nDone.")
