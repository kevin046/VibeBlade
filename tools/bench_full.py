#!/usr/bin/env python3
"""Full benchmark: Baseline vs PowerInfer vs TurboSparse vs Speculative vs PI+Spec on all models."""
import sys, os, ctypes, gc, time
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = 'cpp/build'

_lib = ctypes.CDLL('cpp/build/libllama.so')
from vibeblade.llama_backend import _helper, LlamaCppBackend
from vibeblade.speculative import SpeculativeBackend

# Declare argtypes
_lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
_lib.turbosparse_is_enabled.restype = ctypes.c_bool
_lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
_lib.turbosparse_get_threshold.restype = ctypes.c_float
_lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool, ctypes.c_float]
_lib.powerinfer_is_enabled.restype = ctypes.c_bool
_lib.rotorquant_set_enabled.argtypes = [ctypes.c_bool]
_lib.rotorquant_is_enabled.restype = ctypes.c_bool

MODELS = {
    'Qwen2.5-0.5B': '/home/ubuntu/VibeBlade/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    'Llama-3.2-1B':  '/home/ubuntu/VibeBlade/models/Llama-3.2-1B-Instruct-Q4_K_S.gguf',
    'Qwen3.5-MoE':   '/home/ubuntu/VibeBlade/models/Qwen3.5-MoE-0.87B-Instruct-Q4_K_M.gguf',
}
PROMPT = "The quick brown fox jumps over the lazy dog. In the year 2025, artificial intelligence"
MAX_TOKENS = 16
N_THREADS = 4

def reset_all():
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False, 0.1)
    _lib.rotorquant_set_enabled(False)
    _helper.override_model_params(1, 0, 0)

def bench_normal(model_path):
    reset_all()
    b = LlamaCppBackend()
    b.load(model_path, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    t0 = time.time()
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    elapsed = time.time() - t0
    result = {'tok': len(out.tokens), 'tps': out.tokens_per_second, 'time': elapsed}
    b.free(); del b; gc.collect()
    return result

def bench_pi(model_path):
    reset_all()
    _lib.powerinfer_set_enabled(True, 0.1)
    b = LlamaCppBackend()
    b.load(model_path, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    t0 = time.time()
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    elapsed = time.time() - t0
    result = {'tok': len(out.tokens), 'tps': out.tokens_per_second, 'time': elapsed}
    b.free(); del b; gc.collect()
    return result

def bench_ts(model_path):
    reset_all()
    _lib.turbosparse_set_enabled(True)
    _lib.turbosparse_set_threshold(0.05)
    b = LlamaCppBackend()
    b.load(model_path, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    t0 = time.time()
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    elapsed = time.time() - t0
    result = {'tok': len(out.tokens), 'tps': out.tokens_per_second, 'time': elapsed}
    b.free(); del b; gc.collect()
    return result

def bench_spec(model_path):
    reset_all()
    spec = SpeculativeBackend(draft_n=4, draft_max=5)
    spec.load(model_path, n_ctx=256, n_threads=N_THREADS)
    t0 = time.time()
    out = spec.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
    elapsed = time.time() - t0
    st = spec.spec_stats
    result = {
        'tok': len(out.tokens), 'tps': out.tokens_per_second, 'time': elapsed,
        'accept': st.n_draft_accepted, 'drafted': st.n_draft_generated,
        'rate': f"{st.acceptance_rate:.0%}",
    }
    spec.free(); del spec; gc.collect()
    return result

def bench_pi_spec(model_path):
    reset_all()
    _lib.powerinfer_set_enabled(True, 0.1)
    spec = SpeculativeBackend(draft_n=4, draft_max=5)
    spec.load(model_path, n_ctx=256, n_threads=N_THREADS)
    t0 = time.time()
    out = spec.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
    elapsed = time.time() - t0
    st = spec.spec_stats
    result = {
        'tok': len(out.tokens), 'tps': out.tokens_per_second, 'time': elapsed,
        'accept': st.n_draft_accepted, 'drafted': st.n_draft_generated,
        'rate': f"{st.acceptance_rate:.0%}",
    }
    spec.free(); del spec; gc.collect()
    return result

CONFIGS = [
    ('Baseline',      bench_normal),
    ('PowerInfer',    bench_pi),
    ('TurboSparse',   bench_ts),
    ('Speculative',   bench_spec),
    ('PI+Spec',       bench_pi_spec),
]

print(f"{'Model':<16} {'Config':<14} {'Tok':>4} {'t/s':>8} {'Time':>6} {'Notes'}")
print("-" * 80)

for model_name, model_path in MODELS.items():
    print(f"\n>>> {model_name}")
    for config_name, bench_fn in CONFIGS:
        try:
            r = bench_fn(model_path)
            notes = ""
            if 'rate' in r:
                notes = f"accept={r['rate']} ({r['accept']}/{r['drafted']})"
            print(f"{model_name:<16} {config_name:<14} {r['tok']:>4} {r['tps']:>8.3f} {r['time']:>5.1f}s {notes}")
        except Exception as e:
            print(f"{model_name:<16} {config_name:<14} ERROR: {e}")
        reset_all()

print("\nDone.")
