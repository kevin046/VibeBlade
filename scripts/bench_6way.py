#!/usr/bin/env python3
"""Clean 6-config benchmark with proper state isolation.

Usage:
  python scripts/bench_6way.py /path/to/model.gguf

Sets C argtypes BEFORE any _helper calls. Resets all optimizations
before each config. Reports baseline-relative speedups.

Tests: Baseline, TurboSparse, Speculative, Spec+TS, PowerInfer, PI+TS
"""
import sys, os, ctypes, gc, time

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

# Set helper argtypes BEFORE any use
_helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_helper.override_model_params(1, 0, 0)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/VibeBlade/models/tinyllama-1.1b-q4km.gguf"
PROMPT = "The quick brown fox jumps over the lazy dog. In the year 2025, artificial intelligence"
MAX_TOKENS = 16
N_THREADS = 4

def disable_all():
    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False, 0.1)
    _lib.rotorquant_set_enabled(False)

def run_bench(name, fn):
    disable_all()
    gc.collect()
    r = fn()
    return r

# Config implementations
def fn_baseline():
    b = LlamaCppBackend()
    b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    b.free(); gc.collect()
    return {'tok': len(out.tokens), 'tps': out.tokens_per_second}

def fn_turbosparse():
    _lib.turbosparse_set_enabled(True)
    _lib.turbosparse_set_threshold(0.05)
    b = LlamaCppBackend()
    b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    b.free(); gc.collect()
    return {'tok': len(out.tokens), 'tps': out.tokens_per_second}

def fn_speculative():
    spec = SpeculativeBackend(draft_n=4, draft_max=5)
    spec.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    out = spec.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
    st = spec.spec_stats
    spec.free(); gc.collect()
    return {'tok': len(out.tokens), 'tps': out.tokens_per_second,
            'accept': st.n_draft_accepted, 'drafted': st.n_draft_generated,
            'rate': f"{st.acceptance_rate:.0%}"}

def fn_spec_ts():
    spec = SpeculativeBackend(draft_n=4, draft_max=5)
    spec.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    spec.set_turbosparse(True, threshold=0.05)
    out = spec.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, speculative=True)
    st = spec.spec_stats
    spec.free(); gc.collect()
    return {'tok': len(out.tokens), 'tps': out.tokens_per_second,
            'accept': st.n_draft_accepted, 'drafted': st.n_draft_generated,
            'rate': f"{st.acceptance_rate:.0%}"}

def fn_powerinfer():
    _lib.powerinfer_set_enabled(True, 0.1)
    b = LlamaCppBackend()
    b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    b.free(); gc.collect()
    return {'tok': len(out.tokens), 'tps': out.tokens_per_second}

def fn_pi_ts():
    _lib.powerinfer_set_enabled(True, 0.1)
    _lib.turbosparse_set_enabled(True)
    _lib.turbosparse_set_threshold(0.05)
    b = LlamaCppBackend()
    b.load(MODEL, n_ctx=256, n_threads=N_THREADS)
    b._set_sampler(temperature=0.0)
    out = b.generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    b.free(); gc.collect()
    return {'tok': len(out.tokens), 'tps': out.tokens_per_second}

CONFIGS = [
    ('Baseline',    fn_baseline),
    ('TurboSparse', fn_turbosparse),
    ('Speculative', fn_speculative),
    ('Spec+TS',     fn_spec_ts),
    ('PowerInfer',  fn_powerinfer),
    ('PI+TS',       fn_pi_ts),
]

print(f"\n{'Config':<16} {'Tok':>4} {'t/s':>8} {'Speedup':>7} {'Notes'}")
print("=" * 70)

results = {}
for name, fn in CONFIGS:
    r = run_bench(name, fn)
    sp = f"{r['tps']/results['Baseline']['tps']:.2f}x" if 'Baseline' in results else "—"
    notes = f"accept={r.get('rate','N/A')} ({r.get('accept',0)}/{r.get('drafted',0)})" if 'rate' in r else ''
    print(f"{name:<16} {r['tok']:>4} {r['tps']:>8.3f} {sp:>7}  {notes}")
    results[name] = r

print()
if 'Baseline' in results:
    base = results['Baseline']['tps']
    print("Speedup table (vs Baseline):")
    for name, r in results.items():
        if name != 'Baseline':
            print(f"  {name:<16}: {r['tps']/base:.2f}x")
print("\nDone.")