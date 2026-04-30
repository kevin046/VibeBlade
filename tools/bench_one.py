#!/usr/bin/env python3
"""Quick single-config benchmark with debug."""
import sys, os, time, gc
sys.path.insert(0, '/home/ubuntu/VibeBlade')
os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/VibeBlade/cpp/build'

from vibeblade.llama_backend import LlamaCppBackend, _helper

MODEL = "/home/ubuntu/VibeBlade/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
PROMPT = "The capital of France is"
N_TOKENS = 32

LABEL = sys.argv[1] if len(sys.argv) > 1 else "test"
MLOCK = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
TS = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
PI = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
TS_TH = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01

gc.collect()
print(f"[{LABEL}] Loading...", flush=True)
_helper.override_model_params(1, int(MLOCK), 0)
b = LlamaCppBackend()
b.load(MODEL, n_ctx=512, n_threads=4)
print(f"[{LABEL}] Model loaded. TS={TS} PI={PI}", flush=True)
b.set_turbosparse(TS, threshold=TS_TH)
b.set_powerinfer(PI, hot_budget=0.1)

print(f"[{LABEL}] Warmup...", flush=True)
_ = b.generate(PROMPT, max_tokens=2, temperature=0.0)
print(f"[{LABEL}] Running {N_TOKENS} tokens...", flush=True)

t0 = time.time()
out = b.generate(PROMPT, max_tokens=N_TOKENS, temperature=0.0)
elapsed = time.time() - t0

n_gen = len(out.tokens)
tps = n_gen / elapsed if elapsed > 0 else 0
print(f"[{LABEL}] tok={n_gen} t/s={tps:.3f} dec={out.time_decode:.2f}s total={elapsed:.1f}s", flush=True)
del b
gc.collect()
