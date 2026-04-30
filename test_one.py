"""Quality test — direct execution, one model at a time. No subprocess."""
import os, time, sys
os.environ['LLAMA_LOG'] = '0'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibeblade.llama_backend import LlamaCppBackend
from vibeblade.speculative import SpeculativeBackend

MODEL = sys.argv[1]  # path
MNAME = sys.argv[2]  # display name

PROMPTS = [
    "Explain why the sky is blue in simple terms.",
    "Write a short poem about programming.",
    "What are three benefits of drinking water?",
]

print(f"\n===== {MNAME} =====", flush=True)
for i, prompt in enumerate(PROMPTS):
    print(f"\n--- Prompt {i+1}: {prompt[:40]}... ---", flush=True)
    base_text = ""; base_tps = 0
    
    # Baseline
    try:
        b = LlamaCppBackend(); b.load(MODEL)
        t0 = time.time()
        r = b.generate(prompt, max_tokens=64, temperature=0.0, add_bos=True)
        dt = time.time() - t0
        base_text = r.text.strip()
        base_tps = r.tokens_per_second
        print(f"[llama.cpp] {len(r.tokens)} toks, {base_tps:.2f} t/s, stop={r.stop_reason}", flush=True)
        print(f">>> {base_text[:350]}", flush=True)
        b.free()
    except Exception as e:
        print(f"[llama.cpp] ERROR: {e}", flush=True)
    
    # Spec+TurboSparse
    try:
        s = SpeculativeBackend(); s.load(MODEL); s.set_turbosparse(True, threshold=0.05)
        t0 = time.time()
        r = s.generate(prompt, max_tokens=64, temperature=0.0, add_bos=True, speculative=True)
        dt = time.time() - t0
        spec_text = r.text.strip()
        spec_tps = r.tokens_per_second
        speedup = f"{spec_tps/base_tps:.1f}x" if base_tps > 0 else "N/A"
        accept = getattr(r, "speculative_acceptance_rate", "N/A")
        print(f"[Spec+TS]  {len(r.tokens)} toks, {spec_tps:.2f} t/s ({speedup}) accept={accept}, stop={r.stop_reason}", flush=True)
        print(f">>> {spec_text[:350]}", flush=True)
        s.free()
    except Exception as e:
        print(f"[Spec+TS]  ERROR: {e}", flush=True)
        spec_text = ""
    
    ok = len(base_text) > 30 and len(spec_text) > 30
    print(f"[quality]  {'✓ COHERENT' if ok else '⚠ SHORT/GARBLED'}", flush=True)

print("\nDONE", flush=True)
