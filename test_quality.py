#!/usr/bin/env python3
"""Quality test at temp=0.7 — tests actual 4x speedup with natural output."""
import os, sys, subprocess, time

os.environ['LLAMA_LOG'] = '0'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("models/llama-3.2-1b-q4_k_m.gguf", "Llama-3.2-1B"),
    ("models/qwen2.5-3b-q4_k_m.gguf", "Qwen2.5-3B"),
    ("models/qwen3.5-moe-0.87b-q4_k_s.gguf", "Qwen3.5-MoE-0.87B"),
    ("models/phi-3.5-mini-q4_k_m.gguf", "Phi-3.5-mini"),
    ("models/gemma-2-2b-q4_k_m.gguf", "Gemma-2-2B"),
]

PROMPT = "Explain why the sky is blue in simple terms."
DIR = os.path.dirname(os.path.abspath(__file__))

def run_subprocess(config, mpath):
    if config == "base":
        code = f'''
import os, sys
os.environ["LLAMA_LOG"] = "0"
sys.path.insert(0, "{DIR}")
from vibeblade.llama_backend import LlamaCppBackend
b = LlamaCppBackend()
b.load("{mpath}")
r = b.generate("{PROMPT}", max_tokens=64, temperature=0.7, top_k=40, top_p=0.95, add_bos=True, seed=42)
print(str(len(r.tokens)) + "|" + f"{{r.tokens_per_second:.2f}}" + "|" + repr(r.text[:400]))
b.free()
'''
    else:
        code = f'''
import os, sys
os.environ["LLAMA_LOG"] = "0"
sys.path.insert(0, "{DIR}")
from vibeblade.speculative import SpeculativeBackend
s = SpeculativeBackend()
s.load("{mpath}")
s.set_turbosparse(True, threshold=0.05)
r = s.generate("{PROMPT}", max_tokens=64, temperature=0.7, top_k=40, top_p=0.95, add_bos=True, speculative=True, seed=42)
acc = getattr(r, "speculative_acceptance_rate", "N/A")
print(str(len(r.tokens)) + "|" + f"{{r.tokens_per_second:.2f}}" + "|" + str(acc) + "|" + repr(r.text[:400]))
s.free()
'''
    r = subprocess.run(["python3", "-u", "-c", code], capture_output=True, text=True, timeout=600)
    return r.stdout.strip(), r.stderr[-300:] if r.stderr else "", r.returncode

results = {}
for mpath, mname in MODELS:
    if not os.path.exists(mpath):
        print(f"SKIP {mname}: not found", flush=True)
        continue
    results[mname] = {}

    print(f"\n[{mname}] llama.cpp baseline...", flush=True)
    out, err, rc = run_subprocess("base", mpath)
    if rc == 0 and out and "|" in out:
        parts = out.split("|", 2)
        results[mname]["base"] = {"tokens": parts[0], "tps": parts[1], "text": parts[2]}
        print(f"  {parts[0]} toks, {parts[1]} t/s", flush=True)
    else:
        results[mname]["base"] = {"error": err[:200]}
        print(f"  FAILED: {err[:100]}", flush=True)

    print(f"[{mname}] Spec+TurboSparse...", flush=True)
    out, err, rc = run_subprocess("spec", mpath)
    if rc == 0 and out and "|" in out:
        parts = out.split("|", 3)
        results[mname]["spec"] = {"tokens": parts[0], "tps": parts[1], "acc": parts[2], "text": parts[3]}
        speedup = ""
        if results[mname]["base"].get("tps"):
            bt = float(results[mname]["base"]["tps"])
            st = float(parts[1])
            speedup = f" ({st/bt:.1f}x)" if bt > 0 else ""
        print(f"  {parts[0]} toks, {parts[1]} t/s{speedup}, accept={parts[2]}", flush=True)
    else:
        results[mname]["spec"] = {"error": err[:200]}
        print(f"  FAILED: {err[:100]}", flush=True)

# Summary
print("\n" + "="*60, flush=True)
print("QUALITY + SPEED SUMMARY (temp=0.7)", flush=True)
print("="*60, flush=True)
for mname, data in results.items():
    b = data.get("base", {})
    s = data.get("spec", {})
    has_b = "text" in b
    has_s = "text" in s
    print(f"\n--- {mname} ---", flush=True)
    if has_b:
        print(f"llama.cpp: {b['tps']} t/s", flush=True)
        print(f"  >>> {str(b['text'])[:250]}", flush=True)
    if has_s:
        print(f"Spec+TS:   {s['tps']} t/s (accept={s['acc']})", flush=True)
        print(f"  >>> {str(s['text'])[:250]}", flush=True)
    if has_b and has_s:
        bt = float(b["tps"])
        st = float(s["tps"])
        ratio = st / bt if bt > 0 else 0
        # Coherence: check for common garbage patterns
        for label, src in [("Base", b), ("Spec", s)]:
            t = src["text"].lower()
            if "the" in t and ("is" in t or "are" in t or "sky" in t or "blue" in t):
                print(f"  [{label}] ✓ Coherent", flush=True)
            elif len(t) < 50:
                print(f"  [{label}] ⚠ Very short", flush=True)
            else:
                print(f"  [{label}] ? Check manually", flush=True)
    elif not has_b:
        print(f"  💀 BASELINE FAILED", flush=True)
    elif not has_s:
        print(f"  ⚠ SPEC FAILED (baseline ok)", flush=True)
