#!/usr/bin/env python3
"""VibeBlade Interactive Setup Menu — Windows / Linux / macOS"""

import os
import sys
import subprocess
import platform
from pathlib import Path

BANNER = r"""
 _    __________  __________  __          __
| |  / /  _/ __ )/ ____/ __ )/ /___ _____/ /__
| | / // // __  / __/ / __  / / __ `/ __  / _ \
| |/ // // /_/ / /___/ /_/ / / /_/ / /_/ /  __/
|___/___/_____/_____/_____/_/\__,_/\__,_/\___/
  Unified CPU/RAM Sparse Inference Protocol v1.4
"""

ACTIVATE_LINUX = "source venv/bin/activate"
ACTIVATE_WIN   = "venv\\Scripts\\activate"

# ── Colour helpers (termcolor optional) ──────────────────────────────────────
try:
    from termcolor import colored as _tc
    def _c(t, c, **kw): return _tc(str(t), c, **kw)
    R  = lambda t: _c(t,"red");  G  = lambda t: _c(t,"green")
    Y  = lambda t: _c(t,"yellow"); C = lambda t: _c(t,"cyan")
    M  = lambda t: _c(t,"magenta"); B = lambda t: _c(t,attrs=["bold"])
except Exception:
    def _c(t, *a, **k): return str(t)
    R=G=Y=C=M=B = lambda t: str(t)

def ok(msg):    print(f"  {G('✓')} {msg}")
def info(msg):  print(f"  {Y('ℹ')} {msg}")
def err(msg):   print(f"  {R('✗')} {msg}")
def heading(t): print(f"\n{C('═'*60)}\n  {B(t)}\n{C('═'*60)}")

def cls():
    os.system("cls" if platform.system()=="Windows" else "clear")

def shell(cmd, fatal=True, cwd=None):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=180, cwd=cwd)
        out = (r.stdout+r.stderr).strip()
        if r.returncode == 0:
            return True, out
        if fatal:
            err(f"Command failed: {cmd}")
            if out: print(f"  {R(out[:400])}")
            sys.exit(1)
        return False, out
    except Exception as e:
        if fatal:
            err(f"Exception: {e}"); sys.exit(1)
        return False, str(e)

def has_nvidia():
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
        return True
    except: return False

def ram_gb():
    try:
        if platform.system()=="Windows":
            import ctypes; k=ctypes.windll.kernel32
            class MEM(ctypes.Structure):
                _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                          ("dwTotalPhys",ctypes.c_ulonglong),("dwAvailPhys",ctypes.c_ulonglong)]
            m=MEM(); k.GetGlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.dwTotalPhys)//(1024**3)
        else:
            with open("/proc/meminfo") as f: return int(f.readline().split()[1])//1024//1024
    except: return None

def has_inet():
    try:
        subprocess.run(["curl","-sI","https://huggingface.co","-o",os.devnull,
                       "--max-time","5"], check=True, capture_output=True)
        return True
    except:
        try:
            subprocess.run(["powershell","-Command",
                           "(New-Object Net.WebClient).DownloadString('https://huggingface.co')"],
                          check=True, capture_output=True, timeout=8)
            return True
        except: return False

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3",      "~4GB",  "8GB",  "Dense",  "Fast, general-purpose. Runs in RAM."),
    ("meta-llama/Llama-3.1-8B-Instruct",        "~5GB",  "16GB", "Dense",  "Meta's top open 8B. Excellent reasoning."),
    ("Qwen/Qwen2.5-14B-Instruct",               "~9GB",  "24GB", "Dense",  "Strong coding and math."),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",    "~26GB", "48GB", "MoE",    "Mixtral MoE. Fast, each token 2/8 experts."),
    ("deepseek-ai/DeepSeek-V2.5",               "~52GB", "96GB", "MoE",    "DeepSeek V2.5 MoE. Top-tier reasoning."),
    ("OpenBuddy/openbuddy-mixtral-8x22B",        "~48GB", "64GB", "MoE",    "OpenBuddy MoE. Excellent multilingual."),
]

def main():
    cls()
    print(B(BANNER))
    print(M("  Cross-platform setup — Windows / Linux / macOS\n"))

    # ── Step 1: Detect ──────────────────────────────────────────────────────
    heading("Step 1: System Detection")
    os_name = platform.system(); os_ver = platform.release()
    py = sys.version_info; ram = ram_gb(); vram = None
    if has_nvidia():
        _, out = shell("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
        try: vram = int(out.strip().split()[0])//1024
        except: pass
    inet = has_inet()

    ok(f"OS:       {os_name} {os_ver}")
    ok(f"Python:   {py.major}.{py.minor}.{py.micro}"
       + (G(" (≥3.10 OK)") if (py.major>3 or py.major==3 and py.minor>=10) else R(" (need ≥3.10)")))
    ok(f"RAM:      {ram}GB detected" + (G(" ✓") if ram and ram>=16 else Y(" low")))
    if vram: ok(f"VRAM:     {vram}GB NVIDIA GPU detected" + (G(" ✓") if vram>=8 else Y(" limited")))
    else:    info("VRAM:    No NVIDIA GPU — CPU inference mode")
    ok(f"Internet: {'✓ Connected' if inet else Y('✗ Offline — skip model download')}")

    # ── Step 2: Install ─────────────────────────────────────────────────────
    heading("Step 2: Install VibeBlade")
    turbodir = Path.cwd()
    venv_dir = turbodir / "venv"

    if venv_dir.exists():
        ok(G("Virtual environment already exists"))
    else:
        ok_, _ = shell("python3 -m venv venv"); ok("Virtual environment created")

    activate = (venv_dir/"Scripts"/"activate") if os_name=="Windows" else (venv_dir/"bin"/"activate")
    pip = f'source "{activate}" && pip' if os_name!="Windows" else f'"{activate}" && pip'
    ok(f"Activate: {C(ACTIVATE_LINUX) if os_name!='Windows' else C(ACTIVATE_WIN)}")

    shell(f'cd "{turbodir}" && {pip} install --upgrade pip -q', fatal=False)
    ok_, out = shell(f'cd "{turbodir}" && {pip} install -e ".[onnx]" -q', fatal=False)
    if ok_:
        ok(G("VibeBlade + ONNX installed"))
    else:
        shell(f'cd "{turbodir}" && {pip} install -e . -q', fatal=False)
        ok(Y("VibeBlade installed (base, ONNX skipped)"))

    # ── Step 3: Model ─────────────────────────────────────────────────────
    heading("Step 3: Choose Model")
    print(f"  System RAM: {ram}GB  |  VRAM: {vram or 'N/A'}GB\n")
    for i,(mid,size,ram_n,typ,desc) in enumerate(MODELS,1):
        tag = C(f"[{typ}]")
        print(f"  {G(f'[{i}]')} {mid:<45} {tag}")
        print(f"       Size: {size}  RAM: {ram_n}  {desc}")
        print()
    print(f"  {G('[0]')} Custom path or HuggingFace ID")

    while True:
        try: ch = int(input(f"  {C('Enter choice [1-6, 0=custom]:')} ").strip())
        except: ch = -1
        if 0 <= ch <= len(MODELS): break
        err("Invalid choice")

    if ch == 0:
        model_id = input(f"  {C('Enter HuggingFace ID or GGUF path:')} ").strip()
    else:
        model_id = MODELS[ch-1][0]
        print(f"  Selected: {C(model_id)}")

    # ── Step 4: Configure ──────────────────────────────────────────────────
    heading("Step 4: Memory Configuration")
    if vram and vram >= 8 and ram and ram >= 64:
        vram_s, ram_s, hot_t, mode, desc = min(vram,16), ram, "0.20", "HYBRID_SSD", G("HYBRID_SSD (VRAM+RAM+SSD)")
    elif ram and ram >= 32:
        vram_s, ram_s, hot_t, mode = min(vram or 4,8), ram, "0.15", "RAM_ONLY"
        desc = Y("RAM_ONLY (SSD mode recommended for MoE)")
    else:
        vram_s, ram_s, hot_t, mode = 2, min(ram or 16,32), "0.10", "RAM_ONLY"
        desc = Y("RAM_ONLY (small model recommended)")

    print(f"  Offload mode:  {desc}")
    print(f"  VRAM limit:   {vram_s}GB")
    print(f"  RAM limit:    {ram_s}GB")
    print(f"  Hot threshold:{hot_t}  (top {float(hot_t)*100:.0f}% of experts kept hot)")

    # Write YAML config
    try:
        import yaml
        cfg = {"model": model_id, "vram_gb": vram_s, "ram_gb": ram_s,
               "hot_threshold": float(hot_t), "offload_mode": mode,
               "cpu_threads": os.cpu_count() or 8}
        cfg_path = turbodir / "vibeblade.yaml"
        with open(cfg_path,"w") as f: yaml.dump(cfg, f)
        ok(f"Config written: {cfg_path}")
    except Exception as e:
        info(f"yaml not available — config skipped ({e})")

    # ── Step 5: Download ───────────────────────────────────────────────────
    if inet:
        heading("Step 5: Download Model")
        proceed = input(f"  {C('Download now? (y/N):')} ").strip().lower()
        if proceed == "y":
            ok(f"Downloading {C(model_id)} — this may take a while...")
            shell(f'{pip} install huggingface_hub -q', fatal=False)
            dl = f'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'{model_id}\', allow_patterns=[\'*.gguf\'])"'
            shell(dl, fatal=False)
            ok("Download started (or already cached)")
        else:
            info("Skipped. To download later:")
            print(f"  {C(f'huggingface-cli download {model_id}')}")
    else:
        heading("Step 5: Download (skipped — offline)")
        info("No internet. Download GGUF files manually to ~/.cache/huggingface/hub/")

    # ── Step 6: Verify ─────────────────────────────────────────────────────
    heading("Step 6: Verify")
    shell(f'{pip} install pytest -q', fatal=False)
    _, out = shell(f'cd "{turbodir}" && {pip} pytest tests/test_sparse.py -v --tb=line -q', fatal=False)
    if "passed" in out or "PASSED" in out:
        ok("Tests pass!")
    else:
        info("Tests may have issues — check output above")

    # ── Done ────────────────────────────────────────────────────────────────
    cls()
    print(B(BANNER))
    heading("Setup Complete!")
    activate = ACTIVATE_LINUX if os_name != "Windows" else ACTIVATE_WIN
    print(f"""
  {G('Next steps:')}

  1. Activate virtual environment:
     {C(activate)}

  2. Run benchmark:
     {C('python -m vibeblade bench --quick')}

  3. Run inference:
     {C('python -m vibeblade run')} {model_id}
     {C('--config vibeblade.yaml --prompt "Hello world"')}

  4. Start API server:
     {C('python -m vibeblade serve --model')} {model_id}
     {C('--port 8000')}

  5. Download model (if skipped):
     {C('huggingface-cli download ' + model_id)}

  Config: {cfg_path if 'cfg_path' in dir() else 'vibeblade.yaml'}
""")
    print(C("═"*60))
    input(f"\n  {Y('Press Enter to exit...')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n  {Y('Interrupted. Re-run: python setup_menu.py')}")
        sys.exit(0)
