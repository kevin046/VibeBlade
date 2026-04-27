#!/usr/bin/env python3
"""
VibeBlade Interactive Setup Wizard
Interactive TTY setup wizard for Windows / Linux / macOS

Developed by VibeDrift Inc.
https://vibedrift.com | https://github.com/kevin046/VibeBlade

Requirements:
    pip install prompt_toolkit rich

Usage:
    python setup_wizard.py
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path

# ── Rich for pretty output ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    console = Console()
except ImportError:
    console = None
    def rich_assert(s): pass

def print(msg="", style=None):
    if console:
        console.print(msg, style=style or None)
    else:
        print(msg)

def panel(title, body, style="bold cyan"):
    if console:
        console.print(Panel(body, title=title, style=style, expand=False))
    else:
        print(f"=== {title} ===\n{body}")

def table(headers, rows, title=""):
    if not console:
        return
    t = Table(title=title, show_header=True, header_style="bold cyan")
    for h in headers:
        t.add_column(h)
    for row in rows:
        t.add_row(*[str(c) for c in row])
    console.print(t)

# ── Prompt_toolkit for interactive input ─────────────────────────────────────
HAS_PT = False
try:
    from prompt_toolkit.shortcuts import (
        radiolist_dialog, checkboxlist_dialog,
        input_dialog, message_dialog, confirm,
    )
    from prompt_toolkit.styles import Style
    HAS_PT = True
except ImportError:
    pass

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = Style.from_dict({
    "question":     "fg:cyan bold",
    "answer":       "fg:yellow",
    "warning":      "fg:yellow",
    "error":        "fg:red bold",
    "success":      "fg:green bold",
    "panel.border": "fg:cyan",
    "table.header": "fg:cyan bold",
}) if HAS_PT else None

PROMPT_SYMBOL = "[?]"
PROMPT_STYLE  = "class:question"

# ── System Detection ───────────────────────────────────────────────────────────
def _ensure_psutil():
    """Try to import psutil; if missing, auto-install it."""
    try:
        import psutil  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import importlib.metadata as _im
        # Only auto-install if running from an installed package (not a random script)
        _im.distribution("vibeblade")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "psutil>=5.9.0"],
            check=False, capture_output=True, timeout=30,
        )
        import psutil  # noqa: F401
        return True
    except Exception:
        pass
    return False

def get_ram_gb():
    """Detect total physical RAM in GB. Returns None on failure."""
    system = platform.system()

    # Method 0: psutil — works on all platforms, most reliable
    if _ensure_psutil():
        try:
            import psutil
            return int(psutil.virtual_memory().total) // (1024**3)
        except Exception:
            pass

    if system == "Windows":
        # Method 1: wmic (most reliable, available on all Windows)
        try:
            out = subprocess.check_output(
                ["wmic", "OS", "get", "TotalVisibleMemorySize", "/value"],
                text=True, timeout=5,
            )
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("TotalVisibleMemorySize="):
                    kb = int(line.split("=", 1)[1])
                    return kb // 1024
        except Exception:
            pass

        # Method 2: ctypes GetGlobalMemoryStatusEx
        try:
            import ctypes
            k = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength",        ctypes.c_ulong),
                    ("dwMemoryLoad",    ctypes.c_ulong),
                    ("ullTotalPhys",    ctypes.c_ulonglong),
                    ("ullAvailPhys",    ctypes.c_ulonglong),
                    ("ullTotalPageFile",ctypes.c_ulonglong),
                    ("ullAvailPageFile",ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if k.GetGlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.ullTotalPhys) // (1024**3)
        except Exception:
            pass

        # Method 3: PowerShell CIM
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True, timeout=5,
            )
            return int(out.strip()) // (1024**3)
        except Exception:
            pass

        return None

    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
            )
            return int(out.strip()) // (1024**3)
        except Exception:
            return None

    else:  # Linux
        try:
            with open("/proc/meminfo") as f:
                return int(f.readline().split()[1]) // 1024 // 1024
        except Exception:
            return None

def get_vram_gb():
    """Detect GPU VRAM on NVIDIA (Linux/Windows), Apple Silicon (macOS), or AMD."""
    system = platform.system()
    # NVIDIA — nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        total, used = out.strip().split(",")
        return int(total.strip()) // 1024, int(used.strip()) // 1024
    except Exception:
        pass
    # Apple Silicon — Metal shared memory
    if system == "Darwin":
        try:
            # system_profiler is slow; use ioreg for GPU memory class
            out = subprocess.check_output(
                ["ioreg", "-r", "-c", "AppleARMGraphics",
                 "-d", "2"],
                text=True, timeout=5,
            )
            # Parse "gpu-memory-size" = 0x<bytes>
            for line in out.splitlines():
                if "gpu-memory-size" in line and "= " in line:
                    hexbytes = line.split("= ")[-1].strip()
                    if hexbytes.startswith("0x"):
                        total_bytes = int(hexbytes, 16)
                        return total_bytes // (1024**3), None
        except Exception:
            pass
        # Fallback: read from system_profiler SPDisplaysDataType
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                text=True, timeout=10,
            )
            import json as _json
            data = _json.loads(out)
            for gpu_info in data.get("SPDisplaysDataType", []):
                metal_str = gpu_info.get("Metal", "")
                if "GB" in metal_str:
                    gb = int(metal_str.split()[0])
                    return gb, None
        except Exception:
            pass
    # AMD — rocm-smi (Linux)
    if system == "Linux":
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                text=True, timeout=5,
            )
            for line in out.splitlines():
                if "VRAM" in line and "Used" in line:
                    parts = [p.strip() for p in line.split(",")]
                    # columns: GPU, VRAM Total, VRAM Used
                    total = int(parts[1].replace("MB", "").strip()) // 1024
                    used = int(parts[2].replace("MB", "").strip()) // 1024
                    return total, used
        except Exception:
            pass
    return None, None

def get_gpu_name():
    """Detect GPU name on NVIDIA, Apple Silicon, or AMD."""
    system = platform.system()
    # NVIDIA
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return out.strip()
    except Exception:
        pass
    # Apple Silicon
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                text=True, timeout=10,
            )
            import json as _json
            data = _json.loads(out)
            for gpu_info in data.get("SPDisplaysDataType", []):
                chip = gpu_info.get("chipset-model", "")
                if chip:
                    return chip
                name = gpu_info.get("sppci_model", "")
                if name:
                    return name
        except Exception:
            pass
    # AMD — rocm-smi
    if system == "Linux":
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname"],
                text=True, timeout=5,
            )
            for line in out.splitlines():
                if "Card series" in line or "GPU" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        return parts[-1].strip()
        except Exception:
            pass
    return None

def detect_accel_backend():
    """Auto-detect the best acceleration backend from hardware.

    Returns (extra_suffix, human_name) — extra_suffix like '.[gpu-metal]' or ''.
    """
    system = platform.system()
    gpu = get_gpu_name()

    # Apple Silicon → Metal backend
    if system == "Darwin" and gpu and ("Apple" in gpu or "M1" in gpu or "M2" in gpu
                                       or "M3" in gpu or "M4" in gpu):
        return ".[gpu-metal]", "Apple Metal (CoreML)"

    # NVIDIA GPU → CUDA / ONNX CUDA
    if gpu:
        try:
            subprocess.check_output(["nvidia-smi"], timeout=3,
                                    stderr=subprocess.DEVNULL)
            return ".[grammar]", "NVIDIA CUDA"
        except Exception:
            pass

    # AMD GPU → Vulkan backend (Linux)
    if system == "Linux" and gpu:
        try:
            subprocess.check_output(["rocm-smi"], timeout=3,
                                    stderr=subprocess.DEVNULL)
            return ".[gpu-vulkan]", "AMD ROCm / Vulkan"
        except Exception:
            pass

    # CPU-only — check for AVX-512 or AVX2
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            if "avx512" in cpuinfo.lower():
                return ".[grammar]", "CPU AVX-512 optimized"
            if "avx2" in cpuinfo.lower():
                return ".[grammar]", "CPU AVX2 optimized"
        except Exception:
            pass
    elif system == "Darwin":
        return ".[grammar]", "Apple CPU (NEON)"

    # Fallback — base install, works everywhere
    return "", "Base (NumPy, universal)"

def has_ssd():
    """Detect whether an SSD is available. Returns True/False."""
    system = platform.system()

    # Method 0: psutil — works on all platforms
    if _ensure_psutil():
        try:
            import psutil
            for part in psutil.disk_partitions():
                if part.fstype and part.mountpoint:
                    try:
                        usage = psutil.disk_usage(part.mountpoint)
                        if usage.total > 10**9:  # > 1GB = usable
                            return True
                    except Exception:
                        pass
            return False
        except Exception:
            pass

    if system == "Darwin":
        return True

    if system == "Windows":
        # Method 1: PowerShell Get-PhysicalDisk (Win 8+ / Server 2012+)
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-PhysicalDisk | Select-Object -First 1 "
                 "-ExpandProperty MediaType"],
                text=True, timeout=5,
            )
            media = out.strip().lower()
            if "ssd" in media:
                return True
            if "hdd" in media:
                return False
        except Exception:
            pass

        # Method 2: wmic diskdrive (available on all Windows versions)
        try:
            out = subprocess.check_output(
                ["wmic", "diskdrive", "get", "MediaType"],
                text=True, timeout=5,
            )
            for line in out.splitlines():
                line = line.strip().upper()
                if "SSD" in line or "SOLID STATE" in line:
                    return True
                if "FIXED HARD DISK" in line:
                    # Fixed hard disk media type — likely SSD on modern systems
                    # but could be HDD. Continue to next check.
                    continue
                if "HARD DRIVE" in line:
                    return False
        except Exception:
            pass

        # Method 3: Any fixed disk letter exists = assume SSD on modern Windows
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            if os.path.isdir(f"{letter}:\\"):
                return True

        return False

    else:  # Linux
        for blk in os.listdir("/sys/block"):
            rot_path = f"/sys/block/{blk}/queue/rotational"
            if os.path.isfile(rot_path):
                try:
                    with open(rot_path) as f:
                        if int(f.read().strip()) == 0:
                            return True
                except (ValueError, OSError):
                    pass
        return shutil.disk_usage("/").free > 50 * (1024**3)

def get_disk_free_gb(path=None):
    if path is None:
        path = "C:\\" if platform.system() == "Windows" else "/"
    try:
        return shutil.disk_usage(path).free // (1024**3)
    except Exception:
        return None

def check_python():
    v = sys.version_info
    return v.major > 3 or (v.major == 3 and v.minor >= 10)

def has_inet():
    """Check internet connectivity using platform-appropriate methods."""
    import urllib.request
    import urllib.error
    urls = [
        "https://huggingface.co",
        "https://github.com",
        "https://google.com",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            continue
    return False

def run(args: str | list[str], fatal: bool = False, cwd: str | None = None) -> tuple[bool, str]:
    """Run a command safely without shell=True. Accepts string or list."""
    try:
        cmd_list: list[str]
        if isinstance(args, str):
            import shlex
            cmd_list = shlex.split(args)
        else:
            cmd_list = list(args)
        r = subprocess.run(
            cmd_list,
            shell=False,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=cwd,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:
        return (False, str(e)) if not fatal else (print(f"[ERROR] {e}") or sys.exit(1))

# ── Model Catalogue ────────────────────────────────────────────────────────────
# Format: (repo_id, display_name, type, size_label, ram_min, vram_min, quant_default, description)
# Ordered: MoE first (VibeBlade's strength), then dense.
MODELS = [
    # --- MoE (VibeBlade excels here) ---
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",
     "Mixtral 8×7B ⭐", "MoE", "~26GB", 32, 8, "Q4_K_M",
     "Golden standard MoE. 8 experts, 2 active/token. Runs great with VibeBlade tiering."),
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct",
     "Llama 4 Scout (109B MoE)", "MoE", "~40GB", 48, 8, "Q4_K_M",
     "Meta's Llama 4. 16 experts, fast inference. VibeBlade showcase."),
    ("deepseek-ai/DeepSeek-R1-0528-Q4_K_M-GGUF",
     "DeepSeek R1", "MoE", "~67GB", 64, 8, "Q4_K_M",
     "DeepSeek's reasoning model. Long chain-of-thought. 67B MoE."),
    ("Qwen/Qwen3-235B-A22B",
     "Qwen 3 235B MoE", "MoE", "~90GB", 96, 8, "Q4_K_M",
     "Alibaba's MoE flagship. 22B active params, massive quality."),
    ("meta-llama/Llama-4-Maverick-17B-128E-Instruct",
     "Llama 4 Maverick (400B MoE)", "MoE", "~110GB", 128, 16, "Q4_K_M",
     "Meta's flagship. 128 experts, SOTA open-source quality."),
    ("MiniMaxAI/MiniMax-M2.7-01",
     "MiniMax M2.7 (456B MoE) ⭐", "MoE", "~115GB", 128, 16, "Q4_K_M",
     "VibeBlade showcase. 456B params, best MoE efficiency."),
    # --- Dense (VibeBlade still helps via sparsity + tiering) ---
    ("microsoft/Phi-4-mini-instruct",
     "Phi-4 Mini (3.8B)", "Dense", "~2.5GB", 8, 0, "Q4_K_M",
     "Microsoft. Tiny but punchy — great for edge devices."),
    ("meta-llama/Llama-3.1-8B-Instruct",
     "Llama 3.1 8B", "Dense", "~5GB", 8, 0, "Q4_K_M",
     "Most popular 8B model, excellent all-rounder."),
    ("mistralai/Mistral-7B-Instruct-v0.3",
     "Mistral 7B v0.3", "Dense", "~4GB", 8, 0, "Q4_K_M",
     "Fast & versatile. Battle-tested, huge community."),
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503",
     "Mistral Small 3.1 (24B)", "Dense", "~14GB", 24, 4, "Q4_K_M",
     "Mistral's best small model. Vision + text, excellent quality."),
    ("Qwen/Qwen3-32B",
     "Qwen 3 32B", "Dense", "~19GB", 32, 4, "Q4_K_M",
     "Alibaba's latest. Top-tier coding, math, multilingual."),
    ("google/gemma-3-27b-it",
     "Gemma 3 27B", "Dense", "~16GB", 32, 4, "Q4_K_M",
     "Google's latest Gemma. Strong reasoning, multilingual."),
    ("custom", "✏️  Custom / Browse HuggingFace", "Custom", "?GB", 4, 0, "Q4_K_M",
     "Enter any HuggingFace repo ID or local GGUF path."),
]

# ── Categorized Model Selection ───────────────────────────────────────────────
def recommend_models(ram_gb, vram_gb):
    """Return list of (index, model) filtered to what user can realistically run."""
    # Always include "custom" (last index)
    custom_idx = len(MODELS) - 1
    filtered = []
    for i, m in enumerate(MODELS):
        if m[0] == "custom":
            continue
        if ram_gb and ram_gb >= m[4]:
            filtered.append((i, m))
    # Always append custom option at end
    filtered.append((custom_idx, MODELS[custom_idx]))
    return filtered

# ── Dialog Helpers (prompt_toolkit) ──────────────────────────────────────────
def radio(title, options, default=0):
    """Show a radio-list dialog and return the return_value from the selected option.
    
    Options format: [(display_label, return_value), ...]
    Default is an index into the options list.
    
    Returns return_value from the selected option, or None if cancelled.
    """
    if HAS_PT:
        # radiolist_dialog values format: (value, text) — value is returned
        dialog_values = [(ret, display) for display, ret in options]
        return radiolist_dialog(
            title=title,
            text="Use arrow keys to navigate, Enter to confirm:",
            values=dialog_values,
            default=options[default][1] if default < len(options) else None,
        ).run()
    else:
        print(f"\n  {title}")
        for i, (_ret, label) in enumerate(options):
            print(f"    {i+1}. {label}")
        while True:
            try:
                ch = int(input("  Choice: ").strip())
                if 1 <= ch <= len(options):
                    return options[ch - 1][1]
            except Exception:
                pass

def checkbox(title, options, defaults=None):
    """Show a checkbox dialog and return list of selected values.
    
    Options format: [(display_label, return_value), ...]
    Defaults is a list of return_values to pre-check.
    """
    if HAS_PT:
        # checkboxlist_dialog values format: (value, text)
        dialog_values = [(ret, display) for display, ret in options]
        return checkboxlist_dialog(
            title=title,
            text="Use space to toggle, Enter to confirm:",
            values=dialog_values,
            default_values=defaults or [],
        ).run()
    else:
        print(f"\n  {title} (comma-separated numbers)")
        for i, (_ret, label) in enumerate(options):
            print(f"    {i+1}. {label} {'[default]' if defaults and _ret in defaults else ''}")
        while True:
            try:
                nums = input("  Choices (e.g. 1,3): ").strip()
                selected = [options[int(n) - 1][1] for n in nums.split(",") if n.strip().isdigit()]
                return selected
            except Exception:
                pass

def confirm(title, text="", default=True):
    if HAS_PT:
        msg = title if not text else f"{title}\n{text}"
        return confirm(message=msg)
    else:
        ch = input(f"\n  {title} {'[Y/n]: ' if default else '[y/N]: '}")
        return ch.lower() in ("y","yes") if ch else default

def text_input(title, text="", default=""):
    if HAS_PT:
        return input_dialog(title=title, text=text, default=default).run() or default
    else:
        v = input(f"\n  {title} [{default}]: ").strip()
        return v if v else default

def message(title, text="", style="info"):
    if HAS_PT:
        message_dialog(title=title, text=text,).run()
    else:
        print(f"\n  [{style.upper()}] {title}: {text}")

def spinner(text, coro_fn, *args, **kwargs):
    """Run a coroutine with a spinner."""
    if console:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
            task = p.add_task(text, total=None)
            result = coro_fn(*args, **kwargs)
            p.update(task, completed=True)
            return result
    else:
        print(f"\n  {text}...")
        result = coro_fn(*args, **kwargs)
        print("  Done.")
        return result

# ── Offload Strategy Recommendation ─────────────────────────────────────────
def recommend_offload(ram_gb, vram_gb, model_type, total_experts=8):
    """Return recommended offload config."""
    if vram_gb and vram_gb >= 8:
        vram_s = min(vram_gb, 16)
    else:
        vram_s = 0

    ram_s  = ram_gb or 64
    ssd_en = (ram_gb or 999) < 64 or has_ssd()

    # Expert offloading
    if model_type == "MoE":
        hot_frac  = 0.20  # top 20% of experts in VRAM
        ram_frac  = 0.50  # 50% in RAM
        ssd_frac  = 0.30  # 30% on SSD
    elif model_type == "Dense":
        hot_frac  = 0.10
        ram_frac  = 0.70
        ssd_frac  = 0.20
    else:
        hot_frac, ram_frac, ssd_frac = 0.15, 0.60, 0.25

    return {
        "vram_gb":     vram_s,
        "ram_gb":      ram_s,
        "ssd_enabled": ssd_en,
        "hot_frac":    hot_frac,
        "ram_frac":    ram_frac,
        "ssd_frac":    ssd_frac,
    }

def describe_offload(cfg):
    lines = []
    lines.append(f"  VRAM:  {cfg['vram_gb']}GB allocated")
    lines.append(f"  RAM:   {cfg['ram_gb']}GB allocated")
    lines.append(f"  SSD:   {'Enabled' if cfg['ssd_enabled'] else 'Disabled'}")
    lines.append(f"  Expert split: {cfg['hot_frac']:.0%} hot / {cfg['ram_frac']:.0%} warm / {cfg['ssd_frac']:.0%} cold")
    if cfg['vram_gb'] > 0:
        lines.append("  Strategy: HOT (VRAM) → WARM (RAM) → COLD (SSD)")
    else:
        lines.append("  Strategy: WARM (RAM) → COLD (SSD) → RAM_ONLY mode")
    return "\n".join(lines)

# ── Memory Tier Selector ───────────────────────────────────────────────────────
def interactive_offload(ram_gb, vram_gb, model_type):
    """Step-by-step offload config with recommendations."""
    recommended = recommend_offload(ram_gb, vram_gb, model_type)
    defaults = {
        "vram": str(recommended["vram_gb"]),
        "ram":  str(recommended["ram_gb"]),
        "ssd":  "yes" if recommended["ssd_enabled"] else "no",
        "hot":  str(int(recommended["hot_frac"]*100)),
    }

    print()
    panel("Recommended Offload Configuration",
          describe_offload(recommended), "bold green")

    if confirm("Accept recommended configuration?", default=True):
        return recommended

    print()
    # VRAM
    if vram_gb and vram_gb > 0:
        vram_choices = [(f"{i}GB", i) for i in range(0, min(vram_gb+1, 33), 2)]
        vram_sel = radio("How much VRAM to allocate for hot experts?",
                          vram_choices, default=int(defaults["vram"])//2)
        vram_gb_cfg = vram_sel
    else:
        vram_gb_cfg = 0

    # RAM
    ram_choices = []
    step = 16 if ram_gb >= 64 else 8
    for r in range(8, min(ram_gb+1, 257), step):
        ram_choices.append((f"{r}GB", r))
    if ram_gb not in [x[1] for x in ram_choices]:
        ram_choices.append((f"{ram_gb}GB", ram_gb))
    default_ram = min(defaults["ram"], ram_gb) if ram_gb else int(defaults["ram"])
    default_idx = max(0, next((i for i,x in enumerate(ram_choices) if x[1]<=default_ram), 0))
    ram_sel = radio("How much RAM to allocate for warm experts?",
                    ram_choices, default=ram_choices[min(default_idx, len(ram_choices)-1)][1])
    ram_gb_cfg = ram_sel

    # SSD
    ssd_enabled = confirm("Enable SSD tier for cold experts? (Recommended if RAM < 64GB)",
                          default=recommended["ssd_enabled"])

    # Hot threshold
    hot_choices = [("10% — Aggressive (VRAM focused)",  10),
                   ("15% — Balanced (recommended)",       15),
                   ("20% — Generous (better quality)",    20),
                   ("25% — Maximum (more VRAM used)",     25)]
    hot_sel = radio("What fraction of experts to keep hot in VRAM?",
                    hot_choices, default=15)

    return {
        "vram_gb":     vram_gb_cfg,
        "ram_gb":      ram_gb_cfg,
        "ssd_enabled": ssd_enabled,
        "hot_frac":    hot_sel / 100,
        "ram_frac":    1.0 - (hot_sel/100),
        "ssd_frac":    0.0 if not ssd_enabled else 0.25,
    }

# ── Model Selector ─────────────────────────────────────────────────────────────
def search_huggingface(query: str, limit: int = 10) -> list[tuple[str, str, str]]:
    """Search HuggingFace for GGUF models. Returns list of (repo_id, name, size_hint)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        results = api.list_models(
            search=query,
            limit=limit,
            sort="downloads",
            filter=["gguf"],
        )
        models = []
        for m in results:
            repo_id = m.id
            name = repo_id.split("/")[-1][:40]
            # Rough size hint from tags or safetensors
            size_hint = ""
            if hasattr(m, "safetensors") and m.safetensors:
                total = sum(s.get("size", 0) for s in m.safetensors.values() if isinstance(s, dict))
                if total > 0:
                    gb = total / (1024**3)
                    size_hint = f"~{gb:.0f}GB"
            if not size_hint and hasattr(m, "tags"):
                for tag in (m.tags or []):
                    if "B" in tag and any(c.isdigit() for c in tag):
                        size_hint = tag
                        break
            models.append((repo_id, name, size_hint or "?GB"))
        return models
    except Exception as e:
        print(f"  [dim]Search error: {e}[/dim]")
        return []

def interactive_model_select(ram_gb, vram_gb):
    """Show categorized model picker with hardware awareness + HuggingFace search."""
    candidates = recommend_models(ram_gb, vram_gb)

    while True:
        print()
        panel("Model Selection",
              f"Showing {len(candidates) - 1} models your hardware can run.\\n"
              f"RAM: {ram_gb}GB | VRAM: {vram_gb or 0}GB",
              "bold cyan")

        # Show categorized table
        if console:
            tbl = Table(title="Recommended Models", show_header=True)
            tbl.add_column("#", style="dim", width=3)
            tbl.add_column("Model", style="cyan")
            tbl.add_column("Type", style="magenta")
            tbl.add_column("Size", style="yellow")
            tbl.add_column("RAM", style="green")
            tbl.add_column("VRAM", style="green")
            tbl.add_column("Description")
            for i, (_, m) in enumerate(candidates):
                tbl.add_row(
                    str(i + 1),
                    m[0].split("/")[-1][:30],
                    f"[{m[2]}]",
                    m[3],
                    f"≥{m[4]}GB",
                    f"≥{m[5]}GB",
                    m[7][:50],
                )
            console.print(tbl)
        else:
            for i, (_, m) in enumerate(candidates):
                print(f"  {i+1}. [{m[2]}] {m[1]} — {m[3]}, RAM≥{m[4]}GB, VRAM≥{m[5]}GB")

        print()
        choice = radio(
            "Select a model",
            [(f"{i+1}. {m[1]} ({m[3]}) [{m[2]}]", i) for i, (_, m) in enumerate(candidates)],
            default=0,
        )
        if choice is None:
            continue
        if not isinstance(choice, int) or choice < 0 or choice >= len(candidates):
            continue
        selected = candidates[choice][1]

        # If user picked "Custom / Browse HuggingFace", offer sub-options
        if selected[0] == "custom":
            mode = radio("How would you like to find a model?", [
                ("🔍  Search HuggingFace for GGUF models", "search"),
                ("✏️  Enter a HuggingFace repo ID manually", "manual"),
                ("↩️  Go back to recommended list", "back"),
            ], default=0)
            if mode == "back":
                continue
            if mode == "manual":
                repo_id = text_input(
                    "Enter HuggingFace model ID or local GGUF path",
                    default="",
                )
                if repo_id:
                    # Build a synthetic model tuple for manual entry
                    return (
                        repo_id,
                        repo_id.split("/")[-1][:30],
                        "Custom",
                        "?GB",
                        ram_gb or 8,
                        0,
                        "Q4_K_M",
                        f"Custom model: {repo_id}",
                    )
                continue
            # mode == "search"
            query = text_input("Search query (e.g. 'llama 4', 'gemma', 'mistral')", default="gguf instruct")
            if not query:
                continue
            print(f"\n  Searching HuggingFace for '{query}'...")
            hf_results = search_huggingface(query, limit=10)
            if not hf_results:
                print("  [yellow]No results found. Try a different query.[/yellow]")
                continue
            # Show results
            if console:
                tbl2 = Table(title=f"HuggingFace Results for '{query}'", show_header=True)
                tbl2.add_column("#", style="dim", width=3)
                tbl2.add_column("Model", style="cyan")
                tbl2.add_column("Repo ID", style="dim")
                tbl2.add_column("Size", style="yellow")
                for i, (repo_id, name, size) in enumerate(hf_results):
                    tbl2.add_row(str(i + 1), name, repo_id, size)
                console.print(tbl2)
            else:
                for i, (repo_id, name, size) in enumerate(hf_results):
                    print(f"  {i+1}. {name} ({size}) — {repo_id}")
            print()
            hf_idx = radio("Select a model",
                           [(f"{i+1}. {name} ({size}) — {repo_id}", i)
                            for i, (repo_id, name, size) in enumerate(hf_results)],
                           default=0)
            repo_id, name, size = hf_results[hf_idx]
            return (
                repo_id,
                name,
                "Custom",
                size,
                ram_gb or 8,
                0,
                "Q4_K_M",
                f"HuggingFace: {repo_id}",
            )

        return selected

# ── Quantization Selector ─────────────────────────────────────────────────────
QUANT_OPTIONS = [
    ("Q4_K_M — Recommended (4-bit, best quality/size)",        "Q4_K_M"),
    ("Q5_K_M — Better quality, 15% larger",                   "Q5_K_M"),
    ("Q6_K  — Near-lossless, 30% larger",                     "Q6_K"),
    ("Q8_0  — FP16 quality, 2× larger (for quality work)",   "Q8_0"),
    ("Q3_K_M — Smaller, slight quality loss",                 "Q3_K_M"),
    ("Q2_K  — Smallest, notable quality loss",                "Q2_K"),
]

def interactive_quant_select():
    sel = radio("Select quantization level", QUANT_OPTIONS, default=0)
    return sel

# ── Download Models via HF Hub ────────────────────────────────────────────────
def download_model(model_id, quant="Q4_K_M", progress_cb=None):
    """Download a GGUF model using huggingface_hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False, "huggingface_hub not installed. Run: pip install huggingface_hub"

    api = HfApi()
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # List files in the repo
        models = api.list_repo_files(model_id, repo_type="model")
        gguf_files = [f for f in models if f.lower().endswith(".gguf")]
        if not gguf_files:
            return False, f"No GGUF files found in {model_id}. Try a different repo."

        # Pick file matching quant
        target = next((f for f in gguf_files if quant.upper() in f.upper()), gguf_files[0])
        print(f"  Selected: {target}")

        if progress_cb:
            progress_cb(f"Downloading {model_id} / {target}...")

        # Stream download
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=model_id,
            filename=target,
            repo_type="model",
            local_dir=str(cache_dir / model_id.replace("/","_")),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True, path
    except Exception as e:
        return False, str(e)

# ── Verify / Run Tests ─────────────────────────────────────────────────────────
def verify_install():
    print()
    print("  [1/2] Installing pytest...")
    ok, _ = run("pip install pytest -q", fatal=False)
    if ok:
        print("    ✓ pytest installed")
    else:
        print("    ✗ pytest install failed")

    print("  [2/2] Running test suite...")
    ok, out = run("pytest tests/test_sparse.py -v --tb=line -q 2>&1 | tail -5", fatal=False)
    if ok and "passed" in out.lower():
        print("    ✓ All tests pass!")
        return True
    else:
        print(f"    ! Tests: {out[-200:]}")
        return False

# ── Write Config ───────────────────────────────────────────────────────────────
def write_config(model_id, offload, quant, output="vibeblade.yaml"):
    cfg = {
        "model":        model_id,
        "vram_gb":      offload["vram_gb"],
        "ram_gb":       offload["ram_gb"],
        "ssd_enabled":  offload["ssd_enabled"],
        "hot_threshold": offload["hot_frac"],
        "quantization": quant,
        "cpu_threads":  os.cpu_count() or 8,
    }
    # Try YAML, fallback to JSON
    try:
        import yaml
        with open(output, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except ImportError:
        with open(output.replace(".yaml",".json"), "w") as f:
            json.dump(cfg, f, indent=2)
    return cfg

# ── CLI Help System ─────────────────────────────────────────────────────────────
WIZARD_HELP = """
╔══════════════════════════════════════════════════════════════╗
║                  VibeBlade Wizard Help                      ║
║             Developed by VibeDrift Inc.                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [H]   Show this help screen                                 ║
║  [Q]   Quit the wizard                                       ║
║  [↑↓]  Navigate options (in interactive mode)                ║
║  [↵]   Confirm / Continue                                    ║
║  [Tab] Cycle through choices                                 ║
║                                                              ║
║  CLI Commands (run from terminal outside the wizard):        ║
║                                                              ║
║  python -m vibeblade wizard     Run this setup wizard       ║
║  python -m vibeblade serve      Start inference server      ║
║  python -m vibeblade run        Run single inference        ║
║  python -m vibeblade bench      Benchmark your system       ║
║                                                              ║
║  Quick Start:                                                ║
║    git clone https://github.com/kevin046/VibeBlade         ║
║    cd VibeBlade && python -m vibeblade wizard              ║
║                                                              ║
║  Docs:   github.com/kevin046/VibeBlade                     ║
║  Maker:  vibedrift.com                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""".strip()


def show_help():
    """Display the help screen."""
    if console:
        from rich.panel import Panel as RichPanel
        console.print(RichPanel(WIZARD_HELP, border_style="cyan", padding=(1, 2)))
    else:
        print(WIZARD_HELP)


def pause(msg="Press [Enter] to continue..."):
    """Like input() but catches [H] for help and [Q] to quit."""
    while True:
        try:
            response = input(f"\n  {msg}  [H] Help  [Q] Quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if response == "h":
            show_help()
            continue
        if response == "q":
            print("\n  [dim]Wizard cancelled. Run 'python -m vibeblade wizard' to restart.[/dim]")
            sys.exit(0)
        return


# ── Main Wizard ────────────────────────────────────────────────────────────────
def main():
    os.system("cls" if platform.system()=="Windows" else "clear")

    print()
    print("[bold cyan]" + "="*60 + "[/bold cyan]")
    print("[bold cyan]     VIBEBlade SETUP WIZARD[/bold cyan]")
    print("[bold cyan]     Interactive Guided Setup[/bold cyan]")
    print("[bold cyan]" + "="*60 + "[/bold cyan]")
    print()

    if not HAS_PT:
        print("[yellow]NOTE: pip install prompt_toolkit rich for full TTY experience[/yellow]")
        print()

    # ── Welcome ──────────────────────────────────────────────────────────────
    if console:
        console.print(Panel(
            "[bold]Welcome to VibeBlade![/bold]\n\n"
            "This wizard will:\n"
            "  1. Detect your hardware (RAM, VRAM, GPU)\n"
            "  2. Recommend the best model for your system\n"
            "  3. Configure memory offloading (VRAM / RAM / SSD)\n"
            "  4. Set up and verify your installation\n"
            "  5. Generate a ready-to-run [cyan]vibeblade.yaml[/cyan] config\n\n"
            "[dim]Works on Windows, Linux, and macOS[/dim]\n"
            "[dim]Developed by [bold]VibeDrift Inc.[/bold] — vibedrift.com[/dim]",
            title="[bold green]Let's get started[/bold green]",
            style="bold cyan",
        ))
    else:
        print("=== VIBEBlade SETUP WIZARD ===")
        print("This wizard will detect hardware, configure memory offloading,")
        print("and generate a vibeblade.yaml config file.\n")
        print("Developed by VibeDrift Inc. — vibedrift.com\n")

    pause("Press [Enter] to begin")

    # ── Step 1: Detect Hardware ───────────────────────────────────────────────
    os.system("cls" if platform.system()=="Windows" else "clear")
    print("[bold cyan]STEP 1: Hardware Detection[/bold cyan]")
    print()

    ram  = get_ram_gb()
    vram_total, vram_used = get_vram_gb()
    gpu  = get_gpu_name()
    ssd  = has_ssd()
    disk = get_disk_free_gb()
    inet = has_inet()
    py_ok = check_python()

    # If RAM/SSD detection failed, ask user manually
    if ram is None:
        print("[yellow]⚠ Automatic RAM detection failed.[/yellow]")
        try:
            ans = input("  Enter your total RAM in GB (e.g. 32): ").strip()
            if ans and ans.isdigit():
                ram = int(ans)
        except (EOFError, KeyboardInterrupt):
            pass
    if not ssd:
        print("[yellow]⚠ Automatic SSD detection failed.[/yellow]")
        try:
            ans = input("  Do you have an SSD? (Y/n): ").strip().lower()
            if ans != "n":
                ssd = True
        except (EOFError, KeyboardInterrupt):
            pass

    if console:
        tbl = Table(title="System Summary", show_header=False, box=None)
        tbl.add_column("Property", style="cyan", width=22)
        tbl.add_column("Value",   style="white")
        tbl.add_row("OS",           f"{platform.system()} {platform.release()}")
        tbl.add_row("Python",       f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                                     + (" ✓" if py_ok else " ✗ NEED ≥3.10"))
        tbl.add_row("RAM",          f"{ram}GB" + (" ✓" if ram and ram>=16 else " LOW") if ram else "Unknown")
        if vram_total:
            tbl.add_row("VRAM", f"{vram_total}GB")
        else:
            tbl.add_row("VRAM", "None detected")
        if gpu:
            tbl.add_row("GPU", gpu)
        tbl.add_row("SSD", "Available ✓" if ssd else "Not detected")
        tbl.add_row("Disk (free)",  f"{disk}GB" if disk else "Unknown")
        tbl.add_row("Internet",     "✓ Connected" if inet else "✗ Offline")
        console.print(tbl)
    else:
        print(f"  OS:       {platform.system()} {platform.release()}")
        print(f"  Python:   {sys.version_info.major}.{sys.version_info.minor} {'OK' if py_ok else 'NEED 3.10+'}")
        print(f"  RAM:      {ram}GB" if ram else "  RAM:      Unknown")
        print(f"  VRAM:     {vram_total}GB" if vram_total else "  VRAM:     None")
        if gpu:
            print(f"  GPU:      {gpu}")
        print(f"  SSD:      {'Available' if ssd else 'Not detected'}")
        print(f"  Disk:     {disk}GB free" if disk else "  Disk:     Unknown")
        print(f"  Internet: {'Connected' if inet else 'Offline'}")

    if not py_ok:
        print("\n[red]ERROR: Python 3.10+ required. Please upgrade Python.[/red]")
        sys.exit(1)

    pause("Press [Enter] to continue")

    # ── Step 2: Model Selection ──────────────────────────────────────────────
    os.system("cls" if platform.system()=="Windows" else "clear")
    print("[bold cyan]STEP 2: Choose Model[/bold cyan]")
    print()

    model = interactive_model_select(ram, vram_total or 0)
    print(f"\n  [green]Selected:[/green] {model[1]} ({model[0]})")
    print(f"             Type: {model[2]} | Size: {model[3]} | Min RAM: {model[4]}GB")

    custom_id = None
    if model[0] == "custom":
        custom_id = text_input("Enter HuggingFace model ID or local GGUF path",
                                default="")
        if not custom_id:
            print("[red]No model specified — exiting.[/red]")
            sys.exit(1)

    pause("Press [Enter] to continue")

    # ── Step 3: Offload Config ───────────────────────────────────────────────
    os.system("cls" if platform.system()=="Windows" else "clear")
    print("[bold cyan]STEP 3: Memory Offloading[/bold cyan]")
    print()

    model_type = model[2]
    offload = interactive_offload(ram, vram_total or 0, model_type)

    print()
    panel("Final Offload Configuration", describe_offload(offload), "bold green")
    print()

    pause("Press [Enter] to continue")

    # ── Step 4: Quantization ─────────────────────────────────────────────────
    os.system("cls" if platform.system()=="Windows" else "clear")
    print("[bold cyan]STEP 4: Quantization[/bold cyan]")
    print()
    quant = interactive_quant_select()
    print(f"\n  [green]Selected:[/green] {quant}")

    pause("Press [Enter] to continue")

    # ── Step 5: Install & Verify ────────────────────────────────────────────
    os.system("cls" if platform.system()=="Windows" else "clear")
    print("[bold cyan]STEP 5: Install VibeBlade[/bold cyan]")
    print()

    turbodir = Path.cwd()
    venv_dir  = turbodir / "venv"

    print(f"  Working directory: {turbodir}")
    print(f"  Installing to:     {venv_dir}")
    print()

    # ── Auto-detect acceleration backend ─────────────────────────────────
    accel_extra, accel_name = detect_accel_backend()
    print(f"  [bold green]Detected acceleration:[/bold green] {accel_name}")
    if accel_extra:
        print(f"  [dim]Installing with extras: pip install -e .{accel_extra}[/dim]")
    else:
        print("  [dim]Installing base package (no GPU extras needed)[/dim]")
    print()

    if not venv_dir.exists():
        print("  Creating virtual environment...")
        ok, out = run(["python3", "-m", "venv", str(venv_dir)], fatal=True)
        print("    ✓ Virtual environment created")

    # Detect pip executable (cross-platform)
    pip_bin = "Scripts" if platform.system() == "Windows" else "bin"
    pip_exe = venv_dir / pip_bin / ("pip.exe" if platform.system() == "Windows" else "pip")

    print("  Installing dependencies...")
    run([str(pip_exe), "install", "--upgrade", "pip", "-q"], fatal=False)

    # Install with auto-detected acceleration backend
    extra = accel_extra
    ok, out = run([str(pip_exe), "install", "-e", f".{extra}", "-q"], fatal=False)
    if ok:
        print(f"    ✓ VibeBlade installed ({accel_name})")
    else:
        # Fallback to base if extras fail
        run([str(pip_exe), "install", "-e", ".", "-q"], fatal=False)
        print("    ✓ VibeBlade installed (base, extras skipped)")

    ok, out = run([str(pip_exe), "install", "prompt_toolkit", "rich", "-q"], fatal=False)
    if ok:
        print("    ✓ UI libraries installed")

    print()
    verify_install()

    pause("Press [Enter] to continue")

    # ── Step 6: Write Config ────────────────────────────────────────────────
    os.system("cls" if platform.system()=="Windows" else "clear")
    print("[bold cyan]STEP 6: Generate Config[/bold cyan]")
    print()

    model_id = custom_id if custom_id else model[0]
    cfg = write_config(model_id, offload, quant)

    print("  Generated [cyan]vibeblade.yaml:[/cyan]")
    if console:
        try:
            import yaml
            syn = Syntax(yaml.dump(cfg, default_flow_style=False), "yaml",
                        theme="monokai", line_numbers=True)
            console.print(syn)
        except Exception:
            print(json.dumps(cfg, indent=2))
    else:
        print(json.dumps(cfg, indent=2))

    # ── Step 7: Optional Download ───────────────────────────────────────────
    if inet and model[0] != "custom":
        os.system("cls" if platform.system()=="Windows" else "clear")
        print("[bold cyan]STEP 7: Download Model (Optional)[/bold cyan]")
        print()
        dl = confirm("Download the model now?",
                      text=f"This will download {model[1]} (~{model[3]}) from HuggingFace.\n"
                           f"You can skip and download manually later.",
                      default=False)
        if dl:
            print()
            ok, path = download_model(model_id, quant)
            if ok:
                print(f"\n  ✓ Model downloaded to:\n    {path}")
            else:
                print(f"\n  ✗ Download failed: {path}")
    else:
        print("[dim]STEP 7: Download skipped (offline or custom model)[/dim]")

    # ── Final Summary ────────────────────────────────────────────────────────
    os.system("cls" if platform.system()=="Windows" else "clear")
    activate = ("venv\\\\Scripts\\\\activate" if platform.system()=="Windows"
                 else "source venv/bin/activate")
    print("[bold green]" + "="*60 + "[/bold green]")
    print("[bold green]       SETUP COMPLETE![/bold green]")
    print("[bold green]" + "="*60 + "[/bold green]")

    if console:
        tbl = Table(title="Your Configuration", show_header=False, box=None)
        tbl.add_column("Setting", style="cyan", width=24)
        tbl.add_column("Value",   style="white")
        tbl.add_row("Model",      model_id)
        tbl.add_row("Type",       model[2])
        tbl.add_row("Quantization", quant)
        tbl.add_row("VRAM",       f"{offload['vram_gb']}GB")
        tbl.add_row("RAM",        f"{offload['ram_gb']}GB")
        tbl.add_row("SSD tier",   "Enabled" if offload["ssd_enabled"] else "Disabled")
        tbl.add_row("Hot experts", f"{offload['hot_frac']:.0%}")
        tbl.add_row("Config file", "vibeblade.yaml")
        console.print(tbl)

    print()
    print("  [bold green]NEXT STEPS:[/bold green]")
    print("  1. Activate environment:")
    print(f"       [yellow]{activate}[/yellow]")
    print()
    print("  2. Run benchmark:")
    print("       [yellow]python -m vibeblade bench --quick[/yellow]")
    print()
    print("  3. Run inference:")
    print("       [yellow]python -m vibeblade run[/yellow]")
    print("            --config vibeblade.yaml[/yellow]")
    print("            --prompt \"Hello world\"[/yellow]")
    print()
    print("  4. Start API server:")
    print("       [yellow]python -m vibeblade serve[/yellow]")
    print(f"            --model {model_id}[/yellow]")
    print("            --port 8000[/yellow]")
    print()
    print("[bold green]" + "="*60 + "[/bold green]")
    print()
    print("  [dim]VibeBlade by [bold]VibeDrift Inc.[/bold] — vibedrift.com | github.com/kevin046/VibeBlade[/dim]")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [yellow]Setup interrupted. Re-run anytime:[/yellow]")
        print("  [cyan]  python setup_wizard.py[/cyan]")
        sys.exit(0)
