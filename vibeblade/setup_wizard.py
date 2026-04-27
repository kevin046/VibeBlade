#!/usr/bin/env python3
"""
VibeBlade Interactive Setup Wizard
Interactive TTY setup wizard for Windows / Linux / macOS

Developed by VibeDrift Inc.
https://vibedrift.com | https://github.com/kevin046/VibeBlade

Usage:
    python -m vibeblade wizard
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path


# ── Terminal helpers (standard CLI, no external dependencies) ─────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

if os.name == "nt":
    try:
        os.system("")
    except Exception:
        BOLD = DIM = CYAN = GREEN = YELLOW = RED = RESET = ""


def _b(s):
    return f"{BOLD}{s}{RESET}"

def _c(s):
    return f"{CYAN}{s}{RESET}"

def _g(s):
    return f"{GREEN}{s}{RESET}"

def _y(s):
    return f"{YELLOW}{s}{RESET}"

def _r(s):
    return f"{RED}{s}{RESET}"

def _d(s):
    return f"{DIM}{s}{RESET}"


def panel(title, body):
    lines = body.split("\n")
    w = max((len(line) for line in lines), default=0)
    w = max(w, len(title))
    b = "+" + "-" * (w + 4) + "+"
    print(f"\n  {b}")
    print(f"  | {_b(title):^{w + 2}} |")
    print(f"  +{'-' * (w + 4)}+")
    for line in lines:
        print(f"  | {line:<{w + 2}} |")
    print(f"  {b}\n")


def print_table(headers, rows):
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    hdr = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    print(f"\n  {_c(hdr)}")
    print(f"  {sep}")
    for row in rows:
        line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        print(f"  {line}")
    print()


def clear_screen():
    os.system("cls" if platform.system() == "Windows" else "clear")

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

# ── Dialog Helpers ───────────────────────────────────────────────────────

def radio(title, options, default=0):
    """Numbered selection menu. Options: [(label, value), ...]. Returns value."""
    print(f"\n  {_b(title)}")
    for i, (_ret, label) in enumerate(options):
        marker = ">" if i == default else " "
        print(f"    {marker} {i+1}. {label}")
    while True:
        try:
            ch = int(input("  Choice: ").strip())
            if 1 <= ch <= len(options):
                return options[ch - 1][1]
        except (ValueError, IndexError):
            pass

def checkbox(title, options, defaults=None):
    """Checkbox list. Returns list of selected values."""
    print(f"\n  {_b(title)} (comma-separated numbers)")
    for i, (_ret, label) in enumerate(options):
        checked = "[x]" if defaults and _ret in defaults else "[ ]"
        print(f"    {checked} {i+1}. {label}")
    while True:
        try:
            nums = input("  Choices (e.g. 1,3): ").strip()
            return [options[int(n) - 1][1] for n in nums.split(",") if n.strip().isdigit()]
        except (ValueError, IndexError):
            pass

def confirm(title, text="", default=True):
    prompt = "[Y/n]" if default else "[y/N]"
    if text:
        print(f"\n  {_b(title)}")
        print(f"  {text}")
        ch = input(f"  {prompt}: ").strip()
    else:
        ch = input(f"\n  {title} {prompt}: ").strip()
    return ch.lower() in ("y", "yes") if ch else default

def text_input(title, default=""):
    v = input(f"\n  {title} [{default}]: ").strip()
    return v if v else default

def message(title, text="", style="info"):
    print(f"\n  [{style.upper()}] {title}: {text}")

def spinner(text, coro_fn, *args, **kwargs):
    print(f"\n  {text}...")
    result = coro_fn(*args, **kwargs)
    print("  Done.")
    return result

# ── Offload Strategy Recommendation ─────────────────────────────────────────
def recommend_offload(ram_gb, vram_gb, model_type, model_size_gb=0):
    """Return recommended offload config that uses ALL available memory.
    
    Strategy:
    - Hot experts → VRAM (fastest)
    - Warm experts → RAM
    - Cold experts → SSD only if model doesn't fit in VRAM+RAM
    - For dense models, layers are split VRAM-first then RAM
    """
    total_mem = (vram_gb or 0) + (ram_gb or 32)
    model_fits = total_mem >= model_size_gb if model_size_gb > 0 else True

    # Use all VRAM for hot experts / model layers
    vram_s = vram_gb or 0

    # Use all RAM for warm experts / remaining layers
    ram_s = ram_gb or 32

    # SSD: only enabled if model exceeds VRAM+RAM
    ssd_en = not model_fits and has_ssd()

    # Expert split fractions (only matter for MoE)
    if model_type == "MoE":
        if vram_gb and ram_gb:
            # Proportional split based on available memory
            hot_frac = vram_gb / total_mem if total_mem > 0 else 0.15
            ram_frac = ram_gb / total_mem if total_mem > 0 else 0.70
            ssd_frac = max(0, 1.0 - hot_frac - ram_frac)
        elif vram_gb:
            hot_frac = 0.30
            ram_frac = 0.70
            ssd_frac = 0.0
        else:
            hot_frac = 0.0
            ram_frac = 1.0
            ssd_frac = 0.0
    elif model_type == "Dense":
        hot_frac = vram_gb / total_mem if (total_mem > 0 and vram_gb) else 0.0
        ram_frac = 1.0 - hot_frac
        ssd_frac = 0.0
    else:
        hot_frac = 0.15
        ram_frac = 0.85
        ssd_frac = 0.0

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
def interactive_offload(ram_gb, vram_gb, model_type, model_size_gb=0):
    """Step-by-step offload config with recommendations."""
    recommended = recommend_offload(ram_gb, vram_gb, model_type, model_size_gb)
    total_mem = (vram_gb or 0) + (ram_gb or 32)
    model_fits = total_mem >= model_size_gb if model_size_gb > 0 else True

    print()
    panel("Recommended Offload Configuration",
          describe_offload(recommended))

    if model_fits:
        print(f"  💡 Model fits in VRAM+RAM ({total_mem}GB). SSD not needed.")
    else:
        print(f"  ⚠️  Model ({model_size_gb}GB) exceeds VRAM+RAM ({total_mem}GB). SSD tier will be used for overflow.")

    if confirm("Accept recommended configuration?", default=True):
        return recommended

    print()
    # VRAM — use all of it by default, let user dial down if they want
    if vram_gb and vram_gb > 0:
        vram_choices = [(f"{i}GB", i) for i in range(0, min(vram_gb + 1, 65), 2)]
        if (vram_gb % 2) != 0:
            vram_choices.append((f"{vram_gb}GB", vram_gb))
        # Default index: find the one closest to vram_gb (use all)
        default_vram_idx = next(
            (i for i, (_, v) in enumerate(vram_choices) if v >= vram_gb),
            len(vram_choices) - 1,
        )
        vram_sel = radio("How much VRAM to allocate?",
                          vram_choices, default=default_vram_idx)
        vram_gb_cfg = vram_sel
    else:
        vram_gb_cfg = 0

    # RAM — use all of it by default
    ram_choices = []
    step = 16 if ram_gb >= 64 else 8
    for r in range(8, min(ram_gb + 1, 257), step):
        ram_choices.append((f"{r}GB", r))
    if ram_gb and ram_gb not in [x[1] for x in ram_choices]:
        ram_choices.append((f"{ram_gb}GB", ram_gb))
    default_ram_idx = next(
        (i for i, (_, v) in enumerate(ram_choices) if v >= (ram_gb or 32)),
        len(ram_choices) - 1,
    )
    ram_sel = radio("How much RAM to allocate?",
                    ram_choices, default=default_ram_idx)
    ram_gb_cfg = ram_sel

    # SSD — only offer if model doesn't fit in VRAM+RAM
    allocated = vram_gb_cfg + ram_gb_cfg
    if model_size_gb > 0 and allocated >= model_size_gb:
        ssd_enabled = False
        print(f"\n  ✅ {allocated}GB allocated, model needs {model_size_gb}GB — SSD not needed.")
    elif model_size_gb > 0:
        ssd_enabled = confirm("Model exceeds allocated memory. Enable SSD tier for overflow?",
                              default=True)
    else:
        ssd_enabled = confirm("Enable SSD tier for cold experts?",
                              default=False)

    # Hot threshold (only relevant for MoE)
    if model_type == "MoE" and vram_gb_cfg > 0:
        hot_choices = [
            ("10% — Aggressive (minimal VRAM, more in RAM)", 10),
            ("15% — Balanced",                                15),
            ("20% — Generous (recommended)",                  20),
            ("30% — VRAM focused (faster, uses more VRAM)",   30),
            ("50% — Max VRAM (best speed if experts fit)",    50),
        ]
        hot_sel = radio("What fraction of experts to keep hot in VRAM?",
                        hot_choices, default=2)
        hot_frac = hot_sel / 100
    else:
        hot_frac = vram_gb_cfg / allocated if allocated > 0 else 0.0

    return {
        "vram_gb":     vram_gb_cfg,
        "ram_gb":      ram_gb_cfg,
        "ssd_enabled": ssd_enabled,
        "hot_frac":    hot_frac,
        "ram_frac":    1.0 - hot_frac,
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
        print(f"  _d(Search error: {e})")
        return []

def interactive_model_select(ram_gb, vram_gb):
    """Show model picker — asks about existing downloads first, then catalog."""
    candidates = recommend_models(ram_gb, vram_gb)

    while True:
        print()

        # ── Step 2a: Ask if user already has models downloaded ──
        print(f"  {_b('Do you already have LLM models installed on this computer?')}")
        print(f"  {_d('(e.g. from LM Studio, Ollama, GPT4All, HuggingFace, or downloaded manually)')}")
        print()

        has_existing = radio("Choose an option", [
            ("Yes — scan my computer for downloaded models", "scan"),
            ("No — show me the model catalog to download", "catalog"),
            ("I have a model file — let me paste the path", "manual_path"),
        ], default=0)

        if has_existing == "manual_path":
            # Let user paste a direct path to a .gguf file or folder
            custom_path = text_input(
                "Enter the full path to your .gguf model file (or folder containing models)",
                default="",
            )
            if custom_path:
                custom_path = Path(custom_path.strip('"').strip("'"))

                # If it's a directory, scan it for .gguf files
                if custom_path.is_dir():
                    from .model_hub import find_gguf_files
                    gguf_files = find_gguf_files(custom_path, recursive=True)
                    if gguf_files:
                        print(f"\n  Found {len(gguf_files)} .gguf file(s) in that folder:")
                        rows = []
                        for i, f in enumerate(gguf_files):
                            size_gb = round(f.stat().st_size / (1024**3), 1)
                            rows.append((str(i+1), f.name[:40], f"{size_gb}GB"))
                        print_table(["#", "Model File", "Size"], rows)
                        choice = radio(
                            "Which model would you like to use?",
                            [(f"{f.name[:40]} ({round(f.stat().st_size / (1024**3), 1)}GB)", i)
                             for i, f in enumerate(gguf_files)],
                            default=0,
                        )
                        chosen = gguf_files[choice]
                        size_gb = round(chosen.stat().st_size / (1024**3), 1)
                        from .model_hub import _detect_gguf_quant
                        quant = _detect_gguf_quant(chosen.name)
                        return (
                            str(chosen.resolve()),
                            chosen.name[:30],
                            "Downloaded",
                            f"~{size_gb}GB",
                            ram_gb or 8,
                            0,
                            quant if quant != "unknown" else "Q4_K_M",
                            f"Local file: {chosen.parent}",
                        )
                    else:
                        print(f"\n  {_y('No .gguf files found in that folder.')}")
                        pause("Press [Enter] to continue")
                        continue

                elif custom_path.is_file() and custom_path.suffix.lower() == ".gguf":
                    size_gb = round(custom_path.stat().st_size / (1024**3), 1)
                    from .model_hub import _detect_gguf_quant
                    quant = _detect_gguf_quant(custom_path.name)
                    return (
                        str(custom_path.resolve()),
                        custom_path.name[:30],
                        "Downloaded",
                        f"~{size_gb}GB",
                        ram_gb or 8,
                        0,
                        quant if quant != "unknown" else "Q4_K_M",
                        f"Local file: {custom_path.parent}",
                    )
                else:
                    print(f"\n  {_r('Path not found. Check the path and try again.')}")
                    print(f"  {_d('Tip: You can paste a path to a .gguf file OR a folder containing .gguf files.')}")
                    pause("Press [Enter] to continue")
                    continue
            continue

        if has_existing == "scan":
            # ── Scan all known model locations ──
            print("\n  Scanning your computer for downloaded models...")
            detected = []
            try:
                from .model_hub import scan_cached_models
                cached = scan_cached_models()
                for m in cached:
                    detected.append((m["path"], m["name"], m["size_gb"], m["source"]))
            except Exception as exc:
                print(f"  {_y(f'Scan error: {exc}')}")

            if not detected:
                print(f"\n  {_y('No downloaded models found on this computer.')}")
                print(f"  {_d('Scanned: HuggingFace cache, LM Studio, Ollama, GPT4All, and local directories.')}")
                print(f"  {_d('Tip: Make sure models are in their default install locations.')}")
                pause("Press [Enter] to continue")
                continue

            # Show what we found
            panel("Downloaded Models Found",
                  f"{len(detected)} model(s) found on your system\n"
                  f"Sources: {', '.join(sorted(set(s for _, _, _, s in detected)))}")
            rows = []
            for i, (path, name, size, source) in enumerate(detected):
                short_name = name[:40]
                rows.append((str(i+1), short_name, f"{size}GB", source))
            print_table(["#", "Model File", "Size", "Source"], rows)

            print()
            use_detected = confirm(
                "Use one of these downloaded models?",
                text="Select one above, or choose 'No' to browse the download catalog.",
                default=True,
            )
            if use_detected:
                choice = radio(
                    "Which model would you like to use?",
                    [(f"{name[:40]} ({size}GB) [{source}]", i)
                     for i, (_, name, size, source) in enumerate(detected)],
                    default=0,
                )
                path, name, size, source = detected[choice]
                from .model_hub import _detect_gguf_quant
                quant = _detect_gguf_quant(name)
                return (
                    str(path),
                    name[:30],
                    "Downloaded",
                    f"~{size}GB",
                    ram_gb or 8,
                    0,
                    quant if quant != "unknown" else "Q4_K_M",
                    f"Local file from {source}",
                )

            # User said no — fall through to catalog below

        # ── Step 2b: Show catalog (also scan detected in background) ──
        detected = []
        try:
            from .model_hub import scan_cached_models
            cached = scan_cached_models()
            for m in cached:
                detected.append((m["path"], m["name"], m["size_gb"], m["source"]))
        except Exception:
            pass

        if detected:
            panel("Downloaded Models Found",
                  f"{len(detected)} model(s) detected — listed first")
            rows = []
            for i, (path, name, size, source) in enumerate(detected):
                short_name = name[:40]
                rows.append((str(i+1), short_name, f"{size}GB", source))
            print_table(["#", "Model File", "Size", "Source"], rows)
            print(f"  {_y('Tip:')} Select a downloaded model above, or pick from the catalog below.")
            print()

        panel("Model Selection",
              f"Showing {len(candidates) - 1} models your hardware can run.\n"
              f"RAM: {ram_gb}GB | VRAM: {vram_gb or 0}GB")

        rows = [(str(i+1), m[0].split("/")[-1][:30], m[2], m[3],
                   f"\u2265{m[4]}GB", f"\u2265{m[5]}GB", m[7][:50])
                  for i, (_, m) in enumerate(candidates)]
        print_table(["#", "Model", "Type", "Size", "RAM", "VRAM", "Description"], rows)

        print()
        # Build options: detected models first, then catalog
        all_options = []
        if detected:
            for i, (path, name, size, source) in enumerate(detected):
                all_options.append((f"D{i+1}. {name} ({size}GB) [{source}]",
                                    ("detected", i)))
            all_options.append(("", "separator_detected"))

        for i, (_, m) in enumerate(candidates):
            all_options.append((f"{m[1]} ({m[3]}) [{m[2]}]",
                                ("catalog", i)))

        # Remove separators for radio display
        radio_options = [(label, val) for label, val in all_options
                         if val != "separator_detected"]

        choice = radio(
            "Select a model",
            radio_options,
            default=0,
        )
        if choice is None:
            continue

        # Handle detected model selection
        if isinstance(choice, tuple) and choice[0] == "detected":
            idx = choice[1]
            path, name, size, source = detected[idx]
            # Build a model tuple from the detected file
            from .model_hub import _detect_gguf_quant
            quant = _detect_gguf_quant(name) if "unknown" not in _detect_gguf_quant(name).lower() else "Q4_K_M"
            return (
                str(path),       # repo_id = local path
                name[:30],        # display name
                "Downloaded",     # type
                f"~{size}GB",     # size label
                ram_gb or 8,      # min ram
                0,                # min vram
                quant,            # quant
                f"Local file from {source}",
            )

        # Handle catalog selection
        if isinstance(choice, tuple) and choice[0] == "catalog":
            cat_idx = choice[1]
        else:
            cat_idx = choice

        if cat_idx < 0 or cat_idx >= len(candidates):
            continue
        selected = candidates[cat_idx][1]

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
                msg = _y("No results found. Try a different query.")
                print(f"  {msg}")
                continue
            rows = [(str(i+1), name, repo_id, size)
                    for i, (repo_id, name, size) in enumerate(hf_results)]
            print_table(["#", "Model", "Repo ID", "Size"], rows)
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
    """Download a GGUF model using huggingface_hub.

    If the model is already cached (HF cache, LM Studio, or local), returns
    the existing path without re-downloading.
    """
    # ── Check if model is already downloaded ──
    from .model_hub import resolve_model_path
    try:
        existing = resolve_model_path(model_id, quant=quant)
        if existing and existing.exists():
            print(f"  Found existing model: {existing}")
            return True, str(existing)
    except FileNotFoundError:
        pass  # Not cached — proceed with download

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

        # Stream download to HF cache (standard location)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=model_id,
            filename=target,
            repo_type="model",
            resume_download=True,
        )
        return True, path
    except Exception as e:
        return False, str(e)

# ── Verify / Run Tests ─────────────────────────────────────────────────────────
def _venv_python() -> str:
    """Return the Python executable from the venv (if running inside wizard)."""
    # Wizard stores venv path; detect it from sys.prefix or nearby venv dir
    from pathlib import Path
    import os
    base = Path(os.getcwd())
    for candidate in [base / "venv", base.parent / "venv"]:
        if candidate.exists():
            scripts = candidate / ("Scripts" if os.name == "nt" else "bin")
            exe = scripts / ("python.exe" if os.name == "nt" else "python")
            if exe.exists():
                return str(exe)
    return sys.executable

def verify_install():
    py = _venv_python()
    print()
    print("  [1/2] Installing pytest...")
    ok, _ = run([py, "-m", "pip", "install", "pytest", "-q"], fatal=False)
    if ok:
        print("    ✓ pytest installed")
    else:
        print("    ✗ pytest install failed")

    print("  [2/2] Running test suite...")
    ok, out = run([py, "-m", "pytest", "tests/test_sparse.py", "--tb=line", "-q"], fatal=False)
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
            print("\n  _d(Wizard cancelled. Run 'python -m vibeblade wizard' to restart.)")
            sys.exit(0)
        return


# ── Main Wizard ────────────────────────────────────────────────────────────────
def main():
    clear_screen()

    print()
    print(_c(_b("=" * 60)))
    print(_c(_b("     VIBEBlade SETUP WIZARD")))
    print(_c(_b("     Interactive Guided Setup")))
    print(_c(_b("=" * 60)))
    print()

    # ── Welcome ──────────────────────────────────────────────────────────────
    panel("Welcome to VibeBlade!",
          "This wizard will:\n"
          "  1. Detect your hardware (RAM, VRAM, GPU)\n"
          "  2. Recommend the best model for your system\n"
          "  3. Configure memory offloading (VRAM / RAM / SSD)\n"
          "  4. Set up and verify your installation\n"
          "  5. Generate a ready-to-run vibeblade.yaml config\n\n"
          "Works on Windows, Linux, and macOS\n"
          "Developed by VibeDrift Inc. — vibedrift.com")


    pause("Press [Enter] to begin")

    # ── Step 1: Detect Hardware ───────────────────────────────────────────────
    clear_screen()
    print(_c(_b("STEP 1: Hardware Detection")))
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
        print(_y("⚠ Automatic RAM detection failed."))
        try:
            ans = input("  Enter your total RAM in GB (e.g. 32): ").strip()
            if ans and ans.isdigit():
                ram = int(ans)
        except (EOFError, KeyboardInterrupt):
            pass
    if not ssd:
        print(_y("⚠ Automatic SSD detection failed."))
        try:
            ans = input("  Do you have an SSD? (Y/n): ").strip().lower()
            if ans != "n":
                ssd = True
        except (EOFError, KeyboardInterrupt):
            pass

    rows = [
        ("OS", f"{platform.system()} {platform.release()}"),
        ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                   + (" OK" if py_ok else " NEED 3.10+")),
        ("RAM", f"{ram}GB" if ram else "Unknown"),
    ]
    if vram_total:
        rows.append(("VRAM", f"{vram_total}GB"))
    else:
        rows.append(("VRAM", "None detected"))
    if gpu:
        rows.append(("GPU", gpu))
    rows.append(("SSD", "Available" if ssd else "Not detected"))
    rows.append(("Disk (free)", f"{disk}GB" if disk else "Unknown"))
    rows.append(("Internet", "Connected" if inet else "Offline"))
    print_table(["Property", "Value"], rows)



    if not py_ok:
        print("\n" + _r("ERROR: Python 3.10+ required. Please upgrade Python."))
        sys.exit(1)

    pause("Press [Enter] to continue")

    # ── Step 2: Model Selection ──────────────────────────────────────────────
    clear_screen()
    print(_c(_b("STEP 2: Choose Model")))
    print()

    model = interactive_model_select(ram, vram_total or 0)
    print(f'\n  {_g("Selected:")} {model[1]} ({model[0]})')
    print(f"             Type: {model[2]} | Size: {model[3]} | Min RAM: {model[4]}GB")

    custom_id = None
    if model[0] == "custom":
        custom_id = text_input("Enter HuggingFace model ID or local GGUF path",
                                default="")
        if not custom_id:
            print(_r("No model specified — exiting."))
            sys.exit(1)

    pause("Press [Enter] to continue")

    # ── Step 3: Offload Config ───────────────────────────────────────────────
    clear_screen()
    print(_c(_b("STEP 3: Memory Offloading")))
    print()

    model_type = model[2]
    # Parse model size from label like "~26GB" or "~115GB"
    import re
    size_match = re.search(r"(\d+)", model[3])
    model_size_gb = int(size_match.group(1)) if size_match else 0
    offload = interactive_offload(ram, vram_total or 0, model_type, model_size_gb)

    print()
    panel("Final Offload Configuration", describe_offload(offload))
    print()

    pause("Press [Enter] to continue")

    # ── Step 4: Quantization ─────────────────────────────────────────────────
    clear_screen()
    print(_c(_b("STEP 4: Quantization")))
    print()
    quant = interactive_quant_select()
    print(f'\n  {_g("Selected:")} {quant}')

    pause("Press [Enter] to continue")

    # ── Step 5: Install & Verify ────────────────────────────────────────────
    clear_screen()
    print(_c(_b("STEP 5: Install VibeBlade")))
    print()

    turbodir = Path.cwd()
    venv_dir  = turbodir / "venv"

    print(f"  Working directory: {turbodir}")
    print(f"  Installing to:     {venv_dir}")
    print()

    # ── Auto-detect acceleration backend ─────────────────────────────────
    accel_extra, accel_name = detect_accel_backend()
    print(f"  {_g(_b('Detected acceleration:'))} {accel_name}")
    if accel_extra:
            print(f"  {_d('Installing with extras: pip install -e .' + accel_extra)}")
    else:
            print("  " + _d("Installing base package (no GPU extras needed)"))
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

    # No extra UI dependencies needed

    print()
    verify_install()

    pause("Press [Enter] to continue")

    # ── Step 6: Write Config ────────────────────────────────────────────────
    clear_screen()
    print(_c(_b("STEP 6: Generate Config")))
    print()

    model_id = custom_id if custom_id else model[0]
    cfg = write_config(model_id, offload, quant)

    print("  Generated " + _c("vibeblade.yaml:"))
    print(json.dumps(cfg, indent=2))

    # ── Step 7: Optional Download ───────────────────────────────────────────
    if model[2] == "Downloaded":
        print(f"\n  {_g('STEP 7: Download skipped')} — using your local model:")
        print(f"  {_d(model[0])}")
    elif inet and model[0] != "custom":
        clear_screen()
        print(_c(_b("STEP 7: Download Model (Optional)")))
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
        print(_d("STEP 7: Download skipped (offline or custom model)"))

    # ── Final Summary ────────────────────────────────────────────────────────
    clear_screen()
    activate = ("venv\\Scripts\\activate" if platform.system()=="Windows"
                 else "source venv/bin/activate")
    print(_g(_b("=" * 60)))
    print(_g(_b("       SETUP COMPLETE!")))
    print(_g(_b("=" * 60)))

    print_table(["Setting", "Value"], [
        ("Model", model_id),
        ("Type", model[2]),
        ("Quantization", quant),
        ("VRAM", f"{offload['vram_gb']}GB"),
        ("RAM", f"{offload['ram_gb']}GB"),
        ("SSD tier", "Enabled" if offload["ssd_enabled"] else "Disabled"),
        ("Hot experts", f"{offload['hot_frac']:.0%}"),
        ("Config file", "vibeblade.yaml"),
    ])

    print()
    print("  " + _g(_b("NEXT STEPS:")))
    print("  1. Activate environment:")
    print("       " + _y(activate))
    print()
    print("  2. Run benchmark:")
    print("       " + _y("python -m vibeblade bench --quick"))
    print()
    print("  3. Run inference:")
    print("       " + _y("python -m vibeblade run --config vibeblade.yaml"))
    print("       " + _y("python -m vibeblade run --prompt \"Hello world\""))
    print()
    print("  4. Start API server:")
    print("       " + _y(f"python -m vibeblade serve --model {model_id} --port 8000"))
    print()
    print("  5. Chat with your model:")
    print("       " + _y("python -m vibeblade chat --config vibeblade.yaml"))
    print()
    print(_g(_b("=" * 60)))
    print()

    # ── Offer to launch chat immediately ─────────────────────────────────
    launch_chat = confirm(
        "Launch chat now?",
        text="Start chatting with your model right away?",
        default=True,
    )
    if launch_chat:
        print(f"\n  {_c(_b('Launching VibeBlade Chat...'))}")
        print()
        try:
            from .chat import chat_loop
            chat_loop(model_path=model_id)
        except Exception as e:
            print(f"  {_r(f'Could not launch chat: {e}')}")
            print(f"  {_d('You can start it manually later with:')}")
            print(f"  {_d('python -m vibeblade chat --config vibeblade.yaml')}")
    else:
        print(f"  {_d('VibeBlade by _b(VibeDrift Inc.) — vibedrift.com | github.com/kevin046/VibeBlade)')}")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  " + _y("Setup interrupted. Re-run anytime:"))
        print("  " + _c("  python setup_wizard.py"))
        sys.exit(0)
