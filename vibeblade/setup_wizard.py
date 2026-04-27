#!/usr/bin/env python3
"""
VibeBlade Setup Wizard — Minimal CLI Interface
Streamlined setup like Claude Code / Hermes Agent.

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


def _header(title):
    """Clean minimal header bar."""
    print(f"\n  {_b(title)}")
    print(f"  {'─' * len(_strip(title))}")

def _strip(s):
    """Remove ANSI codes for width calculation."""
    import re
    return re.sub(r'\033\[[0-9;]*m', '', s)

def _kv(key, val):
    """Print a key: value pair with aligned keys."""
    print(f"  {_d(key + ':'):<14} {val}")

def _step(n, total, text):
    """Step indicator: ◆ 1/6 Model Selection"""
    print(f"\n  {_c('◆')} {_d(f'{n}/{total}')} {_b(text)}")


def radio(title, options, default=0):
    """Minimal numbered menu. Returns selected value."""
    print(f"\n  {_b(title)}")
    for i, (key, label) in enumerate(options):
        marker = f"{_g('▸')}" if i == default else " "
        print(f"  {marker} {i+1}. {label}")
    print(f"  {_d(f'(default: {default+1})')}", end="")
    while True:
        try:
            raw = input(f"\n  {_b('>')} ").strip()
            if not raw:
                return options[default][1]
            ch = int(raw)
            if 1 <= ch <= len(options):
                return options[ch - 1][1]
        except (ValueError, IndexError, EOFError, KeyboardInterrupt):
            pass

def confirm(title, text="", default=True):
    """Single-line confirmation."""
    prompt = f"  {title} [{_g('Y')}/{_d('n')}]: " if default else f"  {title} [{_d('y')}/{_g('N')}]: "
    if text:
        print(f"  {_d(text)}")
    try:
        ch = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    return ch in ("y", "yes") if ch else default

def text_input(title, default=""):
    """Inline text prompt."""
    default_hint = _d(f'[{default}]') if default else ""
    prompt = f"  {title}"
    if default_hint:
        prompt += f" {default_hint}"
    prompt += ": "
    try:
        v = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return v if v else default

def message(text, style="info"):
    """Print a status message."""
    icons = {"info": "ℹ", "ok": "✓", "warn": "⚠", "error": "✗"}
    icon = icons.get(style, "●")
    colors = {"info": _d, "ok": _g, "warn": _y, "error": _r}
    fn = colors.get(style, _d)
    print(f"  {fn(f'{icon}  {text}')}")

def pause(msg="Press [Enter] to continue"):
    """Pause for user acknowledgment."""
    try:
        input(f"\n  {_d(msg)}...")
    except (EOFError, KeyboardInterrupt):
        pass


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

    if _ensure_psutil():
        try:
            import psutil
            return int(psutil.virtual_memory().total) // (1024**3)
        except Exception:
            pass

    if system == "Windows":
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
                    ("ullAvailPageFile", ctypes.c_ulonglong),
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

    else:
        try:
            with open("/proc/meminfo") as f:
                return int(f.readline().split()[1]) // 1024 // 1024
        except Exception:
            return None

def get_vram_gb():
    """Detect GPU VRAM on NVIDIA (Linux/Windows), Apple Silicon (macOS), or AMD."""
    system = platform.system()
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
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["ioreg", "-r", "-c", "AppleARMGraphics", "-d", "2"],
                text=True, timeout=5,
            )
            for line in out.splitlines():
                if "gpu-memory-size" in line and "= " in line:
                    hexbytes = line.split("= ")[-1].strip()
                    if hexbytes.startswith("0x"):
                        total_bytes = int(hexbytes, 16)
                        return total_bytes // (1024**3), None
        except Exception:
            pass
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
    if system == "Linux":
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                text=True, timeout=5,
            )
            for line in out.splitlines():
                if "VRAM" in line and "Used" in line:
                    parts = [p.strip() for p in line.split(",")]
                    total = int(parts[1].replace("MB", "").strip()) // 1024
                    used = int(parts[2].replace("MB", "").strip()) // 1024
                    return total, used
        except Exception:
            pass
    return None, None

def get_gpu_name():
    """Detect GPU name on NVIDIA, Apple Silicon, or AMD."""
    system = platform.system()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return out.strip()
    except Exception:
        pass
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

    if system == "Darwin" and gpu and ("Apple" in gpu or "M1" in gpu or "M2" in gpu
                                       or "M3" in gpu or "M4" in gpu):
        return ".[gpu-metal]", "Apple Metal (CoreML)"

    if gpu:
        try:
            subprocess.check_output(["nvidia-smi"], timeout=3,
                                    stderr=subprocess.DEVNULL)
            return ".[grammar]", "NVIDIA CUDA"
        except Exception:
            pass

    if system == "Linux" and gpu:
        try:
            subprocess.check_output(["rocm-smi"], timeout=3,
                                    stderr=subprocess.DEVNULL)
            return ".[gpu-vulkan]", "AMD ROCm / Vulkan"
        except Exception:
            pass

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

    return "", "Base (NumPy, universal)"

def has_ssd():
    """Detect whether an SSD is available. Returns True/False."""
    system = platform.system()

    if _ensure_psutil():
        try:
            import psutil
            for part in psutil.disk_partitions():
                if part.fstype and part.mountpoint:
                    try:
                        usage = psutil.disk_usage(part.mountpoint)
                        if usage.total > 10**9:
                            return True
                    except Exception:
                        pass
            return False
        except Exception:
            pass

    if system == "Darwin":
        return True

    if system == "Windows":
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
                    continue
                if "HARD DRIVE" in line:
                    return False
        except Exception:
            pass

        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            if os.path.isdir(f"{letter}:\\"):
                return True

        return False

    else:
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
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",
     "Mixtral 8x7B", "MoE", "~26GB", 32, 8, "Q4_K_M",
     "Golden standard MoE. 8 experts, 2 active/token."),
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct",
     "Llama 4 Scout (109B MoE)", "MoE", "~40GB", 48, 8, "Q4_K_M",
     "Meta's Llama 4. 16 experts, fast inference."),
    ("deepseek-ai/DeepSeek-R1-0528-Q4_K_M-GGUF",
     "DeepSeek R1", "MoE", "~67GB", 64, 8, "Q4_K_M",
     "Reasoning model. Long chain-of-thought. 67B MoE."),
    ("Qwen/Qwen3-235B-A22B",
     "Qwen 3 235B MoE", "MoE", "~90GB", 96, 8, "Q4_K_M",
     "Alibaba MoE flagship. 22B active params."),
    ("meta-llama/Llama-4-Maverick-17B-128E-Instruct",
     "Llama 4 Maverick (400B MoE)", "MoE", "~110GB", 128, 16, "Q4_K_M",
     "Meta's flagship. 128 experts, SOTA quality."),
    ("MiniMaxAI/MiniMax-M2.7-01",
     "MiniMax M2.7 (456B MoE)", "MoE", "~115GB", 128, 16, "Q4_K_M",
     "VibeBlade showcase. 456B params, best MoE efficiency."),
    ("microsoft/Phi-4-mini-instruct",
     "Phi-4 Mini (3.8B)", "Dense", "~2.5GB", 8, 0, "Q4_K_M",
     "Tiny but punchy — great for edge devices."),
    ("meta-llama/Llama-3.1-8B-Instruct",
     "Llama 3.1 8B", "Dense", "~5GB", 8, 0, "Q4_K_M",
     "Most popular 8B model, excellent all-rounder."),
    ("mistralai/Mistral-7B-Instruct-v0.3",
     "Mistral 7B v0.3", "Dense", "~4GB", 8, 0, "Q4_K_M",
     "Fast & versatile. Battle-tested."),
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503",
     "Mistral Small 3.1 (24B)", "Dense", "~14GB", 24, 4, "Q4_K_M",
     "Vision + text, excellent quality."),
    ("Qwen/Qwen3-32B",
     "Qwen 3 32B", "Dense", "~19GB", 32, 4, "Q4_K_M",
     "Top-tier coding, math, multilingual."),
    ("google/gemma-3-27b-it",
     "Gemma 3 27B", "Dense", "~16GB", 32, 4, "Q4_K_M",
     "Strong reasoning, multilingual."),
    ("custom", "Custom / Browse HuggingFace", "Custom", "?GB", 4, 0, "Q4_K_M",
     "Enter any HuggingFace repo ID or local GGUF path."),
]


def recommend_models(ram_gb, vram_gb):
    """Return list of (index, model) filtered to what user can realistically run."""
    custom_idx = len(MODELS) - 1
    filtered = []
    for i, m in enumerate(MODELS):
        if m[0] == "custom":
            continue
        if ram_gb and ram_gb >= m[4]:
            filtered.append((i, m))
    filtered.append((custom_idx, MODELS[custom_idx]))
    return filtered


# ── Model Selector ─────────────────────────────────────────────────────────────
def search_huggingface(query: str, limit: int = 10) -> list[tuple[str, str, str]]:
    """Search HuggingFace for GGUF models."""
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
    except Exception:
        return []


def interactive_model_select(ram_gb, vram_gb):
    """Streamlined model picker — scan first, then catalog."""
    candidates = recommend_models(ram_gb, vram_gb)
    detected = []

    # Quick scan for existing models
    print(f"  {_d('Scanning for models...')}")
    try:
        from .model_hub import scan_cached_models
        cached = scan_cached_models()
        for m in cached:
            detected.append({"path": m["path"], "name": m["name"],
                            "size_gb": m["size_gb"], "source": m["source"]})
    except Exception:
        pass

    while True:
        if detected:
            print(f"\n  {_g(f'Found {len(detected)} downloaded model(s)')}")
            for i, m in enumerate(detected):
                short = m["name"][:44]
                sz = m["size_gb"]
                src = m["source"]
                print(f"    {i+1}. {_b(short)}  {_d(f'{sz}GB [{src}]')}")

        print(f"\n  {_b('Model catalog')}")
        for i, (_, m) in enumerate(candidates):
            marker = _g("▸") if m[2] == "MoE" else " "
            print(f"    {marker} {_c(str(i+1) + '.')} {m[1]:<32} {_d(f'{m[3]} {m[2]}')}")

        prompt_num = len(detected) + len(candidates)
        print(f"\n  {_d(f'1-{len(detected)} downloaded, {len(detected)+1}-{prompt_num} catalog, or paste a path')}")
        try:
            raw = input(f"  {_b('>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

        if not raw:
            # Default to first detected or first catalog
            if detected:
                return _build_detected_tuple(detected[0], ram_gb)
            return candidates[0][1]

        # Try as number
        try:
            num = int(raw)
            if 1 <= num <= len(detected):
                return _build_detected_tuple(detected[num - 1], ram_gb)
            cat_num = num - len(detected)
            if 1 <= cat_num <= len(candidates):
                selected = candidates[cat_num - 1][1]
                if selected[0] == "custom":
                    return _handle_custom_select(ram_gb)
                return selected
            continue
        except ValueError:
            pass

        # Try as path
        p = Path(raw.strip('"').strip("'"))
        if p.exists():
            return _handle_path_input(p, ram_gb)

        # Try as HF search
        q_text = raw
        search_msg = f'Searching HuggingFace for "{q_text}"...'
        print(f"  {_d(search_msg)}")
        results = search_huggingface(raw, limit=10)
        if not results:
            print(f"  {_y('No results. Try a model number or path.')}")
            continue
        for i, (repo, name, size) in enumerate(results):
            print(f"    {i+1}. {name:<30} {_d(f'{size}')} {_d(f'{repo}')}")
        try:
            pick = input(f"\n  {_b('>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            continue
        if pick and pick.isdigit():
            idx = int(pick) - 1
            if 0 <= idx < len(results):
                repo, name, size = results[idx]
                return (repo, name, "Custom", size, ram_gb or 8, 0, "Q4_K_M",
                        f"HuggingFace: {repo}")
        print(f"  {_y('Invalid selection.')}")
        continue


def _build_detected_tuple(det, ram_gb):
    """Build a model tuple from a detected downloaded model."""
    from .model_hub import _detect_gguf_quant
    quant = _detect_gguf_quant(det["name"])
    return (
        str(det["path"]),
        det["name"][:30],
        "Downloaded",
        f"~{det['size_gb']}GB",
        ram_gb or 8,
        0,
        quant if quant != "unknown" else "Q4_K_M",
        f"Local: {det['source']}",
    )


def _handle_custom_select(ram_gb):
    """Handle custom model selection (HF search or manual entry)."""
    mode = radio("Find a model:", [
        ("Search HuggingFace", "search"),
        ("Enter repo ID or path", "manual"),
        ("Back to catalog", "back"),
    ], default=0)
    if mode == "back":
        return None  # caller should loop
    if mode == "manual":
        repo_id = text_input("Model ID or path", default="")
        if repo_id:
            return (repo_id, repo_id.split("/")[-1][:30], "Custom", "?GB",
                    ram_gb or 8, 0, "Q4_K_M", f"Custom: {repo_id}")
        return None
    # search
    query = text_input("Search query", default="gguf instruct")
    if not query:
        return None
    print(f"  {_d('Searching')}: {_b(query)}")
    results = search_huggingface(query, limit=10)
    if not results:
        print(f"  {_y('No results.')}")
        return None
    for i, (repo, name, size) in enumerate(results):
        print(f"    {i+1}. {name:<30} {_d(f'{size}')} {_d(f'{repo}')}")
    try:
        pick = input(f"\n  {_b('>')} ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if pick and pick.isdigit():
        idx = int(pick) - 1
        if 0 <= idx < len(results):
            repo, name, size = results[idx]
            return (repo, name, "Custom", size, ram_gb or 8, 0, "Q4_K_M",
                    f"HuggingFace: {repo}")
    return None


def _handle_path_input(p, ram_gb):
    """Handle a file/folder path input for model selection."""
    from .model_hub import find_gguf_files, _detect_gguf_quant

    if p.is_dir():
        gguf_files = find_gguf_files(p, recursive=True)
        if not gguf_files:
            print(f"  {_y('No .gguf files in that folder.')}")
            return None
        if len(gguf_files) == 1:
            f = gguf_files[0]
            size_gb = round(f.stat().st_size / (1024**3), 1)
            quant = _detect_gguf_quant(f.name)
            return (str(f.resolve()), f.name[:30], "Downloaded",
                    f"~{size_gb}GB", ram_gb or 8, 0,
                    quant if quant != "unknown" else "Q4_K_M",
                    f"Local: {f.parent}")
        for i, f in enumerate(gguf_files):
            size_gb = round(f.stat().st_size / (1024**3), 1)
            print(f"    {i+1}. {f.name[:44]}  {_d(f'{size_gb}GB')}")
        try:
            pick = input(f"\n  {_b('>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if pick and pick.isdigit():
            idx = int(pick) - 1
            if 0 <= idx < len(gguf_files):
                f = gguf_files[idx]
                size_gb = round(f.stat().st_size / (1024**3), 1)
                quant = _detect_gguf_quant(f.name)
                return (str(f.resolve()), f.name[:30], "Downloaded",
                        f"~{size_gb}GB", ram_gb or 8, 0,
                        quant if quant != "unknown" else "Q4_K_M",
                        f"Local: {f.parent}")
        return None

    if p.is_file() and p.suffix.lower() == ".gguf":
        size_gb = round(p.stat().st_size / (1024**3), 1)
        quant = _detect_gguf_quant(p.name)
        return (str(p.resolve()), p.name[:30], "Downloaded",
                f"~{size_gb}GB", ram_gb or 8, 0,
                quant if quant != "unknown" else "Q4_K_M",
                f"Local: {p.parent}")

    print(f"  {_y('Not a valid .gguf file or folder.')}")
    return None


# ── Offload Strategy ──────────────────────────────────────────────────────────
def recommend_offload(ram_gb, vram_gb, model_type, model_size_gb=0):
    """Return recommended offload config that uses ALL available memory."""
    total_mem = (vram_gb or 0) + (ram_gb or 32)
    model_fits = total_mem >= model_size_gb if model_size_gb > 0 else True

    vram_s = vram_gb or 0
    ram_s = ram_gb or 32
    ssd_en = not model_fits and has_ssd()

    if model_type == "MoE":
        if vram_gb and ram_gb:
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
        "vram_gb": vram_s,
        "ram_gb": ram_s,
        "ssd_enabled": ssd_en,
        "hot_frac": hot_frac,
        "ram_frac": ram_frac,
        "ssd_frac": ssd_frac,
    }


def configure_offload(ram_gb, vram_gb, model_type, model_size_gb=0):
    """Streamlined offload config — show recommendation, accept or tweak."""
    rec = recommend_offload(ram_gb, vram_gb, model_type, model_size_gb)
    total_mem = (vram_gb or 0) + (ram_gb or 32)
    fits = total_mem >= model_size_gb if model_size_gb > 0 else True

    if fits:
        msg = f"{total_mem}GB available, model needs {model_size_gb}GB"
        print(f"  {_g('✓')} {_d(f'{msg} — SSD off')}")
    else:
        print(f"  {_y('⚠')} {_d(f'{model_size_gb}GB model exceeds {total_mem}GB memory — SSD tier on')}")

    _kv("Hot (VRAM)", f"{rec['vram_gb']}GB · {rec['hot_frac']:.0%} experts")
    _kv("Warm (RAM)", f"{rec['ram_gb']}GB · {rec['ram_frac']:.0%} experts")
    if rec["ssd_enabled"]:
        _kv("Cold (SSD)", f"{rec['ssd_frac']:.0%} experts")

    if confirm("Accept this config?", default=True):
        return rec

    # Let user adjust hot fraction (the main knob)
    if model_type == "MoE" and vram_gb:
        choices = [
            ("10% VRAM — faster RAM offload", 10),
            ("20% VRAM — balanced", 20),
            ("30% VRAM — speed focused", 30),
            ("50% VRAM — max speed", 50),
        ]
        sel = radio("Hot expert fraction:", choices, default=1)
        hot_frac = sel / 100
        rec["hot_frac"] = hot_frac
        rec["ram_frac"] = 1.0 - hot_frac

    return rec


# ── Quantization ──────────────────────────────────────────────────────────────
QUANT_OPTIONS = [
    ("Q4_K_M — best quality/size", "Q4_K_M"),
    ("Q5_K_M — better quality, +15% size", "Q5_K_M"),
    ("Q6_K — near-lossless, +30%", "Q6_K"),
    ("Q8_0 — FP16 quality, 2× larger", "Q8_0"),
    ("Q3_K_M — smaller, slight loss", "Q3_K_M"),
    ("Q2_K — smallest, notable loss", "Q2_K"),
]


def pick_quant():
    """Minimal quant selection — one line prompt."""
    names = "/".join(q[1] for q in QUANT_OPTIONS[:4])
    print(f"  {_d(f'Options: {names}')}")
    while True:
        q = text_input("Quantization", default="Q4_K_M")
        q_upper = q.upper()
        if q_upper in [x[1] for x in QUANT_OPTIONS]:
            return q_upper
        quant_msg = f'Unknown quant "{q}". Pick from the list above.'
        print(f"  {_y(quant_msg)}")

# ── Download Models via HF Hub ────────────────────────────────────────────────
def download_model(model_id, quant="Q4_K_M", progress_cb=None):
    """Download a GGUF model using huggingface_hub."""
    from .model_hub import resolve_model_path
    try:
        existing = resolve_model_path(model_id, quant=quant)
        if existing and existing.exists():
            message(f"Using cached model: {existing.name}", "ok")
            return True, str(existing)
    except FileNotFoundError:
        pass

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False, "huggingface_hub not installed. Run: pip install huggingface_hub"

    api = HfApi()
    try:
        models = api.list_repo_files(model_id, repo_type="model")
        gguf_files = [f for f in models if f.lower().endswith(".gguf")]
        if not gguf_files:
            return False, f"No GGUF files found in {model_id}."

        target = next((f for f in gguf_files if quant.upper() in f.upper()), gguf_files[0])
        message(f"Downloading {target}...", "info")

        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=model_id,
            filename=target,
            repo_type="model",
            resume_download=True,
        )
        message(f"Downloaded: {path}", "ok")
        return True, path
    except Exception as e:
        return False, str(e)


# ── Verify / Run Tests ─────────────────────────────────────────────────────────
def _venv_python() -> str:
    """Return the Python executable from the venv."""
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
    ok, _ = run([py, "-m", "pip", "install", "pytest", "-q"], fatal=False)
    ok, out = run([py, "-m", "pytest", "tests/test_sparse.py", "--tb=line", "-q"], fatal=False)
    if ok and "passed" in out.lower():
        message("Tests pass", "ok")
        return True
    else:
        message("Tests skipped (non-blocking)", "warn")
        return True


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
    try:
        import yaml
        with open(output, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except ImportError:
        with open(output.replace(".yaml", ".json"), "w") as f:
            json.dump(cfg, f, indent=2)
    return cfg


# ── Main Wizard ────────────────────────────────────────────────────────────────
def main():
    print()
    print(f"  {_b(_c('VibeBlade'))} {_d('setup')}")

    # ── Detect hardware ────────────────────────────────────────────────────
    ram  = get_ram_gb()
    vram_total, vram_used = get_vram_gb()
    gpu  = get_gpu_name()
    ssd  = has_ssd()
    disk = get_disk_free_gb()
    inet = has_inet()

    # Fallbacks
    if ram is None:
        try:
            ans = input(f"  {_y('RAM (GB)?')} ").strip()
            if ans and ans.isdigit():
                ram = int(ans)
        except (EOFError, KeyboardInterrupt):
            ram = 16

    py_ok = check_python()
    if not py_ok:
        print(f"\n  {_r('Python 3.10+ required.')}")
        sys.exit(1)

    # System summary — one compact block
    print()
    vram_str = f"{vram_total}GB {gpu}" if gpu and vram_total else ("No GPU" if not gpu else f"{vram_total}GB")
    ram_str = f"{ram}GB" if ram else "?GB"
    disk_str = f"{disk}GB free" if disk else "?"
    py_str = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"  {_c(f'{vram_str}')} · {_b(ram_str)} RAM · Py {py_str} · {'SSD' if ssd else 'HDD'} · {disk_str}")

    # ── Model selection ────────────────────────────────────────────────────
    _step(1, 4, "Model")

    model = interactive_model_select(ram, vram_total or 0)
    if model is None:
        # Custom select returned None (back)
        model = MODELS[0]  # fallback

    print(f"\n  {_g('→')} {_b(model[1])}  {_d(f'{model[3]} {model[2]}')}")

    custom_id = None
    if model[0] == "custom":
        custom_id = text_input("Model ID or path")
        if not custom_id:
            print(f"  {_r('No model specified.')}")
            sys.exit(1)

    # ── Offload config ─────────────────────────────────────────────────────
    _step(2, 4, "Memory")

    import re
    size_match = re.search(r"(\d+)", model[3])
    model_size_gb = int(size_match.group(1)) if size_match else 0
    offload = configure_offload(ram, vram_total or 0, model[2], model_size_gb)

    # ── Quantization ───────────────────────────────────────────────────────
    _step(3, 4, "Quantization")
    quant = pick_quant()

    # ── Install + config ───────────────────────────────────────────────────
    _step(4, 4, "Install")

    venv_dir = Path.cwd() / "venv"

    if not venv_dir.exists():
        message("Creating venv...", "info")
        run(["python3", "-m", "venv", str(venv_dir)], fatal=True)
        message("Venv created", "ok")

    # Detect pip
    scripts_dir = "Scripts" if platform.system() == "Windows" else "bin"
    pip_exe = venv_dir / scripts_dir / ("pip.exe" if platform.system() == "Windows" else "pip")
    py_exe = venv_dir / scripts_dir / ("python.exe" if platform.system() == "Windows" else "python")

    if not py_exe.exists():
        py_exe = Path(_venv_python())
        pip_exe = py_exe.parent / ("pip.exe" if platform.system() == "Windows" else "pip")

    message("Installing VibeBlade...", "info")
    run([str(pip_exe), "install", "--upgrade", "pip", "-q"], fatal=False)

    accel_extra, accel_name = detect_accel_backend()
    extra = accel_extra
    ok, _ = run([str(pip_exe), "install", "-e", f".{extra}", "-q"], fatal=False)
    if ok:
        message(f"Installed ({accel_name})", "ok")
    else:
        run([str(pip_exe), "install", "-e", ".", "-q"], fatal=False)
        message("Installed (base)", "ok")

    verify_install()

    # Write config
    model_id = custom_id if custom_id else model[0]
    write_config(model_id, offload, quant)
    message("Config saved → vibeblade.yaml", "ok")

    # Optional download
    if model[2] == "Downloaded":
        pass  # already local
    elif inet and model[0] != "custom":
        if confirm("Download model now?", default=False):
            download_model(model_id, quant)

    # ── Summary + Hatch ────────────────────────────────────────────────────
    print()
    print(f"  {_g(_b('✓ Ready'))}")
    _kv("Model", f"{model[1]} ({quant})")
    _kv("Config", "vibeblade.yaml")
    _kv("Memory", f"{offload['vram_gb']}GB VRAM + {offload['ram_gb']}GB RAM"
                  + (" + SSD" if offload["ssd_enabled"] else ""))

    print(f"\n  {_c(_b('Hatching chat...'))}")
    print()

    try:
        from .chat import chat_loop
        chat_loop(model_path=model_id)
    except Exception as e:
        print(f"  {_r(f'Chat failed: {e}')}")
        print(f"  {_d('Run manually: python -m vibeblade chat --config vibeblade.yaml')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n  {_y('Interrupted. Re-run:')} python -m vibeblade wizard")
        sys.exit(0)
