"""Model discovery — find GGUF files from HuggingFace cache, LM Studio, or local paths.

VibeBlade searches for models in this order:
1. Exact local file path
2. HF cache (~/.cache/huggingface/hub/)
3. LM Studio directory (%LOCALAPPDATA%/LM Studio/ on Windows, ~/.cache/lm-studio/ on Linux)
4. Current working directory
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ── File Classification ──────────────────────────────────────────────────────

# Common GGUF quantization patterns
_GGUF_QUANT_RE = re.compile(
    r"[._-]([IQ]?[QBKM][0-9](?:_[0-9A-Z]+)?)\b",
    re.IGNORECASE,
)

# Direct match for common quant types (more specific, checked first)
_DIRECT_QUANTS = [
    "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_S", "IQ1_S",
    "IQ4_NL", "IQ3_M", "IQ2_M", "IQ1_M",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0",
    "F16", "F32", "BF16", "FP16",
]


def _detect_gguf_quant(filename: str) -> str:
    """Extract quantization type from a GGUF filename.

    Examples:
        "model.Q4_K_M.gguf" -> "Q4_K_M"
        "model-f16.gguf" -> "F16"
        "model.IQ4_XS.gguf" -> "IQ4_XS"
        "model.Q8_0.gguf" -> "Q8_0"
    """
    upper = filename.upper()
    # Try direct match first (most common quant types)
    for q in _DIRECT_QUANTS:
        if q in upper:
            return q
    # Fallback regex
    m = _GGUF_QUANT_RE.search(upper)
    if m:
        return m.group(1).upper()
    return "unknown"


def classify_file(filename: str) -> tuple[str, str]:
    """Classify a model file by format and quantization.

    Returns (format, quant_type).
    """
    name = filename.lower()
    if name.endswith(".gguf"):
        return "gguf", _detect_gguf_quant(filename)
    if name.endswith(".safetensors"):
        if "gptq" in name:
            return "gptq", "GPTQ"
        if "awq" in name:
            return "awq", "AWQ"
        return "safetensors", "full"
    return "unknown", "unknown"


def is_quantized_file(filename: str) -> bool:
    """Check if a filename indicates a quantized model."""
    name = filename.lower()
    if name.endswith(".gguf"):
        q = _detect_gguf_quant(filename)
        return q != "F16" and q != "F32" and q != "BF16" and q != "FP16" and q != "unknown"
    return "awq" in name or "gptq" in name or "int4" in name or "int8" in name


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class QuantizedFile:
    """A quantized model file with metadata."""

    filename: str
    size_bytes: int
    quant_type: str
    format: str

    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        if self.size_bytes >= 1024**3:
            return f"{self.size_bytes / 1024**3:.1f} GB"
        if self.size_bytes >= 1024**2:
            return f"{self.size_bytes / 1024**2:.1f} MB"
        return f"{self.size_bytes} B"


@dataclass
class HubModel:
    """A model on the HuggingFace Hub."""

    model_id: str
    author: str
    formats: list[str]
    quantized_files: list[QuantizedFile]

    @property
    def file_count(self) -> int:
        return len(self.quantized_files)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "author": self.author,
            "formats": self.formats,
            "file_count": self.file_count,
            "quantized_files": [
                {"filename": f.filename, "size": f.size_human, "quant": f.quant_type}
                for f in self.quantized_files
            ],
        }


def find_gguf_files(
    directory: str | Path,
    recursive: bool = True,
    quant_filter: str = "",
) -> list[Path]:
    """Scan a directory for .gguf files.

    Parameters
    ----------
    directory:
        Directory to scan.
    recursive:
        If True, search subdirectories.
    quant_filter:
        If set, only return files matching this quantization (e.g. "Q4_K_M").
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    pattern = "*.gguf"
    if recursive:
        files = sorted(directory.rglob(pattern))
    else:
        files = sorted(directory.glob(pattern))

    if quant_filter:
        q = quant_filter.upper()
        files = [f for f in files if q in f.name.upper()]

    return files


def find_in_hf_cache(
    repo_id: str = "",
    quant_filter: str = "",
) -> list[Path]:
    """Search HuggingFace hub cache for GGUF files.

    HF cache layout: ~/.cache/huggingface/hub/models--<org>--<name>/blobs/
    or: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/

    On Windows: %USERPROFILE%\\.cache\\huggingface\\hub\\
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_root.is_dir():
        return []

    # Normalize repo_id for directory matching: "org/name" -> "org--name"
    repo_prefix = repo_id.replace("/", "--").lower() if repo_id else ""

    results: list[Path] = []
    for model_dir in cache_root.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue

        # Filter by repo_id if provided
        if repo_prefix and repo_prefix not in model_dir.name.lower():
            continue

        # Check snapshots and blobs for .gguf
        snapshots = model_dir / "snapshots"
        if snapshots.is_dir():
            for snap in snapshots.iterdir():
                if snap.is_dir():
                    results.extend(find_gguf_files(snap, recursive=True, quant_filter=quant_filter))

        # Also check blobs (HF stores files as SHA256 hashes)
        blobs = model_dir / "blobs"
        if blobs.is_dir():
            for f in blobs.iterdir():
                if f.suffix == "" and f.stat().st_size > 1_000_000:  # > 1MB, likely a model
                    # Check refs to see if this blob is a .gguf
                    refs_file = model_dir / "refs" / "main"
                    if refs_file.exists():
                        # Parse the pointers file to resolve blob names
                        pass  # Blobs without extensions are harder — skip

    return results


def find_in_lm_studio(
    quant_filter: str = "",
) -> list[Path]:
    """Search LM Studio model directories for GGUF files.

    Windows: %LOCALAPPDATA%/LM Studio/models/
    macOS:   ~/Library/Application Support/LM Studio/models/
    Linux:   ~/.cache/lm-studio/models/
    """
    candidates = []

    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.append(Path(local_app) / "LM Studio" / "models")
        candidates.append(Path.home() / ".lmstudio" / "models")
    elif sys.platform == "darwin":
        candidates.append(Path.home() / "Library" / "Application Support" / "LM Studio" / "models")
        candidates.append(Path.home() / ".lmstudio" / "models")
    else:
        candidates.append(Path.home() / ".cache" / "lm-studio" / "models")
        candidates.append(Path.home() / ".lmstudio" / "models")

    results: list[Path] = []
    for d in candidates:
        results.extend(find_gguf_files(d, recursive=True, quant_filter=quant_filter))

    return results


def find_in_ollama(
    quant_filter: str = "",
) -> list[Path]:
    """Search Ollama model storage for GGUF files.

    Windows: %USERPROFILE%/.ollama/models/manifests/
    macOS/Linux: ~/.ollama/models/manifests/
    Ollama stores models as GGUF blobs inside per-registry subdirectories.
    """
    candidates = []

    ollama_root = Path.home() / ".ollama" / "models"
    # Ollama stores manifests and blobs
    candidates.append(ollama_root / "manifests")
    candidates.append(ollama_root / "blobs")

    results: list[Path] = []
    for d in candidates:
        results.extend(find_gguf_files(d, recursive=True, quant_filter=quant_filter))

    return results


def find_in_gpt4all(
    quant_filter: str = "",
) -> list[Path]:
    """Search GPT4All model directories for GGUF files.

    Windows: %LOCALAPPDATA%/nomic.ai/GPT4All/
    macOS:   ~/Library/Application Support/nomic.ai/GPT4All/
    Linux:   ~/.local/share/nomic.ai/GPT4All/
    """
    candidates = []

    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.append(Path(local_app) / "nomic.ai" / "GPT4All")
    elif sys.platform == "darwin":
        candidates.append(Path.home() / "Library" / "Application Support" / "nomic.ai" / "GPT4All")
    else:
        candidates.append(Path.home() / ".local" / "share" / "nomic.ai" / "GPT4All")

    results: list[Path] = []
    for d in candidates:
        results.extend(find_gguf_files(d, recursive=True, quant_filter=quant_filter))

    return results


def find_in_directory(
    repo_id: str = "",
    quant_filter: str = "",
) -> list[Path]:
    """Search current directory and common local model paths."""
    results: list[Path] = []

    # Current directory
    results.extend(find_gguf_files(Path.cwd(), recursive=True, quant_filter=quant_filter))

    # Common model directories
    for d in ["models", "gguf", "weights", ".models"]:
        p = Path.cwd() / d
        results.extend(find_gguf_files(p, recursive=True, quant_filter=quant_filter))

    return results


def resolve_model_path(
    model_input: str,
    quant: str = "",
) -> Path:
    """Resolve a model identifier to a local .gguf file path.

    Parameters
    ----------
    model_input:
        One of:
        - Absolute or relative path to a .gguf file (returned as-is if it exists)
        - HuggingFace repo ID (e.g. "lmstudio-community/Qwen3.6-35B-A3B-GGUF")
        - Model name fragment (e.g. "Qwen3.6-35B")
    quant:
        Preferred quantization filter (e.g. "Q4_K_M").

    Returns
    -------
    Path to the .gguf file.

    Raises
    ------
    FileNotFoundError if no matching model is found.
    """
    model_path = Path(model_input)

    # 1. Direct file path
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path.resolve()

    # 2. Path without extension — try appending .gguf
    if model_path.is_file():
        return model_path.resolve()

    # 3. Search by repo_id in HF cache
    print(f"  Searching for model: {model_input}")
    found = find_in_hf_cache(repo_id=model_input, quant_filter=quant)
    if found:
        # Pick the best match — prefer files matching quant
        if quant:
            q_match = [f for f in found if quant.upper() in f.name.upper()]
            if q_match:
                found = q_match
        return found[0].resolve()

    # 4. Search LM Studio
    found = find_in_lm_studio(quant_filter=quant)
    if found:
        # Filter by model name if repo_id provided
        if model_input:
            name_parts = model_input.lower().replace("/", "-").split("--")
            name_match = [
                f for f in found
                if any(part.lower() in f.name.lower() for part in name_parts)
            ]
            if name_match:
                found = name_match
        return found[0].resolve()

    # 5. Search Ollama
    found = find_in_ollama(quant_filter=quant)
    if found:
        if model_input:
            name_match = [
                f for f in found
                if model_input.lower() in f.name.lower()
            ]
            if name_match:
                found = name_match
        return found[0].resolve()

    # 6. Search GPT4All
    found = find_in_gpt4all(quant_filter=quant)
    if found:
        if model_input:
            name_match = [
                f for f in found
                if model_input.lower() in f.name.lower()
            ]
            if name_match:
                found = name_match
        return found[0].resolve()

    # 7. Search local directories
    found = find_in_directory(quant_filter=quant)
    if found and model_input:
        name_match = [
            f for f in found
            if model_input.lower() in f.name.lower()
        ]
        if name_match:
            found = name_match
        else:
            # No local file matches the requested model — don't return a random file
            found = []
    if found:
        return found[0].resolve()

    # 8. Nothing found — raise with helpful message
    raise FileNotFoundError(
        f"Could not find model '{model_input}'.\n"
        f"\n"
        f"Searched:\n"
        f"  - Local path: {model_path.resolve()}\n"
        f"  - HuggingFace cache: ~/.cache/huggingface/hub/\n"
        f"  - LM Studio: %LOCALAPPDATA%/LM Studio/models/\n"
        f"  - Ollama: ~/.ollama/models/\n"
        f"  - GPT4All: %LOCALAPPDATA%/nomic.ai/GPT4All/\n"
        f"  - Current directory: {Path.cwd()}\n"
        f"\n"
        f"To download: python -m vibeblade wizard"
    )

def scan_cached_models(quant_filter: str = "") -> list[dict]:
    """Scan all known model locations and return a summary of found models.

    Returns list of {"path": Path, "name": str, "size_gb": float, "source": str}
    """
    seen: set[str] = set()
    models: list[dict] = []

    def _add(files: list[Path], source: str):
        for f in files:
            resolved = str(f.resolve())
            if resolved not in seen:
                seen.add(resolved)
                size_gb = f.stat().st_size / (1024**3)
                models.append({
                    "path": f.resolve(),
                    "name": f.name,
                    "size_gb": round(size_gb, 1),
                    "source": source,
                })

    _add(find_in_hf_cache(quant_filter=quant_filter), "HuggingFace cache")
    _add(find_in_lm_studio(quant_filter=quant_filter), "LM Studio")
    _add(find_in_ollama(quant_filter=quant_filter), "Ollama")
    _add(find_in_gpt4all(quant_filter=quant_filter), "GPT4All")
    _add(find_in_directory(quant_filter=quant_filter), "Local")

    # Sort by size descending
    models.sort(key=lambda m: m["size_gb"], reverse=True)
    return models


def is_model_cached(repo_id: str, quant: str = "") -> bool:
    """Check if a model is already downloaded locally."""
    try:
        resolve_model_path(repo_id, quant=quant)
        return True
    except FileNotFoundError:
        return False


