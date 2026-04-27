"""VibeBlade Model Manager — Local model registry, format detection, and lifecycle.

Scans the local models directory, detects formats (GGUF, safetensors, AWQ, GPTQ, FP16),
and provides a registry for model management.  Also auto-detects models stored by
LM Studio, Ollama, llama.cpp, KoboldCpp, text-generation-webui, HuggingFace Hub
cache, and VibeBlade itself.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = str(Path.home() / ".vibeblade" / "models")
REGISTRY_FILE = "models.json"

# ── External model store directories ─────────────────────────────────────────
#
# Supports: LM Studio, Ollama, llama.cpp, KoboldCpp, text-generation-webui,
# HuggingFace Hub cache, and VibeBlade itself.
#
# Each entry: (source_name, env_var, posix_fallback, windows_expansions)
# - env_var:           checked on ALL platforms first (user override)
# - posix_fallback:    relative to ~/.  Used on Linux + macOS.
# - windows_expansions: list of %VAR% templates expanded on Windows only.

EXTERNAL_DIR_DEFS: list[tuple[str, str | None, str, list[str]]] = [
    # ── LM Studio ──────────────────────────────────────────────────────────
    # LM Studio stores models under ~/.lmstudio/hub/models (v0.3+).
    # Older installs may use .lm-studio or .cache/lm-studio.
    # On Windows, also checks %LOCALAPPDATA%\LM Studio\models (AppImage-style).
    ("lm_studio", "LM_STUDIO_MODELS_DIR", ".lmstudio/hub/models", [
        "{USERPROFILE}\\.lmstudio\\hub\\models",
        "{USERPROFILE}\\.lm-studio\\hub\\models",
        "{USERPROFILE}\\.lmstudio\\models",
        "{USERPROFILE}\\.lm-studio\\models",
        "{LOCALAPPDATA}\\LM Studio\\models",
        "{LOCALAPPDATA}\\lm-studio\\models",
        "{APPDATA}\\LM Studio\\models",
        # Older Linux/macOS fallback
        "~/.cache/lm-studio/models",
    ]),

    # ── Ollama ─────────────────────────────────────────────────────────────
    ("ollama", "OLLAMA_MODELS", ".ollama/models", [
        # Ollama on Windows stores under %LOCALAPPDATA%\Programs\Ollama or
        # %USERPROFILE%\.ollama\models  (note: "models" for Linux/macOS,
        # "models" is the canonical dir on all platforms)
        "{USERPROFILE}\\.ollama\\models",
        "{LOCALAPPDATA}\\Programs\\Ollama\\models",
        "{LOCALAPPDATA}\\Ollama\\models",
    ]),

    # ── llama.cpp ──────────────────────────────────────────────────────────
    # Users typically run llama.cpp from the dir containing their .gguf files,
    # but many keep a dedicated models folder.
    ("llama_cpp", "LLAMA_CPP_MODELS_DIR", "llama.cpp/models", [
        "{USERPROFILE}\\llama.cpp\\models",
        "{USERPROFILE}\\llama-cpp\\models",
        "{USERPROFILE}\\Documents\\llama.cpp\\models",
    ]),

    # ── KoboldCpp ──────────────────────────────────────────────────────────
    # KoboldCpp ships as a portable exe; models go in a "models" subfolder
    # next to the binary, or in a user-chosen location.  Common defaults:
    ("koboldcpp", "KOBOLDCPP_MODELS_DIR", "KoboldCpp/models", [
        "{USERPROFILE}\\KoboldCpp\\models",
        "{USERPROFILE}\\Documents\\KoboldCpp\\models",
        "{USERPROFILE}\\Downloads\\KoboldCpp\\models",
    ]),

    # ── text-generation-webui (oobabooga) ──────────────────────────────────
    # Default install path + the "models" subfolder inside the repo clone.
    ("textgen_webui", "TEXTGEN_MODELS_DIR", "text-generation-webui/models", [
        "{USERPROFILE}\\text-generation-webui\\models",
        "{USERPROFILE}\\Documents\\text-generation-webui\\models",
    ]),

    # ── HuggingFace Hub cache ──────────────────────────────────────────────
    ("huggingface", None, ".cache/huggingface/hub", []),

    # ── VibeBlade own models directory ────────────────────────────────────
    ("vibeblade", "VIBEBlade_MODELS_DIR", ".vibeblade/models", [
        "{USERPROFILE}\\.vibeblade\\models",
    ]),
]


@dataclass
class ModelRecord:
    """A registered model in the local registry."""
    id: str                          # Unique ID (e.g., "bartowski_llama-3.1-8b-instruct-gguf")
    model_id: str                    # HuggingFace model ID (if from HF)
    name: str                        # Human-readable name
    path: str                        # Absolute path to model directory or file
    format: str                      # "gguf", "safetensors", "awq", "gptq"
    quant_type: str                  # "Q4_K_M", "AWQ-4bit", "full", etc.
    size_bytes: int = 0
    added_at: str = ""               # ISO timestamp
    last_used: str = ""              # ISO timestamp
    source: str = "local"            # "local", "lm_studio", "huggingface", "ollama"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.added_at:
            self.added_at = datetime.utcnow().isoformat()

    @property
    def size_human(self) -> str:
        b = self.size_bytes
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def detect_model_format(path: str) -> tuple[str, str]:
    """Detect model format and quantization type from a file or directory.

    Args:
        path: Path to a .gguf file, directory with .safetensors, etc.

    Returns:
        (format, quant_type) e.g. ("gguf", "Q4_K_M")
    """
    p = Path(path)

    if p.is_file():
        if p.suffix.lower() == ".gguf":
            from .model_hub import _detect_gguf_quant
            return "gguf", _detect_gguf_quant(p.name)
        if p.suffix.lower() == ".safetensors":
            return "safetensors", "full"
        if p.suffix.lower() == ".bin":
            # Could be FP16, GPTQ, or AWQ — detect from parent context
            parent_lower = p.parent.name.lower() if p.parent else ""
            if "gptq" in parent_lower:
                return "gptq", "GPTQ-4bit"
            if "awq" in parent_lower:
                return "awq", "AWQ-4bit"
            return "bin", "FP16"
        return "unknown", "unknown"

    if p.is_dir():
        # Check for GGUF files first
        gguf_files = list(p.glob("*.gguf"))
        if gguf_files:
            from .model_hub import _detect_gguf_quant
            best = _pick_largest(gguf_files)
            return "gguf", _detect_gguf_quant(best.name)

        # Check for safetensors
        st_files = list(p.glob("*.safetensors"))
        if st_files:
            return "safetensors", "full"

        # Check for AWQ model files
        for f in p.glob("*"):
            if ".awq." in f.name.lower() or "awq" in f.name.lower():
                return "awq", "AWQ-4bit"

        # Check for GPTQ
        for f in p.glob("*.safetensors"):
            if "gptq" in f.name.lower() or f.parent.name.lower() == "gptq":
                return "gptq", "GPTQ-4bit"

    return "unknown", "unknown"


def _pick_largest(files: list[Path]) -> Path:
    """Pick the largest file from a list."""
    return max(files, key=lambda f: f.stat().st_size)


def _dir_size(path: Path) -> int:
    """Get total size of a directory."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _make_id(model_id: str, name: str) -> str:
    """Create a stable ID from model_id or name."""
    base = model_id or name
    return base.replace("/", "_").replace(" ", "-").lower()


def get_external_model_dirs() -> list[tuple[str, Path]]:
    """Return a list of ``(source_name, path)`` for well-known external model stores.

    Probes all supported tools (LM Studio, Ollama, llama.cpp, KoboldCpp,
    text-generation-webui, HuggingFace Hub, VibeBlade) on every platform.
    Environment variables take priority.  On Windows, additional path
    templates are expanded.  Only directories that actually exist are returned.
    """
    results: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def _add(source: str, p: Path) -> None:
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            results.append((source, p))

    for source_name, env_var, posix_rel, win_templates in EXTERNAL_DIR_DEFS:
        # 1. Environment variable override (all platforms)
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                p = Path(env_val)
                if p.is_dir():
                    _add(source_name, p)
                    continue

        # 2. POSIX fallback (Linux + macOS) — relative to home
        p = Path.home() / posix_rel
        if p.is_dir():
            _add(source_name, p)

        # 3. Windows-specific path templates
        if sys.platform == "win32" and win_templates:
            for tmpl in win_templates:
                try:
                    expanded = os.path.expandvars(tmpl)
                    p = Path(expanded)
                    if p.is_dir():
                        _add(source_name, p)
                except (OSError, ValueError):
                    pass

    return results


def get_all_external_dir_candidates() -> list[dict]:
    """Return ALL candidate directories per source, including non-existent ones.

    Used by the dashboard to show the user which paths were checked and
    which ones exist.  Returns a list of dicts::

        {"source": "lm_studio", "path": "/home/.../.lmstudio/hub/models", "exists": true}
    """
    results: list[dict] = []
    seen: set[str] = set()

    for source_name, env_var, posix_rel, win_templates in EXTERNAL_DIR_DEFS:
        # POSIX fallback
        p = Path.home() / posix_rel
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            results.append({
                "source": source_name,
                "path": key,
                "exists": p.is_dir(),
            })

        # Windows templates
        if sys.platform == "win32" and win_templates:
            for tmpl in win_templates:
                try:
                    expanded = os.path.expandvars(tmpl)
                    p = Path(expanded)
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "source": source_name,
                            "path": key,
                            "exists": p.is_dir(),
                        })
                except (OSError, ValueError):
                    pass

    return results


def _classify_source(path: Path, external_dirs: list[tuple[str, Path]]) -> str:
    """Determine the source label for *path* based on known external dirs."""
    resolved = str(path.resolve())
    for source_name, ext_dir in external_dirs:
        if resolved.startswith(str(ext_dir.resolve())):
            return source_name
    return "local"


class ModelManager:
    """Local model registry and lifecycle manager."""

    def __init__(self, models_dir: str = ""):
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.models_dir / REGISTRY_FILE
        self._registry: dict[str, ModelRecord] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                self._registry = {
                    k: ModelRecord.from_dict(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to load registry: %s", e)
                self._registry = {}

    def _save_registry(self):
        """Persist registry to disk."""
        data = {k: v.to_dict() for k, v in self._registry.items()}
        self.registry_path.write_text(json.dumps(data, indent=2))

    # ── CRUD ──

    def register(
        self,
        path: str,
        name: str = "",
        model_id: str = "",
        format: str = "",
        quant_type: str = "",
        source: str = "local",
    ) -> ModelRecord:
        """Register a model in the local registry.

        Auto-detects format if not specified.
        """
        p = Path(path).resolve()

        if not p.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")

        detected_fmt, detected_quant = detect_model_format(str(p))
        fmt = format or detected_fmt
        quant = quant_type or detected_quant

        if fmt == "unknown":
            logger.warning("Could not detect format for %s, registering as 'unknown'", path)

        # Compute size
        if p.is_file():
            size = p.stat().st_size
        else:
            size = _dir_size(p)

        auto_name = name or p.stem if p.is_file() else p.name

        rec_id = _make_id(model_id, auto_name)
        rec = ModelRecord(
            id=rec_id,
            model_id=model_id,
            name=auto_name,
            path=str(p),
            format=fmt,
            quant_type=quant,
            size_bytes=size,
            source=source,
        )

        self._registry[rec_id] = rec
        self._save_registry()
        logger.info("Registered model: %s (%s, %s, source=%s)", rec_id, fmt, quant, source)
        return rec

    def unregister(self, model_id: str) -> bool:
        """Remove a model from the registry (does not delete files)."""
        if model_id in self._registry:
            del self._registry[model_id]
            self._save_registry()
            return True
        return False

    def get(self, model_id: str) -> Optional[ModelRecord]:
        """Get a model by ID."""
        return self._registry.get(model_id)

    def list_models(
        self,
        fmt: str = "",
        search: str = "",
    ) -> list[ModelRecord]:
        """List all registered models, optionally filtered."""
        models = list(self._registry.values())

        if fmt:
            models = [m for m in models if m.format == fmt]
        if search:
            search_lower = search.lower()
            models = [m for m in models
                      if search_lower in m.name.lower()
                      or search_lower in m.model_id.lower()
                      or search_lower in m.format.lower()]

        # Sort by recently added
        models.sort(key=lambda m: m.added_at, reverse=True)
        return models

    def touch(self, model_id: str):
        """Update last_used timestamp."""
        rec = self._registry.get(model_id)
        if rec:
            rec.last_used = datetime.utcnow().isoformat()
            self._save_registry()

    def scan_directory(self, path: str = "", *, include_external: bool = True) -> list[ModelRecord]:
        """Scan a directory for model files and auto-register them.

        Returns newly registered models.

        If *path* is explicitly provided, that directory is scanned directly
        (the caller is responsible for the path).  If omitted, the default
        ``~/.vibeblade/models/`` directory is scanned.

        By default (*include_external=True*) this also scans well-known
        external model stores (LM Studio, HuggingFace Hub cache, Ollama)
        and registers any ``.gguf`` files found there in-place (no copying).
        """
        new_models = []

        # ── Scan the primary (or explicitly specified) directory ──
        requested = Path(path) if path else self.models_dir
        resolved = requested.resolve()

        if path:
            # User explicitly chose this path — trust it.
            # Still validate it actually exists and is a directory.
            if not resolved.is_dir():
                logger.warning("Scan path does not exist or is not a directory: %s", resolved)
            else:
                new_models.extend(self._scan_dir(resolved, "local"))
        else:
            # Auto-scan: only allow the default models_dir
            allowed = self.models_dir.resolve()
            if str(resolved).startswith(str(allowed)) and resolved.exists():
                new_models.extend(self._scan_dir(resolved, "local"))

        # ── Scan external model stores ──
        if include_external:
            external_dirs = get_external_model_dirs()
            for source_name, ext_dir in external_dirs:
                logger.debug("Scanning external dir: %s (%s)", ext_dir, source_name)
                new_models.extend(self._scan_dir(ext_dir, source_name))

        return new_models

    def scan_sources(self, sources: list[str]) -> list[ModelRecord]:
        """Scan only the specified external sources by name.

        Args:
            sources: list of source names, e.g. ["lm_studio", "ollama", "koboldcpp"]

        Returns:
            Newly registered models from the selected sources.
        """
        new_models: list[ModelRecord] = []
        valid_sources = {s[0] for s in EXTERNAL_DIR_DEFS}
        external_dirs = get_external_model_dirs()

        for source_name in sources:
            if source_name not in valid_sources:
                logger.warning("Unknown source '%s', skipping", source_name)
                continue
            for src, ext_dir in external_dirs:
                if src == source_name:
                    logger.info("Scanning %s at %s", source_name, ext_dir)
                    new_models.extend(self._scan_dir(ext_dir, source_name))

        return new_models

    def _scan_dir(self, base: Path, source: str) -> list[ModelRecord]:
        """Scan a single directory for model files and auto-register them.

        *source* is the label stored on every discovered ``ModelRecord``.

        Supports: ``.gguf``, ``.safetensors``, ``.bin`` (FP16/GPTQ), and
        directory-level detection for HuggingFace cache layouts.
        """
        if not base.exists():
            return []

        registered_paths = {r.path for r in self._registry.values()}
        new_models: list[ModelRecord] = []

        # Scan for GGUF files recursively (the primary format)
        for gguf in base.rglob("*.gguf"):
            gguf_str = str(gguf.resolve())
            if gguf_str not in registered_paths:
                rec = self.register(str(gguf), source=source)
                new_models.append(rec)

        # Scan for .bin files (FP16, GPTQ, AWQ — common in textgen/llama.cpp)
        for binf in base.rglob("*.bin"):
            binf_str = str(binf.resolve())
            if binf_str not in registered_paths:
                # Skip tiny files (< 10 MB) — likely tokenizers, not model weights
                if binf.stat().st_size < 10 * 1024 * 1024:
                    continue
                try:
                    rec = self.register(str(binf), source=source)
                    new_models.append(rec)
                except Exception:
                    pass

        # Scan for safetensors files/directories
        for st_dir in base.iterdir():
            if not st_dir.is_dir():
                continue
            st_str = str(st_dir.resolve())
            if st_str in registered_paths:
                continue
            if list(st_dir.glob("*.safetensors")):
                rec = self.register(str(st_dir), source=source)
                new_models.append(rec)

        # HuggingFace cache: look for snapshot dirs with model files inside
        if source == "huggingface":
            for snap_dir in base.glob("models--*--*/snapshots/*"):
                if snap_dir.is_dir():
                    snap_str = str(snap_dir.resolve())
                    if snap_str in registered_paths:
                        continue
                    has_model = (
                        list(snap_dir.glob("*.gguf"))
                        or list(snap_dir.glob("*.safetensors"))
                        or list(snap_dir.glob("*.bin"))
                    )
                    if has_model:
                        try:
                            rec = self.register(str(snap_dir), source=source)
                            new_models.append(rec)
                        except Exception:
                            pass

        return new_models

    def delete(self, model_id: str, delete_files: bool = False) -> bool:
        """Delete a model from registry. Optionally delete files.

        Security: when delete_files=True, the path is validated to be
        within models_dir to prevent arbitrary file deletion.
        """
        rec = self._registry.get(model_id)
        if not rec:
            return False

        if delete_files:
            p = Path(rec.path).resolve()
            allowed = self.models_dir.resolve()
            if not str(p).startswith(str(allowed)):
                logger.warning("Blocked deletion of non-allowed path: %s (allowed: %s)", p, allowed)
                return False
            if p.exists():
                if p.is_file():
                    p.unlink()
                    logger.info("Deleted file: %s", p)
                else:
                    import shutil
                    shutil.rmtree(p)
                    logger.info("Deleted directory: %s", p)

        del self._registry[model_id]
        self._save_registry()
        return True

    def get_stats(self) -> dict:
        """Get summary statistics."""
        models = list(self._registry.values())
        total_size = sum(m.size_bytes for m in models)
        formats = {}
        for m in models:
            formats[m.format] = formats.get(m.format, 0) + 1
        return {
            "total_models": len(models),
            "total_size_bytes": total_size,
            "total_size_human": _human_size(total_size),
            "formats": formats,
        }

    def export_registry(self) -> dict:
        """Export full registry as dict."""
        return {k: v.to_dict() for k, v in self._registry.items()}


def _human_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"
