"""VibeBlade Model Hub — HuggingFace integration for quantized model discovery & download.

Supports searching, browsing, and downloading pre-quantized models in GGUF,
AWQ, GPTQ, and safetensors formats. Compatible with LM Studio, llama.cpp,
Ollama, KoboldCpp, and other GGUF-consuming platforms.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Quantization format tags on HuggingFace
QUANT_TAGS = {
    "gguf": ["gguf"],
    "awq": ["awq"],
    "gptq": ["gptq", "gptq-4bit", "gptq-8bit"],
    "bnb": ["bitsandbytes", "bnb-4bit", "bnb-8bit"],
}

# Popular quantized model publishers (curated for quality)
POPULAR_QUANT_PUBLISHERS = [
    "bartowski",
    "TheBloke",
    "QuantFactory",
    "ikawrakow",
    "leliuga",
    "MaziyarPanahi",
]

# Recommended GGUF quant sizes (sorted by preference)
GGUF_QUANT_SIZES = [
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q3_K_M",
    "Q2_K",
    "F16",
    "F32",
    "IQ4_XS",
]

# ── Data Classes ──


@dataclass
class QuantizedFile:
    """A single quantized file available for download."""
    filename: str
    size_bytes: int
    quant_type: str  # e.g., "Q4_K_M", "Q5_K_S", "awq-4bit"
    format: str      # "gguf", "awq", "gptq", "safetensors"
    url: str = ""

    @property
    def size_human(self) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if self.size_bytes < 1024:
                return f"{self.size_bytes:.1f} {unit}"
            self.size_bytes /= 1024
        return f"{self.size_bytes:.1f} PB"

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "size_human": self.size_human,
            "quant_type": self.quant_type,
            "format": self.format,
            "url": self.url,
        }


@dataclass
class HubModel:
    """A model on HuggingFace with quantized files."""
    model_id: str
    author: str
    tags: list[str] = field(default_factory=list)
    likes: int = 0
    downloads: int = 0
    last_modified: Optional[str] = None
    quantized_files: list[QuantizedFile] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def name(self) -> str:
        return self.model_id.split("/")[-1]

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "author": self.author,
            "name": self.name,
            "tags": self.tags,
            "likes": self.likes,
            "downloads": self.downloads,
            "last_modified": self.last_modified,
            "formats": self.formats,
            "description": self.description[:200] if self.description else "",
            "quantized_files": [f.to_dict() for f in self.quantized_files],
            "file_count": len(self.quantized_files),
        }


# ── Quant Type Detection ──


def _detect_gguf_quant(filename: str) -> str:
    """Extract quantization type from GGUF filename.

    Examples:
        llama-2-7b-chat.Q4_K_M.gguf → "Q4_K_M"
        model.Q5_K_S.gguf → "Q5_K_S"
        model-f16.gguf → "F16"
        model.IQ4_XS.gguf → "IQ4_XS"
    """
    stem = Path(filename).stem.upper()
    # Try standard GGUF quant patterns
    patterns = [
        r"[.-](Q4_0|Q4_1|Q5_0|Q5_1|Q8_0)",
        r"[.-](Q[234568]_[KXM]_[SMLX])",
        r"[.-](Q[234568]_[KXM])",
        r"[.-](IQ[234]_\w+)",
        r"[.-](F(16|32))",
        r"[.-](BF16)",
        r"[.-](Q[234568]_\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            return m.group(1).upper()
    # Fallback: check if it's a plain GGUF (likely F16)
    return "F16"


def _detect_awq_quant(filename: str) -> str:
    """Detect AWQ quant type from filename."""
    lower = filename.lower()
    if "4bit" in lower or "g128" in lower:
        return "AWQ-4bit"
    if "8bit" in lower or "g64" in lower:
        return "AWQ-8bit"
    return "AWQ"


def _detect_gptq_quant(filename: str) -> str:
    """Detect GPTQ quant type from filename."""
    lower = filename.lower()
    if "4bit" in lower or "g128" in lower or "4-" in lower:
        return "GPTQ-4bit"
    if "8bit" in lower or "g64" in lower or "8-" in lower:
        return "GPTQ-8bit"
    if "3bit" in lower or "3-" in lower:
        return "GPTQ-3bit"
    return "GPTQ"


def classify_file(filename: str) -> tuple[str, str]:
    """Classify a file by format and quantization type.

    Returns:
        (format, quant_type) e.g. ("gguf", "Q4_K_M")
    """
    lower = filename.lower()

    if lower.endswith(".gguf"):
        return "gguf", _detect_gguf_quant(filename)

    if ".awq." in lower or "awq" in lower:
        return "awq", _detect_awq_quant(filename)

    if ".gptq." in lower or "gptq" in lower:
        return "gptq", _detect_gptq_quant(filename)

    if lower.endswith(".safetensors"):
        return "safetensors", "full"

    return "unknown", "unknown"


def is_quantized_file(filename: str) -> bool:
    """Check if a file is a quantized model file."""
    fmt, _ = classify_file(filename)
    return fmt in ("gguf", "awq", "gptq")


def _get_single_gguf_info(filename: str) -> Optional[str]:
    """For 'single-file' GGUF repos, detect the quant type from the one GGUF file."""
    if filename.lower().endswith(".gguf"):
        return _detect_gguf_quant(filename)
    return None


# ── Main Hub Interface ──


class ModelHub:
    """HuggingFace Model Hub client for discovering and downloading quantized models."""

    def __init__(self, token: str = ""):
        from huggingface_hub import HfApi  # lazy: optional dep

        self.api = HfApi(token=token or HF_TOKEN or None)

    def search(
        self,
        query: str = "",
        fmt: str = "gguf",
        limit: int = 20,
        sort: str = "downloads",
        tag: str = "",
    ) -> list[HubModel]:
        """Search HuggingFace for quantized models.

        Args:
            query: Search query (model name, architecture, etc.)
            fmt: Format filter — "gguf", "awq", "gptq", "all"
            limit: Max results
            sort: "downloads" | "likes" | "recent"
            tag: HuggingFace model tag for category filtering
                 (e.g., "moe", "text-generation", "code", "math")

        Returns:
            List of HubModel with quantized file info
        """
        # Build filter — use GGUF tag as the filter
        filters: list[str] = []
        if fmt == "gguf":
            filters.append("gguf")
        elif fmt in ("awq", "gptq"):
            filters.append(fmt)

        # Category tag filter (HuggingFace uses model-tags like "moe", "text-generation")
        if tag and tag.lower() not in ("all", "featured"):
            filters.append(tag)

        filter_val = ",".join(filters) if filters else None

        sort_val = "downloads"
        if sort == "likes":
            sort_val = "likes"
        elif sort == "recent":
            sort_val = "lastModified"

        models = self.api.list_models(
            search=query or None,
            filter=filter_val,
            limit=limit,
            sort=sort_val,
        )

        results = []
        for hf_model in models:
            try:
                hub_model = self._enrich_model(hf_model.id)
                if hub_model.quantized_files:
                    results.append(hub_model)
            except Exception as e:
                logger.debug("Failed to enrich %s: %s", hf_model.id, e)
                continue

        return results

    def search_gguf(
        self,
        query: str = "",
        quant_size: str = "",
        limit: int = 20,
    ) -> list[HubModel]:
        """Search specifically for GGUF models, optionally filtering by quant size."""
        results = self.search(query=query, fmt="gguf", limit=limit)

        if quant_size:
            quant_upper = quant_size.upper()
            filtered = []
            for m in results:
                m.quantized_files = [
                    f for f in m.quantized_files
                    if f.quant_type.upper() == quant_upper
                ]
                if m.quantized_files:
                    filtered.append(m)
            return filtered

        return results

    def get_model(self, model_id: str) -> HubModel:
        """Get detailed info about a specific model including all quantized files."""
        return self._enrich_model(model_id)

    def _enrich_model(self, model_id: str) -> HubModel:
        """Fetch model metadata and list quantized files."""
        from huggingface_hub import model_info  # lazy: optional dep
        from huggingface_hub.utils import RepositoryNotFoundError  # lazy: optional dep

        try:
            info = model_info(model_id, token=self.api.token)
        except RepositoryNotFoundError:
            raise ValueError(f"Model not found on HuggingFace: {model_id}")

        siblings = getattr(info, "siblings", []) or []
        tags = getattr(info, "tags", []) or []

        quantized_files = []
        formats = set()

        for sib in siblings:
            fname = sib.rfilename
            fmt, qtype = classify_file(fname)

            if fmt in ("gguf", "awq", "gptq"):
                size = getattr(sib, "size", 0) or 0
                quantized_files.append(QuantizedFile(
                    filename=fname,
                    size_bytes=size,
                    quant_type=qtype,
                    format=fmt,
                ))
                formats.add(fmt)

        # Sort files by quant quality preference
        quantized_files.sort(key=lambda f: _quant_preference(f.quant_type))

        author = model_id.split("/")[0] if "/" in model_id else "unknown"

        return HubModel(
            model_id=model_id,
            author=author,
            tags=tags,
            likes=getattr(info, "likes", 0) or 0,
            downloads=getattr(info, "downloads", 0) or 0,
            last_modified=str(getattr(info, "last_modified", ""))[:10],
            quantized_files=quantized_files,
            formats=sorted(formats),
            description=getattr(info, "card_data", None) and
                       getattr(info.card_data, "description", "") or "",
        )

    def download(
        self,
        model_id: str,
        filename: str,
        local_dir: str = "",
        quant_type: str = "",
        progress_callback=None,
    ) -> str:
        """Download a file from HuggingFace.

        Args:
            model_id: HF repo (e.g. "TheBloke/Llama-2-7B-GGUF")
            filename: Specific file to download
            local_dir: Where to save. Defaults to ~/.vibeblade/models/
            quant_type: If set, auto-pick the best matching file
            progress_callback: Optional callback(bytes_downloaded, total_bytes)

        Returns:
            Absolute path to the downloaded file.
        """
        from huggingface_hub import hf_hub_download  # lazy: optional dep

        model = self._enrich_model(model_id)

        if not local_dir:
            local_dir = str(Path.home() / ".vibeblade" / "models" / model_id.replace("/", "_"))

        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Auto-select file if quant_type specified but not filename
        if not filename and quant_type:
            model = self.get_model(model_id)
            q_upper = quant_type.upper()
            for f in model.quantized_files:
                if f.quant_type.upper() == q_upper:
                    filename = f.filename
                    break
            if not filename:
                raise ValueError(
                    f"No {quant_type} file found for {model_id}. "
                    f"Available: {[f.quant_type for f in model.quantized_files]}"
                )

        if not filename:
            raise ValueError("Either filename or quant_type must be specified")

        logger.info("Downloading %s/%s → %s", model_id, filename, local_dir)

        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=local_dir,
            token=self.api.token,
        )

        logger.info("Downloaded to %s", path)
        return path

    def download_gguf(
        self,
        model_id: str,
        quant: str = "Q4_K_M",
        local_dir: str = "",
    ) -> str:
        """Download a GGUF model with specified quantization.

        Convenience method — auto-finds the best matching GGUF file.

        Args:
            model_id: HF model ID
            quant: Quant type (e.g., "Q4_K_M", "Q5_K_S")
            local_dir: Save location

        Returns:
            Path to downloaded GGUF file
        """
        return self.download(model_id, filename="", local_dir=local_dir, quant_type=quant)

    def get_compatible_platforms(self, fmt: str = "gguf") -> list[dict]:
        """Get list of platforms compatible with a format.

        Returns platform info with name, format support, and notes.
        """
        platforms = [
            {"name": "llama.cpp", "formats": ["gguf"], "notes": "Universal GGUF engine"},
            {"name": "LM Studio", "formats": ["gguf"], "notes": "Desktop GUI, auto-detects quant"},
            {"name": "Ollama", "formats": ["gguf"], "notes": "CLI/server, uses GGUF internally"},
            {"name": "KoboldCpp", "formats": ["gguf"], "notes": "Browser-based UI, GGUF native"},
            {"name": "TextGen WebUI", "formats": ["gguf", "awq", "gptq"], "notes": "Multi-backend web UI"},
            {"name": "MLX", "formats": ["gguf", "safetensors"], "notes": "Apple Silicon unified memory"},
            {"name": "vLLM", "formats": ["awq", "gptq", "safetensors"], "notes": "Production serving"},
            {"name": "VibeBlade", "formats": ["gguf", "safetensors"], "notes": "CPU/RAM sparse inference"},
        ]
        return [p for p in platforms if fmt in p["formats"]]


def _quant_preference(quant_type: str) -> int:
    """Score for sorting quant types by recommendation order (lower = better)."""
    prefs = {
        "Q4_K_M": 0, "Q5_K_M": 1, "Q5_K_S": 2, "Q6_K": 3,
        "Q4_K_S": 4, "Q8_0": 5, "Q3_K_M": 6, "IQ4_XS": 7,
        "Q2_K": 8, "F16": 9, "F32": 10, "BF16": 11,
    }
    return prefs.get(quant_type.upper(), 50)


# ── Quick CLI helpers ──

def quick_search(query: str, fmt: str = "gguf", limit: int = 10) -> list[HubModel]:
    """One-shot search without instantiating ModelHub."""
    hub = ModelHub()
    return hub.search(query=query, fmt=fmt, limit=limit)


def quick_download(model_id: str, quant: str = "Q4_K_M") -> str:
    """One-shot download GGUF model."""
    hub = ModelHub()
    return hub.download_gguf(model_id, quant)
