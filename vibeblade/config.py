"""VibeBlade Configuration — vibeblade.yaml offload strategy loader.

Supports two modes:
- RAM_ONLY: all cold experts in system RAM (requires 128GB+ RAM for 230B MoE)
- HYBRID_SSD: hot experts in VRAM, medium-heat in RAM buffer, cold on SSD

Example vibeblade.yaml:
  offload_strategy:
    mode: "RAM_ONLY"
    vram_limit: "16GB"
    ram_limit: "256GB"
    hot_threshold: 0.15

  offload_strategy:
    mode: "HYBRID_SSD"
    vram_limit: "16GB"
    ram_limit: "32GB"
    ssd_path: "/mnt/nvme/vibeblade_cache"
    hot_threshold: 0.15
    ram_buffer_ratio: 0.25
    ssd_preemptive_layers: 2
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised for invalid configuration values or unknown keys."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OffloadMode(Enum):
    RAM_ONLY = "RAM_ONLY"
    HYBRID_SSD = "HYBRID_SSD"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(
    r"""^\s*(\d+(?:\.\d+)?)\s*(TB|GB|MB|KB|B|)\s*$""",
    re.IGNORECASE,
)
_SIZE_MULTIPLIERS = {
    "": 1,
    "B": 1,
    "KB": 1 << 10,
    "MB": 1 << 20,
    "GB": 1 << 30,
    "TB": 1 << 40,
}


def parse_size(s: str | int | float) -> int:
    """Convert a human-readable size string like ``"16GB"`` to an integer byte count.

    Also accepts bare ints/floats (treated as raw bytes).
    """
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip()
    if not s:
        raise ConfigError("Empty size string")
    m = _SIZE_RE.match(s)
    if not m:
        raise ConfigError(f"Cannot parse size: {s!r}")
    value = float(m.group(1))
    unit = m.group(2).upper()
    return int(value * _SIZE_MULTIPLIERS[unit])


def _system_ram_bytes() -> int:
    """Return total system RAM in bytes (best-effort)."""
    try:
        # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb << 10
    except (FileNotFoundError, ValueError, IndexError):
        pass
    # Fallback
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        pass
    return 64 << 30  # assume 64 GB


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OffloadConfig:
    """Schema for the ``offload_strategy`` block."""

    mode: OffloadMode = OffloadMode.RAM_ONLY
    vram_limit: int = 16 << 30          # 16 GiB
    ram_limit: int = field(default_factory=_system_ram_bytes)
    hot_threshold: float = 0.15

    # HYBRID_SSD extras
    ssd_path: Optional[str] = None
    ram_buffer_ratio: float = 0.25
    ssd_preemptive_layers: int = 2

    # Known keys (used for validation)
    _KNOWN_KEYS: frozenset = field(
        default=frozenset({
            "mode", "vram_limit", "ram_limit", "hot_threshold",
            "ssd_path", "ram_buffer_ratio", "ssd_preemptive_layers",
        }),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.vram_limit <= 0:
            raise ConfigError(f"vram_limit must be positive, got {self.vram_limit}")
        if self.ram_limit <= 0:
            raise ConfigError(f"ram_limit must be positive, got {self.ram_limit}")
        if not (0.0 < self.hot_threshold <= 1.0):
            raise ConfigError(
                f"hot_threshold must be in (0, 1], got {self.hot_threshold}"
            )
        if self.mode == OffloadMode.HYBRID_SSD:
            if not self.ssd_path:
                raise ConfigError(
                    "ssd_path is required when mode is HYBRID_SSD"
                )
            if not (0.0 < self.ram_buffer_ratio <= 1.0):
                raise ConfigError(
                    f"ram_buffer_ratio must be in (0, 1], got {self.ram_buffer_ratio}"
                )
            if self.ssd_preemptive_layers < 0:
                raise ConfigError(
                    f"ssd_preemptive_layers must be >= 0, got {self.ssd_preemptive_layers}"
                )


@dataclass
class VibeBladeConfig:
    """Top-level configuration container."""

    offload_strategy: OffloadConfig = field(default_factory=OffloadConfig)


# ---------------------------------------------------------------------------
# Simple YAML parser (regex-based, indentation-aware)
# ---------------------------------------------------------------------------

# Matches:  key: value   where value may be a quoted string, number, or bare word
_KEYVAL_RE = re.compile(
    r"""^(\s*)"""                   # leading whitespace (indent)
    r"""([a-z_][a-z0-9_]*)"""       # key
    r""":\s*"""                     # colon separator
    r"""(.*)$""",                   # value (may be empty)
    re.IGNORECASE,
)

# Strips inline comments (but respects # inside quoted strings)
_INLINE_COMMENT_RE = re.compile(
    r"""(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')"""   # quoted string
    r"""|\s+#.*$""",                                   # or whitespace + comment
    re.MULTILINE,
)


def _strip_comments(text: str) -> str:
    """Remove full-line comments and inline comments (outside quotes)."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        # Drop full-line comments and blank lines
        if not stripped or stripped.startswith("#"):
            continue
        # Remove inline comment (simple approach: find last # outside quotes)
        line = _strip_inline_comment(raw_line)
        lines.append(line)
    return "\n".join(lines)


def _strip_inline_comment(line: str) -> str:
    """Strip inline ``# comment`` outside of quoted strings."""
    in_quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quote:
            if ch == "\\" and i + 1 < len(line):
                i += 2  # skip escaped char
                continue
            if ch == in_quote:
                in_quote = None
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "#":
                return line[:i].rstrip()
        i += 1
    return line


def _indent_level(line: str) -> int:
    """Return the number of leading spaces."""
    return len(line) - len(line.lstrip())


def _parse_yaml_value(raw: str):
    """Parse a single YAML value string into a Python object."""
    raw = raw.strip()
    if not raw:
        return None
    # Quoted string
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    # Boolean
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    # None
    if raw.lower() in ("null", "~", "none"):
        return None
    # Integer
    try:
        return int(raw)
    except ValueError:
        pass
    # Float
    try:
        return float(raw)
    except ValueError:
        pass
    # Bare string
    return raw


def _parse_yaml(text: str) -> dict:
    """Minimal YAML parser for flat / one-level-nested dicts.

    Returns a dict of dicts (top-level keys → nested key-value pairs).
    """
    text = _strip_comments(text)
    if not text.strip():
        return {}

    result: dict = {}
    current_block: dict = {}
    current_block_key: str | None = None
    base_indent: int | None = None

    for line in text.splitlines():
        if not line.strip():
            continue
        indent = _indent_level(line)
        m = _KEYVAL_RE.match(line)
        if not m:
            continue  # skip malformed lines

        key = m.group(2).lower()
        value_str = m.group(3).strip()

        # Determine if this is a top-level key (block header) or nested value
        if base_indent is None or indent <= base_indent:
            # Save previous block
            if current_block_key is not None:
                result[current_block_key] = current_block
            current_block_key = key
            current_block = {}
            base_indent = indent

            # If there's a value on the same line, it's a leaf (not a block)
            if value_str:
                current_block = {"_self": _parse_yaml_value(value_str)}
                base_indent = None
        else:
            # Nested key-value inside current block
            current_block[key] = _parse_yaml_value(value_str)

    # Save last block
    if current_block_key is not None:
        result[current_block_key] = current_block

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_OFFLOAD_CONFIG_FIELD_MAP = {
    "mode": ("mode", lambda v: OffloadMode(v) if isinstance(v, str) else v),
    "vram_limit": ("vram_limit", parse_size),
    "ram_limit": ("ram_limit", parse_size),
    "hot_threshold": ("hot_threshold", float),
    "ssd_path": ("ssd_path", str),
    "ram_buffer_ratio": ("ram_buffer_ratio", float),
    "ssd_preemptive_layers": ("ssd_preemptive_layers", int),
}


def _build_offload_config(raw: dict) -> OffloadConfig:
    """Build an ``OffloadConfig`` from a flat dict of key-value strings."""
    known_keys = set(_OFFLOAD_CONFIG_FIELD_MAP.keys())
    extra = set(raw.keys()) - known_keys
    if extra:
        raise ConfigError(f"Unknown offload_strategy keys: {extra}")

    kwargs: dict = {}
    for yaml_key, (field_name, converter) in _OFFLOAD_CONFIG_FIELD_MAP.items():
        if yaml_key in raw:
            kwargs[field_name] = converter(raw[yaml_key])
    return OffloadConfig(**kwargs)


def load_config(path: str) -> VibeBladeConfig:
    """Load a VibeBlade configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a ``vibeblade.yaml`` file.

    Returns
    -------
    VibeBladeConfig
    """
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    parsed = _parse_yaml(text)

    # Top-level known keys
    known_top = {"offload_strategy"}
    extra_top = set(parsed.keys()) - known_top
    if extra_top:
        raise ConfigError(f"Unknown top-level keys: {extra_top}")

    offload_raw: dict = parsed.get("offload_strategy", {})
    if "_self" in offload_raw:
        raise ConfigError(
            "offload_strategy must be a mapping block, not a scalar value"
        )

    offload_config = _build_offload_config(offload_raw)
    return VibeBladeConfig(offload_strategy=offload_config)


def default_config() -> VibeBladeConfig:
    """Return a ``VibeBladeConfig`` with sensible defaults.

    Defaults:
    - mode: RAM_ONLY
    - vram_limit: 16 GiB
    - ram_limit: detected system RAM (fallback 64 GiB)
    - hot_threshold: 0.15
    """
    return VibeBladeConfig(offload_strategy=OffloadConfig())
