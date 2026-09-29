"""vibeblade.ui — shared terminal UI for the VibeBlade CLI.

One source of truth for ANSI styling, panel drawing, and status lines.
Respects NO_COLOR and non-tty stdout (pipes, CI) automatically.
"""
from __future__ import annotations

import os
import shutil
import sys

# ── Capability detection ─────────────────────────────────────────
def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_COLOR = _colors_enabled()

# ── Base ANSI codes (empty when disabled) ────────────────────────
BOLD = "\033[1m" if _COLOR else ""
DIM = "\033[2m" if _COLOR else ""
CYAN = "\033[36m" if _COLOR else ""
GREEN = "\033[32m" if _COLOR else ""
YELLOW = "\033[33m" if _COLOR else ""
RED = "\033[31m" if _COLOR else ""
MAGENTA = "\033[35m" if _COLOR else ""
RESET = "\033[0m" if _COLOR else ""

if os.name == "nt" and _COLOR:
    try:
        os.system("")
    except Exception:
        BOLD = DIM = CYAN = GREEN = YELLOW = RED = MAGENTA = RESET = ""


# ── Style shorthands ─────────────────────────────────────────────
def b(s: str) -> str:     # bold
    return f"{BOLD}{s}{RESET}"


def d(s: str) -> str:     # dim
    return f"{DIM}{s}{RESET}"


def c(s: str) -> str:     # cyan
    return f"{CYAN}{s}{RESET}"


def g(s: str) -> str:     # green
    return f"{GREEN}{s}{RESET}"


def y(s: str) -> str:     # yellow
    return f"{YELLOW}{s}{RESET}"


def r(s: str) -> str:     # red
    return f"{RED}{s}{RESET}"


def m(s: str) -> str:     # magenta
    return f"{MAGENTA}{s}{RESET}"


def ok(s: str = "ok") -> str:
    return f"{GREEN}✓{RESET} {s}"


def warn(s: str) -> str:
    return f"{YELLOW}⚠ {s}{RESET}"


def err(s: str) -> str:
    return f"{RED}✗ {s}{RESET}"


# ── Layout helpers ───────────────────────────────────────────────
def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def _visible_len(s: str) -> int:
    """String length excluding ANSI escape sequences."""
    n, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            j = s.find("m", i)
            i = (j + 1) if j != -1 else len(s)
        else:
            n += 1
            i += 1
    return n


def kv(key: str, value: str, key_width: int = 16) -> str:
    """Aligned 'Key   value' line."""
    return f" {d(key.ljust(key_width))} {value}"


def panel(title: str, lines: list[str], width: int | None = None, accent: str = CYAN) -> str:
    """Draw a rounded box around content lines.

    Lines may contain ANSI codes; padding accounts for visible width only.
    """
    width = width or min(max(term_width() - 2, 40), 78)
    inner = width - 4  # '│ ' + ' │'

    rows = [
        f"{d('╭─')} {accent}{b(title)}{RESET} {d('─' * max(1, width - len(title) - 6))}╮"
        if _COLOR else f"╭─ {title} {'─' * max(1, width - len(title) - 6)}╮"
    ]
    for line in lines:
        vis = _visible_len(line)
        pad = " " * max(0, inner - vis)
        rows.append(f"{d('│')} {line}{pad} {d('│')}")
    rows.append(d("╰" + "─" * (width - 2) + "╯"))
    return "\n".join(rows)


def hr(width: int | None = None) -> str:
    return d("─" * (width or min(term_width(), 64)))


def header(subtitle: str = "") -> str:
    """CLI banner."""
    art = f"{CYAN}{BOLD}◆ VibeBlade{RESET}"
    tag = d("  adaptive memory tiering for LLM inference")
    line = f"{art}{tag}"
    if subtitle:
        line += f"\n{d(subtitle)}"
    return line


def status(label: str, detail: str = "") -> str:
    """One-line progress note: '▸ Label — detail'."""
    arrow = f"{CYAN}▸{RESET}"
    if detail:
        return f" {arrow} {b(label)} {d('—')} {d(detail)}"
    return f" {arrow} {b(label)}"


def bytes_human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def table(headers: list[str], rows: list[list[str]], align_right: list[int] | None = None) -> str:
    """Minimal aligned text table with a rule under headers."""
    align_right = align_right or []
    widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(str(cell)))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            cell = str(cell)
            pad = " " * (widths[i] - _visible_len(cell))
            parts.append(f"{cell}{pad}" if i not in align_right else f"{pad}{cell}")
        return "  " + d("  ").join(parts)

    out = [fmt_row([b(h) if _COLOR else h for h in headers]), d("  " + "─" * (sum(widths) + 2 * (len(widths) - 1)))]
    for row in rows:
        out.append(fmt_row(row))
    return "\n".join(out)
