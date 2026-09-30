"""VibeBlade Expert Heat — persistent cross-run route histogram and learned RAM pin tier.

Model-agnostic implementation of the Deltafin pattern (credit: gavamedia/deltafin,
docs/OPTIMIZATIONS.md "Persistent cross-run expert heat"): every committed MoE
pass folds its authoritative route IDs into a small decayed histogram that
survives process restarts. A byte-budgeted pin tier promotes the hottest
(expert, layer) pairs into RAM *before* inference — learned storage placement,
NOT learned inference:

- the histogram only OBSERVES routes after the router has published them;
- it never supplies weights, route IDs or reduction order;
- a missing / stale / corrupt / locked histogram degrades to "nothing learned";
- recording is infallible on the hot path (best-effort flush, sidecar lock,
  atomic replacement).

File format (v1, little-endian): magic "VBHT" | u16 version | u16 half_life_k |
u32 n_entries | then n_entries × (u32 layer_idx, u32 expert_id, f64 weight).
"""

from __future__ import annotations

import fcntl
import os
import struct
import tempfile
import time
from pathlib import Path

MAGIC = b"VBHT"
VERSION = 1
DEFAULT_HALF_LIFE_PASSES = 8192
DEFAULT_CONFIDENCE_RAMP = 256        # passes before a pin candidate is trusted
DEFAULT_FREQ_FLOOR = 2.0             # × uniform frequency, anti-long-tail guard
LOCK_TIMEOUT_S = 2.0


class ExpertHeat:
    """Decayed per-(layer, expert) route histogram with atomic persistence."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        half_life_passes: int = DEFAULT_HALF_LIFE_PASSES,
        confidence_ramp: int = DEFAULT_CONFIDENCE_RAMP,
        freq_floor: float = DEFAULT_FREQ_FLOOR,
    ) -> None:
        self._path = Path(path)
        self._half_life = max(1, int(half_life_passes))
        self._confidence_ramp = max(0, int(confidence_ramp))
        self._freq_floor = float(freq_floor)
        # {(layer, expert): weight}
        self._weights: dict[tuple[int, int], float] = {}
        self._passes_recorded = 0
        self._dirty = False
        self._last_flush = 0.0
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load or initialize. Any problem = empty histogram (fail-soft)."""
        try:
            if not self._path.exists():
                return
            raw = self._path.read_bytes()
            if len(raw) < 12 or raw[:4] != MAGIC:
                return  # unknown format → nothing learned yet
            version, half_life, n = struct.unpack_from("<HHI", raw, 4)
            if version != VERSION:
                return
            expected = 12 + n * 16
            if len(raw) < expected:
                return  # truncated → ignore, never crash inference
            off = 12
            for _ in range(n):
                layer, expert = struct.unpack_from("<II", raw, off)
                (w,) = struct.unpack_from("<d", raw, off + 8)
                off += 16
                if w > 0.0:
                    self._weights[(layer, expert)] = w
            (passes,) = struct.unpack_from("<Q", raw, off) if len(raw) >= off + 8 else (0,)
            self._passes_recorded = passes
        except (OSError, struct.error):
            self._weights = {}
            self._passes_recorded = 0

    def flush(self) -> bool:
        """Atomic write under a sidecar lock. Best-effort: returns success."""
        if not self._dirty:
            return True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._path.with_suffix(self._path.suffix + ".lock")
            # Merge under lock: another process may have recorded while we ran.
            with open(lock_path, "w") as lock_fh:
                fcntl.flock(lock_fh, fcntl.LOCK_EX)
                try:
                    merged = ExpertHeat(self._path, half_life_passes=self._half_life)
                    for key, w in self._weights.items():
                        merged._weights[key] = merged._weights.get(key, 0.0) + w
                    merged._passes_recorded += self._passes_recorded
                    payload = merged._serialize()
                    # atomic replace
                    fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
                    try:
                        with os.fdopen(fd, "wb") as fh:
                            fh.write(payload)
                            fh.flush()
                            os.fsync(fh.fileno())
                        os.replace(tmp, self._path)
                    except BaseException:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
                        raise
                    # adopt merged state
                    self._weights = merged._weights
                    self._passes_recorded = merged._passes_recorded
                    self._dirty = False
                    self._last_flush = time.time()
                    return True
                finally:
                    fcntl.flock(lock_fh, fcntl.LOCK_UN)
        except OSError:
            return False

    def _serialize(self) -> bytes:
        items = sorted(self._weights.items())
        head = MAGIC + struct.pack("<HHI", VERSION, min(self._half_life, 0xFFFF), len(items))
        buf = bytearray(head)
        for (layer, expert), w in items:
            buf += struct.pack("<IId", layer, expert, w)
        buf += struct.pack("<Q", self._passes_recorded)
        return bytes(buf)

    # ── Recording (hot path — infallible) ───────────────────────────────

    def record_routes(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Fold one authoritative route decision in. Never raises."""
        try:
            decay = 0.5 ** (1.0 / self._half_life)
            if not self._dirty:
                # Apply per-flush decay once per generation of recording
                for k in self._weights:
                    self._weights[k] *= decay
            for eid in expert_ids:
                key = (int(layer_idx), int(eid))
                self._weights[key] = self._weights.get(key, 0.0) + 1.0
            self._passes_recorded += 1
            self._dirty = True
        except (TypeError, ValueError):
            pass  # observability must never break inference

    def maybe_flush(self, interval_s: float = 60.0) -> None:
        if self._dirty and (time.time() - self._last_flush) >= interval_s:
            self.flush()

    # ── Learned pin tier (advisory) ──────────────────────────────────────

    def pin_roster(self, budget_bytes: int, expert_size_bytes: int) -> dict[int, list[int]]:
        """Choose hot (layer → expert IDs) to pin into RAM within a byte budget.

        Guards against the long-tail trap Deltafin measured (naive frequency
        pinning generalized at 41.1% vs 51.9% in-sample): entries below
        `freq_floor ×` the uniform frequency are not eligible, and the whole
        histogram must pass its confidence ramp before anything pins.
        """
        if budget_bytes <= 0 or expert_size_bytes <= 0:
            return {}
        if self._passes_recorded < self._confidence_ramp:
            return {}  # not enough evidence yet
        if not self._weights:
            return {}
        n_layers = max(k[0] for k in self._weights) + 1
        n_experts = max(k[1] for k in self._weights) + 1
        # Long-tail guard, two-sided (Deltafin measured naive frequency pinning
        # generalizing at 41.1% vs 51.9% in-sample — the TAIL is what fails):
        #   floor = min(2 × uniform frequency, 5% of the hottest entry)
        # Either guard alone is sufficient to admit an expert; both together
        # still cut the long tail (B: 51 → 1 kept) without killing medium
        # heat on skewed histograms (A: hot+medium kept, rare dropped).
        total = sum(self._weights.values())
        uniform = total / max(len(self._weights), 1)
        peak = max(self._weights.values())
        floor = min(uniform * self._freq_floor, peak * 0.05)
        ranked = sorted(
            ((w, key) for key, w in self._weights.items() if w >= floor),
            reverse=True,
        )
        max_experts = max(1, budget_bytes // max(expert_size_bytes, 1))
        roster: dict[int, list[int]] = {}
        used = 0
        for w, (layer, expert) in ranked:
            if used >= max_experts:
                break
            roster.setdefault(layer, []).append(expert)
            used += 1
        # silence unused when histogram spans sparse layers
        _ = n_layers, n_experts
        return roster

    # ── Introspection ────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "path": str(self._path),
            "entries": len(self._weights),
            "passes_recorded": self._passes_recorded,
            "confidence_ready": self._passes_recorded >= self._confidence_ramp,
            "dirty": self._dirty,
        }
