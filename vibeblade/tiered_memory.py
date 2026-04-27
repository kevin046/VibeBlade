"""Adaptive Memory Tiering (AMT) — 3-tier hierarchy for MoE expert weights.

Tier A (VRAM): Hot experts, ~1000 GB/s — tracked here, GPU upload in moe_executor.py
Tier B (RAM):  Medium-heat experts, active buffer managed by LRU-K eviction
Tier C (SSD):  Cold experts, deep store — file-per-expert on NVMe

No GPU/CuPy dependencies — all data is numpy arrays.  The actual device
transfer is the caller's responsibility (moe_executor.py).
"""

from __future__ import annotations

import os
import struct
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class MemoryTier(Enum):
    VRAM = "vram"
    RAM = "ram"
    SSD = "ssd"


@dataclass
class ExpertLocation:
    """Tracks where a single expert's weights currently reside."""
    expert_id: int
    layer_idx: int
    tier: MemoryTier
    size_bytes: int = 0
    last_access_time: float = 0.0
    access_count: int = 0


@dataclass
class TieredMemoryStats:
    """Runtime statistics for the tiered memory system."""
    vram_hits: int = 0
    ram_hits: int = 0
    ssd_hits: int = 0
    ssd_loads: int = 0
    evictions_from_ram: int = 0
    evictions_to_ssd: int = 0
    ram_to_vram_transfers: int = 0
    total_ssd_read_bytes: int = 0
    total_ssd_read_ms: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0


# ---------------------------------------------------------------------------
# LRU-K eviction policy
# ---------------------------------------------------------------------------

class LRUKPolicy:
    """LRU-K eviction policy for the RAM buffer.

    Unlike simple LRU, LRU-K requires *K* accesses before an item enters
    the protected set.  One-hit wonders get evicted first.

    Args:
        k: number of recent accesses to track (default 2).
        capacity: maximum number of items the RAM buffer can hold.
    """

    def __init__(self, k: int = 2, capacity: int = 64) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.k = k
        self.capacity = capacity

        # {(layer_idx, expert_id): deque of recent access timestamps}
        self._history: dict[tuple[int, int], deque[float]] = {}

        # Protected set: items with k+ accesses, ordered by oldest K-th
        # access time (evict from front = least-recently-used K-th).
        self._protected: OrderedDict[tuple[int, int], float] = OrderedDict()

        # Probationary set: items with < k accesses (evict from front).
        self._probationary: OrderedDict[tuple[int, int], float] = OrderedDict()

    # -- public API ----------------------------------------------------------

    def access(self, layer_idx: int, expert_id: int) -> None:
        """Record an access to *expert_id* in *layer_idx*."""
        key = (layer_idx, expert_id)
        now = time.monotonic()

        if key in self._history:
            hist = self._history[key]
            hist.append(now)
            # Keep only the K most recent timestamps
            while len(hist) > self.k:
                hist.popleft()

            if len(hist) >= self.k:
                # Promote to protected (or update position)
                self._probationary.pop(key, None)
                kth_time = hist[0]
                # Remove and re-insert so it goes to the back (MRU)
                self._protected.pop(key, None)
                self._protected[key] = kth_time
        else:
            # First access — goes to probationary
            self._history[key] = deque([now])
            self._probationary.pop(key, None)
            self._probationary[key] = now

    def evict(self) -> tuple[int, int] | None:
        """Return the (layer_idx, expert_id) that should be evicted, or
        ``None`` if nothing is tracked."""
        # Prefer evicting from probationary first (one-hit wonders)
        if self._probationary:
            key = next(iter(self._probationary))
            self._remove(key)
            return key
        if self._protected:
            key = next(iter(self._protected))
            self._remove(key)
            return key
        return None

    def should_prefetch(self, layer_idx: int, expert_id: int) -> bool:
        """Return *True* if the expert has been accessed at least *k* times,
        indicating it is a good prefetch candidate."""
        key = (layer_idx, expert_id)
        hist = self._history.get(key)
        return hist is not None and len(hist) >= self.k

    def remove(self, layer_idx: int, expert_id: int) -> None:
        """Explicitly remove an entry (e.g. when it is promoted to VRAM)."""
        self._remove((layer_idx, expert_id))

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether the key is tracked at all."""
        return (layer_idx, expert_id) in self._history

    @property
    def size(self) -> int:
        """Number of items currently tracked."""
        return len(self._history)

    def stats(self) -> dict:
        return {
            "protected_count": len(self._protected),
            "probationary_count": len(self._probationary),
            "total_tracked": len(self._history),
            "k": self.k,
            "capacity": self.capacity,
        }

    # -- internal ------------------------------------------------------------

    def _remove(self, key: tuple[int, int]) -> None:
        self._history.pop(key, None)
        self._protected.pop(key, None)
        self._probationary.pop(key, None)


# ---------------------------------------------------------------------------
# SSD-backed expert store
# ---------------------------------------------------------------------------

class SSDExpertStore:
    """SSD-backed expert weight storage.

    Stores each expert as a raw binary file inside a directory hierarchy:

        {ssd_path}/layer_{L:04d}/expert_{E:04d}.bin

    Binary format (3 matrices: gate, up, down):
        [rows_u32][cols_u32][data_bytes …][rows_u32][cols_u32][data_bytes …][rows_u32][cols_u32][data_bytes …]

    Args:
        ssd_path: root directory for expert binary files.
        num_layers: total transformer layers.
        num_experts: experts per layer.
        preload_threads: threads for async SSD reads (default 4).
    """

    _HEADER_FMT = "<II"  # two uint32s: rows, cols
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 8 bytes

    def __init__(
        self,
        ssd_path: str,
        num_layers: int,
        num_experts: int,
        preload_threads: int = 4,
    ) -> None:
        self._ssd_path = ssd_path
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._executor = ThreadPoolExecutor(max_workers=max(1, preload_threads))
        # Cache of file sizes: {(layer, expert): bytes}
        self._size_cache: dict[tuple[int, int], int] = {}
        os.makedirs(ssd_path, exist_ok=True)

    # -- helpers -------------------------------------------------------------

    def _expert_dir(self, layer_idx: int) -> str:
        return os.path.join(self._ssd_path, f"layer_{layer_idx:04d}")

    def _expert_path(self, layer_idx: int, expert_id: int) -> str:
        return os.path.join(self._expert_dir(layer_idx), f"expert_{expert_id:04d}.bin")

    @staticmethod
    def _write_matrix(f, mat: np.ndarray) -> None:
        """Write a single matrix: [rows_u32][cols_u32][raw data]."""
        rows, cols = mat.shape
        f.write(struct.pack("<II", rows, cols))
        f.write(mat.astype(np.float32).tobytes())

    @staticmethod
    def _read_matrix(data: bytes, offset: int) -> tuple[np.ndarray, int]:
        """Read one matrix starting at *offset*.  Returns (array, new_offset)."""
        rows, cols = struct.unpack_from("<II", data, offset)
        offset += SSDExpertStore._HEADER_SIZE
        n_elements = rows * cols
        nbytes = n_elements * 4  # float32
        mat = np.frombuffer(data, dtype=np.float32, count=n_elements, offset=offset)
        mat = mat.reshape(rows, cols).copy()  # copy so we own the memory
        offset += nbytes
        return mat, offset

    # -- public API ----------------------------------------------------------

    def store_expert(
        self,
        layer_idx: int,
        expert_id: int,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> None:
        """Persist three weight matrices for one expert."""
        d = self._expert_dir(layer_idx)
        os.makedirs(d, exist_ok=True)
        path = self._expert_path(layer_idx, expert_id)
        with open(path, "wb") as f:
            self._write_matrix(f, gate_w)
            self._write_matrix(f, up_w)
            self._write_matrix(f, down_w)
        self._size_cache[(layer_idx, expert_id)] = os.path.getsize(path)

    def load_expert(
        self, layer_idx: int, expert_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load three weight matrices from SSD (synchronous).

        Returns (gate_w, up_w, down_w).
        """
        path = self._expert_path(layer_idx, expert_id)
        with open(path, "rb") as f:
            data = f.read()
        offset = 0
        gate, offset = self._read_matrix(data, offset)
        up, offset = self._read_matrix(data, offset)
        down, offset = self._read_matrix(data, offset)
        self._size_cache[(layer_idx, expert_id)] = len(data)
        return gate, up, down

    def async_load(self, layer_idx: int, expert_id: int) -> Future:
        """Load expert asynchronously.  Returns a :class:`Future` that
        resolves to ``(gate_w, up_w, down_w)``."""
        return self._executor.submit(self.load_expert, layer_idx, expert_id)

    def expert_size_bytes(self, layer_idx: int, expert_id: int) -> int:
        """Return the on-disk size of an expert in bytes, or 0 if unknown."""
        key = (layer_idx, expert_id)
        if key in self._size_cache:
            return self._size_cache[key]
        path = self._expert_path(layer_idx, expert_id)
        if os.path.isfile(path):
            sz = os.path.getsize(path)
            self._size_cache[key] = sz
            return sz
        return 0

    def total_size_bytes(self) -> int:
        """Total bytes used by all stored experts."""
        total = 0
        for layer_num in range(self._num_layers):
            d = self._expert_dir(layer_num)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".bin"):
                        total += os.path.getsize(os.path.join(d, f))
        return total

    def close(self) -> None:
        """Shut down the preload thread pool."""
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Tiered Memory Manager — the main orchestrator
# ---------------------------------------------------------------------------

class TieredMemoryManager:
    """3-tier memory manager for MoE expert weights.

    Priority order: VRAM (hot) > RAM (active buffer) > SSD (deep store).

    When an expert is needed:

    1. Check VRAM cache → instant
    2. Check RAM buffer → fast
    3. Load from SSD → slow, but hidden by predictive pre-fetch

    The manager is pure-CPU / pure-numpy.  The actual GPU upload happens
    in ``moe_executor.py`` — this module only decides *where* weights live
    and handles SSD ↔ RAM transfers.

    Args:
        config: :class:`OffloadConfig` (from ``config.py``).
        num_layers: number of transformer layers.
        num_experts: experts per layer.
        expert_size_bytes: approximate bytes per expert (all 3 matrices).
        hot_cold_map: :class:`HotColdMap` (from ``moe_profiler.py``).
        eviction_policy: optional custom eviction policy from ``eviction.py``.
            If ``None``, uses the default :class:`LRUKPolicy` with k=2.
            Options: :class:`FrequencyAwarePolicy`, :class:`CostBenefitScorer`,
            :class:`AdaptiveBanditPolicy`.
    """

    def __init__(
        self,
        config,
        num_layers: int,
        num_experts: int,
        expert_size_bytes: int,
        hot_cold_map,
        eviction_policy=None,
    ) -> None:
        self._config = config
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._expert_size_bytes = expert_size_bytes
        self._hot_cold_map = hot_cold_map
        self._lock = threading.Lock()

        # Determine how many experts fit in RAM buffer
        ram_limit = config.ram_limit
        # Reserve some headroom; the ram_limit is total system RAM budget
        ram_capacity = max(1, ram_limit // max(expert_size_bytes, 1))

        # Use provided eviction policy or default to LRU-K
        if eviction_policy is not None:
            self._lru_policy = eviction_policy
        else:
            self._lru_policy = LRUKPolicy(k=2, capacity=ram_capacity)

        # VRAM cache: {(layer, expert): (gate, up, down)} — tracks what's
        # "supposed to be in VRAM" as numpy arrays.
        self._vram_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # RAM buffer: same structure, managed by LRU-K
        self._ram_buffer: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

        # Location registry: {(layer, expert): ExpertLocation}
        self._locations: dict[tuple[int, int], ExpertLocation] = {}

        # SSD store (only created for HYBRID_SSD mode)
        self._ssd: Optional[SSDExpertStore] = None
        if config.mode.value == "HYBRID_SSD" and config.ssd_path:
            self._ssd = SSDExpertStore(
                ssd_path=config.ssd_path,
                num_layers=num_layers,
                num_experts=num_experts,
            )

        # Statistics
        self._stats = TieredMemoryStats()
        self._stats_lock = threading.Lock()

        # Populate initial state from hot_cold_map
        self._init_from_hot_cold_map()

    # -- initialisation ------------------------------------------------------

    def _init_from_hot_cold_map(self) -> None:
        """Seed VRAM with hot experts, RAM buffer empty, rest on SSD."""
        now = time.monotonic()
        for layer_idx in range(self._num_layers):
            hot_ids = self._hot_cold_map.hot_experts.get(layer_idx, [])
            cold_ids = self._hot_cold_map.cold_experts.get(layer_idx, [])

            for eid in hot_ids:
                key = (layer_idx, eid)
                self._locations[key] = ExpertLocation(
                    expert_id=eid,
                    layer_idx=layer_idx,
                    tier=MemoryTier.VRAM,
                    size_bytes=self._expert_size_bytes,
                    last_access_time=now,
                    access_count=1,
                )
                # Mark in LRU policy so should_prefetch works for hot experts
                self._lru_policy.access(layer_idx, eid)

            for eid in cold_ids:
                key = (layer_idx, eid)
                self._locations[key] = ExpertLocation(
                    expert_id=eid,
                    layer_idx=layer_idx,
                    tier=MemoryTier.SSD if self._ssd is not None else MemoryTier.RAM,
                    size_bytes=self._expert_size_bytes,
                    last_access_time=now,
                    access_count=0,
                )

    # -- public API ----------------------------------------------------------

    def register_vram_expert(
        self,
        layer_idx: int,
        expert_id: int,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> None:
        """Register that an expert's weights have been loaded into VRAM.

        This populates the VRAM cache so ``get_expert`` can return it
        immediately.  It also removes the expert from RAM buffer if present.
        """
        key = (layer_idx, expert_id)
        with self._lock:
            self._vram_cache[key] = (gate_w, up_w, down_w)
            self._ram_buffer.pop(key, None)
            self._lru_policy.remove(layer_idx, expert_id)
            self._locations[key] = ExpertLocation(
                expert_id=expert_id,
                layer_idx=layer_idx,
                tier=MemoryTier.VRAM,
                size_bytes=self._expert_size_bytes,
                last_access_time=time.monotonic(),
                access_count=self._locations.get(key, ExpertLocation(expert_id, layer_idx, MemoryTier.VRAM)).access_count + 1,
            )

    def get_expert(
        self, layer_idx: int, expert_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Retrieve expert weights, pulling from the highest available tier.

        Returns ``(gate_w, up_w, down_w)`` or ``None`` if the expert has
        never been stored anywhere.
        """
        key = (layer_idx, expert_id)
        now = time.monotonic()

        with self._lock:
            # Tier 1: VRAM
            if key in self._vram_cache:
                loc = self._locations.get(key)
                if loc is not None:
                    loc.last_access_time = now
                    loc.access_count += 1
                with self._stats_lock:
                    self._stats.vram_hits += 1
                return self._vram_cache[key]

            # Tier 2: RAM buffer
            if key in self._ram_buffer:
                # Move to end (MRU)
                self._ram_buffer.move_to_end(key)
                self._lru_policy.access(layer_idx, expert_id)
                loc = self._locations.get(key)
                if loc is not None:
                    loc.last_access_time = now
                    loc.access_count += 1
                with self._stats_lock:
                    self._stats.ram_hits += 1
                return self._ram_buffer[key]

        # Tier 3: SSD
        if self._ssd is not None:
            try:
                t0 = time.monotonic()
                gate, up, down = self._ssd.load_expert(layer_idx, expert_id)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                sz = self._ssd.expert_size_bytes(layer_idx, expert_id)

                with self._lock:
                    self._ensure_ram_capacity(key)
                    self._ram_buffer[key] = (gate, up, down)
                    self._ram_buffer.move_to_end(key)
                    self._lru_policy.access(layer_idx, expert_id)
                    self._locations[key] = ExpertLocation(
                        expert_id=expert_id,
                        layer_idx=layer_idx,
                        tier=MemoryTier.RAM,
                        size_bytes=sz,
                        last_access_time=now,
                        access_count=1,
                    )

                with self._stats_lock:
                    self._stats.ssd_hits += 1
                    self._stats.ssd_loads += 1
                    self._stats.total_ssd_read_bytes += sz
                    self._stats.total_ssd_read_ms += elapsed_ms

                return gate, up, down
            except FileNotFoundError:
                return None

        return None

    def ensure_in_ram(self, layer_idx: int, expert_id: int) -> None:
        """Ensure an expert is in the RAM buffer (load from SSD if needed).

        If the expert is already in VRAM or RAM this is a no-op.
        """
        key = (layer_idx, expert_id)
        with self._lock:
            if key in self._vram_cache or key in self._ram_buffer:
                return

        # Need to load from SSD
        if self._ssd is None:
            return

        try:
            gate, up, down = self._ssd.load_expert(layer_idx, expert_id)
            now = time.monotonic()

            with self._lock:
                self._ensure_ram_capacity(key)
                self._ram_buffer[key] = (gate, up, down)
                self._ram_buffer.move_to_end(key)
                self._lru_policy.access(layer_idx, expert_id)
                self._locations[key] = ExpertLocation(
                    expert_id=expert_id,
                    layer_idx=layer_idx,
                    tier=MemoryTier.RAM,
                    size_bytes=self._expert_size_bytes,
                    last_access_time=now,
                    access_count=1,
                )

            with self._stats_lock:
                self._stats.ssd_loads += 1
        except FileNotFoundError:
            pass

    def prefetch_experts(
        self, layer_idx: int, expert_ids: list[int]
    ) -> list[Future]:
        """Pre-fetch experts that aren't already in VRAM or RAM.

        Returns a list of :class:`Future` objects for SSD-only experts
        (empty list if all are already cached).
        """
        if self._ssd is None:
            return []

        futures: list[Future] = []
        now = time.monotonic()

        for eid in expert_ids:
            key = (layer_idx, eid)
            with self._lock:
                if key in self._vram_cache or key in self._ram_buffer:
                    with self._stats_lock:
                        self._stats.prefetch_hits += 1
                    continue

            # Only prefetch if the expert has been "interesting" before
            if self._lru_policy.should_prefetch(layer_idx, eid):
                fut = self._ssd.async_load(layer_idx, eid)
                futures.append(fut)

                # Post-load callback: put into RAM buffer when done
                def _on_loaded(f: Future, _key=key, _now=now, _eid=eid) -> None:
                    try:
                        gate, up, down = f.result()
                        with self._lock:
                            self._ensure_ram_capacity(_key)
                            self._ram_buffer[_key] = (gate, up, down)
                            self._ram_buffer.move_to_end(_key)
                            self._lru_policy.access(layer_idx, _eid)
                            self._locations[_key] = ExpertLocation(
                                expert_id=_eid,
                                layer_idx=layer_idx,
                                tier=MemoryTier.RAM,
                                size_bytes=self._expert_size_bytes,
                                last_access_time=_now,
                                access_count=1,
                            )
                    except Exception:
                        pass  # swallow errors in prefetch

                fut.add_done_callback(_on_loaded)
                with self._stats_lock:
                    self._stats.prefetch_misses += 1
            else:
                with self._stats_lock:
                    self._stats.prefetch_misses += 1

        return futures

    def update_access(self, layer_idx: int, expert_id: int) -> None:
        """Record that an expert was accessed (for LRU-K bookkeeping)."""
        self._lru_policy.access(layer_idx, expert_id)

    def get_expert_tier(self, layer_idx: int, expert_id: int) -> MemoryTier:
        """Return the current tier of an expert."""
        key = (layer_idx, expert_id)
        loc = self._locations.get(key)
        if loc is not None:
            return loc.tier
        # Check caches directly as fallback
        with self._lock:
            if key in self._vram_cache:
                return MemoryTier.VRAM
            if key in self._ram_buffer:
                return MemoryTier.RAM
        return MemoryTier.SSD

    def stats(self) -> TieredMemoryStats:
        """Return a snapshot of current statistics."""
        with self._stats_lock:
            # Return a copy so the caller can read without races
            return TieredMemoryStats(
                vram_hits=self._stats.vram_hits,
                ram_hits=self._stats.ram_hits,
                ssd_hits=self._stats.ssd_hits,
                ssd_loads=self._stats.ssd_loads,
                evictions_from_ram=self._stats.evictions_from_ram,
                evictions_to_ssd=self._stats.evictions_to_ssd,
                ram_to_vram_transfers=self._stats.ram_to_vram_transfers,
                total_ssd_read_bytes=self._stats.total_ssd_read_bytes,
                total_ssd_read_ms=self._stats.total_ssd_read_ms,
                prefetch_hits=self._stats.prefetch_hits,
                prefetch_misses=self._stats.prefetch_misses,
            )

    def tier_summary(self) -> str:
        """Human-readable summary of where experts currently live."""
        vram_count = len(self._vram_cache)
        ram_count = len(self._ram_buffer)
        ssd_count = (
            self._num_layers * self._num_experts - vram_count - ram_count
        )

        lines = [
            "=== Tiered Memory Summary ===",
            f"  VRAM (Tier A): {vram_count:>5d} experts",
            f"  RAM  (Tier B): {ram_count:>5d} experts",
            f"  SSD  (Tier C): {ssd_count:>5d} experts",
            f"  Total experts: {self._num_layers * self._num_experts}",
            "",
            f"  LRU-K policy: {self._lru_policy.stats()}",
        ]

        s = self.stats()
        lines.extend([
            "",
            "--- Access Statistics ---",
            f"  VRAM hits:            {s.vram_hits}",
            f"  RAM hits:             {s.ram_hits}",
            f"  SSD hits:             {s.ssd_hits}",
            f"  SSD loads:            {s.ssd_loads}",
            f"  Evictions from RAM:   {s.evictions_from_ram}",
            f"  Evictions to SSD:     {s.evictions_to_ssd}",
            f"  RAM→VRAM transfers:   {s.ram_to_vram_transfers}",
            f"  Total SSD read bytes: {s.total_ssd_read_bytes:,}",
            f"  Total SSD read ms:    {s.total_ssd_read_ms:.2f}",
            f"  Prefetch hits:        {s.prefetch_hits}",
            f"  Prefetch misses:      {s.prefetch_misses}",
        ])

        return "\n".join(lines)

    def close(self) -> None:
        """Release resources (SSD thread pool, clear caches)."""
        if self._ssd is not None:
            self._ssd.close()
        self._vram_cache.clear()
        self._ram_buffer.clear()
        self._locations.clear()

    # -- internal ------------------------------------------------------------

    def _ensure_ram_capacity(self, new_key: tuple[int, int]) -> None:
        """Evict from RAM buffer until there is room for *new_key*.

        Must be called while holding ``self._lock``.
        """
        max_ram = self._lru_policy.capacity
        while len(self._ram_buffer) >= max_ram and new_key not in self._ram_buffer:
            victim = self._lru_policy.evict()
            if victim is None:
                # Fallback: evict oldest from RAM buffer directly
                if self._ram_buffer:
                    victim = next(iter(self._ram_buffer))
                else:
                    break
            weights = self._ram_buffer.pop(victim, None)
            if weights is not None:
                loc = self._locations.get(victim)
                if loc is not None:
                    if self._ssd is not None:
                        loc.tier = MemoryTier.SSD
                        # Write back to SSD
                        try:
                            self._ssd.store_expert(
                                victim[0], victim[1],
                                weights[0], weights[1], weights[2],
                            )
                        except OSError:
                            pass
                        with self._stats_lock:
                            self._stats.evictions_to_ssd += 1
                    else:
                        loc.tier = MemoryTier.SSD
                with self._stats_lock:
                    self._stats.evictions_from_ram += 1
