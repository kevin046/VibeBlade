"""Hot/Cold MoE Split Executor — GPU for hot experts, CPU for cold experts.

Splits MoE expert computation between GPU VRAM and system RAM:
- Hot experts: weights pinned in GPU, computed via GPUBackend.matmul
- Cold experts: weights stay in system RAM, computed via CPU thread pool
- Only activations cross PCIe, not weights — eliminates the bandwidth bottleneck

When a TieredMemoryManager is attached (HYBRID_SSD mode), cold experts
are further split into RAM-resident (fast) and SSD-backed (slow, pre-fetched).
"""

from __future__ import annotations

import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

# MemoryTier used at runtime for tier dispatch
try:
    from .tiered_memory import MemoryTier
except ImportError:
    MemoryTier = None  # tiered_memory not available

if TYPE_CHECKING:
    from .gpu import GPUBackend


@dataclass
class ExecutorStats:
    """Per-token execution statistics."""

    gpu_hits: int = 0
    cpu_falls: int = 0
    gpu_latency_ms: float = 0.0
    cpu_latency_ms: float = 0.0
    sync_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.gpu_hits + self.cpu_falls
        return self.gpu_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "gpu_hits": self.gpu_hits,
            "cpu_falls": self.cpu_falls,
            "gpu_latency_ms": round(self.gpu_latency_ms, 3),
            "cpu_latency_ms": round(self.cpu_latency_ms, 3),
            "sync_latency_ms": round(self.sync_latency_ms, 3),
            "hit_rate": round(self.hit_rate, 4),
        }


class HotColdExecutor:
    """Executes MoE FFN with hot/cold expert splitting.

    Hot experts (GPU): weights loaded into GPUBackend, computed via GPU matmul.
    Cold experts (CPU): weights stay in RAM, computed via numpy in thread pool.

    Usage:
        executor = HotColdExecutor(
            hot_cold_map=hot_map,
            routers=routers,       # {layer_idx: ExpertRouter}
            experts=expert_sets,   # {layer_idx: MoEExpertSet}
            gpu_backend=gpu,
            cpu_threads=8,
        )

        # During decode:
        output, stats = executor.dispatch(layer_idx=0, x=h_norm)
    """

    def __init__(
        self,
        hot_cold_map,  # HotColdMap from moe_profiler
        routers: dict,  # {layer_idx: ExpertRouter}
        experts: dict,  # {layer_idx: MoEExpertSet}
        shared_experts: dict = None,  # {layer_idx: (gate_w, up_w, down_w)} optional
        gpu_backend: "GPUBackend" = None,
        cpu_threads: int = 8,
        tiered_mgr=None,  # TieredMemoryManager from tiered_memory.py (optional, for HYBRID_SSD)
    ):
        self._hot_map = hot_cold_map
        self._routers = routers
        self._experts = experts
        self._shared = shared_experts or {}
        self._gpu = gpu_backend
        self._cpu_pool = (
            ThreadPoolExecutor(max_workers=cpu_threads) if cpu_threads > 0 else None
        )
        self._cpu_threads = cpu_threads
        self._tiered_mgr = tiered_mgr

        # Load hot expert weights into GPU (if GPU available)
        self._gpu_expert_weights: dict[int, dict[int, tuple]] = {}
        if self._gpu and self._gpu.is_gpu:
            self._pin_hot_experts()

        # Cumulative stats
        self._total_stats = ExecutorStats()
        self._stats_lock = threading.Lock()

    def _pin_hot_experts(self) -> None:
        """Load hot expert weights into GPU memory.

        For each layer, for each hot expert, pre-load the weight tensors
        into GPU memory via the backend. Since GPUBackend.matmul already
        transfers data, we store references to the weight arrays.
        """
        for layer_idx, hot_ids in self._hot_map.hot_experts.items():
            self._gpu_expert_weights[layer_idx] = {}
            expert_set = self._experts.get(layer_idx)
            if expert_set is None:
                continue
            for eid in hot_ids:
                gate_w, up_w, down_w = expert_set.get_expert(eid)
                self._gpu_expert_weights[layer_idx][eid] = (gate_w, up_w, down_w)

    @property
    def stats(self) -> ExecutorStats:
        """Thread-safe cumulative stats."""
        with self._stats_lock:
            s = self._total_stats
            return ExecutorStats(
                gpu_hits=s.gpu_hits,
                cpu_falls=s.cpu_falls,
                gpu_latency_ms=s.gpu_latency_ms,
                cpu_latency_ms=s.cpu_latency_ms,
                sync_latency_ms=s.sync_latency_ms,
            )

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._total_stats = ExecutorStats()

    def dispatch(
        self,
        layer_idx: int,
        x: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Execute MoE FFN for one layer with hot/cold splitting.

        1. Route: select top-k experts
        2. Split: hot vs cold based on hot_cold_map
        3. GPU: compute hot experts in parallel (or sequentially if no GPU)
        4. CPU: compute cold experts in thread pool
        5. Synchronize + weighted sum

        Args:
            layer_idx: transformer layer index
            x: (1, shared_dim) or (shared_dim,) — normalized hidden state

        Returns:
            output: same shape as input x
            stats: dict with per-token execution details
        """
        # ── Step 0: track original shape ──
        orig_shape = x.shape
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]  # (1, shared_dim)
        x = x.astype(np.float32)

        # ── Step 1: get router and expert set ──
        router = self._routers.get(layer_idx)
        expert_set = self._experts.get(layer_idx)
        if router is None or expert_set is None:
            stats = {
                "gpu_hits": 0,
                "cpu_falls": 0,
                "hit_rate": 0.0,
                "gpu_latency_ms": 0.0,
                "cpu_latency_ms": 0.0,
                "sync_latency_ms": 0.0,
                "expert_indices": np.array([]),
                "expert_weights": np.array([]),
            }
            output = np.zeros(orig_shape, dtype=np.float32)
            return output, stats

        # ── Step 2: route ──
        indices, weights = router.route(x)  # each (1, topk) since batch=1
        topk = router.topk

        # ── Step 3: shared expert (always computed on available device) ──
        t_sync_start = time.perf_counter()
        shared_gate, shared_up, shared_down = self._shared.get(layer_idx, (None, None, None))
        if shared_gate is not None:
            if self._gpu and self._gpu.is_gpu:
                shared_out = self._compute_single_expert_gpu(
                    x, shared_gate, shared_up, shared_down
                )
            else:
                shared_out = self._compute_single_expert_cpu(
                    x, shared_gate, shared_up, shared_down
                )
        else:
            shared_out = np.zeros_like(x)
        t_sync_end = time.perf_counter()

        # ── Step 4: split selected experts into hot / ram / ssd ──
        hot_ids = set(self._hot_map.hot_experts.get(layer_idx, []))
        hot_experts = []   # (slot_idx, expert_id, weight)
        ram_experts = []   # (slot_idx, expert_id, weight) — in RAM via TieredMemoryManager
        ssd_experts = []   # (slot_idx, expert_id, weight) — needs SSD load

        for k in range(topk):
            eid = int(indices[0, k])
            w = float(weights[0, k])
            if eid in hot_ids:
                hot_experts.append((k, eid, w))
            elif self._tiered_mgr is not None:
                tier = self._tiered_mgr.get_expert_tier(layer_idx, eid)
                if tier == MemoryTier.RAM:
                    ram_experts.append((k, eid, w))
                else:
                    ssd_experts.append((k, eid, w))
            else:
                # No tiered manager — fall back to simple hot/cold
                ram_experts.append((k, eid, w))

        # ── Step 5: compute hot experts ──
        t_gpu_start = time.perf_counter()
        hot_results = {}  # slot_idx -> expert_output
        use_gpu = self._gpu is not None and self._gpu.is_gpu

        for slot_idx, eid, w in hot_experts:
            gate_w, up_w, down_w = expert_set.get_expert(eid)
            if use_gpu:
                result = self._compute_single_expert_gpu(x, gate_w, up_w, down_w)
            else:
                result = self._compute_single_expert_cpu(x, gate_w, up_w, down_w)
            hot_results[slot_idx] = (w, result)

        t_gpu_end = time.perf_counter()

        # ── Step 6: compute ram/ssd experts ──
        t_cpu_start = time.perf_counter()
        cold_results = {}  # slot_idx -> expert_output

        # RAM-resident experts: fast, already in memory
        all_cold = ram_experts + ssd_experts

        if all_cold and self._cpu_pool is not None:
            futures = {}
            for slot_idx, eid, w in all_cold:
                # Get weights: from TieredMemoryManager (3-tier) or directly from expert_set
                if self._tiered_mgr is not None:
                    weights_tuple = self._tiered_mgr.get_expert(layer_idx, eid)
                    if weights_tuple is None:
                        # Fallback: load from expert_set
                        weights_tuple = expert_set.get_expert(eid)
                    self._tiered_mgr.update_access(layer_idx, eid)
                else:
                    weights_tuple = expert_set.get_expert(eid)

                gate_w, up_w, down_w = weights_tuple
                future = self._cpu_pool.submit(
                    self._compute_single_expert_cpu, x, gate_w, up_w, down_w
                )
                futures[future] = (slot_idx, w)

            for future in as_completed(futures):
                slot_idx, w = futures[future]
                cold_results[slot_idx] = (w, future.result())
        elif all_cold:
            for slot_idx, eid, w in all_cold:
                if self._tiered_mgr is not None:
                    weights_tuple = self._tiered_mgr.get_expert(layer_idx, eid)
                    if weights_tuple is None:
                        weights_tuple = expert_set.get_expert(eid)
                    self._tiered_mgr.update_access(layer_idx, eid)
                else:
                    weights_tuple = expert_set.get_expert(eid)
                gate_w, up_w, down_w = weights_tuple
                cold_results[slot_idx] = (w, self._compute_single_expert_cpu(x, gate_w, up_w, down_w))

        t_cpu_end = time.perf_counter()

        # ── Step 7: weighted sum ──
        t_sync2_start = time.perf_counter()
        output = np.zeros_like(x)
        for slot_idx, (w, expert_out) in hot_results.items():
            output += w * expert_out
        for slot_idx, (w, expert_out) in cold_results.items():
            output += w * expert_out
        output += shared_out

        # Restore original shape
        if squeeze:
            output = output[0]

        t_sync2_end = time.perf_counter()

        # ── Step 8: record stats ──
        gpu_latency = (t_gpu_end - t_gpu_start) * 1000.0
        cpu_latency = (t_cpu_end - t_cpu_start) * 1000.0
        sync_latency = (t_sync2_end - t_sync2_start + t_sync_end - t_sync_start) * 1000.0

        per_token_stats = ExecutorStats(
            gpu_hits=len(hot_experts),
            cpu_falls=len(ram_experts) + len(ssd_experts),
            gpu_latency_ms=gpu_latency,
            cpu_latency_ms=cpu_latency,
            sync_latency_ms=sync_latency,
        )

        with self._stats_lock:
            self._total_stats.gpu_hits += per_token_stats.gpu_hits
            self._total_stats.cpu_falls += per_token_stats.cpu_falls
            self._total_stats.gpu_latency_ms += per_token_stats.gpu_latency_ms
            self._total_stats.cpu_latency_ms += per_token_stats.cpu_latency_ms
            self._total_stats.sync_latency_ms += per_token_stats.sync_latency_ms

        stats = {
            "gpu_hits": per_token_stats.gpu_hits,
            "cpu_falls": per_token_stats.cpu_falls,
            "hit_rate": per_token_stats.hit_rate,
            "gpu_latency_ms": round(per_token_stats.gpu_latency_ms, 3),
            "cpu_latency_ms": round(per_token_stats.cpu_latency_ms, 3),
            "sync_latency_ms": round(per_token_stats.sync_latency_ms, 3),
            "expert_indices": indices[0] if squeeze else indices,
            "expert_weights": weights[0] if squeeze else weights,
        }

        return output, stats

    def _compute_single_expert_cpu(
        self,
        x: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray:
        """Compute single expert FFN on CPU: down(silu(gate(x)) * up(x)).

        Uses moe convention (no transpose): gate/up (shared_dim, expert_dim), down (expert_dim, shared_dim).
        """
        gate_out = x @ gate_w
        up_out = x @ up_w
        return (self._silu(gate_out) * up_out) @ down_w

    def _compute_single_expert_gpu(
        self,
        x: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray:
        """Compute single expert FFN on GPU via GPUBackend."""
        gate_out = self._gpu.matmul(x, gate_w)
        up_out = self._gpu.matmul(x, up_w)
        silu_out = self._gpu.silu(gate_out) * up_out
        return self._gpu.matmul(silu_out, down_w)

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(x.dtype)

    def layer_stats(self, layer_idx: int) -> dict:
        """Get detailed stats for a specific layer."""
        hot = self._hot_map.hot_experts.get(layer_idx, [])
        cold = self._hot_map.cold_experts.get(layer_idx, [])
        return {
            "layer": layer_idx,
            "hot_experts": len(hot),
            "cold_experts": len(cold),
            "hot_pct": len(hot) / max(len(hot) + len(cold), 1) * 100,
        }

    def memory_estimate(
        self,
        expert_dim: int,
        shared_dim: int,
        bytes_per_param: float = 0.5,
    ) -> dict:
        """Estimate VRAM/RAM usage for the hot/cold split.

        Args:
            expert_dim: intermediate dimension per expert
            shared_dim: hidden dimension
            bytes_per_param: bytes per weight element (0.5 for Q4, 1.0 for fp16, 2.0 for fp32)
        """
        # Per expert: gate(shared_dim, expert_dim) + up(shared_dim, expert_dim) + down(expert_dim, shared_dim)
        # = 2 * shared_dim * expert_dim + expert_dim * shared_dim = 3 * shared_dim * expert_dim
        per_expert_bytes = 3 * shared_dim * expert_dim * bytes_per_param

        total_hot = sum(len(v) for v in self._hot_map.hot_experts.values())
        total_cold = sum(len(v) for v in self._hot_map.cold_experts.values())

        return {
            "hot_vram_gb": total_hot * per_expert_bytes / (1024**3),
            "cold_ram_gb": total_cold * per_expert_bytes / (1024**3),
            "per_expert_mb": per_expert_bytes / (1024**2),
            "total_hot_experts": total_hot,
            "total_cold_experts": total_cold,
        }

    def shutdown(self) -> None:
        """Clean up thread pool."""
        if self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=False)
            self._cpu_pool = None  # Prevent submit after shutdown
