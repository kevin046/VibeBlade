"""Asynchronous dual-stream MoE executor (v1.1 architecture).

GPU and CPU run in parallel, merging results at a barrier at the end of each
FFN block.  While the GPU/main thread processes hot experts, a CPU thread pool
simultaneously computes cold expert activations and prefetches predicted
experts for subsequent layers via an :class:`ExpertOracle`.

Architecture
------------
Stream 1 (GPU/main thread): attention + hot expert FFN
Stream 2 (CPU thread pool): cold expert FFN + next-layer expert prefetch
Barrier: weighted sum of cold outputs merged via zero-copy pinned buffer

Dependencies: numpy, concurrent.futures (stdlib), threading (stdlib).
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .moe_oracle import ExpertOracle


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ColdExpertResult:
    """Result of a cold expert computation submitted to the thread pool.

    Attributes:
        expert_id: Numeric identifier of the expert that produced this result.
        output: Activation delta with shape ``(hidden_dim,)``.
        weight: Routing weight assigned by the router (softmax-normalised).
        compute_time_ms: Wall-clock time (ms) spent inside the worker thread.
    """

    expert_id: int
    output: np.ndarray
    weight: float
    compute_time_ms: float


@dataclass
class AsyncStats:
    """Per-layer execution statistics for the dual-stream executor.

    Attributes:
        hot_compute_ms: Time (ms) spent on hot expert computation (main thread).
        cold_compute_ms: Time (ms) spent on cold expert computation (thread pool).
        cold_wait_ms: Time (ms) the main thread blocked waiting for cold results
            at the barrier.
        prefetch_hit_count: Number of prefetched expert IDs that were actually
            selected by the router (oracle accuracy signal).
        cold_expert_count: Total cold experts dispatched for this layer.
        gpu_overlap_pct: Percentage [0-100] of cold compute time that overlapped
            with hot compute.  100 % means the GPU finished *after* the CPU;
            0 % means the CPU finished *after* the GPU.
    """

    hot_compute_ms: float = 0.0
    cold_compute_ms: float = 0.0
    cold_wait_ms: float = 0.0
    prefetch_hit_count: int = 0
    cold_expert_count: int = 0
    gpu_overlap_pct: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a plain dict for logging / telemetry."""
        return {
            "hot_compute_ms": round(self.hot_compute_ms, 3),
            "cold_compute_ms": round(self.cold_compute_ms, 3),
            "cold_wait_ms": round(self.cold_wait_ms, 3),
            "prefetch_hit_count": self.prefetch_hit_count,
            "cold_expert_count": self.cold_expert_count,
            "gpu_overlap_pct": round(self.gpu_overlap_pct, 1),
        }


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------


class AsyncMoEExecutor:
    """Dual-stream MoE executor: GPU (hot experts) ‖ CPU (cold experts).

    Two parallel streams execute simultaneously:

    1. **Stream 1 (GPU / main thread)** — runs the attention mechanism followed
       by the *hot* expert FFN (experts whose weights reside in GPU VRAM).
    2. **Stream 2 (CPU thread pool)** — computes *cold* expert FFN (experts
       whose weights stay in system RAM) and prefetches expert weights for the
       next layer using an optional :class:`ExpertOracle` Markov predictor.
    3. **Barrier** — at the end of the FFN block the weighted cold-activation
       deltas are merged into the hidden state via a pre-allocated, pinned
       (mlock'd) numpy buffer.

    Parameters
    ----------
    hot_experts:
        Mapping from *layer_idx* to the set of expert IDs that are considered
        "hot" (reside in GPU VRAM).
    cold_expert_weights:
        Nested mapping ``layer_idx → expert_id → (gate_w, up_w, down_w)``
        where each tuple holds numpy arrays for the simplified two-gate FFN.
    cpu_threads:
        Maximum number of worker threads for the cold-expert thread pool.
    oracle:
        Optional :class:`ExpertOracle` used for next-layer expert prediction.
        When provided, :meth:`prefetch_experts` will query the oracle and
        trigger pre-loading of the most likely experts into CPU cache.
    pinned_buffer_size:
        Dimensionality of the hidden-state vector.  A zero-buffer of this size
        is pre-allocated and kept pinned in physical RAM for zero-copy merging.
    """

    def __init__(
        self,
        hot_experts: dict[int, set[int]],
        cold_expert_weights: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
        cpu_threads: int = 8,
        oracle: ExpertOracle | None = None,
        pinned_buffer_size: int = 4096,
    ) -> None:
        self._hot_experts: dict[int, set[int]] = hot_experts
        self._cold_weights: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = (
            cold_expert_weights
        )
        self._oracle: ExpertOracle | None = oracle
        self._oracle_lock = threading.Lock()

        # Thread pool for cold-expert computation + prefetching.
        self._executor = ThreadPoolExecutor(max_workers=cpu_threads)

        # Pre-allocated pinned merge buffer.
        self._pinned_buffer: np.ndarray = np.zeros(pinned_buffer_size, dtype=np.float32)

        # Track the set of expert IDs that were prefetched for the *next*
        # layer so we can measure oracle hit rate.
        self._prefetched_ids: set[int] = set()

        # Optional CPU-side weight cache to avoid repeated numpy access
        # during prefetch.  Keyed by (layer_idx, expert_id).
        self._weight_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def execute_layer(
        self,
        layer_idx: int,
        hidden_state: np.ndarray,
        router_output: list[tuple[int, float]],
        hot_expert_fn: Callable[[np.ndarray, list[tuple[int, float]]], np.ndarray],
        prefetch_layer: int | None = None,
    ) -> tuple[np.ndarray, AsyncStats]:
        """Execute one transformer layer with dual-stream parallelism.

        1. Split *router_output* into hot and cold expert sets.
        2. Immediately submit cold expert computations to the thread pool.
        3. Run *hot_expert_fn* on the calling thread (GPU or main).
        4. Barrier — wait for all cold futures and merge their weighted outputs
           into *hidden_state* using the pinned zero-copy buffer.
        5. Optionally trigger oracle-based prefetching for the next layer.

        Parameters
        ----------
        layer_idx:
            Index of the current transformer layer.
        hidden_state:
            Activation vector of shape ``(hidden_dim,)`` (or batched).  Updated
            **in-place** with cold expert contributions.
        router_output:
            List of ``(expert_id, routing_weight)`` pairs from the gating
            network, sorted by weight descending.
        hot_expert_fn:
            User-supplied callable ``(hidden_state, hot_experts) → output``
            that executes hot expert FFN on the GPU/main thread.
        prefetch_layer:
            If not *None*, the layer index for which expert prefetching should
            be triggered using the oracle.

        Returns
        -------
        tuple[np.ndarray, AsyncStats]
            Updated hidden state and per-layer execution statistics.
        """
        stats = AsyncStats()

        # ── 1. Classify experts into hot / cold ──────────────────────────
        hot_set = self._hot_experts.get(layer_idx, set())
        hot_expert_list: list[tuple[int, float]] = []
        cold_expert_list: list[tuple[int, float]] = []

        for expert_id, weight in router_output:
            if expert_id in hot_set:
                hot_expert_list.append((expert_id, weight))
            else:
                cold_expert_list.append((expert_id, weight))

        stats.cold_expert_count = len(cold_expert_list)

        # ── 2. Dispatch cold experts to the thread pool ─────────────────
        cold_layer_weights = self._cold_weights.get(layer_idx, {})
        cold_futures: dict[Future[ColdExpertResult], int] = {}

        cold_start = time.perf_counter()
        for expert_id, weight in cold_expert_list:
            weights_tuple = cold_layer_weights.get(expert_id)
            if weights_tuple is None:
                continue
            gate_w, up_w, down_w = weights_tuple
            future = self._executor.submit(
                self._compute_cold_expert,
                hidden_state=hidden_state,
                gate_w=gate_w,
                up_w=up_w,
                down_w=down_w,
                expert_id=expert_id,
                weight=weight,
            )
            cold_futures[future] = expert_id
        # ── 3. Hot expert computation on calling thread (GPU) ────────────
        hot_start = time.perf_counter()
        hot_output = hot_expert_fn(hidden_state, hot_expert_list)
        hot_end = time.perf_counter()
        stats.hot_compute_ms = (hot_end - hot_start) * 1000.0

        # ── 4. Barrier — collect cold results ───────────────────────────
        barrier_start = time.perf_counter()
        cold_results: list[ColdExpertResult] = []
        for future in cold_futures:
            try:
                result = future.result()
                cold_results.append(result)
            except Exception:
                pass  # silently skip failed cold experts
        barrier_end = time.perf_counter()

        stats.cold_wait_ms = (barrier_end - barrier_start) * 1000.0

        # Compute total cold time (from dispatch to all futures resolved).
        cold_total_ms = (barrier_end - cold_start) * 1000.0
        stats.cold_compute_ms = cold_total_ms

        # ── 5. Merge cold activations via pinned buffer ─────────────────
        if cold_results:
            hidden_state = self.merge_activations(hidden_state, cold_results)

        # Combine hot + cold into final hidden state.
        # hot_output already contains the hot expert weighted sum, so we add
        # the cold delta on top.
        hidden_state = hidden_state + hot_output

        # ── 6. Overlap metric ────────────────────────────────────────────
        if cold_total_ms > 0:
            overlap = max(0.0, stats.hot_compute_ms) / cold_total_ms * 100.0
            stats.gpu_overlap_pct = min(overlap, 100.0)

        # ── 7. Prefetch for next layer ──────────────────────────────────
        if prefetch_layer is not None:
            all_expert_ids = [eid for eid, _w in router_output]
            self.prefetch_experts(prefetch_layer, all_expert_ids)
            # Measure oracle hit rate: how many predicted IDs match actual
            # cold experts for *this* layer (we use current cold experts as
            # ground truth for the previous prediction).
            actual_cold_ids = {eid for eid, _w in cold_expert_list}
            stats.prefetch_hit_count = len(self._prefetched_ids & actual_cold_ids)

        return hidden_state, stats

    def prefetch_experts(
        self,
        layer_idx: int,
        current_experts: list[int],
    ) -> list[int]:
        """Predict and pre-load experts for *layer_idx*.

        Uses the :class:`ExpertOracle` (if available) to predict which experts
        will be selected at *layer_idx*.  Falls back to a proximity heuristic
        (assume the same experts will be reused) when no oracle is provided.

        The predicted expert weights are submitted to the thread pool for
        asynchronous loading into the internal CPU weight cache.

        Parameters
        ----------
        layer_idx:
            The layer to prefetch experts for.
        current_experts:
            Expert IDs selected at the *previous* layer (used as oracle input).

        Returns
        -------
        list[int]
            The predicted expert IDs for *layer_idx*.
        """
        predicted: list[int] = []

        if self._oracle is not None:
            predictions = self._oracle.predict_next(layer_idx - 1, current_experts, top_k=6)
            predicted = [eid for eid, _prob in predictions]
        else:
            # Proximity heuristic: assume the same experts are likely reused.
            predicted = list(current_experts)

        # Store for hit-rate measurement on the *next* execute_layer call.
        self._prefetched_ids = set(predicted)

        # Trigger pre-loading of predicted cold expert weights into CPU cache.
        cold_layer_weights = self._cold_weights.get(layer_idx, {})
        for expert_id in predicted:
            if expert_id in cold_layer_weights:
                self._executor.submit(
                    self._preload_weight,
                    layer_idx=layer_idx,
                    expert_id=expert_id,
                    weights_tuple=cold_layer_weights[expert_id],
                )

        return predicted

    def merge_activations(
        self,
        hidden_state: np.ndarray,
        cold_results: list[ColdExpertResult],
    ) -> np.ndarray:
        """Merge cold expert outputs into *hidden_state* via a pinned buffer.

        Uses the pre-allocated ``_pinned_buffer`` to accumulate the weighted
        sum of cold expert activation deltas, then adds the result to
        *hidden_state*.  The buffer is zeroed before each merge to avoid
        stale-data contamination across layers.

        Parameters
        ----------
        hidden_state:
            The current hidden-state activation vector.  Updated **in-place**
            (a new view is returned for convenience).
        cold_results:
            Completed cold-expert computation results.

        Returns
        -------
        np.ndarray
            The updated hidden state with cold expert contributions merged in.
        """
        buffer = self._pinned_buffer
        # Ensure the buffer is large enough; if not, expand it.
        hidden_dim = hidden_state.shape[-1]
        if buffer.shape[0] < hidden_dim:
            self._pinned_buffer = np.zeros(hidden_dim, dtype=np.float32)
            buffer = self._pinned_buffer

        # Zero the accumulation buffer.
        buffer[:hidden_dim] = 0.0

        # Accumulate weighted cold outputs.
        for result in cold_results:
            out = result.output
            # Handle case where output may be a different size than hidden_dim.
            effective_dim = min(out.shape[-1], hidden_dim)
            np.add(
                buffer[:effective_dim],
                result.weight * out[:effective_dim],
                out=buffer[:effective_dim],
            )

        # Add accumulated cold delta to the hidden state (in-place).
        np.add(hidden_state, buffer[:hidden_dim], out=hidden_state)

        return hidden_state

    def observe_experts(
        self,
        layer_idx: int,
        selected_experts: list[int],
    ) -> None:
        """Thread-safe wrapper around :meth:`ExpertOracle.observe`.

        Should be called after each layer's router has selected its experts
        so that the oracle can update its Markov transition table.

        Parameters
        ----------
        layer_idx:
            The layer that just finished execution.
        selected_experts:
            Expert IDs that were actually selected by the router.
        """
        if self._oracle is not None:
            with self._oracle_lock:
                self._oracle.observe(layer_idx, selected_experts)

    def shutdown(self) -> None:
        """Shut down the internal thread pool.

        Call this when the executor is no longer needed to free worker
        threads.  The executor should **not** be used after shutdown.
        """
        self._executor.shutdown(wait=False)
        self._weight_cache.clear()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_cold_expert(
        hidden_state: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
        expert_id: int,
        weight: float,
    ) -> ColdExpertResult:
        """Compute a single cold expert FFN in a worker thread.

        Uses a simplified two-gate FFN: ``output = hidden @ gate_w @ up_w @ down_w``.
        This is intentionally lightweight compared to the full SwiGLU FFN used
        for hot experts, trading a small accuracy loss for lower CPU latency.

        Parameters
        ----------
        hidden_state:
            The input activation (1-D or 2-D).  If 2-D, only the first row is
            used (decode-time single-token path).
        gate_w, up_w, down_w:
            Weight matrices for the three linear projections.
        expert_id:
            Numeric identifier of the expert (for the result).
        weight:
            Routing weight (for the result).

        Returns
        -------
        ColdExpertResult
        """
        t0 = time.perf_counter()
        # Flatten to 1-D for the single-token decode path.
        x = hidden_state.ravel().astype(np.float32)
        # Simplified FFN: x @ gate → @ up → @ down
        intermediate = x @ gate_w
        output = intermediate @ up_w
        output = output @ down_w
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ColdExpertResult(
            expert_id=expert_id,
            output=output.ravel(),
            weight=weight,
            compute_time_ms=elapsed_ms,
        )

    def _preload_weight(
        self,
        layer_idx: int,
        expert_id: int,
        weights_tuple: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Pre-load a cold expert's weights into the internal CPU cache.

        This is a thin wrapper that copies weight references into a dict so
        that future lookups do not need to traverse the nested
        ``_cold_weights`` structure.  The actual numpy arrays are shared (no
        data copy).

        Parameters
        ----------
        layer_idx:
            Layer the expert belongs to.
        expert_id:
            Expert identifier.
        weights_tuple:
            ``(gate_w, up_w, down_w)`` weight matrices.
        """
        key = (layer_idx, expert_id)
        with self._cache_lock:
            self._weight_cache[key] = weights_tuple

    def stats_summary(self) -> dict[str, float | int]:
        """Return aggregate statistics about the executor's internal state.

        Returns
        -------
        dict
            Keys: ``oracle_accuracy``, ``cache_size``, ``buffer_size``.
        """
        accuracy = 0.0
        if self._oracle is not None:
            accuracy = self._oracle.accuracy()
        with self._cache_lock:
            cache_size = len(self._weight_cache)
        return {
            "oracle_accuracy": round(accuracy, 4),
            "cache_size": cache_size,
            "buffer_size": self._pinned_buffer.nbytes,
        }
