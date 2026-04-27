"""Advanced MoE optimisations — confidence routing, prefetching, heterogeneous
quantisation, and CPU kernel selection.

These classes build on top of :class:`ExpertRouter` and :class:`HotColdMap`
from the base ``moe`` / ``moe_profiler`` modules to squeeze every last drop of
performance from a tiered-memory MoE setup.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vibeblade.moe import ExpertRouter
    from vibeblade.moe_profiler import HotColdMap

__all__ = [
    "ConfidenceRouter",
    "ContextAwarePrefetcher",
    "HeteroQuantizer",
    "CPUKernelOptimizer",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ConfidenceRouter
# ═══════════════════════════════════════════════════════════════════════════════


class ConfidenceRouter:
    """Wraps :class:`ExpertRouter` with confidence-based dynamic *top-k*.

    If the top-1 expert's routing weight exceeds ``confidence_threshold``, the
    remaining experts are skipped (*early exit*).  This yields a dynamic speed
    boost during simple prose or repetitive code where the router is highly
    confident about a single expert.

    Stats tracked (cumulative):
        * **total_tokens** — number of token-level routing decisions made.
        * **early_exit_count** — tokens where confidence exceeded threshold.
        * **avg_experts_per_token** — running average (dynamic: 1 … max_topk).
        * **saved_experts** — total expert computations saved vs. full topk.
    """

    def __init__(
        self,
        base_router: ExpertRouter,
        confidence_threshold: float = 0.9,
        min_topk: int = 1,
    ) -> None:
        """Initialise the confidence-aware router.

        Args:
            base_router: An :class:`ExpertRouter` instance to wrap.
            confidence_threshold: If top-1 weight ≥ this value, early-exit.
            min_topk: Never route fewer than this many experts per token.
        """
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        if min_topk < 1:
            raise ValueError(f"min_topk must be ≥ 1, got {min_topk}")
        if min_topk > base_router.topk:
            raise ValueError(
                f"min_topk ({min_topk}) cannot exceed base router topk ({base_router.topk})"
            )

        self._router = base_router
        self._threshold = float(confidence_threshold)
        self._min_topk = int(min_topk)

        # Cumulative stats
        self._total_tokens: int = 0
        self._early_exit_count: int = 0
        self._total_experts_used: int = 0  # sum of dynamic topk across tokens
        self._saved_experts: int = 0

    # ── public API ──────────────────────────────────────────────────────

    @property
    def confidence_threshold(self) -> float:
        """Current confidence threshold."""
        return self._threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {value}"
            )
        self._threshold = float(value)

    @property
    def stats(self) -> dict:
        """Cumulative routing statistics.

        Returns:
            Dict with ``total_tokens``, ``early_exit_count``,
            ``avg_experts_per_token``, ``saved_experts``, and
            ``early_exit_rate``.
        """
        avg = (
            self._total_experts_used / self._total_tokens
            if self._total_tokens > 0
            else 0.0
        )
        rate = (
            self._early_exit_count / self._total_tokens
            if self._total_tokens > 0
            else 0.0
        )
        return {
            "total_tokens": self._total_tokens,
            "early_exit_count": self._early_exit_count,
            "avg_experts_per_token": avg,
            "saved_experts": self._saved_experts,
            "early_exit_rate": rate,
        }

    def reset_stats(self) -> None:
        """Reset all cumulative counters to zero."""
        self._total_tokens = 0
        self._early_exit_count = 0
        self._total_experts_used = 0
        self._saved_experts = 0

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Route with confidence early exit.

        Args:
            x: (batch, shared_dim) or (shared_dim,) — hidden states.

        Returns:
            A 3-tuple:

            * **expert_indices** — ``(batch, dynamic_topk)`` int array.
              May be shorter than the base router's ``topk``.
            * **expert_weights** — ``(batch, dynamic_topk)`` float32 array.
            * **per_token_stats** — dict with per-batch arrays
              ``confidence``, ``num_experts``, ``early_exit``.
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        batch_size = x.shape[0]
        max_topk = self._router.topk

        # Step 1: run base router to get full topk + weights
        full_indices, full_weights = self._router.route(x)  # (batch, topk)

        # full_indices / full_weights are (batch, topk) after squeeze handling.
        # The base router already handled the squeeze/unsqueeze for its own
        # return value, but we unsqueezed x above so it returns (batch, topk).
        if full_indices.ndim == 1:
            full_indices = full_indices[np.newaxis, :]
            full_weights = full_weights[np.newaxis, :]

        # Step 2: check top-1 weight vs confidence_threshold
        top1_weights = full_weights[:, 0]  # (batch,)
        is_early_exit = top1_weights >= self._threshold  # (batch,) bool

        # Build dynamic topk per token
        per_token_stats: dict = {
            "confidence": top1_weights.astype(np.float32),
            "num_experts": np.full(batch_size, max_topk, dtype=np.int32),
            "early_exit": is_early_exit.astype(np.bool_),
        }

        # Determine effective topk for each token
        if np.all(is_early_exit):
            # All tokens early-exit — use min_topk
            effective_topk = self._min_topk
            out_indices = full_indices[:, :effective_topk]
            out_weights = full_weights[:, :effective_topk]
            # Re-normalise the truncated weights
            wsum = out_weights.sum(axis=-1, keepdims=True)
            wsum = np.where(wsum < 1e-12, 1.0, wsum)
            out_weights = out_weights / wsum
            per_token_stats["num_experts"][:] = effective_topk
        elif np.any(is_early_exit):
            # Mixed — early-exit tokens get min_topk, others get full topk.
            # Output array uses max_topk columns; early-exit tokens have zeros
            # in the unused columns, and we track the effective count.
            out_indices = full_indices.copy()
            out_weights = full_weights.copy()

            ee_mask = is_early_exit  # (batch,)

            # For early-exit tokens, zero out columns beyond min_topk
            if self._min_topk < max_topk:
                out_weights[ee_mask, self._min_topk:] = 0.0
                # Re-normalise only the min_topk columns for early-exit tokens
                ee_sums = out_weights[ee_mask, : self._min_topk].sum(
                    axis=-1, keepdims=True
                )
                ee_sums = np.where(ee_sums < 1e-12, 1.0, ee_sums)
                out_weights[ee_mask, : self._min_topk] = (
                    out_weights[ee_mask, : self._min_topk] / ee_sums
                )

            per_token_stats["num_experts"][ee_mask] = self._min_topk
        else:
            # No early exits
            out_indices = full_indices
            out_weights = full_weights

        # Step 5: update cumulative stats
        self._total_tokens += batch_size
        self._early_exit_count += int(is_early_exit.sum())
        total_used = int(per_token_stats["num_experts"].sum())
        self._total_experts_used += total_used
        self._saved_experts += batch_size * max_topk - total_used

        if squeeze:
            eff_k = int(per_token_stats["num_experts"][0])
            out_indices = out_indices[0, :eff_k]
            out_weights = out_weights[0, :eff_k]
            per_token_stats = {k: v[0] for k, v in per_token_stats.items()}

        return (
            out_indices.astype(np.int32),
            out_weights.astype(np.float32),
            per_token_stats,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ContextAwarePrefetcher
# ═══════════════════════════════════════════════════════════════════════════════


class ContextAwarePrefetcher:
    """Predicts future expert accesses so they can be pre-loaded.

    While the GPU processes layer *N*'s attention, the CPU should already be
    loading the experts predicted for layers *N+1*, *N+2* based on routing
    patterns.

    Two prediction strategies:

    1. **ROUTING_PROXIMITY** — use the current hidden state to run each
       future layer's router and predict which experts will be needed
       (*commit-based*: consecutive tokens tend to have correlated routes).
    2. **FREQUENCY_BASED** — use historically-observed expert activation
       frequencies per layer (from the profiler).

    The prefetch queue looks ``prefetch_depth`` layers ahead.
    """

    class Strategy(Enum):
        """Prediction strategy for expert prefetching."""

        ROUTING_PROXIMITY = "routing_proximity"
        FREQUENCY_BASED = "frequency_based"

    def __init__(
        self,
        routers: dict[int, ExpertRouter],
        hot_cold_map: HotColdMap | None = None,
        hot_expert_ids: dict[int, list[int]] | None = None,
        prefetch_depth: int = 2,
        strategy: Strategy = Strategy.ROUTING_PROXIMITY,
        frequency_map: dict[int, dict[int, float]] | None = None,
    ) -> None:
        """Initialise the prefetcher.

        Args:
            routers: ``{layer_idx: ExpertRouter}`` for every MoE layer.
            hot_cold_map: A :class:`HotColdMap` instance.  Hot experts are
                skipped during prefetch (already resident in VRAM).
            hot_expert_ids: Fallback hot-set if ``hot_cold_map`` is ``None``.
                ``{layer_idx: [expert_ids]}``.
            prefetch_depth: How many layers ahead to predict.
            strategy: Prediction strategy enum value.
            frequency_map: ``{layer_idx: {expert_id: frequency}}`` from
                profiling (required for :attr:`Strategy.FREQUENCY_BASED`).
        """
        self._routers = dict(routers)
        self._hot_cold_map = hot_cold_map
        self._hot_expert_ids = hot_expert_ids
        self._prefetch_depth = max(1, int(prefetch_depth))
        self._strategy = strategy
        self._frequency_map = frequency_map or {}

        # Resolve hot-set lookup helper
        self._hot_sets: dict[int, set[int]] = {}
        if hot_cold_map is not None:
            for key, val in hot_cold_map.hot_experts.items():
                self._hot_sets[int(key)] = set(int(e) for e in val)
        elif hot_expert_ids is not None:
            for key, val in hot_expert_ids.items():
                self._hot_sets[int(key)] = set(int(e) for e in val)

        # Tracking
        self._prefetch_issued: int = 0
        self._prefetch_hits: int = 0
        self._pending: dict[int, set[int]] = {}  # layer -> set of expert ids
        self._loaded: dict[int, set[int]] = {}  # layer -> set of expert ids
        # Already-requested (in-flight) prefetches to avoid duplicates
        self._in_flight: dict[int, set[int]] = {}

    # ── helpers ─────────────────────────────────────────────────────────

    def _get_hot_set(self, layer_idx: int) -> set[int]:
        """Return the set of hot (VRAM-resident) expert IDs for a layer."""
        return self._hot_sets.get(layer_idx, set())

    # ── public API ──────────────────────────────────────────────────────

    def update_and_predict(
        self,
        layer_idx: int,
        x_norm: np.ndarray,
    ) -> list[tuple[int, int, float]]:
        """Predict which experts future layers will need.

        Called during layer processing.  Returns a prioritised list of
        ``(target_layer, expert_id, score)`` tuples to prefetch.

        Args:
            layer_idx: Index of the layer currently being processed.
            x_norm: Hidden state after attention norm — used as routing
                input for the *ROUTING_PROXIMITY* strategy.

        Returns:
            List of ``(target_layer, expert_id, score)`` sorted by *score*
            descending.  Experts already in the hot set are excluded.
        """
        predictions: list[tuple[int, int, float]] = []

        for delta in range(1, self._prefetch_depth + 1):
            target_layer = layer_idx + delta
            if target_layer not in self._routers:
                continue

            hot_set = self._get_hot_set(target_layer)
            in_flight_set = self._in_flight.get(target_layer, set())
            already_loaded = self._loaded.get(target_layer, set())
            skip = hot_set | in_flight_set | already_loaded

            if self._strategy == self.Strategy.ROUTING_PROXIMITY:
                router = self._routers[target_layer]
                squeeze = x_norm.ndim == 1
                if squeeze:
                    x_in = x_norm[np.newaxis, :]
                else:
                    x_in = x_norm
                # Average logits across batch for a single prediction
                logits = x_in @ router.weight  # (batch, num_experts)
                avg_logits = logits.mean(axis=0)  # (num_experts,)
                # Softmax for probabilities
                exp_logits = np.exp(
                    avg_logits - avg_logits.max()
                )  # numerically stable
                probs = exp_logits / exp_logits.sum()

                topk = min(router.topk, router.num_experts)
                top_indices = np.argsort(probs)[::-1][:topk]
                for eid in top_indices:
                    eid_int = int(eid)
                    if eid_int not in skip:
                        predictions.append(
                            (target_layer, eid_int, float(probs[eid]))
                        )

            elif self._strategy == self.Strategy.FREQUENCY_BASED:
                freq = self._frequency_map.get(target_layer, {})
                if not freq:
                    continue
                # Sort by frequency descending
                sorted_freq = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
                for eid_int, score in sorted_freq:
                    if eid_int not in skip:
                        predictions.append((target_layer, eid_int, score))

        # Sort by score descending
        predictions.sort(key=lambda t: t[2], reverse=True)

        # Track issued prefetches
        for target_layer, eid, _ in predictions:
            self._prefetch_issued += 1
            self._in_flight.setdefault(target_layer, set()).add(eid)
            self._pending.setdefault(target_layer, set()).add(eid)

        return predictions

    def prefetch_callback(
        self,
        layer_idx: int,
        expert_ids: list[int],
    ) -> None:
        """Notify that a prefetch for *expert_ids* at *layer_idx* completed.

        Called by :class:`TieredMemoryManager` (or equivalent) after the
        prefetch transfer finishes.  Updates internal tracking so these
        experts are not re-requested.
        """
        loaded_set = self._loaded.setdefault(layer_idx, set())
        pending_set = self._pending.get(layer_idx, set())
        in_flight_set = self._in_flight.get(layer_idx, set())

        for eid in expert_ids:
            eid_int = int(eid)
            if eid_int in pending_set:
                self._prefetch_hits += 1
            pending_set.discard(eid_int)
            in_flight_set.discard(eid_int)
            loaded_set.add(eid_int)

    @property
    def stats(self) -> dict:
        """Prefetch statistics.

        Returns:
            Dict with ``prefetch_issued``, ``prefetch_hits``, ``hit_rate``,
            ``lookahead_layers``, ``pending_count``.
        """
        rate = (
            self._prefetch_hits / self._prefetch_issued
            if self._prefetch_issued > 0
            else 0.0
        )
        pending_total = sum(len(s) for s in self._pending.values())
        return {
            "prefetch_issued": self._prefetch_issued,
            "prefetch_hits": self._prefetch_hits,
            "hit_rate": rate,
            "lookahead_layers": self._prefetch_depth,
            "pending_count": pending_total,
        }

    def reset_stats(self) -> None:
        """Reset all prefetch tracking counters."""
        self._prefetch_issued = 0
        self._prefetch_hits = 0
        self._pending.clear()
        self._loaded.clear()
        self._in_flight.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# HeteroQuantizer
# ═══════════════════════════════════════════════════════════════════════════════


class HeteroQuantizer:
    """Heterogeneous quantisation: high-bit for hot experts, low-bit for cold.

    Hot experts (VRAM) are kept at original precision (e.g. 4-bit Q4_K_M) for
    maximum quality.  Cold experts (RAM / SSD) are quantised to 2-bit to halve
    bandwidth and fit more in the RAM buffer.

    **2-bit block-quantisation scheme:**

    * **Block size:** 32 values.
    * **Per block:** 1 × ``float16`` scale + 1 × ``float16`` zero-point +
      32 × 2-bit packed values.
    * **Effective:** 4 bytes header + 8 bytes data = 12 bytes per 32 values
      ≈ 3 bits / value.

    Dequantisation is fast (table lookup + scale) and suitable for CPU matmul.
    """

    class QuantLevel(Enum):
        """Quantisation level for an expert."""

        FULL = "full"  # original precision (4-bit GGUF native)
        HALF = "half"  # 2-bit block quantised

    # Number of values per quantisation block
    BLOCK_SIZE = 32
    # Values packed as 2-bit each: 32 values → 8 bytes of packed data
    PACKED_BYTES = BLOCK_SIZE // 4  # 8 bytes
    # Header: 2 × float16 = 4 bytes
    HEADER_BYTES = 4
    # Total bytes per block
    BLOCK_BYTES = HEADER_BYTES + PACKED_BYTES  # 12 bytes

    # 4 quantisation levels for 2-bit representation: 0, 1, 2, 3
    _NUM_LEVELS = 4

    def __init__(
        self,
        hot_expert_ids: dict[int, list[int]],
        block_size: int = 32,
    ) -> None:
        """Initialise the heterogeneous quantiser.

        Args:
            hot_expert_ids: ``{layer_idx: [expert_ids]}`` — experts that
                stay at full precision (resident in VRAM).
            block_size: Number of values per quantisation block.  Must be
                a multiple of 4 (default 32).
        """
        if block_size % 4 != 0:
            raise ValueError(f"block_size must be a multiple of 4, got {block_size}")
        if block_size < 4:
            raise ValueError(f"block_size must be ≥ 4, got {block_size}")

        self._hot_expert_ids: dict[int, set[int]] = {
            int(k): set(int(e) for e in v) for k, v in hot_expert_ids.items()
        }
        self._block_size = int(block_size)
        self._packed_bytes = block_size // 4
        self._header_bytes = 4  # 2 × float16
        self._block_bytes = self._header_bytes + self._packed_bytes

        # Pre-build dequantisation LUT: map 0..3 → normalised float in [-1, 1]
        self._dequant_lut = np.array(
            [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0],
            dtype=np.float32,
        )

    # ── public API ──────────────────────────────────────────────────────

    def quantize_expert(
        self,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantise three expert weight matrices to 2-bit packed format.

        Args:
            gate_w: Gate projection weight matrix.
            up_w: Up projection weight matrix.
            down_w: Down projection weight matrix.

        Returns:
            Tuple of ``(gate_packed, up_packed, down_packed)`` — each a
            ``uint8`` array in the block-quantised packed format.
        """
        return (
            self._quantize_tensor(gate_w),
            self._quantize_tensor(up_w),
            self._quantize_tensor(down_w),
        )

    def dequantize_expert(
        self,
        gate_packed: np.ndarray,
        up_packed: np.ndarray,
        down_packed: np.ndarray,
        orig_shapes: tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...]
        ],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dequantise packed 2-bit tensors back to ``float16``.

        Args:
            gate_packed: Packed gate weights (uint8).
            up_packed: Packed up weights (uint8).
            down_packed: Packed down weights (uint8).
            orig_shapes: Tuple of ``(gate_shape, up_shape, down_shape)``
                specifying the original dimensions of each matrix.

        Returns:
            Tuple of ``(gate_w, up_w, down_w)`` as ``float16`` arrays.
        """
        return (
            self._dequantize_tensor(gate_packed, orig_shapes[0]),
            self._dequantize_tensor(up_packed, orig_shapes[1]),
            self._dequantize_tensor(down_packed, orig_shapes[2]),
        )

    def expert_memory_savings(self, orig_bytes: int) -> float:
        """Fraction of memory saved by 2-bit quantisation.

        Compares original ``float32`` (4 bytes/value) storage against the
        2-bit packed format (≈ 3 bits/value).

        Args:
            orig_bytes: Original size in bytes (assumed float32).

        Returns:
            Fractional saving, e.g. ``0.5`` for 50 % reduction.
        """
        # Per value: original = 4 bytes (f32), quantised = block_bytes / block_size
        bits_per_val_quantized = (self._block_bytes * 8) / self._block_size
        bits_per_val_original = 32.0  # float32
        savings = 1.0 - (bits_per_val_quantized / bits_per_val_original)
        return float(savings)

    # ── internal quantise / dequantise ──────────────────────────────────

    def _quantize_tensor(self, w: np.ndarray) -> np.ndarray:
        """Quantise a single weight tensor to 2-bit packed format.

        Layout per block (block_size values):
            [float16 scale][float16 zero_point][uint8 packed_values ...]

        The float16 header is stored as 4 bytes of raw data followed by the
        packed 2-bit values.
        """
        flat = w.astype(np.float32).ravel()
        n = len(flat)
        bs = self._block_size
        n_blocks = math.ceil(n / bs)

        # Pad to full blocks
        padded = np.empty(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat
        padded[n:] = 0.0

        blocks = padded.reshape(n_blocks, bs)

        # Per-block min / max
        blk_min = blocks.min(axis=1)  # (n_blocks,)
        blk_max = blocks.max(axis=1)  # (n_blocks,)
        blk_range = blk_max - blk_min
        # Avoid division by zero for constant blocks
        blk_range = np.where(blk_range < 1e-8, 1.0, blk_range)

        scale = blk_range / (self._NUM_LEVELS - 1)  # (n_blocks,)
        zero_point = blk_min  # (n_blocks,)

        # Quantise to 0..3
        normalized = (blocks - zero_point[:, np.newaxis]) / scale[:, np.newaxis]
        quantized = np.clip(np.round(normalized), 0, self._NUM_LEVELS - 1).astype(
            np.uint8
        )

        # Pack 4 × 2-bit values into 1 byte (MSB first: val[0] in bits 7-6)
        packed_blocks = np.zeros((n_blocks, self._packed_bytes), dtype=np.uint8)
        for b_idx in range(bs):
            byte_idx = b_idx // 4
            bit_offset = 6 - 2 * (b_idx % 4)  # 6, 4, 2, 0
            packed_blocks[:, byte_idx] |= (quantized[:, b_idx] << bit_offset).astype(
                np.uint8
            )

        # Build output: header (4 bytes) + packed data for each block
        header = np.empty((n_blocks, self._header_bytes), dtype=np.uint8)
        for blk_i in range(n_blocks):
            struct.pack_into(
                "<e", header[blk_i], 0, float(scale[blk_i])
            )
            struct.pack_into(
                "<e", header[blk_i], 2, float(zero_point[blk_i])
            )

        # Concatenate header + packed data per block, then flatten
        block_data = np.concatenate([header, packed_blocks], axis=1)
        return block_data.ravel()

    def _dequantize_tensor(
        self,
        packed: np.ndarray,
        orig_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Dequantise a packed uint8 tensor back to float16."""
        n_values = int(np.prod(orig_shape))
        bs = self._block_size
        n_blocks = math.ceil(n_values / bs)

        block_data = packed.reshape(n_blocks, self._block_bytes)

        # Extract header
        header = block_data[:, : self._header_bytes]
        packed_vals = block_data[:, self._header_bytes :]

        scale = np.empty(n_blocks, dtype=np.float32)
        zero_point = np.empty(n_blocks, dtype=np.float32)
        for blk_i in range(n_blocks):
            scale[blk_i] = struct.unpack_from("<e", header[blk_i], 0)[0]
            zero_point[blk_i] = struct.unpack_from("<e", header[blk_i], 2)[0]

        # Unpack 2-bit values from bytes
        quantized = np.zeros((n_blocks, bs), dtype=np.uint8)
        for b_idx in range(bs):
            byte_idx = b_idx // 4
            bit_offset = 6 - 2 * (b_idx % 4)
            quantized[:, b_idx] = (
                (packed_vals[:, byte_idx] >> bit_offset) & 0x03
            ).astype(np.uint8)

        # Dequantise via LUT
        dequant = (
            self._dequant_lut[quantized] * scale[:, np.newaxis]
            + zero_point[:, np.newaxis]
        )

        flat = dequant.ravel()[:n_values]
        return flat.astype(np.float16).reshape(orig_shape)


# ═══════════════════════════════════════════════════════════════════════════════
# CPUKernelOptimizer
# ═══════════════════════════════════════════════════════════════════════════════


class CPUKernelOptimizer:
    """Detects CPU capabilities and selects the optimal matmul backend.

    Checks for:
        * AVX-512 / AVX2 (x86_64)
        * AMX (Intel Sapphire Rapids+)
        * NEON (aarch64)
        * BLAS library (OpenBLAS, MKL, BLIS, Apple Accelerate)

    Provides:
        * ``optimal_block_size`` for matmul tiling.
        * Whether to use ``float16`` or ``float32`` for CPU compute.
        * Recommended thread count based on cache topology.
    """

    @dataclass
    class CPUInfo:
        """Detected CPU capabilities and optimal settings."""

        arch: str  # "x86_64", "aarch64", or "unknown"
        has_avx512: bool
        has_avx2: bool
        has_amx: bool
        has_neon: bool
        blas_library: str  # "openblas", "mkl", "blis", "accelerate", "none"
        l2_cache_kb: int
        l3_cache_kb: int
        num_cores: int
        optimal_block_size: int
        use_float16: bool
        recommended_threads: int

    def __init__(self) -> None:
        self._cached_info: CPUKernelOptimizer.CPUInfo | None = None

    def detect(self) -> CPUInfo:
        """Detect CPU capabilities via ``/proc/cpuinfo`` and NumPy config.

        Results are cached after the first call.

        Returns:
            A :class:`CPUInfo` dataclass with all detected fields populated.
        """
        if self._cached_info is not None:
            return self._cached_info

        arch = self._detect_arch()
        flags = self._read_cpu_flags()
        caches = self._read_cache_sizes()
        num_cores = self._read_num_cores()
        blas = self._detect_blas()

        has_avx512 = "avx512f" in flags
        has_avx2 = "avx2" in flags
        has_amx = "amx_tile" in flags
        has_neon = arch == "aarch64"

        l2_cache_kb = caches.get("l2", 256)
        l3_cache_kb = caches.get("l3", 0)

        # Optimal block size based on L2 cache and vector width
        # Each block of (block_size × block_size) float32 = block_size^2 × 4 bytes
        # Should fit comfortably in L2 (use ~50 % of L2 for two blocks: A-col + B-row)
        if has_avx512:
            # AVX-512 registers: 512-bit = 64 bytes; good for 16 × float32
            block_size = 128
        elif has_avx2:
            # AVX2: 256-bit = 32 bytes; good for 8 × float32
            block_size = 64
        elif has_neon:
            block_size = 64
        else:
            block_size = 32

        # Clamp block size so 2 blocks fit in half L2
        max_block_for_cache = int(
            math.sqrt((l2_cache_kb * 1024 * 0.5) / (2 * 4))
        )
        block_size = min(block_size, max(32, max_block_for_cache))
        block_size = max(32, block_size)  # floor

        # float16 is beneficial on wide SIMD (AVX-512 has fp16, AVX2 does not
        # natively but numpy can still use it for memory bandwidth reduction).
        use_float16 = has_avx512 or has_neon

        # Thread count: use physical cores, cap at L2-associativity heuristic
        # Conservative: number of physical cores, capped at 8 for cache thrashing
        recommended_threads = min(num_cores, 8)

        info = self.CPUInfo(
            arch=arch,
            has_avx512=has_avx512,
            has_avx2=has_avx2,
            has_amx=has_amx,
            has_neon=has_neon,
            blas_library=blas,
            l2_cache_kb=l2_cache_kb,
            l3_cache_kb=l3_cache_kb,
            num_cores=num_cores,
            optimal_block_size=block_size,
            use_float16=use_float16,
            recommended_threads=recommended_threads,
        )
        self._cached_info = info
        return info

    def optimized_matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        info: CPUInfo | None = None,
    ) -> np.ndarray:
        """Run matrix multiplication with optimal dtype and tiling.

        For large matrices the computation is tiled according to the detected
        ``optimal_block_size`` to maximise L2 cache reuse.  For small matrices
        NumPy's native (BLAS-backed) ``@`` operator is used directly.

        Args:
            a: Left operand, 2-D ``(M, K)``.
            b: Right operand, 2-D ``(K, N)``.
            info: Pre-detected :class:`CPUInfo`.  If ``None``, runs
                :meth:`detect` automatically.

        Returns:
            Product matrix ``C`` of shape ``(M, N)`` as ``float32``.
        """
        if info is None:
            info = self.detect()

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        m, k_dim = a.shape
        k_dim2, n_dim = b.shape
        if k_dim != k_dim2:
            raise ValueError(
                f"Incompatible shapes for matmul: a {a.shape} @ b {b.shape}"
            )

        bs = info.optimal_block_size

        # For matrices smaller than 2× block size, just use numpy directly
        if m <= bs and n_dim <= bs:
            return a @ b

        # Tiled matmul
        c = np.zeros((m, n_dim), dtype=np.float32)

        for row_start in range(0, m, bs):
            row_end = min(row_start + bs, m)
            for col_start in range(0, n_dim, bs):
                col_end = min(col_start + bs, n_dim)
                # Accumulate over K dimension in tiles too
                acc = np.zeros((row_end - row_start, col_end - col_start), dtype=np.float32)
                for k_start in range(0, k_dim, bs):
                    k_end = min(k_start + bs, k_dim)
                    a_block = a[row_start:row_end, k_start:k_end]
                    b_block = b[k_start:k_end, col_start:col_end]
                    acc += a_block @ b_block
                c[row_start:row_end, col_start:col_end] = acc

        return c

    # ── detection helpers ───────────────────────────────────────────────

    @staticmethod
    def _detect_arch() -> str:
        """Detect CPU architecture."""
        try:
            import platform

            machine = platform.machine().lower()
            if machine in ("x86_64", "amd64", "i686", "i386"):
                return "x86_64"
            if machine in ("aarch64", "arm64", "armv8l"):
                return "aarch64"
            return machine or "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _read_cpu_flags() -> set[str]:
        """Read CPU feature flags from /proc/cpuinfo."""
        flags: set[str] = set()
        cpuinfo_path = "/proc/cpuinfo"
        if not os.path.isfile(cpuinfo_path):
            return flags
        try:
            with open(cpuinfo_path, "r") as fh:
                for raw_line in fh:
                    line = raw_line.strip().lower()
                    if line.startswith("flags"):
                        # Format: "flags\t: flag1 flag2 flag3 ..."
                        flag_str = line.split(":", 1)[-1].strip()
                        flags.update(flag_str.split())
                    elif line.startswith("features"):
                        flag_str = line.split(":", 1)[-1].strip()
                        flags.update(flag_str.split())
                    # Only need first CPU core's flags
                    if flags:
                        break
        except OSError:
            pass
        return flags

    @staticmethod
    def _read_cache_sizes() -> dict[str, int]:
        """Read L2/L3 cache sizes from /proc/cpuinfo (in KB)."""
        caches: dict[str, int] = {"l2": 256, "l3": 0}
        cpuinfo_path = "/proc/cpuinfo"
        if not os.path.isfile(cpuinfo_path):
            return caches
        try:
            with open(cpuinfo_path, "r") as fh:
                for raw_line in fh:
                    line = raw_line.strip().lower()
                    if "l2 cache" in line or "cache size" in line:
                        # e.g. "cache size\t: 256 KB"
                        val_str = line.split(":")[-1].strip()
                        caches["l2"] = CPUKernelOptimizer._parse_cache_kb(val_str)
                    elif "l3 cache" in line:
                        val_str = line.split(":")[-1].strip()
                        caches["l3"] = CPUKernelOptimizer._parse_cache_kb(val_str)
        except OSError:
            pass
        return caches

    @staticmethod
    def _parse_cache_kb(s: str) -> int:
        """Parse a cache size string like '256 KB' or '8192 KB' to int KB."""
        s = s.strip().upper()
        value = 0
        multiplier = 1
        for ch in s:
            if ch.isdigit():
                value = value * 10 + int(ch)
            elif ch == "M":
                multiplier = 1024
            elif ch == "G":
                multiplier = 1024 * 1024
            elif ch == "K":
                multiplier = 1
        return value * multiplier

    @staticmethod
    def _read_num_cores() -> int:
        """Detect the number of physical CPU cores."""
        # Try os.sched_getaffinity first (available on Linux)
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass

        # Fallback: /proc/cpuinfo processor count (may be logical cores)
        cpuinfo_path = "/proc/cpuinfo"
        if os.path.isfile(cpuinfo_path):
            try:
                count = 0
                with open(cpuinfo_path, "r") as fh:
                    for raw_line in fh:
                        if raw_line.strip().lower().startswith("processor"):
                            count += 1
                if count > 0:
                    return count
            except OSError:
                pass

        # Final fallback
        try:
            import multiprocessing

            return multiprocessing.cpu_count() or 1
        except Exception:
            return 1

    @staticmethod
    def _detect_blas() -> str:
        """Detect which BLAS library NumPy is linked against."""
        try:
            config = np.show_config()
            # np.show_config prints to stdout and returns None in older numpy,
            # but newer versions may return a string.
            config_str = str(config).lower()
        except Exception:
            config_str = ""

        # Check numpy.__config__ for blas info
        try:
            blas_info = getattr(np, "__config__", None)
            if blas_info is None:
                try:
                    from numpy import __config__ as _nconf

                    blas_info = _nconf
                except ImportError:
                    pass
            if blas_info is not None:
                info_str = str(blas_info).lower()
                config_str += " " + info_str
        except Exception:
            pass

        if "mkl" in config_str:
            return "mkl"
        if "openblas" in config_str:
            return "openblas"
        if "blis" in config_str:
            return "blis"
        if "accelerate" in config_str:
            return "accelerate"
        return "none"
