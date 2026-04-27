"""Phase-specialized MoE scheduling (DuoServe-style).

Manages different memory/expert placement strategies for prefill vs decode
phases of MoE inference. Prefill benefits from spreading experts across RAM
for batch parallelism, while decode aggressively promotes hot experts to VRAM
for low-latency token generation.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any


class InferencePhase(Enum):
    """Current phase of MoE inference."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class PhaseConfig:
    """Memory and compute configuration for a specific inference phase.

    Controls expert placement budgets, routing parameters, prefetch depth,
    and quantization settings tailored to the characteristics of each phase.
    """

    # Expert budget
    max_hot_experts: int  # how many experts to keep in VRAM
    max_warm_experts: int  # how many to keep in RAM (rest → SSD)

    # Routing behavior
    top_k: int  # how many experts to activate per token
    confidence_threshold: float  # early exit threshold (0.0 = always, 1.0 = never)
    min_topk: int  # minimum experts even with early exit

    # Prefetch behavior
    enable_prefetch: bool
    prefetch_depth: int  # how many layers to look ahead

    # Quantization
    hot_quant_bits: int  # 4 or 5
    cold_quant_bits: int  # 2 or 3

    # Concurrency
    cpu_threads: int

    @classmethod
    def prefill_defaults(cls) -> PhaseConfig:
        """Sensible defaults for the prefill phase."""
        return cls(
            max_hot_experts=2,
            max_warm_experts=8,
            top_k=8,
            confidence_threshold=1.0,  # no early exit during prefill
            min_topk=1,
            enable_prefetch=True,
            prefetch_depth=2,
            hot_quant_bits=4,
            cold_quant_bits=3,
            cpu_threads=8,
        )

    @classmethod
    def decode_defaults(cls) -> PhaseConfig:
        """Sensible defaults for the decode phase."""
        return cls(
            max_hot_experts=4,
            max_warm_experts=4,
            top_k=4,
            confidence_threshold=0.9,
            min_topk=1,
            enable_prefetch=True,
            prefetch_depth=4,
            hot_quant_bits=5,
            cold_quant_bits=2,
            cpu_threads=4,
        )


class PhaseScheduler:
    """Manages phase transitions and applies phase-specific configurations.

    Automatically transitions between PREFILL and DECODE phases and adjusts
    memory budgets, routing parameters, and prefetch strategies accordingly.

    DuoServe-style scheduling keeps a small set of hot experts in VRAM during
    decode (for low latency) while spreading experts across warm RAM slots
    during prefill (for batch throughput).
    """

    def __init__(
        self,
        prefill_config: PhaseConfig | None = None,
        decode_config: PhaseConfig | None = None,
        num_experts: int = 16,
        num_layers: int = 80,
    ) -> None:
        self._prefill_config = prefill_config or PhaseConfig.prefill_defaults()
        self._decode_config = decode_config or PhaseConfig.decode_defaults()
        self._num_experts = num_experts
        self._num_layers = num_layers

        self._phase: InferencePhase = InferencePhase.PREFILL
        self._token_count: int = 0

        # Statistics
        self._phase_enter_times: dict[InferencePhase, float] = {}
        self._transition_count: int = 0
        self._total_prefill_tokens: int = 0
        self._total_decode_tokens: int = 0
        self._prefill_time: float = 0.0
        self._decode_time: float = 0.0

        # Per-layer activation frequency tracking for decode expert promotion
        self._activation_freq: dict[int, Counter] = {
            idx: Counter() for idx in range(num_layers)
        }

        self._enter_phase(InferencePhase.PREFILL)

    @property
    def current_phase(self) -> InferencePhase:
        """Current inference phase."""
        return self._phase

    def begin_prefill(self, num_prompt_tokens: int) -> PhaseConfig:
        """Signal start of prefill phase. Returns active config.

        Args:
            num_prompt_tokens: Number of tokens in the input prompt.

        Returns:
            The PhaseConfig to use during prefill.
        """
        if self._phase != InferencePhase.PREFILL:
            self._record_phase_duration()
            self._transition_count += 1
            self._enter_phase(InferencePhase.PREFILL)
        self._token_count += num_prompt_tokens
        return self._prefill_config

    def begin_decode(self) -> PhaseConfig:
        """Signal transition to decode phase. Returns active config.

        Returns:
            The PhaseConfig to use during decode.
        """
        if self._phase != InferencePhase.DECODE:
            self._record_phase_duration()
            self._transition_count += 1
            self._enter_phase(InferencePhase.DECODE)
        return self._decode_config

    def update_expert_budget(
        self,
        hot_cold_map: dict[int, dict[str, set[int]]],
        phase: InferencePhase | None = None,
    ) -> dict[int, dict[str, set[int]]]:
        """Rebalance hot/warm/cold expert assignments for the current phase.

        During prefill: Spread experts across more warm slots (RAM) since
        prefill benefits from batch parallelism rather than cache locality.

        During decode: Aggressively promote frequently-used experts to hot
        (VRAM) for low-latency token generation. Uses activation frequency
        tracking to identify the most important experts.

        Args:
            hot_cold_map: ``{layer_idx: {"hot": {ids}, "warm": {ids}, "cold": {ids}}}``
            phase: Override current phase (useful for testing).

        Returns:
            Updated hot_cold_map with rebalanced assignments.
        """
        active_phase = phase or self._phase

        if active_phase == InferencePhase.DECODE:
            return self._rebalance_for_decode(hot_cold_map)
        else:
            return self._rebalance_for_prefill(hot_cold_map)

    def token_callback(self, is_prefill: bool) -> PhaseConfig:
        """Called for each processed token. Auto-detects phase transitions.

        Tracks cumulative statistics and switches phase when the token type
        changes (e.g. last prefill token → first decode token).

        Args:
            is_prefill: ``True`` if this token is part of the prefill phase.

        Returns:
            The config that should be active for this token.
        """
        if is_prefill:
            if self._phase != InferencePhase.PREFILL:
                self._record_phase_duration()
                self._transition_count += 1
                self._enter_phase(InferencePhase.PREFILL)
            self._total_prefill_tokens += 1
            return self._prefill_config
        else:
            if self._phase != InferencePhase.DECODE:
                self._record_phase_duration()
                self._transition_count += 1
                self._enter_phase(InferencePhase.DECODE)
            self._total_decode_tokens += 1
            return self._decode_config

    def record_expert_activation(
        self,
        layer_idx: int,
        expert_ids: set[int],
    ) -> None:
        """Record which experts were activated at a given layer during decode.

        This feeds into the frequency counters used by
        ``update_expert_budget`` to decide which experts to promote to hot.

        Args:
            layer_idx: Layer index (0-based).
            expert_ids: Set of expert indices activated for the current token.
        """
        if 0 <= layer_idx < self._num_layers:
            self._activation_freq[layer_idx].update(expert_ids)

    def phase_stats(self) -> dict[str, Any]:
        """Return statistics about phase transitions and timing.

        Returns:
            Dictionary with transition counts, token counts, timing, and
            current phase information.
        """
        # Snapshot current phase duration without ending it
        now = time.monotonic()
        current_phase_elapsed = now - self._phase_enter_times.get(self._phase, now)

        stats: dict[str, Any] = {
            "current_phase": self._phase.value,
            "transition_count": self._transition_count,
            "total_tokens": self._token_count + self._total_prefill_tokens + self._total_decode_tokens,
            "prefill_tokens": self._total_prefill_tokens,
            "decode_tokens": self._total_decode_tokens,
            "prefill_time_sec": round(self._prefill_time, 4),
            "decode_time_sec": round(self._decode_time, 4),
            "current_phase_elapsed_sec": round(current_phase_elapsed, 4),
        }
        return stats

    def reset(self) -> None:
        """Reset to initial state, clearing all statistics and tracking."""
        self._phase = InferencePhase.PREFILL
        self._token_count = 0
        self._transition_count = 0
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._prefill_time = 0.0
        self._decode_time = 0.0
        self._activation_freq = {
            idx: Counter() for idx in range(self._num_layers)
        }
        self._phase_enter_times.clear()
        self._enter_phase(InferencePhase.PREFILL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enter_phase(self, phase: InferencePhase) -> None:
        """Mark entry into a phase and record the timestamp."""
        self._phase = phase
        self._phase_enter_times[phase] = time.monotonic()

    def _record_phase_duration(self) -> None:
        """Accumulate elapsed time for the phase we are leaving."""
        now = time.monotonic()
        start = self._phase_enter_times.get(self._phase, now)
        elapsed = now - start
        if self._phase == InferencePhase.PREFILL:
            self._prefill_time += elapsed
        elif self._phase == InferencePhase.DECODE:
            self._decode_time += elapsed

    def _rebalance_for_decode(
        self,
        hot_cold_map: dict[int, dict[str, set[int]]],
    ) -> dict[int, dict[str, set[int]]]:
        """Aggressively promote top experts to hot for decode phase.

        Uses tracked activation frequencies to identify the most frequently
        used experts per layer and promotes them to VRAM up to
        ``max_hot_experts``.
        """
        max_hot = self._decode_config.max_hot_experts
        max_warm = self._decode_config.max_warm_experts
        result: dict[int, dict[str, set[int]]] = {}

        for layer_idx, buckets in hot_cold_map.items():
            all_experts = (
                buckets.get("hot", set())
                | buckets.get("warm", set())
                | buckets.get("cold", set())
            )

            if not all_experts:
                result[layer_idx] = {
                    "hot": set(),
                    "warm": set(),
                    "cold": set(),
                }
                continue

            freq = self._activation_freq.get(layer_idx, Counter())
            # Rank experts by activation frequency, most-used first
            ranked = sorted(all_experts, key=lambda e: freq.get(e, 0), reverse=True)

            n_hot = min(max_hot, len(ranked))
            n_warm = min(max_warm, len(ranked) - n_hot)

            hot_set = set(ranked[:n_hot])
            warm_set = set(ranked[n_hot : n_hot + n_warm])
            cold_set = set(ranked[n_hot + n_warm :])

            result[layer_idx] = {
                "hot": hot_set,
                "warm": warm_set,
                "cold": cold_set,
            }

        return result

    def _rebalance_for_prefill(
        self,
        hot_cold_map: dict[int, dict[str, set[int]]],
    ) -> dict[int, dict[str, set[int]]]:
        """Spread experts across warm slots for prefill phase.

        Prefill benefits from batch parallelism over cache locality, so we
        keep fewer experts in hot VRAM and spread the rest across warm RAM.
        """
        max_hot = self._prefill_config.max_hot_experts
        max_warm = self._prefill_config.max_warm_experts
        result: dict[int, dict[str, set[int]]] = {}

        for layer_idx, buckets in hot_cold_map.items():
            all_experts = (
                buckets.get("hot", set())
                | buckets.get("warm", set())
                | buckets.get("cold", set())
            )

            if not all_experts:
                result[layer_idx] = {
                    "hot": set(),
                    "warm": set(),
                    "cold": set(),
                }
                continue

            # During prefill we don't prioritise by frequency — just
            # keep a small hot set and spread the rest evenly.
            experts_sorted = sorted(all_experts)

            n_hot = min(max_hot, len(experts_sorted))
            n_warm = min(max_warm, len(experts_sorted) - n_hot)

            hot_set = set(experts_sorted[:n_hot])
            warm_set = set(experts_sorted[n_hot : n_hot + n_warm])
            cold_set = set(experts_sorted[n_hot + n_warm :])

            result[layer_idx] = {
                "hot": hot_set,
                "warm": warm_set,
                "cold": cold_set,
            }

        return result
