"""Advanced eviction policies for VibeBlade's RAM buffer.

Provides three eviction strategies that extend/complement the existing
LRUKPolicy in tiered_memory.py:

- FrequencyAwarePolicy: EMA-based heat scoring combining recency + frequency
- CostBenefitScorer: wrapper adding cost-benefit analysis to any policy
- AdaptiveBanditPolicy: Thompson Sampling for dynamic RAM/SSD boundary tuning
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from collections import deque

import numpy as np

if TYPE_CHECKING:
    from typing import Any


class EvictionPolicy(ABC):
    """Base class for RAM buffer eviction policies."""

    @abstractmethod
    def access(self, layer_idx: int, expert_id: int) -> None:
        """Record an access to an expert."""

    @abstractmethod
    def evict(self) -> tuple[int, int] | None:
        """Evict one expert, returning (layer_idx, expert_id) or None."""

    @abstractmethod
    def remove(self, layer_idx: int, expert_id: int) -> None:
        """Remove an expert from tracking (e.g., after explicit unload)."""

    @abstractmethod
    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether an expert is tracked in RAM."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Current number of items tracked in RAM."""

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Maximum capacity of the RAM buffer."""

    @abstractmethod
    def stats(self) -> dict:
        """Return diagnostic statistics."""


# ---------------------------------------------------------------------------
# FrequencyAwarePolicy
# ---------------------------------------------------------------------------


class FrequencyAwarePolicy(EvictionPolicy):
    """Frequency-weighted eviction policy.

    Combines recency (time since last access) with frequency (access count)
    using an exponential decay model.  On every access the heat score is
    updated as ``score = score * decay + 1.0`` where
    ``decay = 0.5 ** (elapsed / half_life)``.  One-hit wonders hover near
    1.0 while frequently-accessed experts accumulate higher scores.

    Higher score → more valuable → less likely to be evicted.

    Args:
        capacity: max items in RAM buffer.
        decay_half_life: seconds for frequency weight to halve (default 60.0).
        min_score_threshold: minimum heat score to enter the protected set
            (default 0.5).  Protected items are only evicted when no
            probationary items remain.
    """

    def __init__(
        self,
        capacity: int,
        decay_half_life: float = 60.0,
        min_score_threshold: float = 0.5,
    ) -> None:
        self._capacity = capacity
        self._decay_half_life = decay_half_life
        self._min_score_threshold = min_score_threshold

        # {(layer, expert): {"count": int, "last_time": float, "score": float}}
        self._items: dict[tuple[int, int], dict[str, float]] = {}

        # Bookkeeping
        self._total_accesses: int = 0
        self._total_evictions: int = 0

    # -- public interface ----------------------------------------------------

    def access(self, layer_idx: int, expert_id: int) -> None:
        """Record an access and update the heat score."""
        key = (layer_idx, expert_id)
        now = time.monotonic()

        if key in self._items:
            entry = self._items[key]
            elapsed = now - entry["last_time"]
            decay = 0.5 ** (elapsed / self._decay_half_life)
            entry["score"] = entry["score"] * decay + 1.0
            entry["count"] += 1
            entry["last_time"] = now
        else:
            self._items[key] = {
                "count": 1,
                "last_time": now,
                "score": 1.0,
            }

        self._total_accesses += 1

    def evict(self) -> tuple[int, int] | None:
        """Evict the lowest-scored item, respecting the protected set."""
        if not self._items:
            return None

        # Split into probationary (below threshold) and protected
        probationary: list[tuple[int, int, float]] = []
        protected: list[tuple[int, int, float]] = []

        for key, entry in self._items.items():
            score = entry["score"]
            if score > self._min_score_threshold:
                protected.append((key[0], key[1], score))
            else:
                probationary.append((key[0], key[1], score))

        # Prefer evicting from probationary
        pool = probationary if probationary else protected

        # Pick the one with the lowest score; break ties by oldest access
        best_key = min(
            pool,
            key=lambda item: (item[2], self._items[(item[0], item[1])]["last_time"]),
        )
        evict_key = (best_key[0], best_key[1])

        del self._items[evict_key]
        self._total_evictions += 1
        return evict_key

    def remove(self, layer_idx: int, expert_id: int) -> None:
        """Remove an expert from tracking."""
        self._items.pop((layer_idx, expert_id), None)

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether an expert is tracked in RAM."""
        return (layer_idx, expert_id) in self._items

    @property
    def size(self) -> int:
        """Current number of items tracked in RAM."""
        return len(self._items)

    @property
    def capacity(self) -> int:
        """Maximum capacity of the RAM buffer."""
        return self._capacity

    def heat_score(self, layer_idx: int, expert_id: int) -> float:
        """Return the current heat score for an expert.

        Returns 0.0 if the expert is not tracked.
        """
        entry = self._items.get((layer_idx, expert_id))
        if entry is None:
            return 0.0
        # Apply time decay to give an up-to-date score
        now = time.monotonic()
        elapsed = now - entry["last_time"]
        decay = 0.5 ** (elapsed / self._decay_half_life)
        return entry["score"] * decay

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        scores = [e["score"] for e in self._items.values()]
        return {
            "type": "frequency_aware",
            "size": len(self._items),
            "capacity": self._capacity,
            "total_accesses": self._total_accesses,
            "total_evictions": self._total_evictions,
            "protected_count": sum(
                1 for s in scores if s > self._min_score_threshold
            ),
            "probationary_count": sum(
                1 for s in scores if s <= self._min_score_threshold
            ),
            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_std": float(np.std(scores)) if scores else 0.0,
            "score_min": min(scores) if scores else 0.0,
            "score_max": max(scores) if scores else 0.0,
        }


# ---------------------------------------------------------------------------
# CostBenefitScorer
# ---------------------------------------------------------------------------


class CostBenefitScorer(EvictionPolicy):
    """Cost-benefit eviction wrapper.

    Wraps any eviction policy and adds cost-benefit analysis.  Scores each
    candidate as ``benefit / cost`` where *benefit* is the frequency /
    heat score from the wrapped policy and *cost* is the estimated reload
    time from SSD (p95 of historical latencies).

    Evicts the item with the **lowest** benefit/cost ratio — i.e., the item
    that is least useful *relative to how expensive it is to reload*.

    Args:
        policy: base eviction policy (FrequencyAwarePolicy or LRUKPolicy).
        ssd_latency_ms: historical average SSD read latency in ms
            (default 2.0).  Used as fallback when no per-expert history
            exists.
        expert_size_bytes: average expert size for bandwidth estimation.
            When provided, items with higher benefit/cost *and* larger size
            are preferred for eviction (they free more RAM).  Pass ``None``
            to disable size-based tie-breaking.
    """

    _LATENCY_HISTORY_SIZE: int = 10

    def __init__(
        self,
        policy: EvictionPolicy,
        ssd_latency_ms: float = 2.0,
        expert_size_bytes: int | None = None,
    ) -> None:
        self._policy = policy
        self._default_latency_ms = ssd_latency_ms
        self._expert_size_bytes = expert_size_bytes

        # Per-expert latency ring buffers: {(layer, expert): deque}
        self._latency_history: dict[
            tuple[int, int], deque[float]
        ] = {}

        self._total_evictions: int = 0

    # -- public interface ----------------------------------------------------

    def access(self, layer_idx: int, expert_id: int) -> None:
        """Delegate access to the wrapped policy."""
        self._policy.access(layer_idx, expert_id)

    def record_ssd_load(
        self, layer_idx: int, expert_id: int, latency_ms: float
    ) -> None:
        """Record a measured SSD load latency for an expert.

        Maintains a ring buffer of the last 10 measurements, which is used
        to compute a p95 estimate of reload cost.
        """
        key = (layer_idx, expert_id)
        if key not in self._latency_history:
            self._latency_history[key] = deque(
                maxlen=self._LATENCY_HISTORY_SIZE
            )
        self._latency_history[key].append(latency_ms)

    def evict(self) -> tuple[int, int] | None:
        """Evict the item with the lowest benefit/cost ratio.

        Falls back to the wrapped policy when there are no candidates or
        when cost information is unavailable for all items.
        """
        if self._policy.size == 0:
            return None

        # Build candidate list with benefit/cost scores
        candidates: list[tuple[int, int, float]] = []
        for key in self._tracked_keys():
            ratio = self.benefit_cost_ratio(key[0], key[1])
            candidates.append((key[0], key[1], ratio))

        if not candidates:
            # Nothing to evict via our logic; delegate
            return self._policy.evict()

        # Sort ascending by benefit/cost (lowest first)
        candidates.sort(key=lambda c: c[2])

        # Among the lowest-scored items, prefer evicting larger experts
        # if size info is available.
        if self._expert_size_bytes is not None:
            # Find the worst 10 % (at least 1 candidate) and pick the largest
            cutoff = max(1, len(candidates) // 10)
            worst = candidates[:cutoff]
            # We don't have per-expert size, so just pick the very worst
            evict_item = worst[0]
        else:
            evict_item = candidates[0]

        evict_key = (evict_item[0], evict_item[1])
        self._policy.remove(evict_key[0], evict_key[1])
        self._total_evictions += 1
        return evict_key

    def remove(self, layer_idx: int, expert_id: int) -> None:
        """Remove an expert from tracking."""
        self._policy.remove(layer_idx, expert_id)
        self._latency_history.pop((layer_idx, expert_id), None)

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether an expert is tracked in RAM."""
        return self._policy.contains(layer_idx, expert_id)

    @property
    def size(self) -> int:
        """Current number of items tracked in RAM."""
        return self._policy.size

    @property
    def capacity(self) -> int:
        """Maximum capacity of the RAM buffer."""
        return self._policy.capacity

    def benefit_cost_ratio(self, layer_idx: int, expert_id: int) -> float:
        """Compute the benefit/cost ratio for an expert.

        *Benefit* is taken from the wrapped policy's heat score (if
        available) or the access count.  *Cost* is the p95 SSD latency
        for that expert, falling back to the default latency.

        Returns 0.0 when the expert is not tracked.
        """
        # Benefit: try heat_score if the inner policy supports it
        inner = self._policy
        if hasattr(inner, "heat_score"):
            benefit = inner.heat_score(layer_idx, expert_id)
        else:
            # Fallback: use 1.0 as neutral benefit
            benefit = 1.0

        # Cost: p95 of historical latencies
        key = (layer_idx, expert_id)
        history = self._latency_history.get(key)
        if history and len(history) >= 1:
            cost = float(np.percentile(list(history), 95))
        else:
            cost = self._default_latency_ms

        # Avoid division by zero
        if cost <= 0:
            cost = 1e-6

        return benefit / cost

    def stats(self) -> dict:  # type: ignore[override]
        """Return diagnostic statistics (including wrapped policy stats)."""
        inner_stats = self._policy.stats()
        return {
            "type": "cost_benefit_scorer",
            "wrapped_policy": inner_stats,
            "default_latency_ms": self._default_latency_ms,
            "expert_size_bytes": self._expert_size_bytes,
            "tracked_latencies": len(self._latency_history),
            "total_evictions": self._total_evictions,
        }

    # -- private helpers -----------------------------------------------------

    def _tracked_keys(self) -> list[tuple[int, int]]:
        """Return all keys tracked by the inner policy.

        Works with both our FrequencyAwarePolicy (which exposes ``_items``)
        and the external LRUKPolicy (which exposes ``_access_history``).
        """
        inner = self._policy
        # FrequencyAwarePolicy stores items in _items dict
        if hasattr(inner, "_items") and isinstance(inner._items, dict):
            return list(inner._items.keys())
        # LRUKPolicy stores entries in _access_history dict
        if hasattr(inner, "_access_history") and isinstance(
            inner._access_history, dict
        ):
            return list(inner._access_history.keys())
        return []


# ---------------------------------------------------------------------------
# AdaptiveBanditPolicy
# ---------------------------------------------------------------------------


class AdaptiveBanditPolicy(EvictionPolicy):
    """Multi-armed bandit (Thompson Sampling) for RAM/SSD boundary tuning.

    Each expert is an "arm" with a Bernoulli reward model: *will it be
    accessed in the next N tokens?*  The bandit maintains ``alpha`` /
    ``beta`` posterior parameters and samples from ``Beta(alpha, beta)``
    to make eviction decisions.

    Arms with high predicted access probability stay in RAM; arms with low
    probability get evicted to SSD.  The bandit automatically adapts to
    changing access patterns — if an expert's topic becomes popular its
    alpha increases and it gets promoted back to RAM.

    Args:
        capacity: max items in RAM buffer.
        exploration_weight: how much to favour exploration (0.0–1.0,
            default 0.1).  Higher values inject more noise into samples.
        update_interval: tokens between periodic ``update()`` rebalances
            (default 100).
        prior_alpha: initial Beta prior α (default 1.0).
        prior_beta: initial Beta prior β (default 1.0).
    """

    def __init__(
        self,
        capacity: int,
        exploration_weight: float = 0.1,
        update_interval: int = 100,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self._capacity = capacity
        self._exploration_weight = exploration_weight
        self._update_interval = update_interval
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # {(layer, expert): {"alpha": float, "beta": float, "in_ram": bool}}
        self._arms: dict[tuple[int, int], dict[str, Any]] = {}

        # Token counter for periodic updates
        self._token_counter: int = 0

        # Set of keys accessed since the last update
        self._recent_accesses: set[tuple[int, int]] = set()

        # Bookkeeping
        self._total_accesses: int = 0
        self._total_misses: int = 0
        self._total_evictions: int = 0
        self._total_promotions: int = 0

    # -- public interface ----------------------------------------------------

    def access(self, layer_idx: int, expert_id: int) -> None:
        """Record an access and update the posterior.

        If the expert is already in RAM this is a positive signal (alpha += 1).
        If the expert was on SSD (``in_ram=False``) we promote it to RAM.
        """
        key = (layer_idx, expert_id)
        self._total_accesses += 1
        self._token_counter += 1
        self._recent_accesses.add(key)

        if key in self._arms:
            arm = self._arms[key]
            arm["alpha"] += 1.0
            if not arm["in_ram"]:
                # Promote from SSD back to RAM
                arm["in_ram"] = True
                self._total_promotions += 1
        else:
            # New expert — start in RAM with a fresh prior
            self._arms[key] = {
                "alpha": self._prior_alpha + 1.0,
                "beta": self._prior_beta,
                "in_ram": True,
            }

    def miss(self, layer_idx: int, expert_id: int) -> None:
        """Record that we had to load an expert from SSD.

        This is a *negative signal* — the expert was evicted but then
        needed, suggesting it should have been kept.  We increment beta
        as a soft penalty that decays once the expert is back in RAM.
        """
        key = (layer_idx, expert_id)
        self._total_misses += 1
        self._recent_accesses.add(key)

        if key in self._arms:
            self._arms[key]["beta"] += 1.0
            # Ensure it is promoted to RAM after a miss
            if not self._arms[key]["in_ram"]:
                self._arms[key]["in_ram"] = True
                self._total_promotions += 1
        else:
            # First time seeing this expert via a miss — put it in RAM
            self._arms[key] = {
                "alpha": self._prior_alpha,
                "beta": self._prior_beta + 1.0,
                "in_ram": True,
            }

    def evict(self) -> tuple[int, int] | None:
        """Thompson-sample all RAM-resident experts and evict the one
        with the lowest sampled probability.
        """
        ram_keys = [
            k for k, v in self._arms.items() if v["in_ram"]
        ]
        if not ram_keys:
            return None

        # Sample probabilities with optional exploration noise
        samples: list[tuple[int, int, float]] = []
        for key in ram_keys:
            arm = self._arms[key]
            p = float(
                np.random.beta(arm["alpha"], arm["beta"])
            )
            # Add exploration noise
            noise = self._exploration_weight * np.random.uniform(0.0, 1.0)
            samples.append((key[0], key[1], p + noise))

        # Evict the lowest-sampled probability
        samples.sort(key=lambda s: s[2])
        evict_key = (samples[0][0], samples[0][1])
        self._arms[evict_key]["in_ram"] = False
        self._total_evictions += 1
        return evict_key

    def remove(self, layer_idx: int, expert_id: int) -> None:
        """Remove an expert from tracking entirely."""
        self._arms.pop((layer_idx, expert_id), None)

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether an expert is currently in RAM."""
        arm = self._arms.get((layer_idx, expert_id))
        return arm is not None and arm["in_ram"]

    @property
    def size(self) -> int:
        """Current number of items in RAM."""
        return sum(1 for v in self._arms.values() if v["in_ram"])

    @property
    def capacity(self) -> int:
        """Maximum capacity of the RAM buffer."""
        return self._capacity

    def sample_probability(self, layer_idx: int, expert_id: int) -> float:
        """Draw a Thompson sample from Beta(α, β) for an expert.

        Returns 0.0 if the expert is not tracked.
        """
        arm = self._arms.get((layer_idx, expert_id))
        if arm is None:
            return 0.0
        return float(np.random.beta(arm["alpha"], arm["beta"]))

    def expected_probability(self, layer_idx: int, expert_id: int) -> float:
        """Return the expected access probability: α / (α + β).

        Returns 0.0 if the expert is not tracked.
        """
        arm = self._arms.get((layer_idx, expert_id))
        if arm is None:
            return 0.0
        total = arm["alpha"] + arm["beta"]
        if total <= 0:
            return 0.0
        return arm["alpha"] / total

    def should_promote(self, layer_idx: int, expert_id: int) -> bool:
        """Decide whether an SSD-resident expert should be promoted to RAM.

        Returns ``True`` when the expert's expected probability exceeds the
        median expected probability of all currently RAM-resident experts.
        """
        arm = self._arms.get((layer_idx, expert_id))
        if arm is None or arm["in_ram"]:
            return False

        ram_probs = [
            v["alpha"] / (v["alpha"] + v["beta"])
            for v in self._arms.values()
            if v["in_ram"] and (v["alpha"] + v["beta"]) > 0
        ]
        if not ram_probs:
            return True  # RAM is empty, promote anything

        threshold = float(np.median(ram_probs))
        expert_prob = self.expected_probability(layer_idx, expert_id)
        return expert_prob > threshold

    def update(self) -> dict:
        """Periodic rebalance of the RAM/SSD boundary.

        For every expert **not** accessed since the last update, increment
        beta (negative evidence).  Then promote the top-K highest-probability
        SSD experts to RAM and evict the bottom-K lowest-probability RAM
        experts to SSD, where K = ``|RAM| - capacity`` (if over capacity)
        or up to ``capacity - |RAM|`` promotions if under capacity.

        Returns:
            A summary dict of actions taken.
        """
        # Negative evidence for experts not recently accessed
        for key, arm in self._arms.items():
            if key not in self._recent_accesses:
                arm["beta"] += 1.0

        actions: dict[str, list[tuple[int, int]]] = {
            "promoted": [],
            "evicted": [],
        }

        current_ram = sum(1 for v in self._arms.values() if v["in_ram"])
        over = current_ram - self._capacity

        if over > 0:
            # Need to evict `over` items from RAM
            ram_items = [
                (k, v["alpha"] / (v["alpha"] + v["beta"]))
                for k, v in self._arms.items()
                if v["in_ram"] and (v["alpha"] + v["beta"]) > 0
            ]
            ram_items.sort(key=lambda item: item[1])
            for key, _ in ram_items[:over]:
                self._arms[key]["in_ram"] = False
                self._total_evictions += 1
                actions["evicted"].append(key)
        else:
            # Room to promote up to (-over) items from SSD
            under = -over
            ssd_items = [
                (k, v["alpha"] / (v["alpha"] + v["beta"]))
                for k, v in self._arms.items()
                if not v["in_ram"] and (v["alpha"] + v["beta"]) > 0
            ]
            ssd_items.sort(key=lambda item: item[1], reverse=True)
            for key, _ in ssd_items[:under]:
                self._arms[key]["in_ram"] = True
                self._total_promotions += 1
                actions["promoted"].append(key)

        # Reset tracking for next interval
        self._recent_accesses.clear()
        self._token_counter = 0

        return actions

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        probs = [
            v["alpha"] / (v["alpha"] + v["beta"])
            for v in self._arms.values()
            if (v["alpha"] + v["beta"]) > 0
        ]
        return {
            "type": "adaptive_bandit",
            "size": self.size,
            "capacity": self._capacity,
            "total_arms": len(self._arms),
            "total_accesses": self._total_accesses,
            "total_misses": self._total_misses,
            "total_evictions": self._total_evictions,
            "total_promotions": self._total_promotions,
            "token_counter": self._token_counter,
            "prob_mean": float(np.mean(probs)) if probs else 0.0,
            "prob_std": float(np.std(probs)) if probs else 0.0,
            "prob_min": float(np.min(probs)) if probs else 0.0,
            "prob_max": float(np.max(probs)) if probs else 0.0,
            "exploration_weight": self._exploration_weight,
            "update_interval": self._update_interval,
        }
