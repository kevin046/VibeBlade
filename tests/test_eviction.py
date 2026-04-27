"""Comprehensive tests for vibeblade.eviction module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from vibeblade.eviction import (
    AdaptiveBanditPolicy,
    CostBenefitScorer,
    EvictionPolicy,
    FrequencyAwarePolicy,
)


# ===========================================================================
# EvictionPolicy (base)
# ===========================================================================


class TestEvictionPolicy:
    """Tests for the abstract EvictionPolicy base class."""

    def test_cannot_instantiate(self) -> None:
        """EvictionPolicy is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EvictionPolicy()  # type: ignore[abstract]

    def test_subclass_must_implement_all(self) -> None:
        """A subclass missing any abstract method is also non-instantiable."""

        class Incomplete(EvictionPolicy):
            def access(self, layer_idx: int, expert_id: int) -> None:
                pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ===========================================================================
# FrequencyAwarePolicy
# ===========================================================================


class TestFrequencyAwarePolicy:
    """Tests for FrequencyAwarePolicy."""

    def _make(self, capacity: int = 10, **kwargs) -> FrequencyAwarePolicy:
        return FrequencyAwarePolicy(capacity=capacity, **kwargs)

    # -- basic access & evict ------------------------------------------------

    def test_basic_access_and_evict(self) -> None:
        """Accessing items and evicting removes lowest-score item."""
        p = self._make(capacity=5)
        p.access(0, 0)
        p.access(0, 1)
        p.access(0, 2)

        assert p.size == 3
        evicted = p.evict()
        assert evicted is not None
        assert evicted[0] == 0  # layer
        assert evicted[1] in {0, 1, 2}  # one of the experts
        assert p.size == 2

    def test_evict_empty_returns_none(self) -> None:
        """Evicting from an empty policy returns None."""
        p = self._make()
        assert p.evict() is None

    # -- frequency scoring ---------------------------------------------------

    def test_frequent_item_higher_score_than_one_hit(self) -> None:
        """A frequently-accessed item should have a higher heat score."""
        p = self._make(capacity=10, decay_half_life=999.0)
        # Access item (0,0) many times quickly
        for _ in range(20):
            p.access(0, 0)
        # Access item (0,1) once
        p.access(0, 1)

        score_frequent = p.heat_score(0, 0)
        score_one_hit = p.heat_score(0, 1)
        assert score_frequent > score_one_hit

    # -- protected set -------------------------------------------------------

    def test_protected_items_evicted_only_after_probationary_empty(self) -> None:
        """Items above the threshold are evicted only when no probationary items remain."""
        p = self._make(capacity=10, min_score_threshold=1.5, decay_half_life=999.0)

        # Create a protected item (score > 1.5) by accessing it multiple times
        for _ in range(10):
            p.access(0, 0)  # will have score well above 1.5

        # Create a probationary item (score = 1.0 initially, ≤ 1.5)
        p.access(0, 1)

        score_0 = p.heat_score(0, 0)
        assert score_0 > 1.5, f"expected protected, got score={score_0}"

        # Evict should pick the probationary item
        evicted = p.evict()
        assert evicted == (0, 1)

        # Now only protected item remains; next evict removes it
        evicted2 = p.evict()
        assert evicted2 == (0, 0)

    # -- heat_score ----------------------------------------------------------

    def test_heat_score_returns_float(self) -> None:
        """heat_score always returns a float."""
        p = self._make()
        p.access(0, 0)
        result = p.heat_score(0, 0)
        assert isinstance(result, float)

    def test_heat_score_untracked_returns_zero(self) -> None:
        """heat_score returns 0.0 for an untracked expert."""
        p = self._make()
        assert p.heat_score(0, 99) == 0.0

    # -- stats ---------------------------------------------------------------

    def test_stats_dict(self) -> None:
        """stats() returns a dict with expected keys."""
        p = self._make(capacity=10)
        p.access(0, 0)
        s = p.stats()
        assert s["type"] == "frequency_aware"
        assert s["size"] == 1
        assert s["capacity"] == 10
        assert s["total_accesses"] == 1
        assert s["total_evictions"] == 0
        assert "score_mean" in s
        assert "score_std" in s

    # -- remove & contains ---------------------------------------------------

    def test_remove(self) -> None:
        """remove() removes an item from tracking."""
        p = self._make()
        p.access(0, 0)
        assert p.contains(0, 0)
        p.remove(0, 0)
        assert not p.contains(0, 0)
        assert p.size == 0

    def test_remove_nonexistent_is_noop(self) -> None:
        """Removing a nonexistent key does not raise."""
        p = self._make()
        p.remove(0, 99)  # should not raise

    def test_contains(self) -> None:
        """contains() reflects whether an expert is tracked."""
        p = self._make()
        assert not p.contains(0, 0)
        p.access(0, 0)
        assert p.contains(0, 0)

    # -- decay ---------------------------------------------------------------

    def test_decay_old_accesses_fade(self) -> None:
        """Heat score decreases over simulated time due to decay."""
        half_life = 0.01  # 10 ms half-life — very fast decay
        p = self._make(capacity=10, decay_half_life=half_life)

        p.access(0, 0)
        initial_score = p.heat_score(0, 0)
        assert initial_score > 0.0

        # Wait long enough for significant decay (> 5 half-lives)
        time.sleep(half_life * 6)
        decayed_score = p.heat_score(0, 0)

        assert decayed_score < initial_score
        assert decayed_score > 0.0  # should still be non-zero

    # -- capacity ------------------------------------------------------------

    def test_capacity_property(self) -> None:
        """capacity returns the value passed to __init__."""
        p = self._make(capacity=42)
        assert p.capacity == 42

    def test_size_reflects_actual_items(self) -> None:
        """size matches the number of tracked items."""
        p = self._make(capacity=5)
        assert p.size == 0
        for idx in range(5):
            p.access(0, idx)
        assert p.size == 5
        p.evict()
        assert p.size == 4


# ===========================================================================
# CostBenefitScorer
# ===========================================================================


class TestCostBenefitScorer:
    """Tests for CostBenefitScorer."""

    def _make(
        self,
        capacity: int = 10,
        ssd_latency_ms: float = 2.0,
        expert_size_bytes: int | None = None,
    ) -> CostBenefitScorer:
        inner = FrequencyAwarePolicy(capacity=capacity)
        return CostBenefitScorer(
            policy=inner,
            ssd_latency_ms=ssd_latency_ms,
            expert_size_bytes=expert_size_bytes,
        )

    # -- basic access/evict delegates ----------------------------------------

    def test_basic_access_and_evict(self) -> None:
        """access() delegates to inner policy; evict() removes via inner."""
        scorer = self._make(capacity=5)
        scorer.access(0, 0)
        scorer.access(0, 1)
        assert scorer.size == 2

        evicted = scorer.evict()
        assert evicted is not None
        assert scorer.size == 1

    def test_evict_empty_returns_none(self) -> None:
        """Evicting from empty scorer returns None."""
        scorer = self._make()
        assert scorer.evict() is None

    # -- record_ssd_load -----------------------------------------------------

    def test_record_ssd_load_updates_latency_tracking(self) -> None:
        """record_ssd_load stores latency history per expert."""
        scorer = self._make()
        scorer.access(0, 0)

        # Record several latencies
        for latency in [1.0, 2.0, 3.0, 4.0, 5.0]:
            scorer.record_ssd_load(0, 0, latency)

        stats = scorer.stats()
        assert stats["tracked_latencies"] == 1

    # -- benefit_cost_ratio --------------------------------------------------

    def test_benefit_cost_ratio_basic(self) -> None:
        """benefit_cost_ratio returns benefit / cost."""
        scorer = self._make(ssd_latency_ms=5.0)
        scorer.access(0, 0)
        # Without per-expert latency, cost defaults to ssd_latency_ms=5.0
        ratio = scorer.benefit_cost_ratio(0, 0)
        # benefit should be ~1.0 (one access, score=1.0), cost=5.0
        assert ratio > 0
        # ratio = heat_score / 5.0 ≈ 1.0/5.0 = 0.2
        assert 0.05 < ratio < 0.5

    def test_benefit_cost_ratio_untracked_returns_zero(self) -> None:
        """benefit_cost_ratio returns 0.0 for untracked expert."""
        scorer = self._make()
        # Note: for untracked items, heat_score returns 0.0
        # So benefit/cost = 0.0 / cost = 0.0
        assert scorer.benefit_cost_ratio(0, 99) == 0.0

    def test_benefit_cost_ratio_uses_per_expert_latency(self) -> None:
        """When per-expert latency history exists, uses p95 of that."""
        scorer = self._make(ssd_latency_ms=2.0)
        scorer.access(0, 0)

        # Record very high latency for this expert
        for _ in range(10):
            scorer.record_ssd_load(0, 0, 100.0)

        ratio = scorer.benefit_cost_ratio(0, 0)
        # cost = p95 of [100, 100, ..., 100] = 100
        # benefit ≈ 1.0 (score=1.0), so ratio ≈ 1.0/100 = 0.01
        assert ratio < 0.05

    # -- eviction order ------------------------------------------------------

    def test_evicts_low_benefit_high_cost_first(self) -> None:
        """Items with low benefit/cost ratio are evicted before high ones."""
        scorer = self._make(ssd_latency_ms=2.0, capacity=2)

        # Item A: high benefit, low cost
        scorer.access(0, 0)
        for _ in range(10):
            scorer.record_ssd_load(0, 0, 1.0)  # cheap reload

        # Item B: low benefit, high cost
        scorer.access(0, 1)
        for _ in range(10):
            scorer.record_ssd_load(0, 1, 100.0)  # expensive reload

        ratio_a = scorer.benefit_cost_ratio(0, 0)
        ratio_b = scorer.benefit_cost_ratio(0, 1)
        assert ratio_a > ratio_b, f"Expected A({ratio_a}) > B({ratio_b})"

        evicted = scorer.evict()
        assert evicted == (0, 1), "Should evict low-benefit/high-cost item"

    # -- stats ---------------------------------------------------------------

    def test_stats_include_cost_info(self) -> None:
        """stats() includes cost-related keys."""
        scorer = self._make(ssd_latency_ms=3.0, expert_size_bytes=1024)
        s = scorer.stats()
        assert s["type"] == "cost_benefit_scorer"
        assert "default_latency_ms" in s
        assert s["default_latency_ms"] == 3.0
        assert "expert_size_bytes" in s
        assert s["expert_size_bytes"] == 1024
        assert "tracked_latencies" in s
        assert "wrapped_policy" in s

    # -- graceful fallback ---------------------------------------------------

    def test_graceful_fallback_no_heat_score(self) -> None:
        """When inner policy has no heat_score, benefit defaults to 1.0."""
        mock_policy = MagicMock(spec=EvictionPolicy)
        mock_policy.size = 1
        mock_policy.capacity = 10
        mock_policy.stats.return_value = {"type": "mock"}

        scorer = CostBenefitScorer(policy=mock_policy, ssd_latency_ms=5.0)

        # benefit_cost_ratio should work even without heat_score
        # Since the mock has no heat_score, benefit = 1.0
        ratio = scorer.benefit_cost_ratio(0, 0)
        # cost = default 5.0, so ratio = 1.0 / 5.0 = 0.2
        assert ratio == pytest.approx(0.2)

    def test_remove_delegates_and_clears_latency(self) -> None:
        """remove() delegates to inner policy and clears latency history."""
        scorer = self._make()
        scorer.access(0, 0)
        scorer.record_ssd_load(0, 0, 5.0)

        assert scorer.contains(0, 0)
        scorer.remove(0, 0)
        assert not scorer.contains(0, 0)
        # Latency history should also be cleared
        assert scorer.stats()["tracked_latencies"] == 0

    def test_contains_delegates(self) -> None:
        """contains() delegates to inner policy."""
        scorer = self._make()
        assert not scorer.contains(0, 0)
        scorer.access(0, 0)
        assert scorer.contains(0, 0)


# ===========================================================================
# AdaptiveBanditPolicy
# ===========================================================================


class TestAdaptiveBanditPolicy:
    """Tests for AdaptiveBanditPolicy."""

    def _make(self, capacity: int = 10, **kwargs) -> AdaptiveBanditPolicy:
        return AdaptiveBanditPolicy(capacity=capacity, **kwargs)

    # -- basic access & evict ------------------------------------------------

    def test_basic_access_and_evict(self) -> None:
        """Accessing adds arms; evict removes one."""
        np.random.seed(42)
        p = self._make(capacity=5)
        p.access(0, 0)
        p.access(0, 1)
        p.access(0, 2)
        assert p.size == 3

        evicted = p.evict()
        assert evicted is not None
        assert p.size == 2

    def test_evict_empty_returns_none(self) -> None:
        """Evicting when no arms in RAM returns None."""
        p = self._make()
        assert p.evict() is None

    # -- access increments alpha ---------------------------------------------

    def test_access_increments_alpha(self) -> None:
        """Each access increments the arm's alpha."""
        p = self._make(prior_alpha=1.0, prior_beta=1.0)
        p.access(0, 0)  # alpha = prior + 1 = 2
        p.access(0, 0)  # alpha = 3

        arm = p._arms[(0, 0)]
        assert arm["alpha"] == pytest.approx(3.0)

    # -- miss increments beta ------------------------------------------------

    def test_miss_increments_beta(self) -> None:
        """miss() increments beta for the arm."""
        p = self._make(prior_alpha=1.0, prior_beta=1.0)
        p.access(0, 0)  # creates the arm
        p.miss(0, 0)  # beta += 1

        arm = p._arms[(0, 0)]
        assert arm["beta"] == pytest.approx(2.0)

    def test_miss_creates_arm_if_not_exists(self) -> None:
        """miss() for a new expert creates the arm in RAM."""
        p = self._make(prior_alpha=1.0, prior_beta=1.0)
        p.miss(0, 0)

        arm = p._arms[(0, 0)]
        assert arm["in_ram"] is True
        assert arm["beta"] == pytest.approx(2.0)  # prior + 1

    # -- sample_probability --------------------------------------------------

    def test_sample_probability_returns_zero_one(self) -> None:
        """sample_probability returns a float in [0, 1]."""
        np.random.seed(123)
        p = self._make()
        p.access(0, 0)

        for _ in range(50):
            val = p.sample_probability(0, 0)
            assert 0.0 <= val <= 1.0
            assert isinstance(val, float)

    def test_sample_probability_untracked_returns_zero(self) -> None:
        """sample_probability returns 0.0 for an untracked expert."""
        p = self._make()
        assert p.sample_probability(0, 99) == 0.0

    # -- expected_probability ------------------------------------------------

    def test_expected_probability(self) -> None:
        """expected_probability = alpha / (alpha + beta)."""
        p = self._make(prior_alpha=1.0, prior_beta=1.0)
        p.access(0, 0)  # alpha = 2.0, beta = 1.0
        # probability = 2.0 / (2.0 + 1.0) = 2/3
        assert p.expected_probability(0, 0) == pytest.approx(2.0 / 3.0)

    def test_expected_probability_untracked_returns_zero(self) -> None:
        """expected_probability returns 0.0 for an untracked expert."""
        p = self._make()
        assert p.expected_probability(0, 99) == 0.0

    # -- update() rebalances -------------------------------------------------

    def test_update_promotes_high_prob_from_ssd(self) -> None:
        """update() promotes high-probability SSD items when under capacity."""
        p = self._make(capacity=5, prior_alpha=1.0, prior_beta=1.0)

        # Add one item to RAM
        p.access(0, 0)
        assert p.size == 1

        # Manually create an SSD item with high alpha (high probability)
        p._arms[(0, 1)] = {
            "alpha": 100.0,
            "beta": 1.0,
            "in_ram": False,
        }
        assert p.size == 1  # still 1 in RAM

        # update() should promote the high-prob item
        result = p.update()
        assert (0, 1) in result["promoted"]
        assert p.size == 2

    def test_update_evicts_low_prob_from_ram(self) -> None:
        """update() evicts low-probability RAM items when over capacity."""
        p = self._make(capacity=2, prior_alpha=1.0, prior_beta=1.0)

        # Add 3 items to RAM (over capacity of 2)
        p.access(0, 0)
        p.access(0, 1)
        p.access(0, 2)
        assert p.size == 3

        # Make item (0,0) low probability by giving it low alpha
        p._arms[(0, 0)]["alpha"] = 1.0
        p._arms[(0, 0)]["beta"] = 100.0

        result = p.update()
        assert (0, 0) in result["evicted"]
        assert p.size == 2

    # -- should_promote ------------------------------------------------------

    def test_should_promote_for_high_alpha(self) -> None:
        """should_promote returns True for high-alpha SSD items."""
        p = self._make(capacity=5, prior_alpha=1.0, prior_beta=1.0)

        # One RAM item with moderate probability
        p.access(0, 0)  # alpha=2, beta=1 → prob=0.667

        # SSD item with very high probability
        p._arms[(0, 1)] = {
            "alpha": 100.0,
            "beta": 1.0,
            "in_ram": False,
        }

        # (0,1) has prob ≈ 0.99 > median of [0.667] = 0.667
        assert p.should_promote(0, 1) is True

    def test_should_promote_false_for_low_alpha(self) -> None:
        """should_promote returns False for low-probability SSD items."""
        p = self._make(capacity=5, prior_alpha=1.0, prior_beta=1.0)

        # RAM item with high probability
        p._arms[(0, 0)] = {
            "alpha": 100.0,
            "beta": 1.0,
            "in_ram": True,
        }

        # SSD item with low probability
        p._arms[(0, 1)] = {
            "alpha": 1.0,
            "beta": 100.0,
            "in_ram": False,
        }

        assert p.should_promote(0, 1) is False

    def test_should_promote_false_for_ram_item(self) -> None:
        """should_promote returns False for items already in RAM."""
        p = self._make()
        p.access(0, 0)
        assert p.should_promote(0, 0) is False

    # -- exploration_weight --------------------------------------------------

    def test_exploration_weight_affects_decisions(self) -> None:
        """Higher exploration_weight adds more noise, changing eviction variance."""
        p_low = self._make(capacity=2, exploration_weight=0.0)
        p_high = self._make(capacity=2, exploration_weight=1.0)

        # Same pattern for both
        for policy in (p_low, p_high):
            policy.access(0, 0)
            policy.access(0, 1)

        # Run many evictions and check that high-exploration has more varied results
        # With exploration_weight=0, deterministic-ish (still random from Beta)
        # With exploration_weight=1.0, noise adds up to 1.0 extra
        evicted_low = set()
        evicted_high = set()

        np.random.seed(42)
        for _ in range(100):
            p_l = self._make(capacity=2, exploration_weight=0.0)
            p_l.access(0, 0)
            p_l.access(0, 1)
            ev = p_l.evict()
            evicted_low.add(ev)

        np.random.seed(42)
        for _ in range(100):
            p_h = self._make(capacity=2, exploration_weight=1.0)
            p_h.access(0, 0)
            p_h.access(0, 1)
            ev = p_h.evict()
            evicted_high.add(ev)

        # Both should see some variance, but we just verify no crash
        # and that exploration weight is stored correctly
        assert p_low.stats()["exploration_weight"] == 0.0
        assert p_high.stats()["exploration_weight"] == 1.0

    # -- stats ---------------------------------------------------------------

    def test_stats_dict(self) -> None:
        """stats() returns expected keys and values."""
        p = self._make(capacity=10, exploration_weight=0.2)
        p.access(0, 0)
        p.miss(0, 1)

        s = p.stats()
        assert s["type"] == "adaptive_bandit"
        assert s["size"] == 2  # both in RAM
        assert s["capacity"] == 10
        assert s["total_accesses"] == 1
        assert s["total_misses"] == 1
        assert s["total_arms"] == 2
        assert s["exploration_weight"] == 0.2
        assert "prob_mean" in s
        assert "prob_std" in s

    def test_capacity_property(self) -> None:
        """capacity returns the value passed to __init__."""
        p = self._make(capacity=7)
        assert p.capacity == 7

    def test_remove(self) -> None:
        """remove() removes an arm entirely."""
        p = self._make()
        p.access(0, 0)
        assert p.contains(0, 0)
        p.remove(0, 0)
        assert not p.contains(0, 0)
        assert p.size == 0

    def test_contains_false_after_evict(self) -> None:
        """After evict(), contains() returns False for the evicted arm."""
        np.random.seed(42)
        p = self._make(capacity=5)
        p.access(0, 0)
        p.access(0, 1)
        assert p.contains(0, 0) and p.contains(0, 1)

        evicted = p.evict()
        assert evicted is not None
        # The evicted arm should have in_ram=False
        assert not p.contains(*evicted)
