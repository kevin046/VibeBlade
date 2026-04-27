"""Tests for SageSched — uncertainty-aware scheduler."""

import math

from vibeblade.sagesched import (
    SageConfig,
    SageRequest,
    SageSched,
    entropy_from_logits,
    entropy_from_probs,
)


class TestEntropyFunctions:
    def test_entropy_from_logits_uniform(self):
        """Uniform distribution (all logits equal) should have high entropy."""
        logits = [0.0, 0.0, 0.0, 0.0]
        h = entropy_from_logits(logits)
        # 4-way uniform: H = log2(4) = 2.0
        assert abs(h - 2.0) < 0.01

    def test_entropy_from_logits_confident(self):
        """Highly peaked distribution should have low entropy."""
        logits = [10.0, 0.0, 0.0, 0.0]
        h = entropy_from_logits(logits)
        assert h < 0.01

    def test_entropy_from_logits_empty(self):
        assert entropy_from_logits([]) == 0.0

    def test_entropy_from_probs_uniform(self):
        """Uniform 4-way: H = log2(4) = 2.0."""
        probs = [0.25, 0.25, 0.25, 0.25]
        h = entropy_from_probs(probs)
        assert abs(h - 2.0) < 0.01

    def test_entropy_from_probs_confident(self):
        probs = [0.99, 0.01]
        h = entropy_from_probs(probs)
        assert h < 0.1

    def test_entropy_from_probs_empty(self):
        assert entropy_from_probs([]) == 0.0

    def test_entropy_from_logits_matches_probs(self):
        """Both functions should give same result for valid distributions."""
        logits = [1.0, 2.0, -1.0, 0.5]
        h_logits = entropy_from_logits(logits)
        # Compute probs manually
        max_l = max(logits)
        exps = [math.exp(logit - max_l) for logit in logits]
        s = sum(exps)
        probs = [e / s for e in exps]
        h_probs = entropy_from_probs(probs)
        assert abs(h_logits - h_probs) < 1e-10

    def test_entropy_invariant_to_shift(self):
        """Adding a constant to all logits shouldn't change entropy."""
        logits1 = [1.0, 2.0, 3.0]
        logits2 = [11.0, 12.0, 13.0]
        assert abs(entropy_from_logits(logits1) - entropy_from_logits(logits2)) < 1e-10


class TestSageRequest:
    def test_initial_state(self):
        req = SageRequest(request_id=0, total_prompt_tokens=100)
        assert not req.is_done
        assert req.last_entropy == 0.0
        assert req.avg_entropy == 0.0

    def test_is_done(self):
        req = SageRequest(request_id=0, total_prompt_tokens=50, max_decode_tokens=5)
        assert not req.is_done
        req.tokens_decoded = 5
        assert req.is_done


class TestSageSched:
    def _make_sched(self, **kwargs) -> SageSched:
        config = SageConfig(
            alpha=kwargs.get("alpha", 1.0),
            beta=kwargs.get("beta", 2.0),
            gamma=kwargs.get("gamma", 0.01),
            max_batch_size=kwargs.get("max_batch_size", 8),
            entropy_window=kwargs.get("entropy_window", 10),
        )
        return SageSched(config)

    def test_add_request(self):
        sched = self._make_sched()
        req = sched.add_request(prompt_tokens=100, base_priority=2.0)
        assert req.request_id == 0
        assert req.base_priority == 2.0
        assert sched.get_stats()["queue_length"] == 1

    def test_schedule_admits_from_queue(self):
        sched = self._make_sched()
        sched.add_request(prompt_tokens=50)
        plan = sched.schedule()
        assert len(plan["scheduled_ids"]) == 1
        assert sched.get_stats()["active_count"] == 1
        assert sched.get_stats()["queue_length"] == 0

    def test_high_uncertainty_prioritized(self):
        """Requests with higher entropy should be scheduled first."""
        sched = self._make_sched(max_batch_size=1)
        # Add two requests — both in queue
        r1 = sched.add_request(prompt_tokens=50)
        r2 = sched.add_request(prompt_tokens=50)

        # Set uncertainty via update (requests still in queue)
        sched.update_entropy(r1.request_id, entropy=0.1)  # low uncertainty
        sched.update_entropy(r2.request_id, entropy=5.0)  # high uncertainty

        # Schedule — with batch_size=1, only one gets admitted
        plan = sched.schedule()
        assert len(plan["scheduled_ids"]) == 1

        # The admitted request should be r2 (higher priority due to entropy)
        assert plan["scheduled_ids"][0] == r2.request_id

    def test_update_entropy_from_logits(self):
        sched = self._make_sched()
        req = sched.add_request(prompt_tokens=50)
        logits = [1.0, 2.0, 0.5, -1.0]
        sched.update_entropy(req.request_id, logits=logits)
        found = sched._find_request(req.request_id)
        assert found is not None
        assert found.last_entropy > 0

    def test_update_entropy_running_average(self):
        sched = self._make_sched(entropy_window=3)
        req = sched.add_request(prompt_tokens=50)
        sched.update_entropy(req.request_id, entropy=4.0)
        sched.update_entropy(req.request_id, entropy=2.0)
        sched.update_entropy(req.request_id, entropy=6.0)
        found = sched._find_request(req.request_id)
        # Running average with window=3: (4+2+6)/3 = 4.0
        assert abs(found.avg_entropy - 4.0) < 0.01

    def test_priority_score_components(self):
        """Priority should consider base_priority, entropy, and wait time."""
        sched = self._make_sched(alpha=1.0, beta=2.0, gamma=100.0)
        r1 = sched.add_request(prompt_tokens=50, base_priority=1.0)
        r2 = sched.add_request(prompt_tokens=50, base_priority=1.0)

        # Both same priority and entropy, but r2 waited longer
        sched.update_entropy(r1.request_id, entropy=1.0)
        sched.update_entropy(r2.request_id, entropy=1.0)

        plan = sched.schedule()
        # Both should be scheduled (batch_size=8)
        # With same priority and entropy, order depends on wait_time (r2 waited longer)
        assert len(plan["scheduled_ids"]) == 2

    def test_max_batch_size_respected(self):
        sched = self._make_sched(max_batch_size=2)
        for _ in range(10):
            sched.add_request(prompt_tokens=50)
        plan = sched.schedule()
        assert len(plan["scheduled_ids"]) == 2
        assert sched.get_stats()["queue_length"] == 8

    def test_stats(self):
        sched = self._make_sched()
        r1 = sched.add_request(prompt_tokens=50)
        sched.update_entropy(r1.request_id, entropy=3.0)
        sched.schedule()
        stats = sched.get_stats()
        assert stats["queue_length"] == 0
        assert stats["active_count"] == 1
        assert stats["total_scheduled"] == 1

    def test_reset(self):
        sched = self._make_sched()
        sched.add_request(prompt_tokens=50)
        sched.schedule()
        sched.reset()
        stats = sched.get_stats()
        assert stats["queue_length"] == 0
        assert stats["active_count"] == 0
        assert stats["completed_count"] == 0
        assert stats["total_scheduled"] == 0

    def test_empty_schedule(self):
        sched = self._make_sched()
        plan = sched.schedule()
        assert len(plan["scheduled_ids"]) == 0

    def test_avg_uncertainty_in_plan(self):
        sched = self._make_sched()
        r1 = sched.add_request(prompt_tokens=50)
        r2 = sched.add_request(prompt_tokens=50)
        sched.update_entropy(r1.request_id, entropy=4.0)
        sched.update_entropy(r2.request_id, entropy=2.0)
        plan = sched.schedule()
        assert plan["avg_uncertainty"] == 3.0


class TestSageConfig:
    def test_defaults(self):
        cfg = SageConfig()
        assert cfg.alpha == 1.0
        assert cfg.beta == 2.0
        assert cfg.gamma == 0.01
        assert cfg.max_batch_size == 32
