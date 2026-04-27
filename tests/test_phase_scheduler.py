"""Tests for vibeblade.phase_scheduler — PhaseScheduler, PhaseConfig, InferencePhase."""

from __future__ import annotations

from vibeblade.phase_scheduler import (
    InferencePhase,
    PhaseConfig,
    PhaseScheduler,
)


# ── PhaseConfig ────────────────────────────────────────────────────────────


class TestPhaseConfig:
    def test_prefill_defaults(self) -> None:
        c = PhaseConfig.prefill_defaults()
        assert c.confidence_threshold == 1.0  # no early exit
        assert c.top_k == 8
        assert c.hot_quant_bits == 4
        assert c.cold_quant_bits == 3

    def test_decode_defaults(self) -> None:
        c = PhaseConfig.decode_defaults()
        assert c.confidence_threshold == 0.9  # early exit
        assert c.top_k == 4
        assert c.hot_quant_bits == 5
        assert c.cold_quant_bits == 2

    def test_decode_has_more_hot_experts(self) -> None:
        prefill = PhaseConfig.prefill_defaults()
        decode = PhaseConfig.decode_defaults()
        assert decode.max_hot_experts > prefill.max_hot_experts

    def test_decode_has_deeper_prefetch(self) -> None:
        prefill = PhaseConfig.prefill_defaults()
        decode = PhaseConfig.decode_defaults()
        assert decode.prefetch_depth > prefill.prefetch_depth

    def test_custom_config(self) -> None:
        c = PhaseConfig(
            max_hot_experts=10,
            max_warm_experts=20,
            top_k=8,
            confidence_threshold=0.85,
            min_topk=2,
            enable_prefetch=True,
            prefetch_depth=5,
            hot_quant_bits=4,
            cold_quant_bits=2,
            cpu_threads=16,
        )
        assert c.max_hot_experts == 10
        assert c.confidence_threshold == 0.85


# ── PhaseScheduler ─────────────────────────────────────────────────────────


def _make_scheduler(
    num_experts: int = 8,
    num_layers: int = 4,
) -> PhaseScheduler:
    return PhaseScheduler(
        num_experts=num_experts,
        num_layers=num_layers,
    )


def _sample_hot_cold_map(num_layers: int = 4, num_experts: int = 8) -> dict:
    """Create a sample hot/warm/cold map."""
    result = {}
    for li in range(num_layers):
        hot = set(range(2))
        warm = set(range(2, 5))
        cold = set(range(5, num_experts))
        result[li] = {"hot": hot, "warm": warm, "cold": cold}
    return result


class TestPhaseScheduler:
    def test_initial_phase(self) -> None:
        s = _make_scheduler()
        # Before any begin_* call, what's the phase?
        assert s.current_phase == InferencePhase.PREFILL  # default

    def test_begin_prefill(self) -> None:
        s = _make_scheduler()
        config = s.begin_prefill(num_prompt_tokens=128)
        assert s.current_phase == InferencePhase.PREFILL
        assert config.top_k == 8
        assert config.confidence_threshold == 1.0

    def test_begin_decode(self) -> None:
        s = _make_scheduler()
        s.begin_prefill(num_prompt_tokens=10)
        config = s.begin_decode()
        assert s.current_phase == InferencePhase.DECODE
        assert config.top_k == 4
        assert config.confidence_threshold == 0.9

    def test_phase_transition(self) -> None:
        s = _make_scheduler()
        s.begin_prefill(10)
        assert s.current_phase == InferencePhase.PREFILL
        s.begin_decode()
        assert s.current_phase == InferencePhase.DECODE

    def test_update_expert_budget_decode_promotes_hot(self) -> None:
        s = _make_scheduler(num_experts=8, num_layers=4)
        s.begin_prefill(10)
        s.begin_decode()

        hcm = _sample_hot_cold_map(4, 8)

        # Record some expert activations to bias the frequency counter
        for _ in range(20):
            s.record_expert_activation(0, [0, 1, 2, 3])

        updated = s.update_expert_budget(hcm)
        # Decode should promote more experts to hot
        hot_count = len(updated[0]["hot"])
        assert hot_count >= 2  # at least the original hot experts

    def test_update_expert_budget_prefill_spreads_experts(self) -> None:
        s = _make_scheduler(num_experts=8, num_layers=4)
        s.begin_prefill(10)
        hcm = _sample_hot_cold_map(4, 8)
        updated = s.update_expert_budget(hcm)
        # All experts should be distributed
        total = len(updated[0]["hot"]) + len(updated[0]["warm"]) + len(updated[0]["cold"])
        assert total == 8

    def test_update_expert_budget_preserves_total_experts(self) -> None:
        s = _make_scheduler(num_experts=8, num_layers=4)
        s.begin_decode()
        hcm = _sample_hot_cold_map(4, 8)
        for li in range(4):
            for _ in range(10):
                s.record_expert_activation(li, list(range(8)))
        updated = s.update_expert_budget(hcm)
        for li in range(4):
            total = len(updated[li]["hot"]) + len(updated[li]["warm"]) + len(updated[li]["cold"])
            assert total == 8

    def test_token_callback_returns_config(self) -> None:
        s = _make_scheduler()
        config = s.token_callback(is_prefill=True)
        assert isinstance(config, PhaseConfig)

    def test_token_callback_auto_transition(self) -> None:
        s = _make_scheduler()
        # Start with prefill
        s.token_callback(is_prefill=True)
        assert s.current_phase == InferencePhase.PREFILL
        # Transition to decode
        s.token_callback(is_prefill=False)
        assert s.current_phase == InferencePhase.DECODE

    def test_phase_stats(self) -> None:
        s = _make_scheduler()
        s.begin_prefill(100)
        s.begin_decode()
        stats = s.phase_stats()
        assert "current_phase" in stats
        assert "transition_count" in stats
        assert stats["transition_count"] >= 1

    def test_reset(self) -> None:
        s = _make_scheduler()
        s.begin_prefill(10)
        s.begin_decode()
        s.reset()
        assert s.current_phase == InferencePhase.PREFILL
        stats = s.phase_stats()
        assert stats["transition_count"] == 0

    def test_record_expert_activation(self) -> None:
        s = _make_scheduler()
        s.record_expert_activation(0, [0, 1, 2])
        s.record_expert_activation(0, [0, 1, 3])
        # Should not raise

    def test_multiple_transitions(self) -> None:
        s = _make_scheduler()
        for _ in range(5):
            s.begin_prefill(10)
            s.begin_decode()
        stats = s.phase_stats()
        # First begin_prefill is a no-op (initial phase is already PREFILL),
        # then each pair: prefill→decode (+1), decode→prefill (+1) = 2 per iteration.
        # 5 iterations × 2 = 10, minus 1 (first prefill doesn't transition) = 9.
        assert stats["transition_count"] == 9
