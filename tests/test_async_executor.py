"""Tests for vibeblade.async_executor — AsyncMoEExecutor, ColdExpertResult, AsyncStats."""

from __future__ import annotations

import numpy as np

from vibeblade.async_executor import AsyncMoEExecutor, AsyncStats, ColdExpertResult


# ── Helpers ────────────────────────────────────────────────────────────────


HIDDEN_DIM = 64


def _make_cold_weights(
    num_layers: int = 2,
    num_experts: int = 4,
) -> dict:
    """Create mock cold expert weights: gate/up/down projections."""
    weights: dict = {}
    for li in range(num_layers):
        layer_weights: dict = {}
        for eid in range(num_experts):
            scale = 0.01
            gate = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * scale
            up = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * scale
            down = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * scale
            layer_weights[eid] = (gate, up, down)
        weights[li] = layer_weights
    return weights


def _make_hot_experts(num_layers: int = 2, hot_ids: set[int] | None = None) -> dict:
    """Create hot expert set mapping."""
    hot = hot_ids if hot_ids is not None else {0, 1}
    return {li: set(hot) for li in range(num_layers)}


def _make_executor(
    num_layers: int = 2,
    num_experts: int = 4,
    hot_ids: set[int] | None = None,
    cpu_threads: int = 2,
) -> AsyncMoEExecutor:
    return AsyncMoEExecutor(
        hot_experts=_make_hot_experts(num_layers, hot_ids),
        cold_expert_weights=_make_cold_weights(num_layers, num_experts),
        cpu_threads=cpu_threads,
        pinned_buffer_size=HIDDEN_DIM,
    )


def _simple_router_output(
    num_experts: int = 4,
    top_k: int = 2,
) -> list:
    """Create mock router output: (expert_id, weight) pairs."""
    weights = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    weights /= weights.sum()
    return [(i, float(weights[i])) for i in range(top_k)]


# ── ColdExpertResult ───────────────────────────────────────────────────────


class TestColdExpertResult:
    def test_create(self) -> None:
        r = ColdExpertResult(
            expert_id=0,
            output=np.zeros(HIDDEN_DIM, dtype=np.float32),
            weight=0.5,
            compute_time_ms=1.0,
        )
        assert r.expert_id == 0
        assert r.weight == 0.5


# ── AsyncStats ─────────────────────────────────────────────────────────────


class TestAsyncStats:
    def test_defaults(self) -> None:
        s = AsyncStats()
        assert s.hot_compute_ms == 0.0
        assert s.cold_expert_count == 0

    def test_to_dict(self) -> None:
        s = AsyncStats(hot_compute_ms=1.5, cold_expert_count=3)
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d["hot_compute_ms"] == 1.5
        assert d["cold_expert_count"] == 3


# ── AsyncMoEExecutor ──────────────────────────────────────────────────────


class TestAsyncMoEExecutor:
    def test_create(self) -> None:
        e = _make_executor()
        assert e._pinned_buffer.shape == (HIDDEN_DIM,)

    def test_execute_layer_basic(self) -> None:
        e = _make_executor(num_layers=2, num_experts=4, hot_ids={0, 1})
        hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)
        router = _simple_router_output(4, top_k=4)

        def hot_fn(state, experts):
            return np.zeros_like(state)

        result, stats = e.execute_layer(0, hidden, router, hot_fn)
        assert result.shape == (HIDDEN_DIM,)
        assert isinstance(stats, AsyncStats)

    def test_execute_layer_cold_experts_computed(self) -> None:
        e = _make_executor(num_layers=2, num_experts=4, hot_ids={0})  # only expert 0 is hot
        hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)
        router = [(0, 0.5), (1, 0.3), (2, 0.2)]  # experts 1, 2 are cold

        def hot_fn(state, experts):
            return np.zeros_like(state)

        result, stats = e.execute_layer(0, hidden, router, hot_fn)
        assert stats.cold_expert_count >= 1  # experts 1 and 2 are cold

    def test_execute_layer_all_hot(self) -> None:
        e = _make_executor(num_layers=2, num_experts=4, hot_ids={0, 1, 2, 3})
        hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)
        router = [(0, 0.5), (1, 0.3), (2, 0.2)]

        def hot_fn(state, experts):
            return np.zeros_like(state)

        _, stats = e.execute_layer(0, hidden, router, hot_fn)
        assert stats.cold_expert_count == 0

    def test_execute_layer_all_cold(self) -> None:
        e = _make_executor(num_layers=2, num_experts=4, hot_ids=set())  # no hot experts
        hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)
        router = [(0, 0.5), (1, 0.5)]

        def hot_fn(state, experts):
            return np.zeros_like(state)

        _, stats = e.execute_layer(0, hidden, router, hot_fn)
        assert stats.cold_expert_count == 2

    def test_execute_layer_no_router_output(self) -> None:
        e = _make_executor()
        hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)

        def hot_fn(state, experts):
            return np.zeros_like(state)

        _, stats = e.execute_layer(0, hidden, [], hot_fn)
        assert stats.cold_expert_count == 0

    def test_execute_layer_missing_cold_weights(self) -> None:
        e = _make_executor(num_layers=2, num_experts=2, hot_ids={0})
        hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)
        # Request expert 99 which has no weights
        router = [(0, 0.5), (99, 0.5)]

        def hot_fn(state, experts):
            return np.zeros_like(state)

        # Should not raise, just skip the missing expert
        _, stats = e.execute_layer(0, hidden, router, hot_fn)
        assert stats.cold_expert_count == 1  # expert 99 counted but skipped

    def test_merge_activations(self) -> None:
        e = _make_executor()
        hidden = np.ones(HIDDEN_DIM, dtype=np.float32) * 10.0
        cold_results = [
            ColdExpertResult(
                expert_id=0,
                output=np.ones(HIDDEN_DIM, dtype=np.float32) * 2.0,
                weight=0.5,
                compute_time_ms=1.0,
            ),
            ColdExpertResult(
                expert_id=1,
                output=np.ones(HIDDEN_DIM, dtype=np.float32) * 4.0,
                weight=0.3,
                compute_time_ms=1.0,
            ),
        ]
        merged = e.merge_activations(hidden, cold_results)
        # Expected: 10 + (0.5 * 2 + 0.3 * 4) = 10 + 2.2 = 12.2
        expected = 10.0 + 0.5 * 2.0 + 0.3 * 4.0
        assert abs(merged[0] - expected) < 1e-4

    def test_merge_activations_empty(self) -> None:
        e = _make_executor()
        hidden = np.ones(HIDDEN_DIM, dtype=np.float32)
        merged = e.merge_activations(hidden, [])
        np.testing.assert_array_equal(merged, hidden)

    def test_stats_tracking(self) -> None:
        e = _make_executor(num_layers=2, num_experts=4, hot_ids={0})
        hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)
        router = [(0, 0.5), (1, 0.3), (2, 0.2)]

        def hot_fn(state, experts):
            return np.zeros_like(state)

        _, stats = e.execute_layer(0, hidden, router, hot_fn)
        assert stats.hot_compute_ms >= 0.0
        assert stats.cold_compute_ms >= 0.0
        assert stats.cold_wait_ms >= 0.0

    def test_stats_summary(self) -> None:
        e = _make_executor()
        summary = e.stats_summary()
        assert isinstance(summary, dict)
        assert "buffer_size" in summary

    def test_shutdown(self) -> None:
        e = _make_executor()
        e.shutdown()  # should not raise

    def test_observe_experts(self) -> None:
        e = _make_executor()
        # Without oracle, observe should be a no-op
        e.observe_experts(0, [0, 1, 2])
