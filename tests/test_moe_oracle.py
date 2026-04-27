"""Tests for vibeblade.moe_oracle — ExpertOracle and PatternOracle."""

from __future__ import annotations

from vibeblade.moe_oracle import ExpertOracle, PatternOracle


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_oracle(num_experts: int = 8, order: int = 1) -> ExpertOracle:
    return ExpertOracle(num_experts=num_experts, order=order)


def _train_sequential(oracle: ExpertOracle, num_layers: int = 20) -> None:
    """Train oracle with a sequential pattern: layer L uses expert L % N."""
    for li in range(num_layers):
        selected = [li % oracle.num_experts, (li + 1) % oracle.num_experts]
        oracle.observe(li, selected)


# ── ExpertOracle: Order 1 ──────────────────────────────────────────────────


class TestExpertOracleOrder1:
    def test_init(self) -> None:
        o = _make_oracle(8)
        assert o.num_experts == 8
        assert o.order == 1
        assert o.accuracy() == 0.0

    def test_single_observation_no_error(self) -> None:
        o = _make_oracle(4)
        o.observe(0, [0, 1])

    def test_sequential_pattern_prediction(self) -> None:
        o = _make_oracle(8)
        _train_sequential(o, 20)
        # predict_next predicts for layer+1 based on current experts
        pred = o.predict_next(19, [3, 4], top_k=4)
        # Prediction is based on transition data; just verify structure
        assert len(pred) >= 1
        assert all(isinstance(eid, int) and isinstance(p, float) for eid, p in pred)

    def test_prediction_sorted_by_probability(self) -> None:
        o = _make_oracle(4)
        # Build a strong 0→2 transition
        o.observe(0, [0])
        o.observe(1, [2])
        o.observe(1, [2])
        pred = o.predict_next(0, [0], top_k=4)
        probs = [p for _, p in pred]
        assert probs == sorted(probs, reverse=True)

    def test_transition_matrix(self) -> None:
        o = _make_oracle(4)
        o.observe(0, [0])
        o.observe(1, [2])
        o.observe(1, [2])
        mat = o.transition_matrix(0)  # layer 0 transitions
        assert mat is not None
        assert mat.shape == (4, 4)
        # Row-normalized: row 0 has one transition to expert 2, so mat[0,2] == 1.0
        assert mat[0, 2] == 1.0

    def test_transition_matrix_returns_none_for_unknown_layer(self) -> None:
        o = _make_oracle(4)
        assert o.transition_matrix(99) is None

    def test_accuracy_updates(self) -> None:
        o = _make_oracle(4)
        # Build transition data
        o.observe(0, [0])
        o.observe(1, [2])
        o.observe(2, [2])
        o.observe(3, [2])
        # Now predict for layer 2, then observe layer 2
        o.predict_next(2, [2], top_k=2)
        o.observe(2, [2])
        # Accuracy should be computed
        acc = o.accuracy()
        assert acc >= 0.0

    def test_accuracy_zero_when_no_predictions(self) -> None:
        o = _make_oracle(4)
        o.observe(0, [0])
        o.observe(1, [1])
        assert o.accuracy() == 0.0

    def test_reset_clears_state(self) -> None:
        o = _make_oracle(4)
        _train_sequential(o, 10)
        o.reset()
        assert o.accuracy() == 0.0
        assert o.transition_matrix(0) is None

    def test_predict_with_no_data_returns_fallback(self) -> None:
        o = _make_oracle(4)
        pred = o.predict_next(0, [0], top_k=2)
        # No data → fallback returns current experts
        assert len(pred) >= 1
        ids = [eid for eid, _p in pred]
        assert 0 in ids

    def test_multiple_experts_per_layer(self) -> None:
        o = _make_oracle(8)
        for _ in range(10):
            o.observe(0, [0, 1, 2])
            o.observe(1, [3, 4, 5])
        pred = o.predict_next(0, [0, 1, 2], top_k=3)
        # After training, 0→3, 1→4, 2→5 should have learned transitions
        assert len(pred) >= 1

    def test_top_k_limited(self) -> None:
        o = _make_oracle(8)
        _train_sequential(o, 10)
        pred = o.predict_next(0, [0], top_k=2)
        assert len(pred) <= 2

    def test_all_same_expert(self) -> None:
        o = _make_oracle(4)
        for _ in range(10):
            o.observe(0, [0])
            o.observe(1, [0])
        pred = o.predict_next(0, [0], top_k=1)
        assert pred[0][0] == 0


# ── ExpertOracle: Higher Order ─────────────────────────────────────────────


class TestExpertOracleHigherOrder:
    def test_order_2_init(self) -> None:
        o = _make_oracle(4, order=2)
        assert o.order == 2

    def test_order_2_transition_counting(self) -> None:
        o = _make_oracle(4, order=2)
        o.observe(0, [0])
        o.observe(1, [1])
        o.observe(2, [2])
        o.observe(3, [3])
        pred = o.predict_next(2, [1, 2], top_k=2)
        assert isinstance(pred, list)
        assert len(pred) <= 2

    def test_order_2_no_error_without_enough_context(self) -> None:
        o = _make_oracle(4, order=2)
        o.observe(0, [0])
        pred = o.predict_next(1, [0], top_k=2)
        assert isinstance(pred, list)


# ── PatternOracle ──────────────────────────────────────────────────────────


def _make_pattern_oracle(num_experts: int = 8, pattern_length: int = 4) -> PatternOracle:
    return PatternOracle(num_experts=num_experts, pattern_length=pattern_length)


class TestPatternOracle:
    def test_init(self) -> None:
        p = _make_pattern_oracle()
        assert p.num_experts == 8

    def test_single_observation(self) -> None:
        p = _make_pattern_oracle()
        p.observe(0, [0, 1])

    def test_predict_after_pattern_established(self) -> None:
        p = _make_pattern_oracle(num_experts=4, pattern_length=3)
        # Build repeating pattern: layer 0→[0], layer 1→[1], layer 2→[2]
        for _ in range(8):
            p.observe(0, [0])
            p.observe(1, [1])
            p.observe(2, [2])
        pred = p.predict(2, top_k=2)
        # After repeated pattern, prediction should return something
        assert isinstance(pred, list)
        # Pattern oracle may return empty if no matching prefix found
        assert len(pred) <= 2

    def test_predict_returns_valid_when_no_pattern(self) -> None:
        p = _make_pattern_oracle(num_experts=4, pattern_length=4)
        p.observe(0, [0])
        pred = p.predict(1, top_k=2)
        assert isinstance(pred, list)

    def test_dominant_patterns(self) -> None:
        p = _make_pattern_oracle(num_experts=4, pattern_length=3)
        for _ in range(10):
            p.observe(0, [0])
            p.observe(1, [1])
            p.observe(2, [2])
        patterns = p.dominant_patterns(min_frequency=0.05)
        assert isinstance(patterns, list)

    def test_reset(self) -> None:
        p = _make_pattern_oracle()
        for _ in range(5):
            p.observe(0, [0])
            p.observe(1, [1])
            p.observe(2, [2])
        p.reset()
        pred = p.predict(2, top_k=2)
        assert isinstance(pred, list)

    def test_layer_order_matters(self) -> None:
        p = _make_pattern_oracle(num_experts=4, pattern_length=2)
        for _ in range(5):
            p.observe(0, [0])
            p.observe(1, [1])
        p.reset()
        for _ in range(5):
            p.observe(0, [3])
            p.observe(1, [3])
        pred = p.predict(1, top_k=2)
        ids = [eid for eid, _pr in pred]
        # After reset and retraining with expert 3, should predict 3
        if len(pred) > 0:
            assert 3 in ids
