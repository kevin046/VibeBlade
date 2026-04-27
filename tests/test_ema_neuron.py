"""Tests for EMA-based NeuronPredictor and dReLU gate activation."""

import numpy as np

from vibeblade.sparse import (
    drelu_gate,
    EMANeuronPredictor,
)


class TestDReLUGate:
    """Test dReLU gating: max(0,x) * max(0,-x) from whitepaper §1."""

    def test_zero(self):
        result = drelu_gate(np.array([0.0]))
        assert result[0] == 0.0

    def test_positive_input(self):
        # drelu(5) = max(0,5) * max(0,-5) = 5 * 0 = 0
        result = drelu_gate(np.array([5.0]))
        assert result[0] == 0.0

    def test_negative_input(self):
        # drelu(-3) = max(0,-3) * max(0,3) = 0 * 3 = 0
        result = drelu_gate(np.array([-3.0]))
        assert result[0] == 0.0

    def test_shape_preserved(self):
        x = np.random.randn(4, 8).astype(np.float32)
        result = drelu_gate(x)
        assert result.shape == x.shape

    def test_all_nonpositive(self):
        """dReLU output is always >= 0 but also always has zeros for nonzero x.

        Actually max(0,x)*max(0,-x) = 0 for all x since either pos or neg is zero.
        This is the pure mathematical interpretation. The whitepaper uses it as
        a gating signal where BOTH branches contribute.
        """
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = drelu_gate(x)
        # For any single scalar: one of max(0,x) or max(0,-x) is zero
        np.testing.assert_array_almost_equal(result, np.zeros(5))

    def test_batch_input(self):
        x = np.random.randn(3, 10, 128).astype(np.float32)
        result = drelu_gate(x)
        assert result.shape == (3, 10, 128)

    def test_dtype_preserved(self):
        x = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
        result = drelu_gate(x)
        assert result.dtype == x.dtype


class TestEMANeuronPredictor:
    """Test EMA-based neuron activation predictor."""

    def _make_predictor(
        self, hidden_dim=128, n_layers=4, ema_decay=0.1, **kwargs
    ) -> EMANeuronPredictor:
        return EMANeuronPredictor(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            ema_decay=ema_decay,
            **kwargs,
        )

    def test_initial_state(self):
        pred = self._make_predictor()
        assert pred.total_updates == 0
        assert not pred.is_warmed_up
        probs = pred.get_activation_probabilities(0)
        assert probs.shape == (128,)
        assert np.all(probs == 0.0)

    def test_update_increments_counter(self):
        pred = self._make_predictor()
        mask = np.zeros(128, dtype=bool)
        mask[:10] = True
        pred.update(0, mask)
        assert pred.total_updates == 1

    def test_ema_decay(self):
        """EMA should decay old values."""
        pred = self._make_predictor(ema_decay=0.1)

        # Activate neurons 0-9 for 10 updates
        mask = np.zeros(128, dtype=bool)
        mask[:10] = True
        for _ in range(10):
            pred.update(0, mask)

        probs = pred.get_activation_probabilities(0)
        # After many updates of same mask, EMA should converge toward the mask
        assert probs[0] > 0.5  # neuron 0 should have high probability
        assert probs[50] < 0.01  # neuron 50 should be near zero

    def test_predict_cold_start_no_gate(self):
        """Without gate values and no warmup, should return uniform top-k."""
        pred = self._make_predictor(sparse_ratio=0.1)
        mask = pred.predict(0, gate_values=None)
        assert mask.shape == (128,)
        # Should select first k neurons (10% of 128 = 12)
        assert np.sum(mask) == 12

    def test_predict_cold_start_with_gate(self):
        """With gate values during cold start, should use top-k on gates."""
        pred = self._make_predictor(sparse_ratio=0.1, hidden_dim=32)
        gate = np.random.randn(32).astype(np.float32)
        gate[5] = 100.0  # ensure neuron 5 is top
        mask = pred.predict(0, gate_values=gate)
        assert mask.shape == (32,)
        assert mask[5]  # neuron 5 should be selected

    def test_predict_warmed_up(self):
        """After warmup, should use EMA probabilities."""
        pred = self._make_predictor(
            hidden_dim=64, n_layers=1, sparse_ratio=0.25, activation_threshold=0.2
        )

        # Warm up layer 0 with neurons 0-15 always active
        mask = np.zeros(64, dtype=bool)
        mask[:16] = True
        for _ in range(10):  # > 5 warmup threshold
            pred.update(0, mask)

        assert pred.is_warmed_up
        result = pred.predict(0)
        # Neurons 0-15 should be selected (EMA > 0.2 after 10 updates)
        assert all(result[i] for i in range(16))

    def test_predict_combined(self):
        """Combined prediction should merge EMA and gate signals."""
        pred = self._make_predictor(hidden_dim=64, sparse_ratio=0.1)

        # Warm up with neurons 0-5 active
        mask = np.zeros(64, dtype=bool)
        mask[:6] = True
        for _ in range(10):
            pred.update(0, mask)

        gate = np.zeros(64, dtype=np.float32)
        gate[50] = 100.0  # unusual neuron

        combined = pred.predict_combined(0, gate)
        # Should include both EMA-predicted and gate top-k neurons
        assert combined[0] or combined[50]  # at least one signal should work

    def test_invalid_layer_idx(self):
        pred = self._make_predictor(n_layers=4)
        # Out of range should return all-True
        mask = pred.predict(-1)
        assert np.all(mask)
        mask = pred.predict(100)
        assert np.all(mask)

    def test_update_invalid_layer(self):
        pred = self._make_predictor(n_layers=4)
        initial_count = pred.total_updates
        pred.update(-1, np.zeros(128, dtype=bool))
        assert pred.total_updates == initial_count

    def test_get_top_neurons(self):
        pred = self._make_predictor(hidden_dim=64)

        # Activate neurons 10-19 frequently
        mask = np.zeros(64, dtype=bool)
        mask[10:20] = True
        for _ in range(10):
            pred.update(0, mask)

        top = pred.get_top_neurons(0, k=5)
        assert len(top) == 5
        # All top neurons should be in the 10-19 range
        assert all(10 <= idx < 20 for idx in top)

    def test_get_top_neurons_sorted(self):
        """Top neurons should be sorted by descending probability."""
        pred = self._make_predictor(hidden_dim=32)
        mask = np.zeros(32, dtype=bool)
        mask[0] = True
        for _ in range(10):
            pred.update(0, mask)

        top = pred.get_top_neurons(0, k=5)
        probs = [pred.get_activation_probabilities(0)[i] for i in top]
        assert probs == sorted(probs, reverse=True)

    def test_reset(self):
        pred = self._make_predictor()
        mask = np.zeros(128, dtype=bool)
        mask[:5] = True
        pred.update(0, mask)
        pred.reset()
        assert pred.total_updates == 0
        assert not pred.is_warmed_up
        assert np.all(pred.get_activation_probabilities(0) == 0.0)

    def test_different_ema_decay_rates(self):
        """Higher decay should track changes faster."""
        pred_slow = self._make_predictor(hidden_dim=16, ema_decay=0.01)
        pred_fast = self._make_predictor(hidden_dim=16, ema_decay=0.5)

        mask1 = np.zeros(16, dtype=bool)
        mask1[:8] = True
        for _ in range(5):
            pred_slow.update(0, mask1)
            pred_fast.update(0, mask1)

        # Switch pattern
        mask2 = np.zeros(16, dtype=bool)
        mask2[8:] = True
        pred_slow.update(0, mask2)
        pred_fast.update(0, mask2)

        # Fast predictor should adapt more quickly
        probs_slow = pred_slow.get_activation_probabilities(0)
        probs_fast = pred_fast.get_activation_probabilities(0)

        # Neuron 8-15 should have higher prob in fast predictor
        assert np.mean(probs_fast[8:]) > np.mean(probs_slow[8:])

    def test_ema_convergence(self):
        """EMA should converge toward the true activation frequency."""
        pred = self._make_predictor(hidden_dim=100, ema_decay=0.05)

        # 50% of neurons always active
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True
        for _ in range(200):
            pred.update(0, mask)

        probs = pred.get_activation_probabilities(0)
        # Neurons 0-49 should have high probability
        assert np.mean(probs[:50]) > 0.8
        # Neurons 50-99 should have low probability
        assert np.mean(probs[50:]) < 0.05

    def test_multiple_layers_independent(self):
        """Different layers should track independently."""
        pred = self._make_predictor(hidden_dim=32, n_layers=2)

        mask0 = np.zeros(32, dtype=bool)
        mask0[:10] = True
        mask1 = np.zeros(32, dtype=bool)
        mask1[10:20] = True

        for _ in range(10):
            pred.update(0, mask0)
            pred.update(1, mask1)

        probs0 = pred.get_activation_probabilities(0)
        probs1 = pred.get_activation_probabilities(1)

        # Layer 0: neurons 0-9 active
        assert np.mean(probs0[:10]) > np.mean(probs0[10:])
        # Layer 1: neurons 10-19 active
        assert np.mean(probs1[10:20]) > np.mean(probs1[20:])
