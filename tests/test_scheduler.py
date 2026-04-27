"""Tests for VibeBlade PowerInfer scheduler."""

import numpy as np
from vibeblade.scheduler import NeuronPredictor, PowerInferScheduler


class TestNeuronPredictor:

    def test_predictor_initial_state(self):
        """Before any update, predict returns all-ones mask."""
        pred = NeuronPredictor(hidden_size=128)
        mask = pred.predict()
        assert mask.shape == (128,)
        assert mask.dtype == bool
        assert np.all(mask)

    def test_predictor_update_and_predict(self):
        """After adding nonzero activations, they should be predicted as active."""
        pred = NeuronPredictor(hidden_size=64, window_size=10)
        # Create activation with strong signal in first 10 neurons
        activations = np.zeros(64, dtype=np.float32)
        activations[:10] = 1.0
        for _ in range(5):
            pred.update(activations)
        mask = pred.predict(threshold=0.01)
        assert np.all(mask[:10])
        assert not np.any(mask[10:])

    def test_predictor_activation_freq(self):
        """Frequency should match actual activation rate."""
        pred = NeuronPredictor(hidden_size=20, window_size=10)
        # Activate neurons 0-9 in every update, 10-19 never
        activations = np.zeros(20, dtype=np.float32)
        activations[:10] = 1.0
        for _ in range(5):
            pred.update(activations)
        freq = pred.get_activation_freq()
        assert freq.shape == (20,)
        assert np.allclose(freq[:10], 1.0)
        assert np.allclose(freq[10:], 0.0)

    def test_predictor_reset(self):
        """After reset, predict returns all-ones again."""
        pred = NeuronPredictor(hidden_size=32)
        # Activate only half the neurons — the other half should not be predicted
        activations = np.zeros(32, dtype=np.float32)
        activations[:16] = 1.0
        pred.update(activations)
        assert not np.all(pred.predict())
        pred.reset()
        mask = pred.predict()
        assert np.all(mask)


class TestPowerInferScheduler:

    def test_scheduler_get_budget_shapes(self):
        """Hot and cold masks should be correct shapes and sum to hidden_size."""
        sched = PowerInferScheduler(hidden_size=256, num_layers=4, hot_budget=0.1)
        # Update predictor with some data first
        for _ in range(3):
            sched.update(0, np.random.randn(256).astype(np.float32))
        hot, cold = sched.get_budget(0)
        assert hot.shape == (256,)
        assert cold.shape == (256,)
        assert hot.dtype == bool
        assert cold.dtype == bool
        assert np.sum(hot) + np.sum(cold) == 256

    def test_scheduler_hot_budget(self):
        """Hot mask should have approximately hot_budget * hidden_size True values."""
        hidden = 1000
        budget = 0.1
        sched = PowerInferScheduler(hidden_size=hidden, num_layers=2, hot_budget=budget)
        # Populate with random activations
        rng = np.random.default_rng(42)
        for _ in range(20):
            sched.update(0, rng.standard_normal(hidden).astype(np.float32))
        hot, cold = sched.get_budget(0)
        expected_hot = int(hidden * budget)
        assert np.sum(hot) == expected_hot
        assert np.sum(cold) == hidden - expected_hot

    def test_scheduler_sparsity(self):
        """Sparsity should be approximately 1 - hot_budget."""
        hidden = 500
        budget = 0.15
        sched = PowerInferScheduler(hidden_size=hidden, num_layers=2, hot_budget=budget)
        rng = np.random.default_rng(123)
        for _ in range(10):
            sched.update(1, rng.standard_normal(hidden).astype(np.float32))
        result = sched.schedule_layer(1, rng.standard_normal(hidden).astype(np.float32))
        assert 'sparsity' in result
        assert abs(result['sparsity'] - (1.0 - budget)) < 0.01

    def test_schedule_layer_returns_dict(self):
        """Verify dict structure and types."""
        sched = PowerInferScheduler(hidden_size=100, num_layers=3, hot_budget=0.2)
        rng = np.random.default_rng(0)
        result = sched.schedule_layer(0, rng.standard_normal(100).astype(np.float32))
        assert isinstance(result, dict)
        assert 'hot_indices' in result
        assert 'cold_indices' in result
        assert 'sparsity' in result
        assert isinstance(result['hot_indices'], np.ndarray)
        assert isinstance(result['cold_indices'], np.ndarray)
        assert isinstance(result['sparsity'], float)
        assert result['hot_indices'].dtype == np.intp or result['hot_indices'].dtype == np.int64
        assert len(result['hot_indices']) + len(result['cold_indices']) == 100

    def test_scheduler_reset(self):
        """After reset, predictors should return default state."""
        sched = PowerInferScheduler(hidden_size=64, num_layers=2)
        sched.update(0, np.ones(64, dtype=np.float32))
        sched.update(0, np.ones(64, dtype=np.float32))
        sched.reset()
        freq = sched.predictors[0].get_activation_freq()
        assert np.allclose(freq, 0.0)
        mask = sched.predictors[0].predict()
        assert np.all(mask)
