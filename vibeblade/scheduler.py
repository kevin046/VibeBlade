"""VibeBlade PowerInfer — CPU neuron-prediction scheduling for sparse inference."""

import numpy as np


class NeuronPredictor:
    """Predicts which neurons activate for the next token based on activation history."""

    def __init__(self, hidden_size: int, window_size: int = 64):
        """Args:
            hidden_size: number of neurons in the layer
            window_size: how many past tokens to consider for prediction
        """
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.activation_history = np.zeros((window_size, hidden_size), dtype=np.float32)
        self._pos = 0

    def update(self, activations: np.ndarray) -> None:
        """Add a new activation vector to history. Shape: (hidden_size,)"""
        self.activation_history[self._pos % self.window_size] = activations
        self._pos += 1

    def predict(self, threshold: float = 0.01) -> np.ndarray:
        """Predict which neurons will activate. Returns boolean mask (hidden_size,).
        Uses mean absolute activation over the window, returns mask where mean > threshold.
        """
        n = min(self._pos, self.window_size)
        if n == 0:
            return np.ones(self.hidden_size, dtype=bool)
        history = self.activation_history[:n]
        mean_activation = np.mean(np.abs(history), axis=0)
        return mean_activation > threshold

    def get_activation_freq(self) -> np.ndarray:
        """Returns frequency of activation (0.0 to 1.0) for each neuron."""
        n = min(self._pos, self.window_size)
        if n == 0:
            return np.zeros(self.hidden_size, dtype=np.float32)
        history = self.activation_history[:n]
        return np.mean(history != 0, axis=0).astype(np.float32)

    def reset(self) -> None:
        """Clear history."""
        self.activation_history.fill(0)
        self._pos = 0


class PowerInferScheduler:
    """Routes computation: hot neurons to fast CPU path, cold neurons to sparse path."""

    def __init__(self, hidden_size: int, num_layers: int, hot_budget: float = 0.1):
        """Args:
            hidden_size: neurons per layer
            num_layers: number of transformer layers
            hot_budget: fraction of neurons to assign to fast (hot) path (0.0 to 1.0)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hot_budget = hot_budget
        self.predictors = [NeuronPredictor(hidden_size) for _ in range(num_layers)]

    def update(self, layer_idx: int, activations: np.ndarray) -> None:
        """Update predictor for a specific layer."""
        if 0 <= layer_idx < self.num_layers:
            self.predictors[layer_idx].update(activations)

    def get_budget(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get hot/cold neuron split for a layer.
        Returns (hot_mask, cold_mask) — boolean arrays (hidden_size,).
        Hot neurons: top-k by activation frequency within hot_budget fraction.
        """
        freq = self.predictors[layer_idx].get_activation_freq()
        k = max(1, int(self.hidden_size * self.hot_budget))
        topk_indices = np.argpartition(freq, -k)[-k:]
        hot_mask = np.zeros(self.hidden_size, dtype=bool)
        hot_mask[topk_indices] = True
        return hot_mask, ~hot_mask

    def schedule_layer(self, layer_idx: int, activations: np.ndarray) -> dict:
        """Generate execution plan for one layer.
        Returns dict with:
            'hot_indices': np.ndarray of indices for fast path
            'cold_indices': np.ndarray of indices for sparse path
            'sparsity': float fraction of neurons that are cold
        """
        hot_mask, cold_mask = self.get_budget(layer_idx)
        self.update(layer_idx, activations)
        return {
            'hot_indices': np.where(hot_mask)[0],
            'cold_indices': np.where(cold_mask)[0],
            'sparsity': float(np.sum(cold_mask) / self.hidden_size),
        }

    def reset(self) -> None:
        """Reset all predictors."""
        for p in self.predictors:
            p.reset()
