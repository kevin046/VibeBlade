"""MoE Expert Activation Profiler — builds hot/cold expert maps for split execution."""

import json
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class HotColdMap:
    """Maps each MoE layer to hot (GPU) and cold (CPU) expert indices."""
    hot_experts: dict[int, list[int]]   # {layer_idx: [expert_ids]}
    cold_experts: dict[int, list[int]]  # {layer_idx: [expert_ids]}
    num_layers: int
    num_experts: int
    profile_tokens: int = 0
    profile_dataset: str = ""

    def hit_rate(self, layer_idx: int, expert_indices: np.ndarray) -> float:
        """What fraction of routed experts were hot (in GPU)?"""
        hot = set(self.hot_experts.get(layer_idx, []))
        return sum(1 for e in expert_indices.flatten() if int(e) in hot) / max(len(expert_indices.flatten()), 1)

    def overall_hit_rate(self) -> float:
        """Average hit rate across all layers."""
        if not self.hot_experts:
            return 0.0
        total = sum(len(v) for v in self.hot_experts.values())
        if self.num_layers == 0:
            return 0.0
        return total / (self.num_layers * self.num_experts)

    def summary(self) -> str:
        """Human-readable stats."""
        lines = [
            f"HotColdMap: {self.num_layers} layers, {self.num_experts} experts",
            f"Profile tokens: {self.profile_tokens}",
            f"Dataset: {self.profile_dataset or '(none)'}",
        ]
        for layer_idx in sorted(self.hot_experts.keys()):
            n_hot = len(self.hot_experts[layer_idx])
            n_cold = len(self.cold_experts.get(layer_idx, []))
            lines.append(f"  Layer {layer_idx:3d}: hot={n_hot}, cold={n_cold}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Serialize to JSON."""
        data = asdict(self)
        # JSON keys must be strings
        data["hot_experts"] = {str(k): v for k, v in data["hot_experts"].items()}
        data["cold_experts"] = {str(k): v for k, v in data["cold_experts"].items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HotColdMap":
        """Deserialize from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        # Convert string keys back to int
        data["hot_experts"] = {int(k): v for k, v in data["hot_experts"].items()}
        data["cold_experts"] = {int(k): v for k, v in data["cold_experts"].items()}
        return cls(**data)


class ExpertProfiler:
    """Offline profiler that records expert activation frequencies during calibration.

    Usage:
        profiler = ExpertProfiler(num_layers=80, num_experts=128)

        # During calibration decode loop:
        for layer_idx in range(num_layers):
            indices, weights = router.route(h_norm)
            profiler.record(layer_idx, indices)

        # After calibration:
        hot_map = profiler.compute_hot_cold_map(hot_ratio=0.1)
    """

    def __init__(self, num_layers: int, num_experts: int):
        # Per-layer activation counters: (num_layers, num_experts) int array
        self._counters = np.zeros((num_layers, num_experts), dtype=np.int64)
        self._total_tokens = 0
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._layer_token_counts = np.zeros(num_layers, dtype=np.int64)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def record(self, layer_idx: int, expert_indices: np.ndarray) -> None:
        """Record expert activations for one layer during one decode step.

        Args:
            layer_idx: which transformer layer
            expert_indices: (batch, topk) or (topk,) — which experts were selected
        """
        flat = expert_indices.flatten().astype(int)
        for e in flat:
            self._counters[layer_idx, e] += 1
        self._layer_token_counts[layer_idx] += 1
        self._total_tokens = max(self._total_tokens, self._layer_token_counts.max())

    def get_activation_freq(self, layer_idx: int) -> np.ndarray:
        """Activation frequency per expert in a layer (0.0 to 1.0)."""
        n = self._layer_token_counts[layer_idx]
        if n == 0:
            return np.zeros(self._num_experts)
        return self._counters[layer_idx] / n

    def get_all_frequencies(self) -> np.ndarray:
        """Returns (num_layers, num_experts) activation frequency matrix."""
        result = np.zeros_like(self._counters, dtype=np.float64)
        for layer_idx in range(self._num_layers):
            result[layer_idx] = self.get_activation_freq(layer_idx)
        return result

    def compute_hot_cold_map(
        self,
        hot_ratio: float = 0.1,
        min_hot_count: int = 0,
    ) -> HotColdMap:
        """Compute hot/cold expert assignment.

        Args:
            hot_ratio: fraction of experts to mark as hot per layer (e.g. 0.1 = top 10%)
            min_hot_count: minimum absolute activation count to be considered hot

        Returns:
            HotColdMap with per-layer hot/cold expert assignments
        """
        hot_experts = {}
        cold_experts = {}
        for layer_idx in range(self._num_layers):
            freq = self.get_activation_freq(layer_idx)
            n_hot = max(1, int(self._num_experts * hot_ratio))
            # Top-k by frequency
            top_indices = np.argsort(freq)[-n_hot:]
            # Filter by min_hot_count
            hot = [int(i) for i in top_indices if self._counters[layer_idx, i] >= min_hot_count]
            hot_experts[layer_idx] = hot
            cold = [i for i in range(self._num_experts) if i not in hot]
            cold_experts[layer_idx] = cold

        return HotColdMap(
            hot_experts=hot_experts,
            cold_experts=cold_experts,
            num_layers=self._num_layers,
            num_experts=self._num_experts,
            profile_tokens=self._total_tokens,
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._counters[:] = 0
        self._total_tokens = 0
        self._layer_token_counts[:] = 0
