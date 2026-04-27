"""VibeBlade KV Cache — Memory-efficient key-value cache for transformer inference."""

import numpy as np


class KVCache:
    """Ring-buffer KV cache for transformer layers."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, max_seq_len: int, dtype=np.float16):
        """Args:
            num_layers: number of transformer layers
            num_heads: number of attention heads
            head_dim: dimension per head
            max_seq_len: maximum sequence length
            dtype: storage dtype (default float16 for memory efficiency)
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Shape: (num_layers, 2, num_heads, max_seq_len, head_dim) — 2 for K and V
        self.cache = np.zeros(
            (num_layers, 2, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self._lengths = np.zeros(num_layers, dtype=np.int64)

    def update(self, layer_idx: int, key: np.ndarray, value: np.ndarray, position: int) -> None:
        """Insert K/V pair at position for a layer.
        key shape: (num_heads, 1, head_dim) or (num_heads, head_dim)
        value shape: same as key
        """
        if position >= self.max_seq_len:
            raise IndexError(f"Position {position} exceeds max_seq_len {self.max_seq_len}")
        if key.ndim == 2:
            key = key[:, np.newaxis, :]
            value = value[:, np.newaxis, :]
        self.cache[layer_idx, 0, :, position:position + 1, :] = key
        self.cache[layer_idx, 1, :, position:position + 1, :] = value
        self._lengths[layer_idx] = max(self._lengths[layer_idx], position + 1)

    def get(self, layer_idx: int, start: int = 0, end: int = None) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve cached K/V for a layer.
        Returns (key, value) each with shape (num_heads, seq_len, head_dim).
        """
        if end is None:
            end = int(self._lengths[layer_idx])
        k = self.cache[layer_idx, 0, :, start:end, :]
        v = self.cache[layer_idx, 1, :, start:end, :]
        return k, v

    @property
    def length(self) -> int:
        """Current cache length (tokens stored)."""
        return int(self._lengths[0])

    def clear(self) -> None:
        """Reset cache to empty."""
        self.cache.fill(0)
        self._lengths.fill(0)

    def memory_usage_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.cache.nbytes
