"""VibeBlade MiniCache — Depth-dimension KV cache compression.

Based on: MiniCache: KV Cache Compression in Depth Dimension for Large Language Models (2405.14366)

Exploits high similarity between adjacent middle-to-deep transformer layers.
States are disentangled into magnitude and direction components; directions
are interpolated across layers while magnitudes are preserved.

Training-free and orthogonal to quantization/sparsity.
Achieves up to 5× compression and ~5× throughput improvement.
"""

from __future__ import annotations

import numpy as np


class MiniCache:
    """Depth-compressed KV cache that stores only a subset of layer KV states.

    For layers identified as similar, only one "representative" KV state is
    stored. Non-representative layers interpolate their KV from neighbors.

    Parameters
    ----------
    num_layers : int
    num_heads : int
    head_dim : int
    max_seq_len : int
    compression_ratio : int
        Store KV for every Nth layer (default 4 = 4× compression).
    start_layer : int
        Start compressing from this layer (default num_layers // 3).
    dtype : np.dtype
        Storage dtype (default float16).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        compression_ratio: int = 4,
        start_layer: int | None = None,
        dtype: np.dtype = np.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.compression_ratio = compression_ratio
        self.start_layer = start_layer if start_layer is not None else num_layers // 3
        self.dtype = dtype

        # Identify representative vs interpolated layers
        self._representative_layers: list[int] = []
        self._layer_to_group: dict[int, int] = {}  # layer_idx -> representative layer
        self._layer_weights: dict[int, float] = {}  # layer_idx -> interpolation weight

        self._build_layer_map()

        # Only store KV for representative layers
        num_reps = len(self._representative_layers)
        # Shape: (num_reps, 2, num_heads, max_seq_len, head_dim)
        self._cache = np.zeros(
            (num_reps, 2, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        # Map: rep_layer -> index in self._cache
        self._rep_idx: dict[int, int] = {
            layer: i for i, layer in enumerate(self._representative_layers)
        }

        self._lengths = np.zeros(num_layers, dtype=np.int64)
        # Track per-layer max position written
        self._max_pos = np.zeros(num_layers, dtype=np.int64)

    def _build_layer_map(self) -> None:
        """Build the compression layer map."""
        # First `start_layer` layers are always stored uncompressed
        for i in range(self.start_layer):
            self._representative_layers.append(i)
            self._layer_to_group[i] = i
            self._layer_weights[i] = 1.0

        # Remaining layers: store every compression_ratio-th layer
        group_start = self.start_layer
        for i in range(self.start_layer, self.num_layers):
            offset = i - group_start
            if offset % self.compression_ratio == 0:
                # This is a representative layer
                self._representative_layers.append(i)
                self._layer_to_group[i] = i
                self._layer_weights[i] = 1.0
            else:
                # Interpolate between the two surrounding representative layers
                group_begin = group_start + (offset // self.compression_ratio) * self.compression_ratio
                # Weight based on distance to nearest representative
                t = offset % self.compression_ratio
                self._layer_to_group[i] = group_begin
                self._layer_weights[i] = 1.0 - (t / self.compression_ratio)

    def update(
        self, layer_idx: int, key: np.ndarray, value: np.ndarray, position: int
    ) -> None:
        """Insert K/V at position. Only stores for representative layers.

        Parameters
        ----------
        layer_idx : int
        key : np.ndarray, shape ``(num_heads, head_dim)`` or ``(num_heads, 1, head_dim)``
        value : np.ndarray, same shape
        position : int
        """
        if position >= self.max_seq_len:
            raise IndexError(f"Position {position} exceeds max_seq_len {self.max_seq_len}")

        if key.ndim == 3:
            key = key[:, 0, :]
            value = value[:, 0, :]

        rep_layer = self._layer_to_group[layer_idx]
        weight = self._layer_weights[layer_idx]
        cache_idx = self._rep_idx[rep_layer]

        if weight >= 1.0:
            # Direct store for representative layers
            self._cache[cache_idx, 0, :, position, :] = key
            self._cache[cache_idx, 1, :, position, :] = value
        else:
            # Blend into representative layer (magnitude-preserving interpolation)
            existing_k = self._cache[cache_idx, 0, :, position, :]
            existing_v = self._cache[cache_idx, 1, :, position, :]

            # Disentangle existing into magnitude + direction
            existing_k_mag = np.linalg.norm(existing_k, axis=-1, keepdims=True) + 1e-8
            existing_v_mag = np.linalg.norm(existing_v, axis=-1, keepdims=True) + 1e-8

            new_k_mag = np.linalg.norm(key, axis=-1, keepdims=True) + 1e-8
            new_v_mag = np.linalg.norm(value, axis=-1, keepdims=True) + 1e-8

            # Interpolate directions, keep max magnitude
            k_dir = (existing_k / existing_k_mag * (1.0 - weight) +
                     key / new_k_mag * weight)
            v_dir = (existing_v / existing_v_mag * (1.0 - weight) +
                     value / new_v_mag * weight)

            k_dir_mag = np.linalg.norm(k_dir, axis=-1, keepdims=True) + 1e-8
            v_dir_mag = np.linalg.norm(v_dir, axis=-1, keepdims=True) + 1e-8

            # Keep the larger magnitude to preserve information
            self._cache[cache_idx, 0, :, position, :] = (
                k_dir / k_dir_mag * np.maximum(existing_k_mag, new_k_mag)
            )
            self._cache[cache_idx, 1, :, position, :] = (
                v_dir / v_dir_mag * np.maximum(existing_v_mag, new_v_mag)
            )

        self._lengths[layer_idx] = max(self._lengths[layer_idx], position + 1)
        # Also update length for the representative layer so interpolated layers can read it
        if rep_layer != layer_idx:
            self._lengths[rep_layer] = max(self._lengths[rep_layer], position + 1)
        self._max_pos[layer_idx] = max(self._max_pos[layer_idx], position)

    def get(
        self, layer_idx: int, start: int = 0, end: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve K/V for a layer (interpolated if not representative).

        Returns
        -------
        (K, V) each shape ``(num_heads, seq_len, head_dim)``, dtype float16
        """
        if end is None:
            # Use the representative layer's length (interpolated layers share it)
            rep_for_len = self._layer_to_group[layer_idx]
            end = int(max(self._lengths[layer_idx], self._lengths[rep_for_len]))

        rep_layer = self._layer_to_group[layer_idx]
        cache_idx = self._rep_idx[rep_layer]

        k = self._cache[cache_idx, 0, :, start:end, :].copy()
        v = self._cache[cache_idx, 1, :, start:end, :].copy()

        if rep_layer != layer_idx:
            # Apply layer-specific scaling for interpolated layers
            # Slightly scale magnitudes based on layer depth
            depth_factor = 1.0 + 0.01 * (layer_idx - rep_layer)
            k = k * depth_factor
            v = v * depth_factor

        return k.astype(np.float16), v.astype(np.float16)

    @property
    def length(self) -> int:
        return int(self._lengths[0])

    def clear(self) -> None:
        self._cache.fill(0)
        self._lengths.fill(0)
        self._max_pos.fill(0)

    def memory_usage_bytes(self) -> int:
        """Memory used by compressed cache."""
        return self._cache.nbytes

    def memory_savings(self) -> float:
        """Fraction of memory saved vs uncompressed."""
        full_size = (
            self.num_layers * 2 * self.num_heads * self.max_seq_len
            * self.head_dim * np.dtype(self.dtype).itemsize
        )
        return 1.0 - (self.memory_usage_bytes() / max(full_size, 1))

    def __repr__(self) -> str:
        return (
            f"MiniCache(layers={self.num_layers}, reps={len(self._representative_layers)}, "
            f"ratio={self.compression_ratio}x, savings={self.memory_savings():.0%})"
        )

    def bulk_load(
        self, layer_idx: int, keys: np.ndarray, values: np.ndarray, start_pos: int = 0
    ) -> None:
        """Load an entire sequence of KV into the cache at once.

        Used after prefill to populate the cache efficiently.

        Parameters
        ----------
        layer_idx : int
        keys : np.ndarray, shape ``(num_heads, seq_len, head_dim)``
        values : np.ndarray, same shape
        start_pos : int
            Starting position in the cache.
        """
        seq_len = keys.shape[1]
        for pos_offset in range(seq_len):
            pos = start_pos + pos_offset
            if pos >= self.max_seq_len:
                break
            self.update(
                layer_idx,
                keys[:, pos_offset, :],
                values[:, pos_offset, :],
                pos,
            )
