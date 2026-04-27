"""VibeBlade KIVI — Tuning-free asymmetric 2-bit KV cache quantization.

Based on: KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (2402.02750)

Keys are quantized per-channel, values per-token. This asymmetric strategy
preserves accuracy better than symmetric quantization because keys exhibit
per-channel variance patterns while values exhibit per-token variance.

Achieves 2.6× memory reduction, enabling 4× larger batch sizes.
"""

from __future__ import annotations

import numpy as np


def quantize_kv_2bit(
    kv: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """2-bit linear quantization along the given axis.

    Maps *kv* to {0, 1, 2, 3} using per-slice (along *axis*) min/max scaling.

    Parameters
    ----------
    kv : np.ndarray
        Input to quantize.
    axis : int
        Axis along which to compute scale (0 = per-token, 1 = per-channel).

    Returns
    -------
    (quantized, scale)
        quantized : uint8 array with values in {0, 1, 2, 3}
        scale : float32 array of per-slice scale factors
    """
    if axis == 0:
        # Per-token: each token gets its own scale → reduce over features → (S,) scales
        kv_min = kv.min(axis=1, keepdims=True)  # (S, 1)
        kv_max = kv.max(axis=1, keepdims=True)  # (S, 1)
        squeeze_axis = 1
    elif axis == 1:
        # Per-channel: each feature gets its own scale → reduce over tokens → (D,) scales
        kv_min = kv.min(axis=0, keepdims=True)  # (1, D)
        kv_max = kv.max(axis=0, keepdims=True)  # (1, D)
        squeeze_axis = 0
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    range_ = kv_max - kv_min
    # Avoid division by zero for constant slices
    range_ = np.where(range_ < 1e-8, 1.0, range_)
    scale = range_ / 3.0  # 4 levels: 0..3

    quantized = np.clip(np.round((kv - kv_min) / scale), 0, 3).astype(np.uint8)
    return quantized, scale.squeeze(axis=squeeze_axis), kv_min.squeeze(axis=squeeze_axis)


def dequantize_kv_2bit(
    quantized: np.ndarray,
    scale: np.ndarray,
    axis: int,
    kv_min: np.ndarray | float | None = None,
) -> np.ndarray:
    """Dequantize 2-bit values back to float16.

    Parameters
    ----------
    quantized : np.ndarray, uint8, values in {0, 1, 2, 3}
    scale : np.ndarray, float32 scale factors
    axis : int (0 = per-token, 1 = per-channel)
    kv_min : optional, min values for offset restoration

    Returns
    -------
    np.ndarray, float16
    """
    min_val = kv_min if kv_min is not None else 0.0
    if axis == 0:
        # Per-token: scale is (S,), min is (S,) → broadcast with newaxis
        return (quantized.astype(np.float16) * scale[:, np.newaxis] + min_val[:, np.newaxis]).astype(np.float16)
    else:
        # Per-channel: scale is (D,), min is (D,) → element-wise with (S,D)
        return (quantized.astype(np.float16) * scale + min_val).astype(np.float16)


def pack_2bit(values: np.ndarray) -> np.ndarray:
    """Pack 2-bit values into uint8 (4 values per byte).

    Parameters
    ----------
    values : np.ndarray, uint8, values in {0, 1, 2, 3}

    Returns
    -------
    np.ndarray, uint8, shape ``(ceil(n / 4),)``
    """
    flat = values.ravel()
    n = flat.size
    out_len = (n + 3) // 4
    # Pad to multiple of 4
    padded = np.zeros(out_len * 4, dtype=np.uint8)
    padded[:n] = flat
    # Pack: byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
    packed = (
        padded[0::4] | (padded[1::4] << 2) | (padded[2::4] << 4) | (padded[3::4] << 6)
    )
    return packed


def unpack_2bit(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack uint8 to 2-bit values.

    Parameters
    ----------
    packed : np.ndarray, uint8
    n : int, number of original 2-bit values

    Returns
    -------
    np.ndarray, uint8, shape ``(n,)``
    """
    flat = packed.ravel()
    out_len = (n + 3) // 4
    valid = flat[:out_len]
    v0 = valid & 0x03
    v1 = (valid >> 2) & 0x03
    v2 = (valid >> 4) & 0x03
    v3 = (valid >> 6) & 0x03
    interleaved = np.zeros(out_len * 4, dtype=np.uint8)
    interleaved[0::4] = v0
    interleaved[1::4] = v1
    interleaved[2::4] = v2
    interleaved[3::4] = v3
    return interleaved[:n]


class KIVICache:
    """KV cache with built-in asymmetric 2-bit quantization.

    Keys are quantized per-channel (axis=1), values per-token (axis=0).
    Stores 2-bit packed uint8 internally, dequantizes on read.

    Parameters
    ----------
    num_layers : int
    num_heads : int
    head_dim : int
    max_seq_len : int
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Packed 2-bit storage: (num_layers, 2, num_heads, max_seq_len, ceil(head_dim/4))
        packed_dim = (head_dim + 3) // 4
        self._k_packed = np.zeros(
            (num_layers, 2, num_heads, max_seq_len, packed_dim), dtype=np.uint8
        )
        # V packed: (num_layers, 2, num_heads, max_seq_len, ceil(head_dim/4))
        # Reuse same layout, index [layer, 0] = K packed, [layer, 1] = V packed

        # Actually let's use separate arrays for clarity
        self._k_packed = np.zeros(
            (num_layers, num_heads, max_seq_len, packed_dim), dtype=np.uint8
        )
        self._v_packed = np.zeros(
            (num_layers, num_heads, max_seq_len, packed_dim), dtype=np.uint8
        )
        # Per-channel scales for K: (num_layers, num_heads, 1, head_dim) -> squeeze to (num_layers, num_heads, head_dim)
        self._k_scales = np.zeros(
            (num_layers, num_heads, head_dim), dtype=np.float32
        )
        # Per-token scales for V: (num_layers, num_heads, max_seq_len)
        self._v_scales = np.zeros(
            (num_layers, num_heads, max_seq_len), dtype=np.float32
        )

        self._lengths = np.zeros(num_layers, dtype=np.int64)

    def update(
        self, layer_idx: int, key: np.ndarray, value: np.ndarray, position: int
    ) -> None:
        """Insert K/V at position with 2-bit quantization.

        Parameters
        ----------
        layer_idx : int
        key : np.ndarray, shape ``(num_heads, head_dim)`` or ``(num_heads, 1, head_dim)``
        value : np.ndarray, same shape
        position : int
        """
        if position >= self.max_seq_len:
            raise IndexError(f"Position {position} >= max_seq_len {self.max_seq_len}")

        if key.ndim == 3:
            key = key[:, 0, :]
            value = value[:, 0, :]  # type: ignore[assignment]

        for h in range(self.num_heads):
            k_h = key[h].astype(np.float32)  # (head_dim,)
            v_h = value[h].astype(np.float32)  # (head_dim,)

            # Quantize K per-channel using running global min/max
            if not hasattr(self, '_k_mins'):
                self._k_mins = np.full(
                    (self.num_layers, self.num_heads, self.head_dim), np.inf, dtype=np.float32
                )
                self._k_maxs = np.full(
                    (self.num_layers, self.num_heads, self.head_dim), -np.inf, dtype=np.float32
                )
            # Update running min/max
            self._k_mins[layer_idx, h, :] = np.minimum(self._k_mins[layer_idx, h, :], k_h)
            self._k_maxs[layer_idx, h, :] = np.maximum(self._k_maxs[layer_idx, h, :], k_h)

            # Compute per-channel scale from running statistics
            k_range = np.maximum(self._k_maxs[layer_idx, h, :] - self._k_mins[layer_idx, h, :], 1e-8)
            k_scale = k_range / 3.0
            k_min = self._k_mins[layer_idx, h, :]
            self._k_scales[layer_idx, h, :] = k_scale

            # Quantize this token's K
            k_q = np.clip(np.round((k_h - k_min) / k_scale), 0, 3).astype(np.uint8)
            self._k_packed[layer_idx, h, position, :] = pack_2bit(k_q.ravel())

            # Quantize V per-token: single scale for the entire head_dim vector
            v_min = float(v_h.min())
            v_max = float(v_h.max())
            v_range = max(v_max - v_min, 1e-8)
            v_scale = v_range / 3.0  # scalar
            v_q = np.clip(np.round((v_h - v_min) / v_scale), 0, 3).astype(np.uint8)
            self._v_packed[layer_idx, h, position, :] = pack_2bit(v_q.ravel())
            self._v_scales[layer_idx, h, position] = v_scale
            # Store v_min for dequantization
            if not hasattr(self, '_v_mins'):
                self._v_mins = np.zeros(
                    (self.num_layers, self.num_heads, self.max_seq_len), dtype=np.float32
                )
            self._v_mins[layer_idx, h, position] = v_min

        self._lengths[layer_idx] = max(self._lengths[layer_idx], position + 1)

    def get(
        self, layer_idx: int, start: int = 0, end: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve dequantized K/V for a layer.

        Returns
        -------
        (K, V) each shape ``(num_heads, seq_len, head_dim)``, dtype float16
        """
        if end is None:
            end = int(self._lengths[layer_idx])

        seq_len = end - start
        if seq_len <= 0:
            return (
                np.empty((self.num_heads, 0, self.head_dim), dtype=np.float16),
                np.empty((self.num_heads, 0, self.head_dim), dtype=np.float16),
            )

        k_out = np.zeros(
            (self.num_heads, seq_len, self.head_dim), dtype=np.float16
        )
        v_out = np.zeros(
            (self.num_heads, seq_len, self.head_dim), dtype=np.float16
        )

        for h in range(self.num_heads):
            # Dequantize K: per-channel
            k_packed_slice = self._k_packed[layer_idx, h, start:end, :]
            k_q = unpack_2bit(k_packed_slice, seq_len * self.head_dim).reshape(
                seq_len, self.head_dim
            )
            k_scale = self._k_scales[layer_idx, h, :]  # (head_dim,)
            k_min = self._k_mins[layer_idx, h, :] if hasattr(self, '_k_mins') else np.zeros(self.head_dim)
            k_out[h] = dequantize_kv_2bit(k_q, k_scale, axis=1, kv_min=k_min)

            # Dequantize V: per-token (scalar scale + min)
            v_packed_slice = self._v_packed[layer_idx, h, start:end, :]
            v_q = unpack_2bit(v_packed_slice, seq_len * self.head_dim).reshape(
                seq_len, self.head_dim
            )
            if hasattr(self, '_v_mins'):
                v_mins = self._v_mins[layer_idx, h, start:end, np.newaxis]  # (seq_len, 1)
                v_scales = self._v_scales[layer_idx, h, start:end, np.newaxis]  # (seq_len, 1)
            else:
                v_mins = 0.0
                v_scales = self._v_scales[layer_idx, h, start:end, np.newaxis]
            v_out[h] = (v_q.astype(np.float32) * v_scales + v_mins).astype(np.float16)

        return k_out, v_out

    @property
    def length(self) -> int:
        return int(self._lengths[0])

    def clear(self) -> None:
        self._k_packed.fill(0)
        self._v_packed.fill(0)
        self._k_scales.fill(0)
        self._v_scales.fill(0)
        self._lengths.fill(0)

    def memory_usage_bytes(self) -> int:
        """Memory used by 2-bit packed cache (not scales)."""
        return self._k_packed.nbytes + self._v_packed.nbytes + self._k_scales.nbytes + self._v_scales.nbytes

    def compression_ratio(self) -> float:
        """Ratio of float16 cache size to this cache size."""
        fp16_size = (
            self.num_layers * 2 * self.num_heads * self.max_seq_len * self.head_dim * 2
        )
        return fp16_size / max(self.memory_usage_bytes(), 1)

    def __repr__(self) -> str:
        return (
            f"KIVICache(layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, max_seq={self.max_seq_len}, "
            f"compression={self.compression_ratio():.1f}x)"
        )
