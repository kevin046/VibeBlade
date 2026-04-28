"""VibeBlade KV Cache — Memory-efficient key-value cache for transformer inference.

Supports two storage modes:
  - float16 (default): fast, 2 bytes per value
  - Q4_0 quantized: ~0.56 bytes per value (3.5x smaller), slight quality loss
"""

from __future__ import annotations

import numpy as np


def _quant_q4_0(arr: np.ndarray) -> np.ndarray:
    """Quantize float32/f16 array to Q4_0 packed format.

    For every block of 32 values:
      - 2 bytes: f16 scale (= max_abs / 7)
      - 16 bytes: packed 4-bit values (nibble = round(x / scale + 8), clamped [0, 15])

    Output shape: (*batch_dims, ceil(kv_dim / 32) * 18)
    """
    block_size = 32
    orig_shape = arr.shape
    kv_dim = arr.shape[-1]
    n_blocks = (kv_dim + block_size - 1) // block_size
    pad = n_blocks * block_size - kv_dim

    if pad > 0:
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)])

    # Reshape last dim into (n_blocks, block_size)
    blocks = arr.reshape(*orig_shape[:-1], n_blocks, block_size)

    # Scale per block
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    scale = np.where(amax > 1e-7, amax / 7.0, 1.0).astype(np.float16)

    # Quantize
    qi = np.clip(
        np.round(blocks / scale.astype(np.float32) + 8.0), 0, 15
    ).astype(np.uint8)

    # Pack nibbles: even index → low nibble, odd index → high nibble
    packed = qi[..., 0::2] | (qi[..., 1::2] << 4)

    # Output: scale (2 bytes) + packed (16 bytes) = 18 bytes per block
    out = np.empty(*orig_shape[:-1], n_blocks * 18, dtype=np.uint8)
    scale_flat = scale.reshape(*orig_shape[:-1], n_blocks, 2)
    packed_flat = packed.reshape(*orig_shape[:-1], n_blocks * 16)
    out[..., :n_blocks * 2] = scale_flat.reshape(*orig_shape[:-1], n_blocks * 2).view(np.uint8)
    out[..., n_blocks * 2:] = packed_flat
    return out


def _dequant_q4_0(packed: np.ndarray, kv_dim: int) -> np.ndarray:
    """Dequantize Q4_0 packed data back to float32.

    Args:
        packed: shape (*batch_dims, n_blocks * 18) uint8
        kv_dim: original dimension (may be < n_blocks * 32)

    Returns:
        float32 array with shape (*batch_dims, kv_dim)
    """
    block_size = 32
    n_blocks = (kv_dim + block_size - 1) // block_size

    # Reshape to separate scale and data
    out_shape = packed.shape[:-1]
    flat = packed.reshape(*out_shape, n_blocks, 18)

    # Extract scale (first 2 bytes per block)
    scale = flat[..., :2].copy().view(np.float16).astype(np.float32)

    # Extract packed nibbles (bytes 2-17 per block)
    nibbles = flat[..., 2:]  # (..., n_blocks, 16)

    # Unpack: low nibble (even), high nibble (odd)
    lo = (nibbles & 0x0F).astype(np.float32)
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32)

    # Interleave: lo[0], hi[0], lo[1], hi[1], ...
    values = np.empty(*out_shape, n_blocks, block_size, dtype=np.float32)
    values[..., 0::2] = lo
    values[..., 1::2] = hi

    # Dequantize
    out = (values - 8.0) * scale
    return out.reshape(*out_shape, n_blocks * block_size)[..., :kv_dim]


class KVCache:
    """Ring-buffer KV cache for transformer layers.

    Supports optional Q4_0 quantization for ~3.5x memory reduction.
    Quantization is lossy but the KV cache is less sensitive to precision
    than weights (paper: "K-V Cache Quantization" — GPTQ-style).

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length.
        dtype: Storage dtype for non-quantized mode (default float16).
        quantize: If True, store KV values in Q4_0 format (50% less memory than
                  float16, 7x less than float32).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype=np.float16,
        quantize: bool = False,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.quantize = quantize

        kv_dim = num_heads * head_dim

        if quantize:
            # Q4_0 storage: ~0.56 bytes per value vs 2 bytes for float16
            n_blocks = (kv_dim + 31) // 32
            bytes_per_dim = n_blocks * 18
            self.k_cache = np.zeros(
                (num_layers, max_seq_len, bytes_per_dim), dtype=np.uint8
            )
            self.v_cache = np.zeros(
                (num_layers, max_seq_len, bytes_per_dim), dtype=np.uint8
            )
            self._kv_dim = kv_dim
        else:
            # Standard float16/float32 storage
            self.k_cache = np.zeros(
                (num_layers, 2, num_heads, max_seq_len, head_dim), dtype=dtype
            )
            self.v_cache = None  # share k_cache buffer (dim 1: 0=K, 1=V)
            self._kv_dim = kv_dim

        self._lengths = np.zeros(num_layers, dtype=np.int64)

    def update(
        self, layer_idx: int, key: np.ndarray, value: np.ndarray,
        position: int,
    ) -> None:
        """Insert K/V pair at position for a layer.

        key shape: (num_heads, 1, head_dim) or (num_heads, head_dim)
        value shape: same as key
        """
        if position >= self.max_seq_len:
            raise IndexError(
                f"Position {position} exceeds max_seq_len {self.max_seq_len}"
            )
        if key.ndim == 2:
            key = key[:, np.newaxis, :]
            value = value[:, np.newaxis, :]

        if self.quantize:
            # Quantize and store as packed uint8
            # Flatten heads: (num_heads, seq, head_dim) → (seq, num_heads * head_dim)
            k_flat = key.transpose(1, 0, 2).reshape(1, self._kv_dim)
            v_flat = value.transpose(1, 0, 2).reshape(1, self._kv_dim)
            self.k_cache[layer_idx, position] = _quant_q4_0(k_flat)
            self.v_cache[layer_idx, position] = _quant_q4_0(v_flat)
        else:
            self.k_cache[layer_idx, 0, :, position:position + 1, :] = key
            self.k_cache[layer_idx, 1, :, position:position + 1, :] = value

        self._lengths[layer_idx] = max(self._lengths[layer_idx], position + 1)

    def get(
        self, layer_idx: int, start: int = 0, end: int = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve cached K/V for a layer.

        Returns (key, value) each with shape (num_heads, seq_len, head_dim).
        """
        if end is None:
            end = int(self._lengths[layer_idx])

        if self.quantize:
            k_packed = self.k_cache[layer_idx, start:end]  # (seq, bytes)
            v_packed = self.v_cache[layer_idx, start:end]
            k_flat = _dequant_q4_0(k_packed, self._kv_dim)  # (seq, kv_dim)
            v_flat = _dequant_q4_0(v_packed, self._kv_dim)
            seq_len = k_flat.shape[0]
            k = k_flat.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
            v = v_flat.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
            return k, v
        else:
            k = self.k_cache[layer_idx, 0, :, start:end, :]
            v = self.k_cache[layer_idx, 1, :, start:end, :]
            return k, v

    @property
    def length(self) -> int:
        """Current cache length (tokens stored)."""
        return int(self._lengths[0])

    def clear(self) -> None:
        """Reset cache to empty."""
        self.k_cache.fill(0)
        if self.v_cache is not None:
            self.v_cache.fill(0)
        self._lengths.fill(0)

    def memory_usage_bytes(self) -> int:
        """Return memory usage in bytes."""
        total = self.k_cache.nbytes
        if self.v_cache is not None:
            total += self.v_cache.nbytes
        elif self.k_cache.ndim == 5:
            # Unified buffer: k_cache already holds both K and V
            pass
        return total

    def memory_saved_pct(self) -> float:
        """Percent memory saved vs float16 baseline."""
        f16_size = (
            2 * self.num_layers * self.max_seq_len
            * self.num_heads * self.head_dim * 2  # 2 for K+V, 2 for float16
        )
        if f16_size == 0:
            return 0.0
        return (1.0 - self.memory_usage_bytes() / f16_size) * 100.0
