"""VibeBlade RotateKV — Hadamard rotation for outlier-aware 2-bit KV quantization.

Based on: RotateKV: Outlier-Aware Rotation for KV Cache Quantization

Applies a block-diagonal Hadamard rotation to KV vectors before quantization,
smoothing channel-wise outliers that otherwise force coarse-grained scales.
Hadamard matrices are orthogonal (H^T = H^{-1}), so inversion is trivial.

Achieves 3.97x memory reduction with < 0.3 PPL degradation at 2-bit.
"""

from __future__ import annotations

import numpy as np


def _sylvester_hadamard(n: int) -> np.ndarray:
    """Build a Sylvester-constructed Hadamard matrix of size n x n.

    Uses recursive construction: H(2n) = [[H(n), H(n)], [H(n), -H(n)]].
    n must be a power of 2.

    Parameters
    ----------
    n : int
        Size of the Hadamard matrix (must be a power of 2).

    Returns
    -------
    np.ndarray, shape ``(n, n)``, dtype float32
        Orthogonal Hadamard matrix (H @ H^T = n * I, so we normalize by 1/sqrt(n)).
    """
    if n == 1:
        return np.array([[1.0]], dtype=np.float32)

    H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    size = 2
    while size < n:
        H = np.block([[H, H], [H, -H]])
        size *= 2

    # Normalize so H is orthogonal: H @ H^T = I
    return H / np.sqrt(n, dtype=np.float32)


def hadamard_rotation_matrix(dim: int, block_size: int = 64) -> np.ndarray:
    """Build a block-diagonal Hadamard rotation matrix.

    Constructs an orthogonal rotation matrix by placing Hadamard blocks
    along the diagonal. If *dim* is not a multiple of *block_size*, the
    last block is truncated to fit.

    Parameters
    ----------
    dim : int
        Full dimension (e.g., head_dim).
    block_size : int
        Size of each Hadamard block. Rounded up to next power of 2.

    Returns
    -------
    np.ndarray, shape ``(dim, dim)``, dtype float32
        Block-diagonal orthogonal rotation matrix.
    """
    # Round up to next power of 2
    bs = 1
    while bs < block_size:
        bs <<= 1

    block = _sylvester_hadamard(bs)
    R = np.zeros((dim, dim), dtype=np.float32)

    pos = 0
    while pos < dim:
        end = min(pos + bs, dim)
        actual_bs = end - pos
        R[pos:end, pos:end] = block[:actual_bs, :actual_bs]
        pos = end

    return R


def rotate_kv(
    kv: np.ndarray,
    hadamard: np.ndarray,
) -> np.ndarray:
    """Apply Hadamard rotation: R @ kv.

    Parameters
    ----------
    kv : np.ndarray
        Shape ``(num_heads, seq_len, head_dim)`` or ``(seq_len, head_dim)``.
    hadamard : np.ndarray
        Shape ``(head_dim, head_dim)`` — the block-diagonal rotation.

    Returns
    -------
    np.ndarray — same shape as *kv*, rotated.
    """
    if kv.ndim == 3:
        # (num_heads, seq_len, head_dim) — rotate each head independently
        return kv @ hadamard.T
    elif kv.ndim == 2:
        # (seq_len, head_dim)
        return kv @ hadamard.T
    else:
        raise ValueError(f"kv must be 2D or 3D, got {kv.ndim}D")


def inverse_rotate_kv(
    kv_rotated: np.ndarray,
    hadamard: np.ndarray,
) -> np.ndarray:
    """Apply inverse Hadamard rotation: R^T @ kv_rotated.

    Since Hadamard is orthogonal, R^{-1} = R^T.

    Parameters
    ----------
    kv_rotated : np.ndarray
        Rotated KV of shape ``(num_heads, seq_len, head_dim)`` or ``(seq_len, head_dim)``.
    hadamard : np.ndarray
        Shape ``(head_dim, head_dim)`` — the block-diagonal rotation.

    Returns
    -------
    np.ndarray — same shape, un-rotated.
    """
    if kv_rotated.ndim == 3:
        return kv_rotated @ hadamard
    elif kv_rotated.ndim == 2:
        return kv_rotated @ hadamard
    else:
        raise ValueError(f"kv_rotated must be 2D or 3D, got {kv_rotated.ndim}D")


class RotateKVCache:
    """KV cache with Hadamard rotation + asymmetric 2-bit KIVI quantization.

    Applies a block-diagonal Hadamard rotation to key and value vectors
    before quantizing with KIVI's asymmetric 2-bit scheme. The rotation
    smooths channel-wise outliers, enabling lower perplexity at extreme
    quantization levels.

    Keys are rotated + quantized per-channel; values are rotated + quantized
    per-token (following KIVI's asymmetric strategy).

    Parameters
    ----------
    num_layers : int
    num_heads : int
    head_dim : int
    max_seq_len : int
    block_size : int
        Hadamard block size (default 64). Larger blocks provide better
        outlier smoothing but cost more compute.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        block_size: int = 64,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size

        # Pre-compute the block-diagonal Hadamard rotation
        self._hadamard = hadamard_rotation_matrix(head_dim, block_size)

        # Internal KIVI-style packed storage
        from .kv_quant import pack_2bit, unpack_2bit

        self._pack_2bit = pack_2bit
        self._unpack_2bit = unpack_2bit

        packed_dim = (head_dim + 3) // 4
        self._k_packed = np.zeros(
            (num_layers, num_heads, max_seq_len, packed_dim), dtype=np.uint8
        )
        self._v_packed = np.zeros(
            (num_layers, num_heads, max_seq_len, packed_dim), dtype=np.uint8
        )
        self._k_scales = np.zeros(
            (num_layers, num_heads, head_dim), dtype=np.float32
        )
        self._v_scales = np.zeros(
            (num_layers, num_heads, max_seq_len), dtype=np.float32
        )
        # Running min/max for per-channel K quantization
        self._k_mins = np.full(
            (num_layers, num_heads, head_dim), np.inf, dtype=np.float32
        )
        self._k_maxs = np.full(
            (num_layers, num_heads, head_dim), -np.inf, dtype=np.float32
        )
        # Per-token V mins
        self._v_mins = np.zeros(
            (num_layers, num_heads, max_seq_len), dtype=np.float32
        )
        self._lengths = np.zeros(num_layers, dtype=np.int64)

    def _quantize_k(
        self, k_rotated: np.ndarray, layer_idx: int, head_idx: int, position: int
    ) -> None:
        """Quantize rotated key per-channel using running min/max."""
        h = k_rotated  # (head_dim,) in float32

        # Update running min/max (per-channel)
        self._k_mins[layer_idx, head_idx, :] = np.minimum(
            self._k_mins[layer_idx, head_idx, :], h
        )
        self._k_maxs[layer_idx, head_idx, :] = np.maximum(
            self._k_maxs[layer_idx, head_idx, :], h
        )

        k_min = self._k_mins[layer_idx, head_idx, :]
        k_range = np.maximum(self._k_maxs[layer_idx, head_idx, :] - k_min, 1e-8)
        k_scale = k_range / 3.0

        self._k_scales[layer_idx, head_idx, :] = k_scale

        k_q = np.clip(np.round((h - k_min) / k_scale), 0, 3).astype(np.uint8)
        self._k_packed[layer_idx, head_idx, position, :] = self._pack_2bit(k_q.ravel())

    def _quantize_v(
        self, v_rotated: np.ndarray, layer_idx: int, head_idx: int, position: int
    ) -> None:
        """Quantize rotated value per-token."""
        h = v_rotated  # (head_dim,) in float32

        v_min = float(h.min())
        v_max = float(h.max())
        v_range = max(v_max - v_min, 1e-8)
        v_scale = v_range / 3.0

        v_q = np.clip(np.round((h - v_min) / v_scale), 0, 3).astype(np.uint8)
        self._v_packed[layer_idx, head_idx, position, :] = self._pack_2bit(v_q.ravel())
        self._v_scales[layer_idx, head_idx, position] = v_scale
        self._v_mins[layer_idx, head_idx, position] = v_min

    def _dequantize_k(
        self, layer_idx: int, head_idx: int, start: int, end: int
    ) -> np.ndarray:
        """Dequantize key and inverse-rotate. Returns (seq_len, head_dim)."""
        seq_len = end - start
        if seq_len <= 0:
            return np.empty((0, self.head_dim), dtype=np.float16)

        k_packed_slice = self._k_packed[layer_idx, head_idx, start:end, :]
        k_q = self._unpack_2bit(k_packed_slice, seq_len * self.head_dim).reshape(
            seq_len, self.head_dim
        )
        k_scale = self._k_scales[layer_idx, head_idx, :]
        k_min = self._k_mins[layer_idx, head_idx, :]

        # Dequantize
        k_deq = (k_q.astype(np.float32) * k_scale + k_min).astype(np.float16)

        # Inverse Hadamard rotation
        k_restored = inverse_rotate_kv(k_deq.astype(np.float32), self._hadamard)
        return k_restored.astype(np.float16)

    def _dequantize_v(
        self, layer_idx: int, head_idx: int, start: int, end: int
    ) -> np.ndarray:
        """Dequantize value and inverse-rotate. Returns (seq_len, head_dim)."""
        seq_len = end - start
        if seq_len <= 0:
            return np.empty((0, self.head_dim), dtype=np.float16)

        v_packed_slice = self._v_packed[layer_idx, head_idx, start:end, :]
        v_q = self._unpack_2bit(v_packed_slice, seq_len * self.head_dim).reshape(
            seq_len, self.head_dim
        )
        v_mins = self._v_mins[layer_idx, head_idx, start:end, np.newaxis]
        v_scales = self._v_scales[layer_idx, head_idx, start:end, np.newaxis]

        # Dequantize
        v_deq = (v_q.astype(np.float32) * v_scales + v_mins).astype(np.float16)

        # Inverse Hadamard rotation
        v_restored = inverse_rotate_kv(v_deq.astype(np.float32), self._hadamard)
        return v_restored.astype(np.float16)

    def update(
        self, layer_idx: int, key: np.ndarray, value: np.ndarray, position: int
    ) -> None:
        """Insert K/V at position with rotation + 2-bit quantization.

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
            value = value[:, 0, :]

        for h in range(self.num_heads):
            k_h = key[h].astype(np.float32)
            v_h = value[h].astype(np.float32)

            # Apply Hadamard rotation before quantization
            k_rot = (k_h @ self._hadamard.T).astype(np.float32)
            v_rot = (v_h @ self._hadamard.T).astype(np.float32)

            # Quantize rotated vectors
            self._quantize_k(k_rot, layer_idx, h, position)
            self._quantize_v(v_rot, layer_idx, h, position)

        self._lengths[layer_idx] = max(self._lengths[layer_idx], position + 1)

    def get(
        self, layer_idx: int, start: int = 0, end: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve dequantized + inverse-rotated K/V for a layer.

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
            k_out[h] = self._dequantize_k(layer_idx, h, start, end)
            v_out[h] = self._dequantize_v(layer_idx, h, start, end)

        return k_out, v_out

    @property
    def hadamard(self) -> np.ndarray:
        """The block-diagonal Hadamard rotation matrix."""
        return self._hadamard

    @property
    def length(self) -> int:
        return int(self._lengths[0])

    def clear(self) -> None:
        self._k_packed.fill(0)
        self._v_packed.fill(0)
        self._k_scales.fill(0)
        self._v_scales.fill(0)
        self._k_mins.fill(np.inf)
        self._k_maxs.fill(-np.inf)
        self._v_mins.fill(0)
        self._lengths.fill(0)

    def memory_usage_bytes(self) -> int:
        """Memory used by 2-bit packed cache (not scales/rotation matrix)."""
        return self._k_packed.nbytes + self._v_packed.nbytes

    def compression_ratio(self) -> float:
        """Ratio of float16 cache size to packed 2-bit size."""
        fp16_size = (
            self.num_layers * 2 * self.num_heads * self.max_seq_len
            * self.head_dim * 2  # float16 = 2 bytes
        )
        return fp16_size / max(self.memory_usage_bytes(), 1)

    def __repr__(self) -> str:
        return (
            f"RotateKVCache(layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, max_seq={self.max_seq_len}, "
            f"block_size={self.block_size}, "
            f"compression={self.compression_ratio():.1f}x)"
        )
