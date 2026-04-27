"""VibeBlade RotorQuant — SO(4) rotation-based 4-bit quantization for weight compression."""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Nibble packing
# ---------------------------------------------------------------------------

def pack_nibbles(values: np.ndarray) -> np.ndarray:
    """Pack an array of 4-bit integers (0–15) into a uint8 array.

    Each byte holds two nibbles: the **low** nibble stores ``values[2*i]`` and
    the **high** nibble stores ``values[2*i + 1]`` (when it exists).

    Parameters
    ----------
    values : np.ndarray
        1-D array of integers in the range [0, 15].  Shape ``(n,)``.

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(ceil(n / 2),)``.
    """
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 1:
        raise ValueError("pack_nibbles expects a 1-D array")
    if values.size > 0 and np.any(values > 15):
        raise ValueError("pack_nibbles: all values must be in 0..15")

    n = values.size
    out_len = math.ceil(n / 2)
    # Pad one element if n is odd so we can vectorise cleanly.
    padded = np.zeros(out_len * 2, dtype=np.uint8)
    padded[:n] = values

    low = padded[0::2]       # values at even positions → low nibble
    high = padded[1::2]      # values at odd  positions → high nibble
    packed = (high << 4) | low
    return packed


def unpack_nibbles(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack a uint8 array produced by :func:`pack_nibbles` back to 4-bit values.

    Parameters
    ----------
    packed : np.ndarray
        uint8 array of shape ``(ceil(n / 2),)``.
    n : int
        Number of original 4-bit values to recover.

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(n,)``.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 1:
        raise ValueError("unpack_nibbles expects a 1-D array")
    expected_bytes = math.ceil(n / 2)
    if packed.size < expected_bytes:
        raise ValueError(
            f"unpack_nibbles: need at least {expected_bytes} bytes for n={n}, "
            f"got {packed.size}"
        )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    # Interleave: [low0, high0, low1, high1, ...]
    out = np.zeros(expected_bytes * 2, dtype=np.uint8)
    out[0::2] = low
    out[1::2] = high
    return out[:n]


# ---------------------------------------------------------------------------
# SO(4) rotor construction
# ---------------------------------------------------------------------------

def build_so4_rotor(weights: np.ndarray) -> np.ndarray:
    """Build a block-diagonal SO(4) Givens rotation for a 4-element weight group.

    For the two pairs ``(w0, w1)`` and ``(w2, w3)``, a 2×2 Givens rotation is
    constructed whose angle maximises the dynamic range of the rotated pair
    before quantisation.  The angle is ``θ = atan2(b, a)`` so that the first
    component of each pair becomes ``sqrt(a² + b²)`` (maximum magnitude).

    The two 2×2 blocks are combined into a 4×4 block-diagonal rotation matrix.

    Parameters
    ----------
    weights : np.ndarray
        1-D array of shape ``(4,)``.

    Returns
    -------
    np.ndarray
        4×4 float32 orthogonal matrix with determinant ≈ 1.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (4,):
        raise ValueError(f"build_so4_rotor expects shape (4,), got {w.shape}")

    R = np.eye(4, dtype=np.float32)

    for i, j in [(0, 1), (2, 3)]:
        a, b = float(w[i]), float(w[j])
        if abs(a) < 1e-12 and abs(b) < 1e-12:
            continue
        theta = math.atan2(b, a)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R[i, i] = cos_t
        R[i, j] = sin_t
        R[j, i] = -sin_t
        R[j, j] = cos_t

    return R


# ---------------------------------------------------------------------------
# 4-bit quantisation / dequantisation
# ---------------------------------------------------------------------------

def quantize_4bit(
    weights: np.ndarray,
    group_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise float32 weights to 4-bit using SO(4) rotor rotation.

    The weight vector is split into groups of *group_size*.  Each group is
    further split into sub-groups of 4, an SO(4) rotation is applied per
    sub-group, then each rotated sub-group is affine-scaled to the integer
    range [0, 15] and packed into nibbles.

    Parameters
    ----------
    weights : np.ndarray
        1-D float32 weight vector.  Length must be a multiple of *group_size*.
    group_size : int
        Number of weights per quantisation group.  Must be divisible by 4.

    Returns
    -------
    packed : np.ndarray
        uint8 nibble-packed codes.  Shape ``(ceil(n / 2),)`` where *n* is the
        total number of weights.
    scales : np.ndarray
        float32 array of shape ``(num_groups, 2, sub_groups)`` where
        ``sub_groups = group_size // 4``.  ``scales[g, 0, s]`` is the minimum
        of the *s*-th sub-group after rotation, and ``scales[g, 1, s]`` is the
        quantisation step size ``(max - min) / 15``.
    rotors : np.ndarray
        float32 array of shape ``(num_groups, sub_groups, 4, 4)`` — one SO(4)
        rotation matrix per sub-group.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 1:
        raise ValueError("quantize_4bit expects a 1-D weight array")
    if group_size % 4 != 0:
        raise ValueError("group_size must be divisible by 4")
    if w.size % group_size != 0:
        raise ValueError(
            f"Weight length ({w.size}) must be divisible by group_size ({group_size})"
        )

    num_groups = w.size // group_size
    sub_groups = group_size // 4

    all_packed: list[np.ndarray] = []
    all_scales = np.empty((num_groups, 2, sub_groups), dtype=np.float32)
    all_rotors = np.empty((num_groups, sub_groups, 4, 4), dtype=np.float32)

    for g in range(num_groups):
        group = w[g * group_size : (g + 1) * group_size]   # (group_size,)
        group_4d = group.reshape(sub_groups, 4)            # (sub_groups, 4)
        quantised = np.empty((sub_groups, 4), dtype=np.uint8)

        for s in range(sub_groups):
            R = build_so4_rotor(group_4d[s])
            rotated = R @ group_4d[s]
            all_rotors[g, s] = R

            lo = float(rotated.min())
            hi = float(rotated.max())
            rng = hi - lo

            if rng < 1e-12:
                # Constant sub-group — map everything to the midpoint (8).
                quantised[s] = np.uint8(8)
                all_scales[g, 0, s] = float(rotated[0])  # the constant value
                all_scales[g, 1, s] = 0.0
            else:
                scale = rng / 15.0
                all_scales[g, 0, s] = lo
                all_scales[g, 1, s] = scale
                quantised[s] = np.clip(
                    np.round((rotated - lo) / scale), 0, 15
                ).astype(np.uint8)

        all_packed.append(pack_nibbles(quantised.reshape(-1)))

    packed_all = np.concatenate(all_packed)
    return packed_all, all_scales, all_rotors


def dequantize_4bit(
    packed: np.ndarray,
    scales: np.ndarray,
    rotors: np.ndarray,
    n: int,
) -> np.ndarray:
    """Reconstruct float32 weights from their 4-bit RotorQuant encoding.

    This is the inverse of :func:`quantize_4bit`.

    Parameters
    ----------
    packed : np.ndarray
        uint8 nibble-packed codes (output of :func:`quantize_4bit`).
    scales : np.ndarray
        float32 array of shape ``(num_groups, 2, sub_groups)`` encoding
        ``(lo, scale)`` pairs per sub-group.
    rotors : np.ndarray
        float32 array of shape ``(num_groups, sub_groups, 4, 4)``.
    n : int
        Original number of weights.

    Returns
    -------
    np.ndarray
        float32 array of shape ``(n,)`` — reconstructed weights.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)
    rotors = np.asarray(rotors, dtype=np.float32)

    num_groups = rotors.shape[0]
    sub_groups = rotors.shape[1]
    group_size = sub_groups * 4

    total_elements = num_groups * group_size
    if total_elements != n:
        raise ValueError(
            f"Mismatch: num_groups({num_groups}) * group_size({group_size})"
            f" = {total_elements} != n({n})"
        )

    unpacked = unpack_nibbles(packed, n).reshape(num_groups, sub_groups, 4)
    lo = scales[:, 0, :]      # (num_groups, sub_groups)
    scale = scales[:, 1, :]    # (num_groups, sub_groups)

    out = np.empty((num_groups, sub_groups, 4), dtype=np.float32)

    for g in range(num_groups):
        for s in range(sub_groups):
            q = unpacked[g, s].astype(np.float32)
            if scale[g, s] < 1e-12:
                # Constant sub-group: every value was the same.
                unrotated = np.full(4, lo[g, s], dtype=np.float32)
            else:
                unrotated = lo[g, s] + q * scale[g, s]
            # Inverse rotation: R^T (orthogonal matrix inverse is transpose).
            R_inv = rotors[g, s].T
            out[g, s] = R_inv @ unrotated

    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def quantization_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Compute root-mean-square error between original and reconstructed weights.

    Parameters
    ----------
    original : np.ndarray
        Reference weight vector.
    reconstructed : np.ndarray
        Reconstructed weight vector.

    Returns
    -------
    float
        RMS error.
    """
    orig = np.asarray(original, dtype=np.float32).ravel()
    recon = np.asarray(reconstructed, dtype=np.float32).ravel()
    if orig.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: original {orig.shape} vs reconstructed {recon.shape}"
        )
    return float(np.sqrt(np.mean((orig - recon) ** 2)))
