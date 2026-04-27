"""VibeBlade backend abstraction — transparent numpy ↔ native C++ dispatch.

Usage:
    from vibeblade.backend import get_backend, set_backend, Backend

    # Force native (falls back to numpy if C++ not built)
    set_backend(Backend.NATIVE)

    # Get the current backend module
    bk = get_backend()
    C = bk.gemm(A, B)
"""

from __future__ import annotations

import enum
import importlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Backend(enum.Enum):
    NUMPY = "numpy"
    NATIVE = "native"


_current_backend: Backend = Backend.NATIVE
_native_module: object | None = None


def _load_native() -> object | None:
    """Try to import the native C++ extension. Returns None if unavailable."""
    global _native_module
    if _native_module is not None:
        return _native_module

    try:
        _native_module = importlib.import_module("vibeblade._vibeblade_native")
        return _native_module
    except ImportError:
        return None


def set_backend(backend: Backend) -> None:
    """Set the compute backend. NATIVE falls back to NUMPY if C++ not built."""
    global _current_backend
    if backend == Backend.NATIVE and _load_native() is None:
        import warnings
        warnings.warn(
            "Native C++ backend not available (build with cpp/build_cpp.sh). "
            "Falling back to numpy.",
            RuntimeWarning,
            stacklevel=2,
        )
        _current_backend = Backend.NUMPY
    else:
        _current_backend = backend


def get_backend() -> Backend:
    """Get the current backend enum."""
    return _current_backend


# ════════════════════════════════════════════════════════════════
#  Public API — drop-in replacements that dispatch to numpy or C++
# ════════════════════════════════════════════════════════════════

def gemm(a: np.ndarray, b: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Matrix multiply: C = alpha * A @ B + beta * C."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.gemm(a, b, alpha, beta)
    # numpy fallback
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if b.dtype != np.float32:
        b = b.astype(np.float32)
    return (alpha * (a @ b)).astype(np.float16)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.rms_norm(x, weight, eps)
    # numpy fallback
    if x.dtype == np.float16:
        x = x.astype(np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * weight).astype(np.float16)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: out = x * sigmoid(x)."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.silu(x)
    # numpy fallback
    return (x * (1.0 / (1.0 + np.exp(-x.astype(np.float32))))).astype(np.float16)


def silu_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused SiLU + multiply: out = silu(a) * b."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.silu_mul(a, b)
    sig = 1.0 / (1.0 + np.exp(-a.astype(np.float32)))
    return (a.astype(np.float32) * sig * b.astype(np.float32)).astype(np.float16)


def fused_sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
               scale: float = -1.0) -> np.ndarray:
    """Fused scaled dot-product attention with online softmax."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.fused_sdpa(Q, K, V, scale)
    # numpy fallback
    d = Q.shape[-1]
    if scale < 0:
        scale = 1.0 / np.sqrt(d)
    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    V32 = V.astype(np.float32)
    attn = Q32 @ K32.T * scale
    attn_max = np.max(attn, axis=-1, keepdims=True)
    attn = np.exp(attn - attn_max)
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    return (attn @ V32).astype(np.float16)


def quantize_2bit(x: np.ndarray, axis: int = 0):
    """Quantize float16 to 2-bit. Returns (packed, scales, mins)."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.quantize_2bit(x, axis)
    # numpy fallback
    S, D = x.shape
    if axis == 1:
        mn = x.min(axis=0).astype(np.float32)
        mx = x.max(axis=0).astype(np.float32)
        scale = np.maximum(mx - mn, 1e-8) / 3.0
        x32 = x.astype(np.float32)
        q = np.clip(np.round((x32 - mn) / scale), 0, 3).astype(np.uint8)
    else:
        mn = x.min(axis=1, keepdims=True).astype(np.float32)
        mx = x.max(axis=1, keepdims=True).astype(np.float32)
        scale = np.maximum(mx - mn, 1e-8) / 3.0
        x32 = x.astype(np.float32)
        q = np.clip(np.round((x32 - mn) / scale), 0, 3).astype(np.uint8)
    # Pack 4 values per byte
    flat = q.ravel()
    padded = np.zeros((len(flat) + 3) // 4 * 4, dtype=np.uint8)
    padded[:len(flat)] = flat
    packed = (padded[0::4]) | (padded[1::4] << 2) | (padded[2::4] << 4) | (padded[3::4] << 6)
    scale_out = scale.ravel().astype(np.float32) if axis == 1 else scale.squeeze().astype(np.float32)
    mn_out = mn.ravel().astype(np.float32) if axis == 1 else mn.squeeze().astype(np.float32)
    return packed, scale_out, mn_out


def dequantize_2bit(packed, scales, mins, S: int, D: int, axis: int = 0):
    """Dequantize 2-bit packed back to float16."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.dequantize_2bit(packed, scales, mins, S, D, axis)
    # numpy fallback
    total = S * D
    bytes_needed = (total + 3) // 4
    p = np.zeros(bytes_needed * 4, dtype=np.uint8)
    actual_packed = np.frombuffer(packed, dtype=np.uint8)
    p[:len(actual_packed)] = actual_packed
    q = ((p[0::4]) & 0x3) | ((p[1::4] >> 2) & 0x3) | ((p[2::4] >> 4) & 0x3) | ((p[3::4] >> 6) & 0x3)
    q = q[:total].reshape(S, D).astype(np.float32)
    if axis == 1:
        result = q * scales + mins
    else:
        result = q * scales[:, np.newaxis] + mins[:, np.newaxis]
    return result.astype(np.float16)


def apply_rope(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Apply rotary positional embeddings."""
    if _current_backend == Backend.NATIVE and _native_module is not None:
        return _native_module.apply_rope(x, freqs)
    # numpy fallback
    seq, heads, dim = x.shape
    half_d = dim // 2
    x32 = x.astype(np.float32).copy()
    for s in range(seq):
        for h in range(heads):
            for i in range(half_d):
                fi = s * half_d + i
                cos_v = freqs[fi, 2 * i] if freqs.ndim == 3 else freqs[fi * 2]
                sin_v = freqs[fi, 2 * i + 1] if freqs.ndim == 3 else freqs[fi * 2 + 1]
                base = h * dim
                x0, x1 = x32[s, base + i], x32[s, base + half_d + i]
                x32[s, base + i] = x0 * cos_v - x1 * sin_v
                x32[s, base + half_d + i] = x0 * sin_v + x1 * cos_v
    return x32.astype(np.float16)


# ════════════════════════════════════════════════════════════════
#  Auto-detect native on import
# ════════════════════════════════════════════════════════════════

def _autodetect() -> None:
    """Try native first, fall back to numpy silently."""
    global _current_backend
    if _load_native() is not None:
        _current_backend = Backend.NATIVE

_autodetect()
