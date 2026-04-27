"""VibeBlade GPU Backends — Metal (Apple Silicon) and Vulkan bridge.

Provides automatic backend selection with NumPy fallback:
- Metal: macOS 13+ / Apple Silicon
- Vulkan: Cross-platform (Linux, Windows, Android)
- NumPy: Pure Python fallback (always available)
"""

from __future__ import annotations

import numpy as np

# Backend state
_METAL_AVAILABLE = False
_VULKAN_AVAILABLE = False

try:
    from vibeblade._vibeblade_metal import MetalBackend
    _METAL_AVAILABLE = True
except ImportError:
    pass

try:
    from vibeblade._vibeblade_vulkan import VulkanBackend
    _VULKAN_AVAILABLE = True
except ImportError:
    pass


def available_backends() -> list[str]:
    """Return list of available GPU backend names."""
    backends = ["numpy"]
    if _METAL_AVAILABLE:
        backends.append("metal")
    if _VULKAN_AVAILABLE:
        backends.append("vulkan")
    return backends


class GPUBackend:
    """Unified GPU backend with automatic selection and NumPy fallback.

    Parameters
    ----------
    backend : str
        One of ``"auto"``, ``"metal"``, ``"vulkan"``, ``"numpy"``.
        ``"auto"`` picks the best available: Metal > Vulkan > NumPy.
    device_index : int
        GPU device index (for multi-GPU systems, Vulkan only).
    """

    def __init__(self, backend: str = "auto", device_index: int = 0):
        if backend == "auto":
            if _METAL_AVAILABLE:
                backend = "metal"
            elif _VULKAN_AVAILABLE:
                backend = "vulkan"
            else:
                backend = "numpy"

        self.backend_name = backend
        self._device = None
        self._device_name = "NumPy CPU"

        if backend == "metal":
            if not _METAL_AVAILABLE:
                raise RuntimeError(
                    "Metal backend not available — requires macOS 13+ / Apple Silicon"
                )
            self._device = MetalBackend()
            self._device_name = self._device.device_name()

        elif backend == "vulkan":
            if not _VULKAN_AVAILABLE:
                raise RuntimeError(
                    "Vulkan backend not available — requires Vulkan SDK 1.2+"
                )
            self._device = VulkanBackend(gpu_index=device_index)
            self._device_name = self._device.device_name()

        elif backend == "numpy":
            pass  # Pure Python, no device needed

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: "
                f"{available_backends()}"
            )

    def __repr__(self) -> str:
        return f"GPUBackend(backend={self.backend_name!r}, device={self._device_name!r})"

    @property
    def is_gpu(self) -> bool:
        """True if using a GPU backend (Metal or Vulkan)."""
        return self.backend_name in ("metal", "vulkan")

    # ── Core operations ─────────────────────

    def drelu(self, x: np.ndarray) -> np.ndarray:
        """dReLU activation: max(x, 0)."""
        x = np.asarray(x, dtype=np.float32)
        if self.backend_name == "numpy":
            return np.maximum(x, 0.0)
        if self.backend_name == "metal":
            return self._device.drelu(x, len(x))
        if self.backend_name == "vulkan":
            out = np.empty_like(x)
            self._device.drelu(x, out, len(x))
            return out
        raise RuntimeError(f"Backend {self.backend_name} not supported")

    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation: x * sigmoid(x)."""
        x = np.asarray(x, dtype=np.float32)
        if self.backend_name == "numpy":
            return x * (1.0 / (1.0 + np.exp(-x)))
        if self.backend_name == "metal":
            return self._device.silu(x, len(x))
        if self.backend_name == "vulkan":
            out = np.empty_like(x)
            self._device.silu(x, out, len(x))
            return out
        raise RuntimeError(f"Backend {self.backend_name} not supported")

    def rms_norm(
        self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """RMS normalization."""
        x = np.asarray(x, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
        dim = weight.shape[0]

        if self.backend_name == "numpy":
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
            return x / rms * weight

        if self.backend_name == "metal":
            return self._device.rms_norm(x, weight, eps, dim)

        if self.backend_name == "vulkan":
            out = np.empty_like(x)
            self._device.rms_norm(x, weight, out, eps, dim)
            return out

        raise RuntimeError(f"Backend {self.backend_name} not supported")

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiply: C = A × B."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if self.backend_name == "numpy":
            return a @ b

        if self.backend_name == "metal":
            M, K = a.shape
            _, N = b.shape
            return self._device.matmul(a, b, M, K, N)

        if self.backend_name == "vulkan":
            M, K = a.shape
            _, N = b.shape
            c = np.empty((M, N), dtype=np.float32)
            self._device.matmul(a, b, c, M, K, N)
            return c

        raise RuntimeError(f"Backend {self.backend_name} not supported")

    def rotor_unpack(
        self, packed: np.ndarray, rotor: np.ndarray, n: int
    ) -> np.ndarray:
        """4-bit weight unpack + SO(4) rotation."""
        if self.backend_name == "numpy":
            from vibeblade.quant import unpack_nibbles
            unpacked = unpack_nibbles(packed, n).astype(np.float32)
            group_size = 4
            for i in range(0, n, group_size):
                g = min(group_size, n - i)
                r = rotor[i:i+g] if rotor.ndim == 1 else np.diag(rotor[:g, :g])
                unpacked[i:i+g] *= r[:g]
            return unpacked

        if self.backend_name == "metal":
            return self._device.rotor_unpack(packed, rotor, n)

        # Vulkan: not yet implemented for rotor_unpack
        raise NotImplementedError(
            "rotor_unpack not yet implemented for Vulkan backend"
        )

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax along an axis."""
        if self.backend_name == "numpy":
            e = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e / np.sum(e, axis=axis, keepdims=True)
        # GPU backends: use numpy for softmax (not yet a dedicated kernel)
        return self.softmax(x, axis)


__all__ = [
    "GPUBackend",
    "available_backends",
]
