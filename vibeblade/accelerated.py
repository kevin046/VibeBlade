"""VibeBlade Accelerated Inference — Auto-routing to fastest available backend.

Platform support matrix:
  ┌──────────┬─────────┬─────────┬─────────┬─────────┐
  │ Platform │ CUDA    │ CoreML  │ CPU     │ Fallback│
  ├──────────┼─────────┼─────────┼─────────┼─────────┤
  │ NVIDIA   │ TensorRT│   —     │ ORT     │ NumPy   │
  │ AMD/Intel│   —     │   —     │ ORT     │ NumPy   │
  │ ARM64    │   —     │ macOS   │ ORT     │ NumPy   │
  │ Other    │   —     │   —     │ ORT     │ NumPy   │
  └──────────┴─────────┴─────────┴─────────┴─────────┘

Usage:
    from vibeblade.accelerated import get_accelerator

    accel = get_accelerator(weights, config)
    output = accel.forward(x, ...)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("vibeblade.accelerated")


@dataclass
class AccelConfig:
    """Configuration for accelerated inference."""
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    hidden_dim: int = 4096
    head_dim: int = 128
    intermediate_dim: int = 11008
    num_threads: int = 4
    max_seq_len: int = 2048
    eps: float = 1e-5


@dataclass
class AccelStats:
    """Runtime statistics."""
    backend: str = "numpy"
    total_tokens: int = 0
    total_ms: float = 0.0
    tokens_per_sec: float = 0.0
    ort_cache_hits: int = 0
    ort_cache_misses: int = 0


class AcceleratedBackend:
    """Auto-routing accelerated inference backend.

    Automatically selects the fastest available backend:
    1. ONNX Runtime (GPU or CPU) — uses optimized BLAS kernels
    2. NumPy — universal fallback

    TensorRT is used when ONNX graphs are explicitly exported to TRT
    (requires NVIDIA GPU + CUDA + TensorRT packages).
    """

    def __init__(
        self,
        weights: dict,
        config: AccelConfig,
        backend: Optional[str] = None,  # force specific backend
        num_threads: int = 4,
    ):
        self.weights = weights
        self.config = config
        self.stats = AccelStats()

        # Import ORT ops
        from vibeblade.onnx_backend import ORTOps, detect_providers, _ORT_AVAILABLE
        from vibeblade.tensorrt_backend import is_available as trt_available

        # Determine backend
        if backend:
            self.stats.backend = backend
        elif trt_available():
            self.stats.backend = "tensorrt"
        elif _ORT_AVAILABLE:
            providers = detect_providers()
            if "CUDAExecutionProvider" in providers:
                self.stats.backend = "ort_cuda"
            elif "CoreMLExecutionProvider" in providers:
                self.stats.backend = "ort_coreml"
            else:
                self.stats.backend = "ort_cpu"
        else:
            self.stats.backend = "numpy"

        # Initialize ORT ops (works on all platforms)
        providers = None
        if self.stats.backend == "ort_cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.stats.backend == "ort_coreml":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        self.ops = ORTOps(providers=providers, num_threads=num_threads)

        # Build RoPE cache
        from vibeblade.transformer import build_rope_cache
        self._rope_cos, self._rope_sin = build_rope_cache(
            config.head_dim, config.max_seq_len
        )

        logger.info(
            "AcceleratedBackend: %s (threads=%d, layers=%d)",
            self.stats.backend, num_threads, config.n_layers,
        )

    def forward_layer(
        self,
        x: np.ndarray,
        layer_idx: int,
        kv_cache_k: Optional[np.ndarray],
        kv_cache_v: Optional[np.ndarray],
        position: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one transformer layer with accelerated ops."""
        cos = self._rope_cos[position : position + 1]
        sin = self._rope_sin[position : position + 1]

        return self.ops.forward_layer(
            x, layer_idx, self.weights, cos, sin,
            kv_cache_k, kv_cache_v, position,
            self.config.n_heads, self.config.n_kv_heads, self.config.head_dim,
        )

    def full_decode(
        self,
        token_ids: np.ndarray,
        kv_caches: Optional[dict] = None,
        start_position: int = 0,
    ) -> tuple[np.ndarray, dict]:
        """Run all layers for a single decode step."""
        if kv_caches is None:
            kv_caches = {}

        # Embed
        x = self.weights["token_embd.weight"][token_ids].reshape(1, -1)
        pos = start_position

        # Through all layers
        for i in range(self.config.n_layers):
            k_cache, v_cache = kv_caches.get(i, (None, None))
            x, k_new, v_new = self.forward_layer(x, i, k_cache, v_cache, pos)
            kv_caches[i] = (k_new, v_new)

        # Final norm + output projection
        x = self.ops.rms_norm(x, self.weights["output_norm.weight"], self.config.eps)
        logits = x @ self.weights["output.weight"].T

        return logits, kv_caches

    def benchmark_tokens(self, n_tokens: int = 50) -> dict:
        """Benchmark single-token decode throughput."""
        token_id = np.array([0])
        kv_caches = {}

        # Warmup
        for i in range(3):
            logits, kv_caches = self.full_decode(token_id, kv_caches, start_position=i)

        # Timed run
        t0 = time.time()
        for i in range(n_tokens):
            logits, kv_caches = self.full_decode(
                token_id, kv_caches, start_position=i + 3
            )
        elapsed = time.time() - t0

        tps = n_tokens / elapsed
        self.stats.total_tokens += n_tokens
        self.stats.total_ms += elapsed * 1000
        self.stats.tokens_per_sec = tps

        if hasattr(self.ops, "_cache") and self.ops._cache:
            self.stats.ort_cache_hits = self.ops._cache.hits
            self.stats.ort_cache_misses = self.ops._cache.misses

        return {
            "backend": self.stats.backend,
            "tokens": n_tokens,
            "elapsed_ms": elapsed * 1000,
            "tokens_per_sec": tps,
            "ms_per_token": elapsed * 1000 / n_tokens,
        }

    def summary(self) -> str:
        """Human-readable backend summary."""
        from vibeblade.onnx_backend import platform_info
        from vibeblade.tensorrt_backend import platform_support as trt_info

        info = platform_info()
        trt = trt_info()

        lines = [
            f"Backend: {self.stats.backend}",
            f"Platform: {info['arch']} ({info['system']})",
            f"ORT providers: {info['providers']}",
            f"TensorRT: {trt['reason']}",
        ]
        if self.stats.total_tokens > 0:
            lines.append(
                f"Throughput: {self.stats.tokens_per_sec:.1f} t/s "
                f"({self.stats.total_tokens} tokens)"
            )
        return "\n".join(lines)


def get_accelerator(
    weights: dict,
    n_layers: int = 32,
    n_heads: int = 32,
    n_kv_heads: Optional[int] = None,
    hidden_dim: int = 4096,
    head_dim: int = 128,
    intermediate_dim: int = 11008,
    num_threads: int = 4,
    backend: Optional[str] = None,
) -> AcceleratedBackend:
    """Create an AcceleratedBackend with auto-detected hardware.

    Convenience function — detects the best available backend and
    returns a ready-to-use AcceleratedBackend instance.
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads

    config = AccelConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        hidden_dim=hidden_dim,
        head_dim=head_dim,
        intermediate_dim=intermediate_dim,
    )

    return AcceleratedBackend(
        weights=weights,
        config=config,
        backend=backend,
        num_threads=num_threads,
    )
