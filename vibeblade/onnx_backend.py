"""VibeBlade ONNX Runtime Backend — Cross-platform optimized inference.

Uses ONNX Runtime's optimized kernels for individual transformer operations
(matmul, rms_norm, silu, softmax). Supports ARM64, x86_64 (AMD/Intel), and
NVIDIA GPUs via CUDA execution provider.

Fallback chain: CUDA → CoreML (ARM macOS) → CPU (all platforms) → NumPy

No hand-built fused graphs — each op is a trivially simple ONNX model
that's impossible to get wrong.
"""

from __future__ import annotations

import logging
import platform
from typing import Optional

import numpy as np

logger = logging.getLogger("vibeblade.onnx")

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.info("onnxruntime not installed — using NumPy fallback")


def detect_providers() -> list[str]:
    """Detect available ORT execution providers for this platform."""
    if not _ORT_AVAILABLE:
        return []

    available = ort.get_available_providers()
    preferred = []

    # Priority: CUDA > TensorRT > CoreML > CPU
    for p in [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if p in available:
            preferred.append(p)

    return preferred


def platform_info() -> dict:
    """Return platform detection info."""
    return {
        "arch": platform.machine(),  # x86_64, aarch64, arm64
        "system": platform.system(),  # Linux, Darwin, Windows
        "ort_available": _ORT_AVAILABLE,
        "providers": detect_providers() if _ORT_AVAILABLE else [],
    }


# ── Tiny ONNX graph builders (one per op) ─────────────────────────────────

def _build_gemm_graph(in_dim: int, out_dim: int) -> bytes:
    """Build y = x @ W^T (Gemm with transB)."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        raise RuntimeError("onnx package required — pip install onnx")

    x_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", in_dim])
    w_in = helper.make_tensor_value_info("W", TensorProto.FLOAT, [out_dim, in_dim])
    y_out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", out_dim])

    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "W", "b"], ["y"], transB=1)],
        "gemm", [x_in, w_in], [y_out],
        [helper.make_tensor("b", TensorProto.FLOAT, [out_dim], [0.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model.SerializeToString()


def _build_rms_norm_graph(dim: int, eps: float = 1e-5) -> bytes:
    """Build RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        raise RuntimeError("onnx package required")

    x_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", dim])
    w_in = helper.make_tensor_value_info("w", TensorProto.FLOAT, [dim])
    y_out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", dim])

    # Use ReduceMean on axes input (opset 18+ style)
    axes_val = helper.make_tensor("axes", TensorProto.INT64, [1], [-1])
    eps_val = helper.make_tensor("eps", TensorProto.FLOAT, [], [eps])

    nodes = [
        helper.make_node("Mul", ["x", "x"], ["x2"]),
        helper.make_node("ReduceMean", ["x2", "axes"], ["mean_sq"], keepdims=1),
        helper.make_node("Add", ["mean_sq", "eps"], ["var"]),
        helper.make_node("Sqrt", ["var"], ["rms"]),
        helper.make_node("Div", ["x", "rms"], ["normed"]),
        helper.make_node("Mul", ["normed", "w"], ["y"]),
    ]

    graph = helper.make_graph(nodes, "rms_norm", [x_in, w_in], [y_out], [axes_val, eps_val])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    return model.SerializeToString()


def _build_silu_graph(dim: int) -> bytes:
    """Build SiLU: y = x * sigmoid(x)."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        raise RuntimeError("onnx package required")

    x_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", dim])
    y_out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", dim])

    nodes = [
        helper.make_node("Sigmoid", ["x"], ["sig"]),
        helper.make_node("Mul", ["x", "sig"], ["y"]),
    ]

    graph = helper.make_graph(nodes, "silu", [x_in], [y_out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model.SerializeToString()


def _build_softmax_graph(dim: int) -> bytes:
    """Build Softmax along last axis."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        raise RuntimeError("onnx package required")

    x_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", dim])
    y_out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", dim])

    graph = helper.make_graph(
        [helper.make_node("Softmax", ["x"], ["y"], axis=-1)],
        "softmax", [x_in], [y_out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model.SerializeToString()


# ── Session cache (one session per unique graph shape) ────────────────────


class _SessionCache:
    """Cache ORT sessions keyed by (op_name, shape_hash)."""

    def __init__(self, providers: list[str], num_threads: int = 4):
        self._cache: dict[tuple, ort.InferenceSession] = {}
        self._providers = providers
        self.opts = ort.SessionOptions()
        self.opts.intra_op_num_threads = num_threads
        self.opts.inter_op_num_threads = num_threads
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple, graph_fn) -> ort.InferenceSession:
        if key in self._cache:
            self.hits += 1
            return self._cache[key]

        self.misses += 1
        onnx_bytes = graph_fn()
        # Validate then strip shapes to avoid dynamic dim issues
        import onnx
        model = onnx.load_from_string(onnx_bytes)
        onnx.checker.check_model(model)
        for inp in model.graph.input:
            inp.type.tensor_type.ClearField("shape")
        for out in model.graph.output:
            out.type.tensor_type.ClearField("shape")
        onnx_bytes = model.SerializeToString()

        session = ort.InferenceSession(onnx_bytes, self.opts, providers=self._providers)
        self._cache[key] = session
        return session


# ── ORT-accelerated operations ────────────────────────────────────────────


class ORTOps:
    """ONNX Runtime accelerated operations with NumPy fallback.

    Each method tries ORT first, falls back to NumPy if ORT unavailable
    or if the operation fails for any reason.
    """

    def __init__(self, providers: Optional[list[str]] = None, num_threads: int = 4):
        if not _ORT_AVAILABLE:
            self.enabled = False
            self._cache = None
            logger.info("ORT ops disabled (onnxruntime not installed)")
            return

        self.enabled = True
        self._providers = providers or detect_providers()
        self._cache = _SessionCache(self._providers, num_threads)
        logger.info(
            "ORT ops enabled with providers=%s, threads=%d",
            self._providers, num_threads,
        )

    @property
    def provider(self) -> str:
        if not self.enabled or not self._providers:
            return "numpy"
        return self._providers[0]

    def gemm(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """y = x @ W^T with optional ORT acceleration."""
        if not self.enabled:
            return x @ W.T

        try:
            x = x.astype(np.float32)
            W = W.astype(np.float32)
            key = ("gemm", x.shape[1], W.shape[0])
            sess = self._cache.get(key, lambda: _build_gemm_graph(x.shape[1], W.shape[0]))
            (y,) = sess.run(None, {"x": x, "W": W})
            return y
        except Exception:
            return x @ W.T

    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """RMSNorm with optional ORT acceleration."""
        if not self.enabled:
            return self._rms_norm_numpy(x, weight, eps)

        try:
            x = x.astype(np.float32)
            weight = weight.astype(np.float32)
            key = ("rms_norm", x.shape[-1])
            sess = self._cache.get(key, lambda: _build_rms_norm_graph(x.shape[-1], eps))
            (y,) = sess.run(None, {"x": x.reshape(1, -1), "w": weight})
            return y.reshape(x.shape)
        except Exception:
            return self._rms_norm_numpy(x, weight, eps)

    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation with optional ORT acceleration."""
        if not self.enabled:
            return x * (1.0 / (1.0 + np.exp(-x)))

        try:
            x = x.astype(np.float32)
            key = ("silu", x.shape[-1])
            sess = self._cache.get(key, lambda: _build_silu_graph(x.shape[-1]))
            (y,) = sess.run(None, {"x": x.reshape(1, -1)})
            return y.reshape(x.shape)
        except Exception:
            return x * (1.0 / (1.0 + np.exp(-x)))

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax with optional ORT acceleration (last axis only)."""
        if axis != -1 and axis != x.ndim - 1:
            # ORT softmax only handles last axis; fall back for others
            return self._softmax_numpy(x, axis)

        if not self.enabled:
            return self._softmax_numpy(x, axis)

        try:
            x = x.astype(np.float32)
            key = ("softmax", x.shape[-1])
            sess = self._cache.get(key, lambda: _build_softmax_graph(x.shape[-1]))
            (y,) = sess.run(None, {"x": x.reshape(-1, x.shape[-1])})
            return y.reshape(x.shape)
        except Exception:
            return self._softmax_numpy(x, axis)

    # ── NumPy fallbacks ──

    @staticmethod
    def _rms_norm_numpy(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        x = x.astype(np.float32)
        ms = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(ms + eps) * weight

    @staticmethod
    def _softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x.astype(np.float32)
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    # ── Forward pass using ORT-accelerated ops ──

    def forward_layer(
        self,
        x: np.ndarray,
        layer_idx: int,
        weights: dict,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        kv_cache_k: Optional[np.ndarray],
        kv_cache_v: Optional[np.ndarray],
        position: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one transformer layer using ORT-accelerated ops.

        Mirrors transformer.forward_token exactly, using ORT for matmul/norm/silu.
        Falls back to NumPy for any op that fails.
        """
        from vibeblade.transformer import rope, attention

        prefix = f"blk.{layer_idx}"

        # Pre-attention RMSNorm
        h = self.rms_norm(x, weights[f"{prefix}.attn_norm.weight"])

        # QKV projections (ORT-accelerated matmul)
        q = self.gemm(h, weights[f"{prefix}.attn_q.weight"])
        k_new = self.gemm(h, weights[f"{prefix}.attn_k.weight"])
        v_new = self.gemm(h, weights[f"{prefix}.attn_v.weight"])

        # Reshape + RoPE (same as forward_token)
        q = q.reshape(1, n_heads, head_dim)
        k_new = k_new.reshape(1, n_kv_heads, head_dim)
        q = rope(q, rope_cos, rope_sin).reshape(1, n_heads * head_dim)
        k_new = rope(k_new, rope_cos, rope_sin).reshape(1, n_kv_heads * head_dim)

        # KV cache (same as forward_token)
        if kv_cache_k is not None:
            k_full = np.concatenate([kv_cache_k, k_new.reshape(n_kv_heads, 1, head_dim)], axis=1)
            v_full = np.concatenate([kv_cache_v, v_new.reshape(n_kv_heads, 1, head_dim)], axis=1)
        else:
            k_full = k_new.reshape(n_kv_heads, 1, head_dim)
            v_full = v_new.reshape(n_kv_heads, 1, head_dim)

        # Attention (reuse transformer.py's GQA-aware attention)
        k_seq = k_full.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim)
        v_seq = v_full.transpose(1, 0, 2).reshape(-1, n_kv_heads * head_dim)
        attn_out = attention(q, k_seq, v_seq, n_heads, n_kv_heads)

        # Output projection + residual
        attn_out = self.gemm(attn_out, weights[f"{prefix}.attn_output.weight"])
        h = x + attn_out

        # Pre-FFN RMSNorm
        h_norm = self.rms_norm(h, weights[f"{prefix}.ffn_norm.weight"])

        # SwiGLU FFN (ORT-accelerated)
        gate = self.gemm(h_norm, weights[f"{prefix}.ffn_gate.weight"])
        up = self.gemm(h_norm, weights[f"{prefix}.ffn_up.weight"])
        silu_out = self.silu(gate)
        hidden = silu_out * up
        ffn_out = self.gemm(hidden, weights[f"{prefix}.ffn_down.weight"])

        # Final residual
        output = h + ffn_out

        return output, k_full, v_full

    @property
    def cache_stats(self) -> dict:
        if not self._cache:
            return {"hits": 0, "misses": 0, "sessions": 0}
        return {
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "sessions": len(self._cache._cache),
        }
