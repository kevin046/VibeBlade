"""VibeBlade TensorRT Backend — GPU-accelerated inference (NVIDIA only).

Converts ONNX graphs to TensorRT engines for maximum GPU throughput.
Requires: NVIDIA GPU, CUDA 11+, TensorRT 8+.

Gracefully disabled on non-NVIDIA platforms — falls back to ORT CPU or NumPy.
"""

from __future__ import annotations

import logging
import platform

import numpy as np

logger = logging.getLogger("vibeblade.tensorrt")

_TRT_AVAILABLE = False
try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
    _TRT_VERSION = trt.__version__
except ImportError:
    _TRT_VERSION = None


def is_available() -> bool:
    """Check if TensorRT is available and usable."""
    if not _TRT_AVAILABLE:
        return False
    if platform.machine() not in ("x86_64", "AMD64"):
        logger.debug("TensorRT requires x86_64 — this is %s", platform.machine())
        return False
    return True


def platform_support() -> dict:
    """Return TensorRT support status."""
    return {
        "available": is_available(),
        "version": _TRT_VERSION,
        "arch": platform.machine(),
        "reason": (
            "ready"
            if is_available()
            else "NVIDIA GPU + CUDA + tensorrt package required"
            if platform.machine() in ("x86_64", "AMD64")
            else f"unsupported architecture: {platform.machine()}"
        ),
    }


class TensorRTEngine:
    """TensorRT inference engine (stub — falls back to NumPy).

    When TensorRT is available, converts ONNX bytes → TRT engine and
    runs inference. Otherwise, all operations fall back to NumPy.
    """

    def __init__(self, fp16: bool = True, max_workspace: int = 1 << 30):
        self.fp16 = fp16
        self.max_workspace = max_workspace
        self._engine = None
        self._context = None

        if is_available():
            import tensorrt as trt
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            logger.info(
                "TensorRT %s ready (fp16=%s, workspace=%dMB)",
                _TRT_VERSION, fp16, max_workspace >> 20,
            )
        else:
            logger.info("TensorRT not available — using NumPy fallback")

    def build_from_onnx(self, onnx_bytes: bytes) -> bool:
        """Build TRT engine from ONNX model bytes."""
        if not is_available():
            return False

        try:
            import tensorrt as trt

            network = self._runtime.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self._logger)
            if not parser.parse(onnx_bytes):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                logger.error("ONNX parse errors: %s", errors)
                return False

            config = self._runtime.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace)
            if self.fp16 and self._runtime.has_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            builder = self._runtime.create_builder(network)
            self._engine = builder.build_serialized_network(network, config)
            if self._engine is None:
                logger.error("TRT engine build failed")
                return False

            self._context = self._engine.create_execution_context()
            logger.info("TRT engine built successfully")
            return True

        except Exception as e:
            logger.error("TRT build failed: %s", e)
            return False

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run inference on the TRT engine."""
        if self._context is None:
            raise RuntimeError("No TRT engine loaded")

        import tensorrt as trt

        # Allocate output buffers
        outputs = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._context.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            buf = np.empty(shape, dtype=dtype)
            self._context.set_tensor_address(name, buf.ctypes.data)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                outputs.append(buf)

        # Set input addresses
        for name, arr in inputs.items():
            self._context.set_tensor_address(name, arr.ctypes.data)

        if not self._context.execute_async_v3(0):
            raise RuntimeError("TRT execution failed")

        return outputs

    def gemm(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Fallback gemm — TRT only used via build_from_onnx + run."""
        return x @ W.T

    @property
    def active(self) -> bool:
        return self._engine is not None
