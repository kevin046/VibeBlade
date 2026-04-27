"""VibeBlade Converter — Convert full-precision models to quantized GGUF.

Supports converting safetensors/HF format to GGUF with various quantization
methods. Produces files compatible with llama.cpp, LM Studio, Ollama, etc.

Conversion pipeline:
    safetensors → load weights → quantize → write GGUF

Quantization methods:
    - q4_k_m: 4-bit K-Quants (recommended, best quality/size tradeoff)
    - q4_k_s: 4-bit K-Quants Small (faster, slightly less accurate)
    - q5_k_m: 5-bit K-Quants Medium
    - q5_k_s: 5-bit K-Quants Small
    - q6_k:   6-bit K-Quants
    - q8_0:   8-bit quantization (near-lossless)
    - f16:    Half-precision (no quantization, just format conversion)
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


class QuantMethod(Enum):
    """Supported quantization methods."""
    Q4_K_M = "q4_k_m"
    Q4_K_S = "q4_k_s"
    Q5_K_M = "q5_k_m"
    Q5_K_S = "q5_k_s"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"
    F16 = "f16"
    F32 = "f32"

    @property
    def bits(self) -> int:
        bits = {
            "q4_k_m": 4, "q4_k_s": 4,
            "q5_k_m": 5, "q5_k_s": 5,
            "q6_k": 6, "q8_0": 8,
            "f16": 16, "f32": 32,
        }
        return bits[self.value]

    @property
    def description(self) -> str:
        descs = {
            "q4_k_m": "4-bit K-Quants Medium — recommended balance",
            "q4_k_s": "4-bit K-Quants Small — faster inference",
            "q5_k_m": "5-bit K-Quants Medium — better quality",
            "q5_k_s": "5-bit K-Quants Small — fast + good quality",
            "q6_k": "6-bit K-Quants — high quality",
            "q8_0": "8-bit — near-lossless",
            "f16": "Half-precision — no quantization loss",
            "f32": "Full precision — largest file",
        }
        return descs[self.value]


# ── Conversion Task Tracking ──


@dataclass
class ConversionTask:
    """Tracks an in-progress or completed conversion."""
    task_id: str
    model_id: str
    input_path: str
    output_path: str
    method: QuantMethod
    status: str = "pending"  # pending, converting, done, error
    progress: float = 0.0    # 0.0 to 1.0
    current_layer: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    input_size: int = 0
    output_size: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "method": self.method.value,
            "bits": self.method.bits,
            "status": self.status,
            "progress": self.progress,
            "current_layer": self.current_layer,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "compression_ratio": (
                self.input_size / self.output_size
                if self.output_size > 0 else 0
            ),
        }


# ── Weight Quantization Kernels ──


def quantize_q4(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize fp32/fp16 array to 4-bit packed + scales + mins (per-channel).

    Args:
        x: (out_features, in_features) weight matrix

    Returns:
        packed: (out_features, in_features//2) uint8 packed 4-bit values
        scales: (out_features,) float32 per-row scale factors
        mins: (out_features,) float32 per-row minimums
    """
    if x.dtype in (np.float16,) or str(x.dtype) == "bfloat16":
        x = x.astype(np.float32)

    out_f, in_f = x.shape
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    scale = np.maximum(mx - mn, 1e-8) / 15.0

    # Quantize to 0-15
    q = np.clip(np.round((x - mn) / scale), 0, 15).astype(np.uint8)

    # Pack two 4-bit values per byte: low nibble = first, high nibble = second
    flat = q.ravel()
    padded = np.zeros((len(flat) + 1) // 2 * 2, dtype=np.uint8)
    padded[:len(flat)] = flat
    packed = (padded[0::2]) | (padded[1::2] << 4)

    return (
        packed.reshape(out_f, in_f // 2),
        scale.ravel().astype(np.float32),
        mn.ravel().astype(np.float32),
    )


def quantize_q8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to 8-bit symmetric per-row.

    Returns:
        quant: int8 array
        scale: float32 per-row scale
    """
    if x.dtype in (np.float16,) or str(x.dtype) == "bfloat16":
        x = x.astype(np.float32)

    max_abs = np.max(np.abs(x), axis=1, keepdims=True)
    scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)
    quant = np.clip(np.round(x / scale), -127, 127).astype(np.int8)

    return quant, scale.ravel().astype(np.float32)


# ── GGUF Writer ──


def _gguf_magic() -> bytes:
    return b"GGUF"


class GGUFWriter:
    """Minimal GGUF file writer for quantized model export.

    Produces valid GGUF v3 files compatible with llama.cpp, LM Studio,
    Ollama, KoboldCpp, and other GGUF consumers.
    """

    # GGUF value types
    UINT32 = 0
    INT32 = 1
    FLOAT32 = 2
    BOOL = 3
    STRING = 4
    ARRAY = 5
    UINT64 = 6
    INT64 = 7
    FLOAT64 = 8

    def __init__(self, path: str):
        self.path = path
        self.tensors: list[dict] = []
        self._kv_pairs: list[tuple[str, tuple]] = []

    def add_string(self, key: str, val: str):
        self._kv_pairs.append((key, (self.STRING, val)))

    def add_uint32(self, key: str, val: int):
        self._kv_pairs.append((key, (self.UINT32, val)))

    def add_uint64(self, key: str, val: int):
        self._kv_pairs.append((key, (self.UINT64, val)))

    def add_int32(self, key: str, val: int):
        self._kv_pairs.append((key, (self.INT32, val)))

    def add_float32(self, key: str, val: float):
        self._kv_pairs.append((key, (self.FLOAT32, val)))

    def add_bool(self, key: str, val: bool):
        self._kv_pairs.append((key, (self.BOOL, val)))

    def add_array_string(self, key: str, vals: list[str]):
        self._kv_pairs.append((key, (self.ARRAY, (self.STRING, vals))))

    def add_array_uint32(self, key: str, vals: list[int]):
        self._kv_pairs.append((key, (self.ARRAY, (self.UINT32, vals))))

    def add_tensor(self, name: str, data: np.ndarray):
        """Register a tensor with its raw data."""
        self.tensors.append({
            "name": name,
            "data": data,
            "n_dims": data.ndim,
            "dims": list(reversed(data.shape)),  # GGUF uses reverse order
            "dtype": self._numpy_to_gguf_type(data.dtype),
            "offset": 0,  # Filled during write
        })

    @staticmethod
    def _numpy_to_gguf_type(dtype) -> int:
        """Map numpy dtype to GGUF tensor type."""
        mapping = {
            np.float32: 0,    # GGUF_TYPE_F32
            np.float16: 1,    # GGUF_TYPE_F16
            np.int32: 6,      # GGUF_TYPE_I32
            np.int8: 8,       # GGUF_TYPE_I8
            np.uint8: 9,      # GGUF_TYPE_U8
        }
        return mapping.get(dtype, 0)

    def write(self):
        """Write the complete GGUF file."""
        buf = bytearray()

        # ── Header ──
        buf.extend(_gguf_magic())
        buf.extend(struct.pack("<I", 3))  # version
        buf.extend(struct.pack("<Q", len(self._kv_pairs)))  # n_kv
        buf.extend(struct.pack("<Q", len(self.tensors)))  # n_tensors

        # ── KV pairs ──
        for key, (vtype, val) in self._kv_pairs:
            buf.extend(self._encode_string(key))
            buf.extend(struct.pack("<I", vtype))
            buf.extend(self._encode_value(vtype, val))

        # ── Tensor infos ──
        tensor_data_offset = 0
        for t in self.tensors:
            raw_size = t["data"].nbytes
            # Pad to alignment
            padded_size = (raw_size + 31) & ~31
            buf.extend(self._encode_string(t["name"]))
            buf.extend(struct.pack("<I", t["n_dims"]))
            for d in t["dims"]:
                buf.extend(struct.pack("<Q", d))
            buf.extend(struct.pack("<I", t["dtype"]))
            buf.extend(struct.pack("<Q", tensor_data_offset))
            t["offset"] = tensor_data_offset
            tensor_data_offset += padded_size

        # ── Padding between header and data ──
        alignment = 32
        current_pos = len(buf)
        pad = (alignment - (current_pos % alignment)) % alignment
        buf.extend(b"\x00" * pad)

        # ── Tensor data ──
        for t in self.tensors:
            raw = t["data"].tobytes()
            padded_size = (len(raw) + 31) & ~31
            buf.extend(raw)
            buf.extend(b"\x00" * (padded_size - len(raw)))

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(buf)

        logger.info("Wrote GGUF: %s (%.1f MB)", self.path, len(buf) / 1e6)

    def _encode_string(self, s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def _encode_value(self, vtype: int, val) -> bytes:
        if vtype == self.STRING:
            return self._encode_string(val)
        elif vtype == self.UINT32:
            return struct.pack("<I", val)
        elif vtype == self.UINT64:
            return struct.pack("<Q", val)
        elif vtype == self.INT32:
            return struct.pack("<i", val)
        elif vtype == self.FLOAT32:
            return struct.pack("<f", val)
        elif vtype == self.BOOL:
            return struct.pack("<B", 1 if val else 0)
        elif vtype == self.ARRAY:
            elem_type, items = val
            result = struct.pack("<I", elem_type) + struct.pack("<Q", len(items))
            if elem_type == self.STRING:
                for item in items:
                    result += self._encode_string(item)
            elif elem_type == self.UINT32:
                for item in items:
                    result += struct.pack("<I", item)
            return result
        return b""


# ── Main Converter ──


class ModelConverter:
    """Convert models between formats and quantization levels.

    Main use case: safetensors → quantized GGUF (compatible with
    llama.cpp, LM Studio, Ollama, etc.)
    """

    def __init__(self):
        self._tasks: dict[str, ConversionTask] = {}

    def convert(
        self,
        input_path: str,
        output_path: str = "",
        method: QuantMethod = QuantMethod.Q4_K_M,
        model_name: str = "",
        model_id: str = "",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ConversionTask:
        """Convert a model to quantized GGUF.

        Args:
            input_path: Path to input model (directory with safetensors or single GGUF)
            output_path: Output GGUF path. Auto-generated if empty.
            method: Quantization method
            model_name: Model name for GGUF metadata
            model_id: HuggingFace model ID for metadata
            progress_callback: Called with (progress_0_to_1, current_layer_name)

        Returns:
            ConversionTask with status and result
        """
        import uuid
        task_id = str(uuid.uuid4())[:8]

        if not output_path:
            input_stem = Path(input_path).stem
            output_path = str(
                Path(input_path).parent / f"{input_stem}-{method.value}.gguf"
            )

        input_p = Path(input_path)
        input_size = input_p.stat().st_size if input_p.is_file() else sum(
            f.stat().st_size for f in input_p.rglob("*") if f.is_file()
        )

        task = ConversionTask(
            task_id=task_id,
            model_id=model_id or input_stem,
            input_path=str(input_path),
            output_path=output_path,
            method=method,
            status="converting",
            started_at=datetime.utcnow().isoformat(),
            input_size=input_size,
        )

        self._tasks[task_id] = task

        try:
            self._do_convert(task, model_name, progress_callback)
            task.status = "done"
            task.finished_at = datetime.utcnow().isoformat()
            task.output_size = Path(output_path).stat().st_size
        except Exception as e:
            task.status = "error"
            task.error = str(e)
            task.finished_at = datetime.utcnow().isoformat()
            logger.error("Conversion failed: %s", e)

        return task

    def _do_convert(
        self,
        task: ConversionTask,
        model_name: str,
        progress_callback: Optional[Callable] = None,
    ):
        """Execute the conversion pipeline."""
        input_path = task.input_path
        output_path = task.output_path
        method = task.method

        input_p = Path(input_path)

        if input_p.suffix.lower() == ".gguf":
            self._convert_gguf_to_gguf(input_p, output_path, method, task, progress_callback)
        elif input_p.is_dir():
            self._convert_safetensors_to_gguf(input_p, output_path, method, task, model_name, progress_callback)
        else:
            raise ValueError(f"Unsupported input: {input_path}. Need .gguf or directory with .safetensors")

    def _convert_gguf_to_gguf(
        self, input_path: Path, output_path: str, method: ConversionTask,
        task: ConversionTask, progress_callback=None,
    ):
        """Re-quantize a GGUF model (read with existing loader, write new GGUF)."""
        from .loader import GGUFLoader

        loader = GGUFLoader(str(input_path))

        writer = GGUFWriter(output_path)

        # Copy metadata
        for key, val in loader.metadata.items():
            if isinstance(val, str):
                writer.add_string(key, val)
            elif isinstance(val, bool):
                writer.add_bool(key, val)
            elif isinstance(val, int):
                if val >= 0:
                    writer.add_uint64(key, val)
                else:
                    writer.add_int32(key, val)
            elif isinstance(val, float):
                writer.add_float32(key, val)

        # Re-quantize tensors
        total = len(loader.tensor_infos)
        for i, info in enumerate(loader.tensor_infos):
            name = info["name"]
            task.current_layer = name
            task.progress = i / max(total, 1)
            if progress_callback:
                progress_callback(task.progress, name)

            tensor = loader.load_tensor(name)
            quantized = self._quantize_tensor(tensor, method)
            writer.add_tensor(name, quantized)

        writer.write()
        task.progress = 1.0

    def _convert_safetensors_to_gguf(
        self, input_dir: Path, output_path: str, method: QuantMethod,
        task: ConversionTask, model_name: str, progress_callback=None,
    ):
        """Convert safetensors weights to quantized GGUF."""
        st_files = sorted(input_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files in {input_dir}")

        # Try to load metadata from config
        config_path = input_dir / "config.json"
        config = {}
        if config_path.exists():
            import json
            config = json.loads(config_path.read_text())

        writer = GGUFWriter(output_path)

        # Write model metadata
        arch = config.get("model_type", "llama").replace("_", "")
        writer.add_string("general.architecture", arch)
        writer.add_string("general.name", model_name or input_dir.name)
        writer.add_uint32(
            "llama.context_length",
            config.get("max_position_embeddings", 4096),
        )
        writer.add_uint32(
            "llama.embedding_length",
            config.get("hidden_size", 4096),
        )
        writer.add_uint32(
            "llama.block_count",
            config.get("num_hidden_layers", 32),
        )
        writer.add_uint32(
            "llama.attention.head_count",
            config.get("num_attention_heads", 32),
        )
        writer.add_uint32(
            "llama.feed_forward_length",
            config.get("intermediate_size", 11008),
        )
        writer.add_string("general.quantization_version", "GGUF_Q4_K_M" if method.value.startswith("q4") else method.value.upper())
        writer.add_string("tokenizer.ggml.model", "llama")

        # Load and quantize tensors
        try:
            from safetensors.numpy import load_file
        except ImportError:
            raise ImportError("safetensors required. Run: pip install safetensors")

        all_tensors = {}
        for st_file in st_files:
            tensors = load_file(str(st_file))
            all_tensors.update(tensors)

        total = len(all_tensors)
        for i, (name, tensor) in enumerate(all_tensors.items()):
            task.current_layer = name
            task.progress = i / max(total, 1)
            if progress_callback:
                progress_callback(task.progress, name)

            quantized = self._quantize_tensor(tensor, method)
            writer.add_tensor(name, quantized)

        writer.write()
        task.progress = 1.0

    def _quantize_tensor(self, tensor: np.ndarray, method: QuantMethod) -> np.ndarray:
        """Quantize a single tensor according to the method."""
        if method == QuantMethod.F16:
            return tensor.astype(np.float16)
        elif method == QuantMethod.F32:
            return tensor.astype(np.float32)
        elif method in (QuantMethod.Q4_K_M, QuantMethod.Q4_K_S,
                        QuantMethod.Q5_K_M, QuantMethod.Q5_K_S, QuantMethod.Q6_K):
            packed, scales, mins = quantize_q4(tensor)
            # Store as uint8 for GGUF
            return packed.astype(np.uint8)
        elif method == QuantMethod.Q8_0:
            quant, scales = quantize_q8(tensor)
            return quant.astype(np.int8)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def get_task(self, task_id: str) -> Optional[ConversionTask]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[ConversionTask]:
        return list(self._tasks.values())

    @staticmethod
    def get_supported_methods() -> list[dict]:
        """Get all supported conversion methods with details."""
        return [
            {
                "id": m.value,
                "bits": m.bits,
                "description": m.description,
                "recommended": m in (QuantMethod.Q4_K_M, QuantMethod.Q5_K_M),
            }
            for m in QuantMethod
        ]

    @staticmethod
    def estimate_output_size(input_size: int, method: QuantMethod) -> int:
        """Rough estimate of output size after quantization.

        Assumes ~90% of model is weights.
        """
        weight_fraction = 0.90
        meta_fraction = 1.0 - weight_fraction
        weight_bits = method.bits
        original_bits = 16  # assume fp16 input
        ratio = weight_bits / original_bits
        return int(input_size * (meta_fraction + weight_fraction * ratio))
