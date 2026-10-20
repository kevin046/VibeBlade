"""NVFP4 safetensors checkpoint loader.

Reads HF-style quantized checkpoints (modelopt / TensorRT-LLM layout):
  model.safetensors      — FP4-packed weights   (suffix .weight, packed uint8)
  model.scales           — per-16-block FP8-E4M3 scale factors
  model.scales_weight or quantization_config.tensor_scale — FP32 global scale
  config.json            — architecture params
  quantization_config in config.json or hf_quant_config.json:
    {"quant_method": "modelopt", "fmt": "nvfp4" | "packed_nvfp4",
     "kernel_options": {"nvfp4": true ...}}

Works without torch/safetensors installed (falls back to numpy safetensors
reader for FP8/FP4/FP16 tensors — the formats we need are all raw bytes).
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SAFE_DTYPES = {
    "F64": np.dtype("float64"),
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
    "BF16": np.dtype("uint16"),   # numpy has no bf16; raw view
    "I64": np.dtype("int64"),
    "I32": np.dtype("int32"),
    "I16": np.dtype("int16"),
    "I8": np.dtype("int8"),
    "U8": np.dtype("uint8"),
    "BOOL": np.dtype("bool"),
}


def _read_safetensors(path: str) -> dict[str, tuple[np.dtype, tuple, memoryview]]:
    """Minimal safetensors reader (header JSON + raw buffer)."""
    out: dict[str, tuple[np.dtype, tuple, memoryview]] = {}
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len))
        base = f.tell()
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dt = meta["dtype"]
            shape = tuple(meta["shape"])
            start, end = meta["data_offsets"]
            if dt not in SAFE_DTYPES:
                continue  # skip exotic (e.g. F4 enums some writers use)
            out[name] = (SAFE_DTYPES[dt], shape, memoryview(f)[base + start:base + end])
    return out


@dataclass
class NVFP4Checkpoint:
    path: str
    config: dict
    tensors: dict = field(default_factory=dict)   # name → (packed, scales)
    tensor_scale: float = 1.0
    # convenience views
    layers: list = field(default_factory=list)


def is_nvfp4_checkpoint(path: str) -> bool:
    """Detect NVFP4 quantized checkpoint dirs/files."""
    p = os.path.expanduser(path)
    if os.path.isdir(p):
        cfg_path = None
        for cand in ("config.json", "hf_quant_config.json"):
            if os.path.exists(os.path.join(p, cand)):
                cfg_path = os.path.join(p, cand)
                break
        if cfg_path is None:
            return False
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            return False
        q = cfg.get("quantization_config") or cfg
        if (q.get("quant_method") or "").lower() in ("modelopt", "fbgemm") or \
           (q.get("fmt") or "").lower() in ("nvfp4", "packed_nvfp4", "fp4"):
            return True
        if os.path.exists(os.path.join(p, "hf_quant_config.json")):
            try:
                hq = json.load(open(os.path.join(p, "hf_quant_config.json")))
                return (hq.get("quantization", {}).get("quant_method", "")).lower() == "modelopt" \
                    or "nvfp4" in str(hq).lower()
            except Exception:
                return False
        return False
    if p.endswith(".safetensors"):
        # single-file heuristic: presence of .scales tensors
        try:
            tensors = _read_safetensors(p)
            return any(k.endswith(".weight_scale_2") or k.endswith(".scales") for k in tensors)
        except Exception:
            return False
    return False


def load_nvfp4_checkpoint(path: str, lazy: bool = True) -> NVFP4Checkpoint:
    """Load an NVFP4 checkpoint into packed numpy arrays.

    Each quantized linear becomes tensors[name] = {
        "packed": (out, in/2) uint8, "scales": (out, in/16) uint8 codes
    }
    Global tensor_scale read from quantization config when present.
    """
    p = os.path.expanduser(path)
    ckpt = NVFP4Checkpoint(path=p, config={})

    files = []
    if os.path.isdir(p):
        cfg_file = os.path.join(p, "config.json")
        if os.path.exists(cfg_file):
            ckpt.config = json.load(open(cfg_file))
        files = sorted(
            f for f in os.listdir(p)
            if f.endswith(".safetensors") and not f.startswith(".")
        )
        files = [os.path.join(p, f) for f in files]
    else:
        files = [p]

    qcfg = ckpt.config.get("quantization_config", {}) or {}
    ckpt.tensor_scale = float(qcfg.get("tensor_scale", 1.0) or 1.0)

    for fp in files:
        raw = _read_safetensors(fp)
        for name, (dt, shape, buf) in raw.items():
            arr = np.frombuffer(buf, dtype=dt)
            if name.endswith("_scale_2") or name.endswith(".scales"):
                base = name.rsplit("_scale_2", 1)[0].rsplit(".scales", 1)[0]
                ckpt.tensors.setdefault(base, {})["scales"] = arr.reshape(shape).copy()
            elif name.endswith(".weight") or name.endswith(".weight_packed"):
                base = name.rsplit(".weight_packed", 1)[0].rsplit(".weight", 1)[0]
                ckpt.tensors.setdefault(base, {})["packed"] = arr.reshape(shape).copy()
            elif not lazy:
                ckpt.tensors.setdefault(name, {})["dense"] = arr.reshape(shape).copy()
    return ckpt
