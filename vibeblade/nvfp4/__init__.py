"""VibeBlade NVFP4 — native NVFP4 (4-bit float, E2M1 + FP8 block scales) inference.

Why this exists:
  Blackwell (GB10 / B-series) tensor cores execute NVFP4 GEMMs natively.
  VibeBlade's C++ path is GGUF-integer-quant only; this package adds the
  NVFP4 checkpoint format (HF safetensors, modelopt-style) as a first-class
  native path:

    weights stay packed FP4 in memory
      → torch._scaled_mm on Blackwell (sm_100/sm_120) for tensor-core GEMMs
      → numpy dequant fallback elsewhere (correct, portable, for tests)

Modules:
  packing  — E2M1 nibble encode/decode, FP8-E4M3 scales, block dequant
  gemm     — nvfp4_gemm() dispatch: tensor-core path vs dequant fallback
  loader   — safetensors NVFP4 checkpoint reader + config detection
  engine   — Qwen-style decoder using NVFP4 projections + generate()
  dflash2  — generalized DFlash/DFlash2 layer-hook math for any target depth
"""

from .packing import (
    E2M1_VALUES,
    decode_fp4_packed,
    encode_fp4,
    fp8_e4m3_to_float32,
    dequant_nvfp4,
    pack_weights,
)
from .gemm import nvfp4_gemm, has_fp4_tensor_cores, swizzle_block_scales
from .loader import is_nvfp4_checkpoint, load_nvfp4_checkpoint, NVFP4Checkpoint
from .dflash2 import dflash2_layer_indices, DFlash2Config

__all__ = [
    "E2M1_VALUES", "decode_fp4_packed", "encode_fp4",
    "fp8_e4m3_to_float32", "dequant_nvfp4", "pack_weights",
    "nvfp4_gemm", "has_fp4_tensor_cores", "swizzle_block_scales",
    "is_nvfp4_checkpoint", "load_nvfp4_checkpoint", "NVFP4Checkpoint",
    "dflash2_layer_indices", "DFlash2Config",
]
