"""NVFP4 GEMM dispatch — Blackwell tensor cores when available, portable fallback.

Strategy:
  1. If torch is present AND the device reports sm_100/sm_121 (Blackwell),
     use torch._scaled_mm with FP4 operands + block scales — native NVFP4
     tensor-core matmul, the same instruction path SGLang/TRT-LLM use.
  2. Otherwise dequantize to BF16/FP32 and run a plain matmul — correct on
     any CPU/GPU, used for tests, ARM boxes, and pre-Blackwell GPUs.
"""

from __future__ import annotations

import numpy as np

from .packing import BLOCK, dequant_nvfp4, fp8_e4m3_to_float32

_TORCH_WARNED = False


def has_fp4_tensor_cores() -> bool:
    """True when torch._scaled_mm supports FP4 on the current device."""
    global _TORCH_WARNED
    try:
        import torch
        from torch._C import _scaled_mm  # noqa: F401  (probe)
    except Exception:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability(0)
        # Blackwell: sm_100 (B200), sm_103/107/110/120 (GB10/B-series variants)
        if cap[0] >= 10:
            return True
        return False
    except Exception:
        if not _TORCH_WARNED:
            _TORCH_WARNED = True
        return False


def swizzle_block_scales(scales: np.ndarray, m: int, n: int,
                         block_k: int = BLOCK) -> np.ndarray:
    """Reorder block scales into the layout torch._scaled_mm expects on
    Blackwell (128x4 tiles of scale factors, padded to 4 columns).

    scales: (ceil(m/block_m) * ceil(n/block_k),) — row-major blocks
    Returns swizzled array of the same total length padded to tile grid.
    """
    block_m = 128
    rows = (m + block_m - 1) // block_m
    cols = (n + block_k - 1) // block_k
    padded_cols = (cols + 3) // 4 * 4
    grid = np.zeros(rows * padded_cols, dtype=scales.dtype)
    base = scales.reshape(rows, cols)
    for r in range(rows):
        for c in range(cols):
            tile_r, tile_c = r // 4, c // 4
            in_r, in_c = r % 4, c % 4
            grid[tile_r * (padded_cols * 4) + tile_c * 16 + in_r * 4 + in_c] = base[r, c]
    return grid


def nvfp4_gemm(x: np.ndarray, packed_w: np.ndarray, scales: np.ndarray,
               tensor_scale: float = 1.0) -> np.ndarray:
    """x @ dequant(W).T with NVFP4 weights.

    x: (batch, in_dim) float32/float16 activations
    packed_w: (out_dim, in_dim/2) uint8 packed FP4 (row-major output dim)
    scales: (out_dim * in_dim/BLOCK,) uint8 FP8 codes, row-major blocks

    Dispatches to torch._scaled_mm on Blackwell, else dequant + matmul.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)

    if has_fp4_tensor_cores():
        try:
            return _gemm_scaled_mm(x, packed_w, scales, tensor_scale)
        except Exception:
            pass  # fall through to portable path on any layout mismatch

    # Portable: dequant W row-block-wise, plain matmul
    out_dim = packed_w.shape[0]
    in_dim = packed_w.shape[1] * 2
    w = _dequant_matrix(packed_w, scales, tensor_scale)
    return (x @ w.T).astype(np.float32)


def _dequant_matrix(packed_w: np.ndarray, scales: np.ndarray,
                    tensor_scale: float) -> np.ndarray:
    out_dim, half_in = packed_w.shape
    in_dim = half_in * 2
    blocks_per_row = in_dim // BLOCK
    if scales.shape[0] != out_dim * blocks_per_row:
        raise ValueError(
            f"scales {scales.shape[0]} != out_dim*blocks/row {out_dim * blocks_per_row}"
        )
    vals = np.empty((out_dim, in_dim), dtype=np.float32)
    block_scales = fp8_e4m3_to_float32(scales).reshape(out_dim, blocks_per_row)
    from .packing import E2M1_LUT
    lo = E2M1_LUT[packed_w & 0x0F]
    hi = E2M1_LUT[packed_w >> 4]
    pairs = np.stack([lo, hi], axis=-1).reshape(out_dim, in_dim)
    expanded = np.repeat(block_scales, BLOCK, axis=1)[:, :in_dim]
    vals = pairs * expanded * np.float32(tensor_scale)
    return vals


def _gemm_scaled_mm(x, packed_w, scales, tensor_scale):
    """Blackwell path: torch._scaled_mm with float4_e2m1fn_x2 + e4m3 scales."""
    import torch

    out_dim, half_in = packed_w.shape
    in_dim = half_in * 2

    xt = torch.from_numpy(x).to("cuda", torch.bfloat16 if x.dtype == np.float32 else torch.float16)
    w_packed = torch.from_numpy(packed_w).to("cuda", torch.uint8)
    sc = torch.from_numpy(scales).to("cuda", torch.uint8)

    w_f4 = w_packed.view(torch.float4_e2m1fn_x2)  # (out_dim, in_dim/2)
    sc_f8 = sc.view(torch.float8_e4m3fn)

    blocks_per_row = in_dim // BLOCK
    sc_mat = sc_f8.reshape(out_dim, blocks_per_row)

    # scale_mm requires swizzled scale layout on Blackwell
    sw = swizzle_block_scales(scales, out_dim, in_dim)
    sc_sw = torch.from_numpy(sw).to("cuda", torch.uint8).view(torch.float8_e4m3fn)

    y = torch._scaled_mm(
        xt,                                   # (batch, in_dim) bf16
        w_f4.T,                               # (in_dim, out_dim) fp4 column-major
        scale_a=torch.ones(xt.shape[0], device="cuda", dtype=torch.bfloat16),
        scale_b=sc_sw,
        out_dtype=torch.bfloat16,
    )
    return (y.float().cpu().numpy()) * np.float32(tensor_scale)
