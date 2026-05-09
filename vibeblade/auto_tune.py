"""
VibeBlade PI+TS Auto-Tuner

Automatically selects optimal PowerInfer hot_budget and TurboSparse threshold
based on model size and architecture. Sets global C flags BEFORE model load
(critical — llama.cpp reads these flags at load time).

MoE-aware: detects MoE models from GGUF metadata and uses active-param count
(expert_count / total_experts × estimated_params) instead of raw file size.

Usage:
    from vibeblade.auto_tune import auto_tune, OptimizationProfile

    # Before loading the model:
    profile = auto_tune("path/to/model.gguf")

    # Then load normally — optimizations are already set
    backend = LlamaCppBackend()
    backend.load("path/to/model.gguf", ...)

Empirically calibrated on Oracle A1 ARM64 (4 cores, NEON) with:
  - TinyLlama-1.1B (Q4_K_M):  grid-search best PI=0.10, TS=0.01
  - Llama-3.2-1B (Q4_K_S):    grid-search best PI=0.20, TS=0.20
  - Phi-2-2.7B (Q4_K_M):      grid-search best PI=0.05, TS=0.05
  - Qwen2.5-MoE 2×1.5B (Q4_K_M): grid-search best PI=0.15, TS=0.05 → 3.26x
"""
import ctypes
import logging
import os
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Q4_K_M ≈ 4.85 bits per param → ~0.61 bytes/param
# Q4_K_S ≈ 4.56 bits per param → ~0.57 bytes/param
# Average: ~0.6 bytes/param for Q4 quantization
BYTES_PER_PARAM_Q4 = 0.6

# GGUF magic number (little-endian)
GGUF_MAGIC = 0x46554747


@dataclass
class OptimizationProfile:
    """Optimal PI+TS parameters for a model class."""
    pi_budget: float       # PowerInfer hot neuron fraction (via set_hot_budget)
    ts_threshold: float    # TurboSparse activation cutoff
    expected_speedup: float  # Approximate speedup vs baseline
    notes: str = ""


# ── Empirical calibration by estimated param count ──────────────
# The PI budget (hot neuron fraction) is the critical knob:
#   - Too high → more compute, no speed benefit
#   - Too low → quality loss
# Larger models need tighter budgets because their FFN layers are bigger.
#
# MoE models use active params (experts_used / total_experts × total_params)
# which puts them in a smaller, more aggressive bucket — matching empirical results.
#
# TS threshold (activation cutoff) has a wide "good enough" range per model
# once PI is set correctly. We pick a stable midrange value.

_PROFILES = [
    # (max_params_B, profile) — checked in order, first match wins
    # Note: MoE models with high active-expert ratio (e.g. 2/2) estimate close
    # to total file size and may land slightly above their true param tier.
    # The 5B boundary for 2-4B accommodates Qwen2.5-MoE (4.09B actual params).
    (1.0,  OptimizationProfile(0.15, 0.01, 1.84, "sub-1B dense")),
    (2.0,  OptimizationProfile(0.10, 0.01, 3.61, "1-2B dense")),
    (5.0,  OptimizationProfile(0.15, 0.05, 3.26, "2-4B dense / MoE")),
    (8.0,  OptimizationProfile(0.05, 0.02, 1.30, "5-8B — conservative")),
    (float('inf'), OptimizationProfile(0.03, 0.02, 1.20, "8B+ — very conservative")),
]


def _read_gguf_moe_info(path: str) -> tuple[float, int, int] | None:
    """Read MoE expert metadata from GGUF header.

    Returns (n_params_billion, n_expert_used, n_expert_total) or None if not MoE.
    Parses the GGUF KV section for qwen2moe/llama/mixtral expert fields.
    Handles both GGUF v2 (uint32 string lengths) and v3 (uint64 string lengths).
    """
    # GGUF scalar type sizes (from gguf.h enum gguf_type)
    _TYPE_SIZES = {
        0: 1,   # GGUF_TYPE_UINT8
        1: 1,   # GGUF_TYPE_INT8
        2: 2,   # GGUF_TYPE_UINT16
        3: 2,   # GGUF_TYPE_INT16
        4: 4,   # GGUF_TYPE_UINT32
        5: 4,   # GGUF_TYPE_INT32
        6: 4,   # GGUF_TYPE_FLOAT32
        7: 1,   # GGUF_TYPE_BOOL
        10: 8,  # GGUF_TYPE_UINT64
        11: 8,  # GGUF_TYPE_INT64
        12: 8,  # GGUF_TYPE_FLOAT64
    }

    try:
        real_path = os.path.realpath(path)
        with open(real_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                return None
            version = struct.unpack('<I', f.read(4))[0]
            if version < 2:
                return None

            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]

            # GGUF v3 uses uint64 string lengths; v2 uses uint32
            use_u64 = version >= 3
            str_len_fmt = '<Q' if use_u64 else '<I'
            str_len_size = 8 if use_u64 else 4

            n_expert = None
            n_expert_used = None

            for _ in range(n_kv):
                # Key string: length + bytes
                key_len = struct.unpack(str_len_fmt, f.read(str_len_size))[0]
                key = f.read(key_len).decode('utf-8', errors='replace')
                vtype = struct.unpack('<I', f.read(4))[0]

                if vtype in _TYPE_SIZES:
                    # Fixed-size scalar
                    data = f.read(_TYPE_SIZES[vtype])
                    if 'expert_count' in key and vtype == 4:
                        n_expert = struct.unpack('<I', data)[0]
                    elif 'expert_used_count' in key and vtype == 4:
                        n_expert_used = struct.unpack('<I', data)[0]
                elif vtype == 8:  # STRING
                    slen = struct.unpack(str_len_fmt, f.read(str_len_size))[0]
                    f.read(slen)
                elif vtype == 9:  # ARRAY
                    arr_type = struct.unpack('<I', f.read(4))[0]
                    arr_len = struct.unpack('<Q', f.read(8))[0]
                    elem_size = _TYPE_SIZES.get(arr_type, 0)
                    if elem_size > 0:
                        f.read(arr_len * elem_size)
                    elif arr_type == 8:  # string array
                        for _ in range(arr_len):
                            sl = struct.unpack(str_len_fmt, f.read(str_len_size))[0]
                            f.read(sl)
                    else:
                        return None  # Unknown array element type
                else:
                    return None  # Unknown value type

            if n_expert is not None and n_expert_used is not None and n_expert > 0:
                file_size = os.path.getsize(real_path)
                total_params_b = file_size / BYTES_PER_PARAM_Q4 / 1e9
                return (total_params_b, n_expert_used, n_expert)
    except Exception as e:
        logger.debug(f"GGUF MoE parse failed: {e}")
    return None


def estimate_params_from_file(path: str) -> float:
    """Estimate active param count (billions) from GGUF file.

    For MoE models: reads expert metadata and scales by active expert ratio.
    For dense models: uses Q4 bit-density heuristic.
    """
    moe_info = _read_gguf_moe_info(path)
    if moe_info:
        total_params_b, n_expert_used, n_expert_total = moe_info
        # Active params = total × (experts_used / total_experts)
        # Shared experts are always active, but this ratio is a good approximation
        active_ratio = n_expert_used / n_expert_total
        active_params_b = total_params_b * active_ratio
        logger.info(
            f"MoE detected: {n_expert_used}/{n_expert_total} experts, "
            f"total ~{total_params_b:.2f}B → active ~{active_params_b:.2f}B"
        )
        return active_params_b

    # Dense model: file size heuristic
    real_path = os.path.realpath(path)
    size = os.path.getsize(real_path)
    return size / BYTES_PER_PARAM_Q4 / 1e9


def get_profile(n_params_billion: float) -> OptimizationProfile:
    """Select the best PI+TS profile for a given param count."""
    for max_params, profile in _PROFILES:
        if n_params_billion <= max_params:
            return profile
    return _PROFILES[-1][1]  # fallback: largest model profile


def auto_tune(model_path: str) -> OptimizationProfile:
    """Auto-tune PI+TS optimization flags for a model.

    MUST be called BEFORE LlamaCppBackend.load() — the C library reads
    these global flags at model load time, not at generation time.

    Args:
        model_path: Path to .gguf model file.

    Returns:
        OptimizationProfile that was applied.
    """
    from vibeblade.llama_backend import _lib

    n_params = estimate_params_from_file(model_path)
    profile = get_profile(n_params)

    # Set C library argtypes
    _lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
    _lib.turbosparse_set_threshold.argtypes = [ctypes.c_float]
    _lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool]
    _lib.powerinfer_set_hot_budget.argtypes = [ctypes.c_float]

    # Apply flags
    _lib.turbosparse_set_enabled(True)
    _lib.turbosparse_set_threshold(ctypes.c_float(profile.ts_threshold))
    _lib.powerinfer_set_enabled(True)
    _lib.powerinfer_set_hot_budget(ctypes.c_float(profile.pi_budget))
    _lib.powerinfer_reset()

    logger.info(
        f"Auto-tune: {os.path.basename(model_path)} ~{n_params:.2f}B active params "
        f"→ PI={profile.pi_budget}, TS={profile.ts_threshold} "
        f"(expected {profile.expected_speedup:.2f}x, {profile.notes})"
    )
    return profile


def disable_all():
    """Disable all optimizations — call before baseline benchmarks."""
    from vibeblade.llama_backend import _lib

    _lib.turbosparse_set_enabled.argtypes = [ctypes.c_bool]
    _lib.powerinfer_set_enabled.argtypes = [ctypes.c_bool]

    _lib.turbosparse_set_enabled(False)
    _lib.powerinfer_set_enabled(False)
