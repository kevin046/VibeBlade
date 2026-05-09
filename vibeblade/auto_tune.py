"""
VibeBlade PI+TS Auto-Tuner

Automatically selects optimal PowerInfer hot_budget and TurboSparse threshold
based on model size. Sets global C flags BEFORE model load (critical — llama.cpp
reads these flags at load time).

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
"""
import ctypes
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Q4_K_M ≈ 4.85 bits per param → ~0.61 bytes/param
# Q4_K_S ≈ 4.56 bits per param → ~0.57 bytes/param
# Average: ~0.6 bytes/param for Q4 quantization
BYTES_PER_PARAM_Q4 = 0.6


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
# TS threshold (activation cutoff) has a wide "good enough" range per model
# once PI is set correctly. We pick a stable midrange value.

_PROFILES = [
    # (max_params_B, profile) — checked in order, first match wins
    (1.0,  OptimizationProfile(0.15, 0.01, 1.84, "sub-1B dense")),
    (2.0,  OptimizationProfile(0.10, 0.01, 3.12, "1-2B dense")),
    (4.0,  OptimizationProfile(0.05, 0.05, 1.58, "2-4B dense")),
    (8.0,  OptimizationProfile(0.05, 0.02, 1.30, "4-8B — conservative")),
    (float('inf'), OptimizationProfile(0.03, 0.02, 1.20, "8B+ — very conservative")),
]


def estimate_params_from_file(path: str) -> float:
    """Estimate param count (billions) from GGUF file size.

    Uses Q4 bit-density heuristic: params ≈ file_bytes / 0.6.
    This is approximate but sufficient for threshold selection.
    """
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
        f"Auto-tune: {os.path.basename(model_path)} ~{n_params:.2f}B params "
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
