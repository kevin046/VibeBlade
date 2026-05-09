"""Tests for PI+TS auto-tuner."""
import pytest
import os
import tempfile
from vibeblade.auto_tune import (
    OptimizationProfile,
    estimate_params_from_file,
    get_profile,
    BYTES_PER_PARAM_Q4,
)


class TestEstimateParams:
    def test_tinyllama_size(self):
        # TinyLlama Q4_K_M = 638MB → ~1.06B params
        est = estimate_params_from_file("models/tinyllama-1.1b-q4km.gguf")
        assert 0.8 < est < 1.5

    def test_phi2_size(self):
        # Phi-2 Q4_K_M = 1.8GB → ~3.0B params
        est = estimate_params_from_file("models/phi-2.Q4_K_M.gguf")
        assert 2.0 < est < 4.0

    def test_synthetic_file(self):
        # Create a temp file of known size
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b'\x00' * int(0.6 * 1e9))  # 600MB → ~1B params
            path = f.name
        try:
            est = estimate_params_from_file(path)
            assert 0.9 < est < 1.1
        finally:
            os.unlink(path)


class TestGetProfile:
    @pytest.mark.parametrize("params,expected_pi,expected_ts", [
        (0.5,  0.15, 0.01),   # sub-1B
        (1.1,  0.10, 0.01),   # 1-2B
        (1.5,  0.10, 0.01),   # 1-2B
        (2.5,  0.05, 0.05),   # 2-4B
        (3.8,  0.05, 0.05),   # 2-4B
        (5.0,  0.05, 0.02),   # 4-8B
        (7.0,  0.05, 0.02),   # 4-8B
        (10.0, 0.03, 0.02),   # 8B+
        (70.0, 0.03, 0.02),   # 8B+
    ])
    def test_profile_selection(self, params, expected_pi, expected_ts):
        p = get_profile(params)
        assert p.pi_budget == expected_pi
        assert p.ts_threshold == expected_ts

    def test_all_profiles_have_valid_params(self):
        for max_p, profile in [
            (1.0,  OptimizationProfile(0.15, 0.01, 1.84, "sub-1B")),
            (2.0,  OptimizationProfile(0.10, 0.01, 3.12, "1-2B")),
            (4.0,  OptimizationProfile(0.05, 0.05, 1.58, "2-4B")),
            (8.0,  OptimizationProfile(0.05, 0.02, 1.30, "4-8B")),
            (float('inf'), OptimizationProfile(0.03, 0.02, 1.20, "8B+")),
        ]:
            assert profile.pi_budget > 0
            assert profile.ts_threshold > 0
            assert profile.expected_speedup > 0


class TestAutoTuneIntegration:
    def test_auto_tune_returns_profile(self):
        """Integration test: auto_tune returns valid profile for real model."""
        from vibeblade.auto_tune import auto_tune, disable_all
        import ctypes

        profile = auto_tune("models/tinyllama-1.1b-q4km.gguf")
        assert isinstance(profile, OptimizationProfile)
        assert profile.pi_budget > 0
        assert profile.ts_threshold > 0

        # Clean up
        disable_all()

    def test_disable_all(self):
        """disable_all should not crash."""
        from vibeblade.auto_tune import disable_all
        disable_all()  # Should not raise
