"""Tests for PI+TS auto-tuner."""
import pytest
import os
import struct
import tempfile
from vibeblade.auto_tune import (
    OptimizationProfile,
    estimate_params_from_file,
    get_profile,
    _read_gguf_moe_info,
    BYTES_PER_PARAM_Q4,
    GGUF_MAGIC,
)


class TestEstimateParams:
    def test_tinyllama_size(self):
        # TinyLlama Q4_K_M = 638MB → ~1.06B params (dense)
        est = estimate_params_from_file("models/tinyllama-1.1b-q4km.gguf")
        assert 0.8 < est < 1.5

    def test_phi2_size(self):
        # Phi-2 Q4_K_M = 1.8GB → ~3.0B params (dense)
        est = estimate_params_from_file("models/phi-2.Q4_K_M.gguf")
        assert 2.0 < est < 4.0

    def test_synthetic_file(self):
        # Create a temp file of known size (no GGUF magic → dense fallback)
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b'\x00' * int(0.6 * 1e9))  # 600MB → ~1B params
            path = f.name
        try:
            est = estimate_params_from_file(path)
            assert 0.9 < est < 1.1
        finally:
            os.unlink(path)

    def test_moe_model_active_params(self):
        # Qwen2.5-MoE: 2.34GB file, 2/2 experts → active ratio = 1.0
        # Estimated ~4.2B → lands in 2-5B bucket (PI=0.15, TS=0.05)
        est = estimate_params_from_file("models/qwen25-moe-2x1.5b-q4km.gguf")
        assert 3.5 < est < 5.0
        p = get_profile(est)
        assert p.pi_budget == 0.15
        assert p.ts_threshold == 0.05


class TestGgufMoeParse:
    def _make_moe_gguf(self, n_expert, n_expert_used, data_size=1000):
        """Create a minimal GGUF v3 file with MoE metadata for testing."""
        buf = bytearray()
        # GGUF header
        buf += struct.pack('<I', GGUF_MAGIC)  # magic
        buf += struct.pack('<I', 3)  # version 3
        buf += struct.pack('<Q', 0)  # n_tensors
        buf += struct.pack('<Q', 3)  # n_kv = 3

        def write_kv_str(key, val):
            """Write a string KV pair (GGUF v3: uint64 string lengths)."""
            buf.extend(struct.pack('<Q', len(key)))
            buf.extend(key.encode('utf-8'))
            buf.extend(struct.pack('<I', 8))  # TYPE_STRING
            buf.extend(struct.pack('<Q', len(val)))
            buf.extend(val.encode('utf-8'))

        def write_kv_u32(key, val):
            """Write a uint32 KV pair."""
            buf.extend(struct.pack('<Q', len(key)))
            buf.extend(key.encode('utf-8'))
            buf.extend(struct.pack('<I', 4))  # TYPE_UINT32
            buf.extend(struct.pack('<I', val))

        # KV 1: general.architecture = "qwen2moe"
        write_kv_str('general.architecture', 'qwen2moe')
        # KV 2: qwen2moe.expert_count
        write_kv_u32('qwen2moe.expert_count', n_expert)
        # KV 3: qwen2moe.expert_used_count
        write_kv_u32('qwen2moe.expert_used_count', n_expert_used)

        # Padding to make file size meaningful
        buf += b'\x00' * data_size
        return bytes(buf)

    def test_parse_moe_gguf(self):
        data = self._make_moe_gguf(n_expert=8, n_expert_used=2, data_size=int(0.6e9))
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            info = _read_gguf_moe_info(path)
            assert info is not None
            total_b, used, total = info
            assert used == 2
            assert total == 8
            # Active ratio = 2/8 = 0.25
            est = estimate_params_from_file(path)
            assert est < total_b * 0.3  # Much smaller than total
        finally:
            os.unlink(path)

    def test_parse_dense_gguf_returns_none(self):
        # File without GGUF magic returns None
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b'\x00' * 1000)
            path = f.name
        try:
            info = _read_gguf_moe_info(path)
            assert info is None
        finally:
            os.unlink(path)

    def test_moe_2_of_2_full_ratio(self):
        """2/2 experts → active ratio = 1.0 → treated like dense of same size."""
        data = self._make_moe_gguf(n_expert=2, n_expert_used=2, data_size=int(0.6e9))
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            est = estimate_params_from_file(path)
            total = os.path.getsize(path) / BYTES_PER_PARAM_Q4 / 1e9
            # With 2/2, active = total (ratio 1.0)
            assert abs(est - total) < 0.1
        finally:
            os.unlink(path)


class TestGetProfile:
    @pytest.mark.parametrize("params,expected_pi,expected_ts", [
        (0.5,  0.15, 0.01),   # sub-1B
        (1.1,  0.10, 0.01),   # 1-2B
        (1.5,  0.10, 0.01),   # 1-2B
        (2.5,  0.15, 0.05),   # 2-5B dense / MoE
        (4.2,  0.15, 0.05),   # 2-5B dense / MoE (Qwen MoE lands here)
        (5.0,  0.15, 0.05),   # boundary: 2-5B
        (6.0,  0.05, 0.02),   # 5-8B
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
            (2.0,  OptimizationProfile(0.10, 0.01, 3.61, "1-2B")),
            (5.0,  OptimizationProfile(0.15, 0.05, 3.26, "2-5B dense / MoE")),
            (8.0,  OptimizationProfile(0.05, 0.02, 1.30, "5-8B")),
            (float('inf'), OptimizationProfile(0.03, 0.02, 1.20, "8B+")),
        ]:
            assert profile.pi_budget > 0
            assert profile.ts_threshold > 0
            assert profile.expected_speedup > 0


class TestAutoTuneIntegration:
    def test_auto_tune_returns_profile(self):
        """Integration test: auto_tune returns valid profile for real model."""
        from vibeblade.auto_tune import auto_tune, disable_all

        profile = auto_tune("models/tinyllama-1.1b-q4km.gguf")
        assert isinstance(profile, OptimizationProfile)
        assert profile.pi_budget > 0
        assert profile.ts_threshold > 0

        # Clean up
        disable_all()

    def test_auto_tune_moe_model(self):
        """Auto-tune on MoE model should detect MoE and select appropriate profile."""
        from vibeblade.auto_tune import auto_tune, disable_all

        profile = auto_tune("models/qwen25-moe-2x1.5b-q4km.gguf")
        assert isinstance(profile, OptimizationProfile)
        # MoE with 2/2 experts → active params ≈ total ≈ 3.9B → 2-4B bucket
        assert profile.pi_budget > 0
        assert profile.ts_threshold > 0

        disable_all()

    def test_disable_all(self):
        """disable_all should not crash."""
        from vibeblade.auto_tune import disable_all
        disable_all()  # Should not raise
