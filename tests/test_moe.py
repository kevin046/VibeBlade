"""Tests for vibeblade.moe — Mixture of Experts module.

All tests use numpy only, no GPU required.
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, "/home/ubuntu/vibeblade")

from vibeblade.moe import (
    MoEConfig,
    ExpertRouter,
    MoEExpertSet,
    detect_moe_config,
    moe_ffn_silu,
    load_moe_weights_from_layer,
    _silu,
    _dense_ffn,
    _softmax,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_weights(
    num_experts=4,
    shared_dim=16,
    expert_dim=32,
    topk=2,
    seed=42,
):
    """Build router + expert weights with deterministic seed."""
    rng = np.random.default_rng(seed)
    router_w = rng.standard_normal((shared_dim, num_experts)).astype(np.float32)
    gate_w = rng.standard_normal((num_experts, shared_dim, expert_dim)).astype(np.float32)
    up_w   = rng.standard_normal((num_experts, shared_dim, expert_dim)).astype(np.float32)
    down_w = rng.standard_normal((num_experts, expert_dim, shared_dim)).astype(np.float32)
    return router_w, gate_w, up_w, down_w


# ═══════════════════════════════════════════════════════════════════════════════
# MoEConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoEConfig:
    def test_basic(self):
        cfg = MoEConfig(num_experts=8, num_active=2, expert_dim=128, shared_dim=64)
        assert cfg.num_experts == 8
        assert cfg.num_active == 2
        assert cfg.router_topk == 2  # defaults to num_active

    def test_router_topk_override(self):
        cfg = MoEConfig(num_experts=8, num_active=2, expert_dim=128,
                        shared_dim=64, router_topk=4)
        assert cfg.router_topk == 4

    def test_from_tensors(self):
        router_w, gate_w, up_w, down_w = make_weights(num_experts=6, shared_dim=32, expert_dim=64)
        cfg = MoEConfig.from_tensors(router_w, gate_w, up_w, down_w, num_active=3)
        assert cfg.num_experts == 6
        assert cfg.expert_dim == 64
        assert cfg.shared_dim == 32
        assert cfg.num_active == 3

    def test_repr(self):
        cfg = MoEConfig(num_experts=8, num_active=2, expert_dim=128, shared_dim=64)
        r = repr(cfg)
        assert "num_experts=8" in r
        assert "expert_dim=128" in r


# ═══════════════════════════════════════════════════════════════════════════════
# _softmax
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoftmax:
    def test_1d(self):
        x = np.array([1.0, 2.0, 3.0])
        s = _softmax(x)
        assert s.shape == (3,)
        assert abs(s.sum() - 1.0) < 1e-6
        # Monotonically increasing
        assert s[2] > s[1] > s[0]

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 0.0]])
        s = _softmax(x, axis=-1)
        assert s.shape == (2, 2)
        assert np.allclose(s.sum(axis=-1), 1.0)

    def test_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        s = _softmax(x)
        assert not np.any(np.isnan(s))
        assert not np.any(np.isinf(s))
        assert abs(s.sum() - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpertRouter:
    def test_route_1d(self):
        router_w, _, _, _ = make_weights(num_experts=4, shared_dim=16)
        router = ExpertRouter(router_w, topk=2)
        x = np.random.randn(16).astype(np.float32)
        indices, weights = router.route(x)
        assert indices.shape == (2,)
        assert weights.shape == (2,)
        assert np.allclose(weights.sum(), 1.0)
        assert np.all(indices >= 0)
        assert np.all(indices < 4)

    def test_route_2d(self):
        router_w, _, _, _ = make_weights(num_experts=8, shared_dim=32)
        router = ExpertRouter(router_w, topk=3)
        x = np.random.randn(5, 32).astype(np.float32)
        indices, weights = router.route(x)
        assert indices.shape == (5, 3)
        assert weights.shape == (5, 3)
        # Each row's weights should sum to 1
        assert np.allclose(weights.sum(axis=-1), 1.0)

    def test_route_weights_nonnegative(self):
        router_w, _, _, _ = make_weights(num_experts=4, shared_dim=16)
        router = ExpertRouter(router_w, topk=2)
        x = np.random.randn(16).astype(np.float32)
        _, weights = router.route(x)
        assert np.all(weights >= 0)

    def test_predict(self):
        router_w, _, _, _ = make_weights(num_experts=4, shared_dim=16)
        router = ExpertRouter(router_w, topk=2)
        x = np.random.randn(16).astype(np.float32)
        logits = router.predict(x)
        assert logits.shape == (4,)
        # Same result for batch-1
        logits_b = router.predict(x[np.newaxis, :])
        assert np.allclose(logits, logits_b[0])

    def test_topk_consistency(self):
        """Top-k indices should correspond to the highest logits."""
        router_w, _, _, _ = make_weights(num_experts=8, shared_dim=16, seed=0)
        router = ExpertRouter(router_w, topk=2)
        x = np.random.randn(16).astype(np.float32)
        indices, _ = router.route(x)
        logits = router.predict(x)
        top2 = np.argsort(logits)[-2:]
        assert set(indices.tolist()) == set(top2.tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# MoEExpertSet
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoEExpertSet:
    def test_basic(self):
        _, gate_w, up_w, down_w = make_weights(num_experts=4, shared_dim=16, expert_dim=32)
        es = MoEExpertSet(gate_w, up_w, down_w)
        assert es.num_experts == 4
        assert es.expert_dim == 32
        assert es.shared_dim == 16

    def test_get_expert(self):
        _, gate_w, up_w, down_w = make_weights(num_experts=4, shared_dim=16, expert_dim=32)
        es = MoEExpertSet(gate_w, up_w, down_w)
        g, u, d = es.get_expert(2)
        assert g.shape == (16, 32)
        assert u.shape == (16, 32)
        assert d.shape == (32, 16)
        # Should be the same as the original slice
        np.testing.assert_array_equal(g, gate_w[2])

    def test_get_experts_batch(self):
        _, gate_w, up_w, down_w = make_weights(num_experts=4, shared_dim=16, expert_dim=32)
        es = MoEExpertSet(gate_w, up_w, down_w)
        indices = np.array([0, 3, 1, 2])
        g, u, d = es.get_experts_batch(indices)
        assert g.shape == (4, 16, 32)
        assert u.shape == (4, 16, 32)
        assert d.shape == (4, 32, 16)
        np.testing.assert_array_equal(g[0], gate_w[0])
        np.testing.assert_array_equal(g[1], gate_w[3])

    def test_shape_mismatch(self):
        gate_w = np.zeros((4, 16, 32), dtype=np.float32)
        # down_w should be (num_experts, expert_dim, shared_dim) = (4, 32, 16)
        # Use (4, 32, 8) — shared_dim 8 ≠ gate's shared_dim 16
        bad_down = np.zeros((4, 32, 8), dtype=np.float32)
        with pytest.raises(AssertionError):
            MoEExpertSet(gate_w, gate_w, bad_down)


# ═══════════════════════════════════════════════════════════════════════════════
# moe_ffn_silu
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoEFFnSiLU:
    def test_1d_input(self):
        """Single-token (1D) input produces 1D output."""
        router_w, gate_w, up_w, down_w = make_weights(
            num_experts=4, shared_dim=16, expert_dim=32, seed=0
        )
        router = ExpertRouter(router_w, topk=2)
        experts = MoEExpertSet(gate_w, up_w, down_w)
        x = np.random.randn(16).astype(np.float32)
        out, stats = moe_ffn_silu(x, router, experts)
        assert out.shape == (16,)
        assert stats["num_experts"] == 4
        assert stats["topk"] == 2
        assert stats["has_shared_expert"] is False
        # moe_ffn_silu converts 1D to 2D internally, so stats are always batched
        assert stats["expert_indices"].shape == (1, 2)

    def test_2d_input(self):
        """Batch (2D) input produces 2D output."""
        router_w, gate_w, up_w, down_w = make_weights(
            num_experts=4, shared_dim=16, expert_dim=32, seed=0
        )
        router = ExpertRouter(router_w, topk=2)
        experts = MoEExpertSet(gate_w, up_w, down_w)
        x = np.random.randn(5, 16).astype(np.float32)
        out, stats = moe_ffn_silu(x, router, experts)
        assert out.shape == (5, 16)
        assert stats["batch_size"] == 5
        assert stats["expert_indices"].shape == (5, 2)

    def test_with_shared_expert(self):
        rng = np.random.default_rng(7)
        router_w, gate_w, up_w, down_w = make_weights(
            num_experts=4, shared_dim=16, expert_dim=32, seed=0
        )
        # Shared expert weights use MoE convention: (shared_dim, expert_dim) for gate/up
        # and (expert_dim, shared_dim) for down — same as per-expert weights
        sh_gate = rng.standard_normal((16, 32)).astype(np.float32)
        sh_up   = rng.standard_normal((16, 32)).astype(np.float32)
        sh_down = rng.standard_normal((32, 16)).astype(np.float32)

        router = ExpertRouter(router_w, topk=2)
        experts = MoEExpertSet(gate_w, up_w, down_w)
        x = np.random.randn(16).astype(np.float32)

        out_no_shared, _ = moe_ffn_silu(x, router, experts)
        out_shared, stats = moe_ffn_silu(x, router, experts, sh_gate, sh_up, sh_down)

        assert stats["has_shared_expert"] is True
        # With shared expert, output should differ from without
        assert not np.allclose(out_no_shared, out_shared)

    def test_deterministic(self):
        """Same input → same output."""
        router_w, gate_w, up_w, down_w = make_weights(seed=123)
        router = ExpertRouter(router_w, topk=2)
        experts = MoEExpertSet(gate_w, up_w, down_w)
        x = np.random.randn(16).astype(np.float32)
        out1, _ = moe_ffn_silu(x, router, experts)
        out2, _ = moe_ffn_silu(x, router, experts)
        np.testing.assert_array_equal(out1, out2)

    def test_no_nan(self):
        router_w, gate_w, up_w, down_w = make_weights(
            num_experts=8, shared_dim=64, expert_dim=128, seed=0
        )
        router = ExpertRouter(router_w, topk=4)
        experts = MoEExpertSet(gate_w, up_w, down_w)
        x = np.random.randn(32, 64).astype(np.float32)
        out, _ = moe_ffn_silu(x, router, experts)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


# ═══════════════════════════════════════════════════════════════════════════════
# detect_moe_config
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectMoEConfig:
    def test_gguf_layout(self):
        rng = np.random.default_rng(0)
        weights = {
            "blk.0.ffn_gate_inp.weight": rng.standard_normal((16, 4)).astype(np.float32),
            "blk.0.ffn_gate_exps.weight": rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.0.ffn_up_exps.weight":   rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.0.ffn_down_exps.weight": rng.standard_normal((4, 32, 16)).astype(np.float32),
        }
        cfg = detect_moe_config(weights)
        assert cfg is not None
        assert cfg.num_experts == 4
        assert cfg.expert_dim == 32
        assert cfg.shared_dim == 16
        assert cfg.num_active == 2  # default

    def test_gguf_with_shared(self):
        rng = np.random.default_rng(0)
        weights = {
            "blk.0.ffn_gate_inp.weight":  rng.standard_normal((16, 4)).astype(np.float32),
            "blk.0.ffn_gate_exps.weight":  rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.0.ffn_up_exps.weight":    rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.0.ffn_down_exps.weight":  rng.standard_normal((4, 32, 16)).astype(np.float32),
            "blk.0.ffn_gate_shunt.weight": rng.standard_normal((16, 32)).astype(np.float32),
        }
        cfg = detect_moe_config(weights)
        assert cfg is not None
        assert cfg.has_shared_expert is True

    def test_alt_layout(self):
        rng = np.random.default_rng(0)
        weights = {
            "blk.0.ffn_router.weight":       rng.standard_normal((16, 3)).astype(np.float32),
            "blk.0.expert.0.ffn_gate.weight": rng.standard_normal((32, 16)).astype(np.float32),
            "blk.0.expert.0.ffn_up.weight":   rng.standard_normal((32, 16)).astype(np.float32),
            "blk.0.expert.0.ffn_down.weight": rng.standard_normal((16, 32)).astype(np.float32),
            "blk.0.expert.1.ffn_gate.weight": rng.standard_normal((32, 16)).astype(np.float32),
            "blk.0.expert.1.ffn_up.weight":   rng.standard_normal((32, 16)).astype(np.float32),
            "blk.0.expert.1.ffn_down.weight": rng.standard_normal((16, 32)).astype(np.float32),
            "blk.0.expert.2.ffn_gate.weight": rng.standard_normal((32, 16)).astype(np.float32),
            "blk.0.expert.2.ffn_up.weight":   rng.standard_normal((32, 16)).astype(np.float32),
            "blk.0.expert.2.ffn_down.weight": rng.standard_normal((16, 32)).astype(np.float32),
        }
        cfg = detect_moe_config(weights)
        assert cfg is not None
        assert cfg.num_experts == 3
        assert cfg.shared_dim == 16
        assert cfg.expert_dim == 32

    def test_dense_returns_none(self):
        weights = {
            "blk.0.ffn_gate.weight": np.zeros((16, 32), dtype=np.float32),
            "blk.0.ffn_up.weight":   np.zeros((16, 32), dtype=np.float32),
            "blk.0.ffn_down.weight": np.zeros((32, 16), dtype=np.float32),
        }
        cfg = detect_moe_config(weights)
        assert cfg is None

    def test_empty_returns_none(self):
        cfg = detect_moe_config({})
        assert cfg is None


# ═══════════════════════════════════════════════════════════════════════════════
# load_moe_weights_from_layer
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadMoEWeights:
    def test_gguf_consolidated(self):
        rng = np.random.default_rng(0)
        weights = {
            "blk.3.ffn_gate_inp.weight": rng.standard_normal((16, 4)).astype(np.float32),
            "blk.3.ffn_gate_exps.weight": rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.3.ffn_up_exps.weight":   rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.3.ffn_down_exps.weight": rng.standard_normal((4, 32, 16)).astype(np.float32),
        }
        router_w, expert_set, extra = load_moe_weights_from_layer(weights, 3)
        assert router_w is not None
        assert expert_set is not None
        assert router_w.shape == (16, 4)
        assert expert_set.num_experts == 4

    def test_gguf_with_shared(self):
        rng = np.random.default_rng(0)
        weights = {
            "blk.0.ffn_gate_inp.weight":  rng.standard_normal((16, 4)).astype(np.float32),
            "blk.0.ffn_gate_exps.weight":  rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.0.ffn_up_exps.weight":    rng.standard_normal((4, 16, 32)).astype(np.float32),
            "blk.0.ffn_down_exps.weight":  rng.standard_normal((4, 32, 16)).astype(np.float32),
            "blk.0.ffn_gate_shunt.weight": rng.standard_normal((16, 32)).astype(np.float32),
            "blk.0.ffn_up_shunt.weight":   rng.standard_normal((16, 32)).astype(np.float32),
            "blk.0.ffn_down_shunt.weight": rng.standard_normal((32, 16)).astype(np.float32),
        }
        router_w, expert_set, extra = load_moe_weights_from_layer(weights, 0)
        assert router_w is not None
        assert expert_set is not None
        assert "shared_gate" in extra
        assert "shared_up" in extra
        assert "shared_down" in extra

    def test_alt_per_expert(self):
        rng = np.random.default_rng(0)
        weights = {}
        weights["blk.5.ffn_router.weight"] = rng.standard_normal((16, 3)).astype(np.float32)
        for e in range(3):
            weights[f"blk.5.expert.{e}.ffn_gate.weight"] = rng.standard_normal((32, 16)).astype(np.float32)
            weights[f"blk.5.expert.{e}.ffn_up.weight"]   = rng.standard_normal((32, 16)).astype(np.float32)
            weights[f"blk.5.expert.{e}.ffn_down.weight"] = rng.standard_normal((16, 32)).astype(np.float32)

        router_w, expert_set, extra = load_moe_weights_from_layer(weights, 5)
        assert router_w is not None
        assert expert_set is not None
        assert expert_set.num_experts == 3

    def test_dense_layer_returns_none(self):
        weights = {
            "blk.0.ffn_gate.weight": np.zeros((16, 32), dtype=np.float32),
            "blk.0.ffn_up.weight":   np.zeros((16, 32), dtype=np.float32),
            "blk.0.ffn_down.weight": np.zeros((32, 16), dtype=np.float32),
        }
        router_w, expert_set, extra = load_moe_weights_from_layer(weights, 0)
        assert router_w is None
        assert expert_set is None

    def test_end_to_end_route_and_ffn(self):
        """Build weights → load → route → FFN → check output shape & no NaN."""
        rng = np.random.default_rng(42)
        num_experts, shared_dim, expert_dim, topk = 8, 64, 128, 2
        weights = {
            "blk.0.ffn_gate_inp.weight": rng.standard_normal((shared_dim, num_experts)).astype(np.float32),
            "blk.0.ffn_gate_exps.weight": rng.standard_normal((num_experts, shared_dim, expert_dim)).astype(np.float32),
            "blk.0.ffn_up_exps.weight":   rng.standard_normal((num_experts, shared_dim, expert_dim)).astype(np.float32),
            "blk.0.ffn_down_exps.weight": rng.standard_normal((num_experts, expert_dim, shared_dim)).astype(np.float32),
        }
        router_w, expert_set, _ = load_moe_weights_from_layer(weights, 0)
        router = ExpertRouter(router_w, topk=topk)
        x = rng.standard_normal((shared_dim,)).astype(np.float32)
        out, stats = moe_ffn_silu(x, router, expert_set)
        assert out.shape == (shared_dim,)
        assert not np.any(np.isnan(out))
        assert stats["num_experts"] == num_experts
        assert stats["topk"] == topk


# ═══════════════════════════════════════════════════════════════════════════════
# _silu / _dense_ffn consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_silu_matches_transformer(self):
        """Verify _silu matches transformer.silu."""
        from vibeblade.transformer import silu
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 3.0]).astype(np.float32)
        np.testing.assert_allclose(_silu(x), silu(x), rtol=1e-6)

    def test_dense_ffn_1d(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(16).astype(np.float32)
        # MoE convention: gate/up (shared_dim, expert_dim), down (expert_dim, shared_dim)
        g = rng.standard_normal((16, 32)).astype(np.float32)
        u = rng.standard_normal((16, 32)).astype(np.float32)
        d = rng.standard_normal((32, 16)).astype(np.float32)
        out = _dense_ffn(x, g, u, d)
        assert out.shape == (16,)

    def test_dense_ffn_2d(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 16)).astype(np.float32)
        # MoE convention: gate/up (shared_dim, expert_dim), down (expert_dim, shared_dim)
        g = rng.standard_normal((16, 32)).astype(np.float32)
        u = rng.standard_normal((16, 32)).astype(np.float32)
        d = rng.standard_normal((32, 16)).astype(np.float32)
        out = _dense_ffn(x, g, u, d)
        assert out.shape == (5, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
