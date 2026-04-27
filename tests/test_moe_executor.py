"""Tests for Hot/Cold MoE Split Executor."""

import numpy as np
from dataclasses import dataclass

from vibeblade.moe import ExpertRouter, MoEExpertSet
from vibeblade.moe_executor import HotColdExecutor, ExecutorStats

# Try importing the real HotColdMap; fall back to inline stub
try:
    from vibeblade.moe_profiler import HotColdMap
except ImportError:
    @dataclass
    class HotColdMap:
        hot_experts: dict[int, list[int]]
        cold_experts: dict[int, list[int]]
        num_layers: int = 1
        num_experts: int = 4
        profile_tokens: int = 100


# ── Fixtures ─────────────────────────────────────────────────────────────────


def make_router(shared_dim: int = 16, num_experts: int = 4, topk: int = 2) -> ExpertRouter:
    """Create a router with deterministic weights that route predictably."""
    rng = np.random.RandomState(42)
    weight = rng.randn(shared_dim, num_experts).astype(np.float32)
    # Bias expert 0 heavily so it's always selected first
    weight[:, 0] += 3.0
    weight[:, 1] += 1.0
    return ExpertRouter(weight, topk=topk)


def make_expert_set(
    num_experts: int = 4, shared_dim: int = 16, expert_dim: int = 32
) -> MoEExpertSet:
    """Create a small MoEExpertSet with random weights."""
    rng = np.random.RandomState(123)
    gate = rng.randn(num_experts, shared_dim, expert_dim).astype(np.float32) * 0.1
    up = rng.randn(num_experts, shared_dim, expert_dim).astype(np.float32) * 0.1
    down = rng.randn(num_experts, expert_dim, shared_dim).astype(np.float32) * 0.1
    return MoEExpertSet(gate, up, down)


def make_shared_expert(shared_dim: int = 16, expert_dim: int = 32):
    """Create shared expert weights (gate_w, up_w, down_w)."""
    rng = np.random.RandomState(99)
    gate_w = rng.randn(shared_dim, expert_dim).astype(np.float32) * 0.1
    up_w = rng.randn(shared_dim, expert_dim).astype(np.float32) * 0.1
    down_w = rng.randn(expert_dim, shared_dim).astype(np.float32) * 0.1
    return (gate_w, up_w, down_w)


def make_executor(
    shared_dim: int = 16,
    num_experts: int = 4,
    topk: int = 2,
    expert_dim: int = 32,
    hot_ids: list[int] | None = None,
    gpu_backend=None,
    cpu_threads: int = 4,
    shared_expert: bool = False,
) -> HotColdExecutor:
    """Create a fully configured HotColdExecutor for testing."""
    if hot_ids is None:
        hot_ids = [0]
    cold_ids = [i for i in range(num_experts) if i not in hot_ids]

    router = make_router(shared_dim, num_experts, topk)
    expert_set = make_expert_set(num_experts, shared_dim, expert_dim)

    hot_cold_map = HotColdMap(
        hot_experts={0: hot_ids},
        cold_experts={0: cold_ids},
        num_layers=1,
        num_experts=num_experts,
    )

    shared = {}
    if shared_expert:
        shared[0] = make_shared_expert(shared_dim, expert_dim)

    return HotColdExecutor(
        hot_cold_map=hot_cold_map,
        routers={0: router},
        experts={0: expert_set},
        shared_experts=shared,
        gpu_backend=gpu_backend,
        cpu_threads=cpu_threads,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDispatchNoGPU:
    """Test dispatch with no GPU (NumPy fallback path)."""

    def test_dispatch_returns_correct_shape_1d(self):
        """1D input returns 1D output."""
        executor = make_executor(shared_dim=16, num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        output, stats = executor.dispatch(layer_idx=0, x=x)
        assert output.shape == (16,), f"Expected (16,), got {output.shape}"

    def test_dispatch_returns_correct_shape_2d(self):
        """2D input (batch=1) returns 2D output."""
        executor = make_executor(shared_dim=16, num_experts=4, topk=2)
        x = np.random.randn(1, 16).astype(np.float32)
        output, stats = executor.dispatch(layer_idx=0, x=x)
        assert output.shape == (1, 16), f"Expected (1, 16), got {output.shape}"

    def test_dispatch_stats_has_required_keys(self):
        """Stats dict has all expected keys."""
        executor = make_executor()
        x = np.random.randn(16).astype(np.float32)
        _, stats = executor.dispatch(layer_idx=0, x=x)
        for key in [
            "gpu_hits", "cpu_falls", "hit_rate",
            "gpu_latency_ms", "cpu_latency_ms", "sync_latency_ms",
            "expert_indices", "expert_weights",
        ]:
            assert key in stats, f"Missing key '{key}' in stats"

    def test_dispatch_returns_finite_output(self):
        """Output should not contain NaN or Inf."""
        executor = make_executor(shared_dim=16, num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        output, _ = executor.dispatch(layer_idx=0, x=x)
        assert np.all(np.isfinite(output)), "Output contains NaN or Inf"

    def test_dispatch_missing_layer_returns_zeros(self):
        """Dispatching to a non-existent layer returns zeros."""
        executor = make_executor()
        x = np.random.randn(16).astype(np.float32)
        output, stats = executor.dispatch(layer_idx=99, x=x)
        assert output.shape == (16,)
        assert np.allclose(output, 0.0)


class TestHotColdSplitting:
    """Test that hot/cold classification is tracked correctly."""

    def test_hot_expert_tracked_as_gpu_hit(self):
        """When all experts are hot, all dispatched experts should be gpu_hits."""
        executor = make_executor(hot_ids=[0, 1, 2, 3], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        _, stats = executor.dispatch(layer_idx=0, x=x)
        assert stats["gpu_hits"] == 2, f"Expected 2 gpu_hits, got {stats['gpu_hits']}"
        assert stats["cpu_falls"] == 0

    def test_cold_expert_tracked_as_cpu_fall(self):
        """When all experts are cold, all dispatched experts should be cpu_falls."""
        executor = make_executor(hot_ids=[], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        _, stats = executor.dispatch(layer_idx=0, x=x)
        assert stats["cpu_falls"] == 2, f"Expected 2 cpu_falls, got {stats['cpu_falls']}"
        assert stats["gpu_hits"] == 0

    def test_all_hot_no_cpu_falls(self):
        """When ALL experts are hot, cpu_falls should be 0."""
        # Make every expert hot so no matter what the router selects
        executor = make_executor(hot_ids=[0, 1, 2, 3], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        _, stats = executor.dispatch(layer_idx=0, x=x)
        assert stats["cpu_falls"] == 0, f"Expected 0 cpu_falls, got {stats['cpu_falls']}"
        assert stats["gpu_hits"] == 2

    def test_all_cold_no_gpu_hits(self):
        """When ALL experts are cold, gpu_hits should be 0."""
        executor = make_executor(hot_ids=[], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        _, stats = executor.dispatch(layer_idx=0, x=x)
        assert stats["gpu_hits"] == 0, f"Expected 0 gpu_hits, got {stats['gpu_hits']}"
        assert stats["cpu_falls"] == 2

    def test_hit_rate_mixed(self):
        """Mixed hot/cold: hit_rate should be between 0 and 1."""
        executor = make_executor(hot_ids=[0], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        _, stats = executor.dispatch(layer_idx=0, x=x)
        assert 0.0 <= stats["hit_rate"] <= 1.0

    def test_cumulative_stats_across_dispatches(self):
        """Multiple dispatches accumulate stats correctly."""
        # All experts hot → every dispatch is 2 gpu_hits, 0 cpu_falls
        executor = make_executor(hot_ids=[0, 1, 2, 3], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        for _ in range(5):
            executor.dispatch(layer_idx=0, x=x)
        cum = executor.stats
        assert cum.gpu_hits == 10  # 2 hot * 5 dispatches
        assert cum.cpu_falls == 0


class TestSharedExpert:
    """Test shared expert inclusion."""

    def test_shared_expert_changes_output(self):
        """With shared expert, output should differ from without."""
        rng = np.random.RandomState(77)
        x = rng.randn(16).astype(np.float32)

        executor_no_shared = make_executor(shared_dim=16, shared_expert=False)
        executor_with_shared = make_executor(shared_dim=16, shared_expert=True)

        out_no, _ = executor_no_shared.dispatch(layer_idx=0, x=x.copy())
        out_yes, _ = executor_with_shared.dispatch(layer_idx=0, x=x.copy())

        # Shared expert adds a non-zero contribution
        assert not np.allclose(out_no, out_yes), "Shared expert should change output"

    def test_shared_expert_output_finite(self):
        """Shared expert output should be finite."""
        executor = make_executor(shared_dim=16, shared_expert=True)
        x = np.random.randn(16).astype(np.float32)
        output, _ = executor.dispatch(layer_idx=0, x=x)
        assert np.all(np.isfinite(output))


class TestMemoryEstimate:
    """Test memory estimation."""

    def test_memory_estimate_keys(self):
        """memory_estimate returns expected keys."""
        executor = make_executor(num_experts=4, shared_dim=16, expert_dim=32)
        est = executor.memory_estimate(expert_dim=32, shared_dim=16)
        for key in ["hot_vram_gb", "cold_ram_gb", "per_expert_mb", "total_hot_experts", "total_cold_experts"]:
            assert key in est, f"Missing key '{key}'"

    def test_memory_estimate_values(self):
        """Verify arithmetic for known dimensions with fp32."""
        executor = make_executor(num_experts=4, shared_dim=16, expert_dim=32)
        # Per expert: 3 * 16 * 32 = 1536 params
        # fp32 = 4 bytes → 1536 * 4 = 6144 bytes = 0.005859375 MB
        est = executor.memory_estimate(expert_dim=32, shared_dim=16, bytes_per_param=4.0)
        per_expert = 3 * 16 * 32 * 4.0
        assert abs(est["per_expert_mb"] - per_expert / (1024**2)) < 1e-10

    def test_memory_estimate_hot_cold_split(self):
        """1 hot + 3 cold → hot_vram and cold_ram split correctly."""
        executor = make_executor(num_experts=4, hot_ids=[0], shared_dim=16, expert_dim=32)
        est = executor.memory_estimate(expert_dim=32, shared_dim=16, bytes_per_param=2.0)
        assert est["total_hot_experts"] == 1
        assert est["total_cold_experts"] == 3
        # cold_ram should be 3x hot_vram
        assert abs(est["cold_ram_gb"] - 3 * est["hot_vram_gb"]) < 1e-15

    def test_memory_estimate_q4_quantization(self):
        """Q4 quantization (0.5 bytes/param) gives smaller sizes."""
        executor = make_executor(num_experts=4, shared_dim=4096, expert_dim=14336)
        est = executor.memory_estimate(expert_dim=14336, shared_dim=4096, bytes_per_param=0.5)
        assert est["hot_vram_gb"] > 0
        assert est["cold_ram_gb"] > 0


class TestLayerStats:
    """Test per-layer statistics."""

    def test_layer_stats_keys(self):
        executor = make_executor(hot_ids=[0, 1], num_experts=4)
        ls = executor.layer_stats(0)
        assert "layer" in ls
        assert "hot_experts" in ls
        assert "cold_experts" in ls
        assert "hot_pct" in ls

    def test_layer_stats_values(self):
        executor = make_executor(hot_ids=[0, 1], num_experts=4)
        ls = executor.layer_stats(0)
        assert ls["layer"] == 0
        assert ls["hot_experts"] == 2
        assert ls["cold_experts"] == 2
        assert abs(ls["hot_pct"] - 50.0) < 1e-10

    def test_layer_stats_missing_layer(self):
        executor = make_executor(hot_ids=[0], num_experts=4)
        ls = executor.layer_stats(99)
        assert ls["hot_experts"] == 0
        assert ls["cold_experts"] == 0
        assert ls["hot_pct"] == 0.0


class TestResetStats:
    """Test that stats can be reset."""

    def test_reset_clears_cumulative(self):
        executor = make_executor(hot_ids=[0, 1, 2, 3], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        executor.dispatch(layer_idx=0, x=x)
        assert executor.stats.gpu_hits > 0
        executor.reset_stats()
        assert executor.stats.gpu_hits == 0
        assert executor.stats.cpu_falls == 0
        assert executor.stats.gpu_latency_ms == 0.0

    def test_stats_accumulate_after_reset(self):
        executor = make_executor(hot_ids=[0, 1, 2, 3], num_experts=4, topk=2)
        x = np.random.randn(16).astype(np.float32)
        executor.dispatch(layer_idx=0, x=x)
        executor.reset_stats()
        executor.dispatch(layer_idx=0, x=x)
        assert executor.stats.gpu_hits == 2


class TestHitRateCalculation:
    """Test ExecutorStats hit_rate edge cases."""

    def test_hit_rate_all_hot(self):
        s = ExecutorStats(gpu_hits=5, cpu_falls=0)
        assert s.hit_rate == 1.0

    def test_hit_rate_all_cold(self):
        s = ExecutorStats(gpu_hits=0, cpu_falls=5)
        assert s.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        s = ExecutorStats(gpu_hits=3, cpu_falls=1)
        assert abs(s.hit_rate - 0.75) < 1e-10

    def test_hit_rate_zero_total(self):
        s = ExecutorStats(gpu_hits=0, cpu_falls=0)
        assert s.hit_rate == 0.0

    def test_hit_rate_single_hit(self):
        s = ExecutorStats(gpu_hits=1, cpu_falls=0)
        assert s.hit_rate == 1.0


class TestExecutorStatsSerialization:
    """Test ExecutorStats.to_dict()."""

    def test_to_dict_keys(self):
        s = ExecutorStats(gpu_hits=3, cpu_falls=1, gpu_latency_ms=0.5, cpu_latency_ms=1.2, sync_latency_ms=0.1)
        d = s.to_dict()
        expected_keys = {"gpu_hits", "cpu_falls", "gpu_latency_ms", "cpu_latency_ms", "sync_latency_ms", "hit_rate"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        s = ExecutorStats(gpu_hits=3, cpu_falls=1, gpu_latency_ms=0.5, cpu_latency_ms=1.2, sync_latency_ms=0.1)
        d = s.to_dict()
        assert d["gpu_hits"] == 3
        assert d["cpu_falls"] == 1
        assert d["gpu_latency_ms"] == 0.5
        assert d["cpu_latency_ms"] == 1.2
        assert d["sync_latency_ms"] == 0.1
        assert abs(d["hit_rate"] - 0.75) < 1e-4

    def test_to_dict_rounds_floats(self):
        s = ExecutorStats(gpu_latency_ms=0.1234567)
        d = s.to_dict()
        assert d["gpu_latency_ms"] == 0.123

    def test_to_dict_from_empty_stats(self):
        s = ExecutorStats()
        d = s.to_dict()
        assert d["gpu_hits"] == 0
        assert d["cpu_falls"] == 0
        assert d["hit_rate"] == 0.0


class TestShutdown:
    """Test executor cleanup."""

    def test_shutdown_does_not_raise(self):
        executor = make_executor(cpu_threads=4)
        executor.shutdown()
        # Dispatching after shutdown should still work (sequential fallback)
        x = np.random.randn(16).astype(np.float32)
        output, stats = executor.dispatch(layer_idx=0, x=x)
        assert output.shape == (16,)

    def test_shutdown_with_no_threads(self):
        executor = make_executor(cpu_threads=0)
        executor.shutdown()  # Should not raise
