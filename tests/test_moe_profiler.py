"""Tests for MoE Expert Activation Profiler."""

import json
import os
import tempfile

import numpy as np

from vibeblade.moe_profiler import ExpertProfiler, HotColdMap


class TestRecordActivations:
    """Record activations and verify counters."""

    def test_single_record(self):
        profiler = ExpertProfiler(num_layers=4, num_experts=8)
        profiler.record(0, np.array([3, 5]))
        # Each expert in the topk list gets +1 per call
        assert profiler._counters[0, 3] == 1
        assert profiler._counters[0, 5] == 1
        assert profiler._counters[0, 0] == 0
        assert profiler._layer_token_counts[0] == 1
        assert profiler.total_tokens == 1

    def test_multiple_records_same_layer(self):
        profiler = ExpertProfiler(num_layers=2, num_experts=4)
        profiler.record(0, np.array([0, 1]))
        profiler.record(0, np.array([0, 2]))
        profiler.record(0, np.array([1, 3]))
        assert profiler._counters[0, 0] == 2
        assert profiler._counters[0, 1] == 2
        assert profiler._counters[0, 2] == 1
        assert profiler._counters[0, 3] == 1
        assert profiler._layer_token_counts[0] == 3

    def test_record_different_layers(self):
        profiler = ExpertProfiler(num_layers=3, num_experts=4)
        profiler.record(0, np.array([0, 1]))
        profiler.record(1, np.array([2, 3]))
        assert profiler._counters[0, 0] == 1
        assert profiler._counters[1, 2] == 1
        assert profiler.total_tokens == 1  # max across layers

    def test_batch_input_2d(self):
        """2D expert_indices: (batch, topk)."""
        profiler = ExpertProfiler(num_layers=2, num_experts=8)
        # batch=4, topk=2
        indices = np.array([
            [0, 1],
            [2, 3],
            [0, 2],
            [1, 3],
        ])
        profiler.record(0, indices)
        # Expert 0 activated in tokens 0,2 → count 2
        assert profiler._counters[0, 0] == 2
        assert profiler._counters[0, 1] == 2
        assert profiler._counters[0, 2] == 2
        assert profiler._counters[0, 3] == 2
        assert profiler._layer_token_counts[0] == 1  # one record() call


class TestGetActivationFreq:
    """Test get_activation_freq returns correct frequencies."""

    def test_basic_freq(self):
        profiler = ExpertProfiler(num_layers=2, num_experts=4)
        profiler.record(0, np.array([0, 0]))  # expert 0 picked twice in one step
        profiler.record(0, np.array([1, 2]))
        freq = profiler.get_activation_freq(0)
        # 2 records total; expert 0: 2/2=1.0, expert 1: 1/2=0.5, expert 2: 1/2=0.5, expert 3: 0/2=0.0
        assert freq[0] == 1.0
        assert freq[1] == 0.5
        assert freq[2] == 0.5
        assert freq[3] == 0.0

    def test_empty_layer_returns_zeros(self):
        profiler = ExpertProfiler(num_layers=3, num_experts=4)
        freq = profiler.get_activation_freq(2)
        assert np.allclose(freq, 0.0)
        assert len(freq) == 4

    def test_sums_to_reasonable(self):
        profiler = ExpertProfiler(num_layers=1, num_experts=8)
        for _ in range(10):
            profiler.record(0, np.array([3, 5]))
        freq = profiler.get_activation_freq(0)
        assert np.isclose(freq[3], 1.0)
        assert np.isclose(freq[5], 1.0)
        assert np.isclose(freq[0], 0.0)


class TestGetAllFrequencies:
    """Test get_all_frequencies returns the full matrix."""

    def test_shape(self):
        profiler = ExpertProfiler(num_layers=5, num_experts=10)
        freqs = profiler.get_all_frequencies()
        assert freqs.shape == (5, 10)

    def test_values(self):
        profiler = ExpertProfiler(num_layers=2, num_experts=4)
        profiler.record(0, np.array([0, 1]))
        profiler.record(1, np.array([2, 3]))
        freqs = profiler.get_all_frequencies()
        assert np.isclose(freqs[0, 0], 1.0)
        assert np.isclose(freqs[0, 1], 1.0)
        assert np.isclose(freqs[0, 2], 0.0)
        assert np.isclose(freqs[1, 2], 1.0)


class TestComputeHotColdMap:
    """Test compute_hot_cold_map with known patterns."""

    def test_hot_experts_are_most_frequent(self):
        """The top-10% by frequency should be the hot experts."""
        num_experts = 10
        profiler = ExpertProfiler(num_layers=2, num_experts=num_experts)
        # Expert 9 gets all the love on layer 0
        for _ in range(100):
            profiler.record(0, np.array([9, 8]))
        # Expert 9: 100 activations, expert 8: 100, rest: 0
        hot_map = profiler.compute_hot_cold_map(hot_ratio=0.2)
        # 20% of 10 = 2 hot experts
        assert len(hot_map.hot_experts[0]) == 2
        assert 9 in hot_map.hot_experts[0]
        assert 8 in hot_map.hot_experts[0]
        # Cold should have the remaining 8
        assert len(hot_map.cold_experts[0]) == 8
        assert 0 in hot_map.cold_experts[0]

    def test_all_layers_present(self):
        profiler = ExpertProfiler(num_layers=4, num_experts=8)
        for layer in range(4):
            profiler.record(layer, np.array([layer, layer]))  # each layer activates its own index
        hot_map = profiler.compute_hot_cold_map(hot_ratio=0.25)
        for layer in range(4):
            assert layer in hot_map.hot_experts
            assert layer in hot_map.cold_experts

    def test_metadata(self):
        profiler = ExpertProfiler(num_layers=3, num_experts=16)
        profiler.record(0, np.array([0, 1]))
        hot_map = profiler.compute_hot_cold_map()
        assert hot_map.num_layers == 3
        assert hot_map.num_experts == 16
        assert hot_map.profile_tokens == 1


class TestHotRatioEdgeCases:
    """Test hot_ratio edge cases (0.0, 1.0)."""

    def test_hot_ratio_zero(self):
        """hot_ratio=0.0 → int(0)=0 → max(1, 0)=1 hot expert per layer."""
        profiler = ExpertProfiler(num_layers=2, num_experts=8)
        for _ in range(10):
            profiler.record(0, np.array([0, 1]))
        hot_map = profiler.compute_hot_cold_map(hot_ratio=0.0)
        # n_hot = max(1, int(8 * 0.0)) = max(1, 0) = 1
        assert len(hot_map.hot_experts[0]) == 1

    def test_hot_ratio_one(self):
        """hot_ratio=1.0 → all experts are hot."""
        profiler = ExpertProfiler(num_layers=1, num_experts=6)
        for _ in range(5):
            profiler.record(0, np.array([0, 1]))
        hot_map = profiler.compute_hot_cold_map(hot_ratio=1.0)
        assert len(hot_map.hot_experts[0]) == 6
        assert len(hot_map.cold_experts[0]) == 0


class TestMinHotCount:
    """Test min_hot_count filtering."""

    def test_filters_out_low_count(self):
        """Experts with count < min_hot_count should not be hot even if in top-k."""
        profiler = ExpertProfiler(num_layers=1, num_experts=4)
        # Expert 3: 5 activations, expert 0: 2 activations, rest: 0
        for _ in range(5):
            profiler.record(0, np.array([3, 3]))
        for _ in range(2):
            profiler.record(0, np.array([0, 0]))
        # hot_ratio=0.5 → 2 hot experts, top-2 are [0,3]
        # min_hot_count=4 → only expert 3 passes (count=10 >= 4), expert 0 has count=4
        # Actually expert 0 has count=4, so use min_hot_count=5 to filter it
        hot_map = profiler.compute_hot_cold_map(hot_ratio=0.5, min_hot_count=5)
        assert 3 in hot_map.hot_experts[0]
        assert 0 not in hot_map.hot_experts[0]

    def test_high_min_count_removes_all(self):
        """Very high min_hot_count should produce empty hot set (but n_hot>=1 from argsort)."""
        profiler = ExpertProfiler(num_layers=1, num_experts=4)
        profiler.record(0, np.array([0, 1]))
        hot_map = profiler.compute_hot_cold_map(hot_ratio=0.25, min_hot_count=999)
        # top-k gives 1 candidate, but its count is 1 < 999, so hot = []
        assert hot_map.hot_experts[0] == []
        assert len(hot_map.cold_experts[0]) == 4


class TestHitRate:
    """Test HotColdMap.hit_rate()."""

    def test_full_hit(self):
        hot_map = HotColdMap(
            hot_experts={0: [0, 1]},
            cold_experts={0: [2, 3]},
            num_layers=1,
            num_experts=4,
        )
        rate = hot_map.hit_rate(0, np.array([0, 1]))
        assert rate == 1.0

    def test_no_hit(self):
        hot_map = HotColdMap(
            hot_experts={0: [0, 1]},
            cold_experts={0: [2, 3]},
            num_layers=1,
            num_experts=4,
        )
        rate = hot_map.hit_rate(0, np.array([2, 3]))
        assert rate == 0.0

    def test_partial_hit(self):
        hot_map = HotColdMap(
            hot_experts={0: [0, 1, 2]},
            cold_experts={0: [3, 4]},
            num_layers=1,
            num_experts=5,
        )
        rate = hot_map.hit_rate(0, np.array([1, 4]))
        assert rate == 0.5

    def test_missing_layer(self):
        hot_map = HotColdMap(
            hot_experts={0: [0]},
            cold_experts={0: [1]},
            num_layers=2,
            num_experts=2,
        )
        # Layer 99 not in hot_experts → empty hot set → 0% hit
        rate = hot_map.hit_rate(99, np.array([0, 1]))
        assert rate == 0.0

    def test_empty_indices(self):
        hot_map = HotColdMap(
            hot_experts={0: [0]},
            cold_experts={0: [1]},
            num_layers=1,
            num_experts=2,
        )
        # Empty array → denominator = max(0, 1) = 1 → 0/1 = 0.0
        rate = hot_map.hit_rate(0, np.array([]))
        assert rate == 0.0


class TestOverallHitRate:
    """Test HotColdMap.overall_hit_rate()."""

    def test_basic(self):
        hot_map = HotColdMap(
            hot_experts={0: [0, 1], 1: [0]},
            cold_experts={0: [2, 3], 1: [1, 2, 3]},
            num_layers=2,
            num_experts=4,
        )
        # (2 + 1) / (2 * 4) = 3/8 = 0.375
        assert np.isclose(hot_map.overall_hit_rate(), 3.0 / 8.0)

    def test_empty(self):
        hot_map = HotColdMap(
            hot_experts={},
            cold_experts={},
            num_layers=0,
            num_experts=0,
        )
        assert hot_map.overall_hit_rate() == 0.0


class TestSaveLoad:
    """Test HotColdMap.save() / load() roundtrip."""

    def test_roundtrip(self):
        hot_map = HotColdMap(
            hot_experts={0: [0, 1, 2], 1: [5, 7]},
            cold_experts={0: [3, 4], 1: [0, 1, 2, 3, 4, 6]},
            num_layers=2,
            num_experts=8,
            profile_tokens=1000,
            profile_dataset="test-dataset",
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            hot_map.save(path)
            loaded = HotColdMap.load(path)
            assert loaded.hot_experts == hot_map.hot_experts
            assert loaded.cold_experts == hot_map.cold_experts
            assert loaded.num_layers == hot_map.num_layers
            assert loaded.num_experts == hot_map.num_experts
            assert loaded.profile_tokens == hot_map.profile_tokens
            assert loaded.profile_dataset == hot_map.profile_dataset
        finally:
            os.unlink(path)

    def test_json_keys_are_strings(self):
        """Verify the serialized JSON has string keys."""
        hot_map = HotColdMap(
            hot_experts={0: [0], 1: [1]},
            cold_experts={0: [1], 1: [0]},
            num_layers=2,
            num_experts=2,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            hot_map.save(path)
            with open(path) as f:
                data = json.load(f)
            # Keys must be strings in JSON
            assert all(isinstance(k, str) for k in data["hot_experts"])
        finally:
            os.unlink(path)


class TestReset:
    """Test reset() clears state."""

    def test_clears_counters(self):
        profiler = ExpertProfiler(num_layers=3, num_experts=4)
        profiler.record(0, np.array([0, 1]))
        profiler.record(1, np.array([2, 3]))
        assert profiler.total_tokens > 0

        profiler.reset()

        assert profiler.total_tokens == 0
        assert np.all(profiler._counters == 0)
        assert np.all(profiler._layer_token_counts == 0)

    def test_usable_after_reset(self):
        profiler = ExpertProfiler(num_layers=2, num_experts=4)
        profiler.record(0, np.array([0, 1]))
        profiler.reset()
        profiler.record(0, np.array([3, 3]))
        assert profiler._counters[0, 3] == 2
        assert profiler._counters[0, 0] == 0
        assert profiler.total_tokens == 1
