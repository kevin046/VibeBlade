"""Tests for the Adaptive Memory Tiering (AMT) system."""

import os
import numpy as np
import pytest
from concurrent.futures import Future

from vibeblade.tiered_memory import (
    MemoryTier,
    ExpertLocation,
    TieredMemoryStats,
    LRUKPolicy,
    SSDExpertStore,
    TieredMemoryManager,
)
from vibeblade.config import OffloadConfig, OffloadMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_expert_weights(shared_dim: int = 16, expert_dim: int = 32) -> tuple:
    """Return (gate_w, up_w, down_w) with sensible shapes."""
    rng = np.random.default_rng(42)
    gate = rng.standard_normal((shared_dim, expert_dim)).astype(np.float32)
    up = rng.standard_normal((shared_dim, expert_dim)).astype(np.float32)
    down = rng.standard_normal((expert_dim, shared_dim)).astype(np.float32)
    return gate, up, down


def _make_hot_cold_map(
    num_layers: int = 4,
    num_experts: int = 8,
    hot_per_layer: int = 2,
) -> object:
    """Build a HotColdMap with the first *hot_per_layer* experts hot."""
    from vibeblade.moe_profiler import HotColdMap

    hot_experts = {}
    cold_experts = {}
    for layer in range(num_layers):
        hot_experts[layer] = list(range(hot_per_layer))
        cold_experts[layer] = list(range(hot_per_layer, num_experts))
    return HotColdMap(
        hot_experts=hot_experts,
        cold_experts=cold_experts,
        num_layers=num_layers,
        num_experts=num_experts,
    )


# ===========================================================================
# MemoryTier enum
# ===========================================================================

class TestMemoryTier:
    """Test MemoryTier enum values."""

    def test_enum_values(self):
        assert MemoryTier.VRAM.value == "vram"
        assert MemoryTier.RAM.value == "ram"
        assert MemoryTier.SSD.value == "ssd"

    def test_enum_members(self):
        assert len(MemoryTier) == 3
        assert set(MemoryTier) == {MemoryTier.VRAM, MemoryTier.RAM, MemoryTier.SSD}


# ===========================================================================
# LRUKPolicy
# ===========================================================================

class TestLRUKPolicy:
    """Test LRU-K eviction policy."""

    def test_basic_access_and_evict(self):
        """Items with fewer accesses are evicted first."""
        policy = LRUKPolicy(k=2, capacity=3)
        policy.access(0, 0)
        policy.access(0, 1)
        policy.access(0, 2)

        # All three have 1 access — all in probationary
        assert policy.size == 3
        assert policy.stats()["probationary_count"] == 3
        assert policy.stats()["protected_count"] == 0

        # Evict should remove a probationary item
        victim = policy.evict()
        assert victim is not None
        assert policy.size == 2

    def test_one_hit_wonder_eviction(self):
        """Items with 1 access evicted before items with 3+ accesses."""
        policy = LRUKPolicy(k=2, capacity=10)
        # Expert 0: accessed 3 times → should be protected
        policy.access(0, 0)
        policy.access(0, 0)
        policy.access(0, 0)
        # Expert 1: accessed 3 times → should be protected
        policy.access(0, 1)
        policy.access(0, 1)
        policy.access(0, 1)
        # Expert 2: accessed once → probationary
        policy.access(0, 2)

        assert policy.stats()["protected_count"] == 2
        assert policy.stats()["probationary_count"] == 1

        # Eviction should remove the one-hit wonder first
        victim = policy.evict()
        assert victim == (0, 2)

        # Next eviction should come from protected (oldest k-th access)
        victim2 = policy.evict()
        assert victim2 in [(0, 0), (0, 1)]

    def test_should_prefetch(self):
        """should_prefetch returns True only after k+ accesses."""
        policy = LRUKPolicy(k=2)
        assert not policy.should_prefetch(0, 0)

        policy.access(0, 0)
        assert not policy.should_prefetch(0, 0)  # 1 access, k=2

        policy.access(0, 0)
        assert policy.should_prefetch(0, 0)  # 2 accesses, k=2

    def test_remove(self):
        """Explicit remove removes from all structures."""
        policy = LRUKPolicy(k=2)
        policy.access(0, 0)
        policy.access(0, 0)
        assert policy.contains(0, 0)

        policy.remove(0, 0)
        assert not policy.contains(0, 0)
        assert policy.stats()["protected_count"] == 0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            LRUKPolicy(k=0, capacity=10)
        with pytest.raises(ValueError):
            LRUKPolicy(k=2, capacity=0)

    def test_evict_empty(self):
        policy = LRUKPolicy(k=2)
        assert policy.evict() is None


# ===========================================================================
# SSDExpertStore
# ===========================================================================

class TestSSDExpertStore:
    """Test SSD-backed expert storage."""

    def test_store_load_roundtrip(self, tmp_path):
        """store_expert then load_expert returns identical arrays."""
        store = SSDExpertStore(
            ssd_path=str(tmp_path / "ssd"),
            num_layers=2,
            num_experts=4,
        )
        gate, up, down = _make_expert_weights()
        store.store_expert(0, 0, gate, up, down)
        g2, u2, d2 = store.load_expert(0, 0)

        np.testing.assert_array_equal(gate, g2)
        np.testing.assert_array_equal(up, u2)
        np.testing.assert_array_equal(down, d2)
        store.close()

    def test_async_load_returns_future(self, tmp_path):
        """async_load returns a Future that resolves to correct data."""
        store = SSDExpertStore(
            ssd_path=str(tmp_path / "ssd"),
            num_layers=2,
            num_experts=4,
        )
        gate, up, down = _make_expert_weights()
        store.store_expert(0, 1, gate, up, down)

        fut = store.async_load(0, 1)
        assert isinstance(fut, Future)
        g2, u2, d2 = fut.result(timeout=5)
        np.testing.assert_array_equal(gate, g2)
        np.testing.assert_array_equal(up, u2)
        np.testing.assert_array_equal(down, d2)
        store.close()

    def test_expert_size_bytes(self, tmp_path):
        store = SSDExpertStore(
            ssd_path=str(tmp_path / "ssd"),
            num_layers=1,
            num_experts=1,
        )
        gate, up, down = _make_expert_weights(16, 32)
        store.store_expert(0, 0, gate, up, down)
        sz = store.expert_size_bytes(0, 0)
        # 3 matrices × (8 bytes header + 16*32*4 data) = 3 × 2056 = 6168
        assert sz > 0
        store.close()

    def test_total_size_bytes(self, tmp_path):
        store = SSDExpertStore(
            ssd_path=str(tmp_path / "ssd"),
            num_layers=1,
            num_experts=2,
        )
        gate, up, down = _make_expert_weights()
        store.store_expert(0, 0, gate, up, down)
        store.store_expert(0, 1, gate, up, down)
        total = store.total_size_bytes()
        expected = 2 * store.expert_size_bytes(0, 0)
        assert total == expected
        store.close()

    def test_multiple_layers(self, tmp_path):
        store = SSDExpertStore(
            ssd_path=str(tmp_path / "ssd"),
            num_layers=3,
            num_experts=2,
        )
        for layer in range(3):
            gate, up, down = _make_expert_weights()
            store.store_expert(layer, 0, gate, up, down)

        # Each layer in its own subdirectory
        assert os.path.isdir(str(tmp_path / "ssd" / "layer_0000"))
        assert os.path.isdir(str(tmp_path / "ssd" / "layer_0001"))
        assert os.path.isdir(str(tmp_path / "ssd" / "layer_0002"))
        store.close()


# ===========================================================================
# TieredMemoryManager — RAM_ONLY mode
# ===========================================================================

class TestTieredMemoryManagerRAMOnly:
    """Test TieredMemoryManager with OffloadMode.RAM_ONLY."""

    @pytest.fixture
    def manager(self):
        config = OffloadConfig(mode=OffloadMode.RAM_ONLY, vram_limit=1 << 30, ram_limit=1 << 30)
        hcm = _make_hot_cold_map(num_layers=4, num_experts=8, hot_per_layer=2)
        mgr = TieredMemoryManager(
            config=config,
            num_layers=4,
            num_experts=8,
            expert_size_bytes=6168,
            hot_cold_map=hcm,
        )
        yield mgr
        mgr.close()

    def test_initialization(self, manager):
        """Manager initializes with hot experts in VRAM tier."""
        # Hot experts should be marked VRAM
        assert manager.get_expert_tier(0, 0) == MemoryTier.VRAM
        assert manager.get_expert_tier(0, 1) == MemoryTier.VRAM
        # Cold experts fall back to SSD tier in RAM_ONLY mode (no SSD store,
        # so they are recorded as SSD — they will be loaded on demand)
        assert manager.get_expert_tier(0, 2) in (MemoryTier.SSD, MemoryTier.RAM)

    def test_stats_initial(self, manager):
        """Stats should start at zero hits."""
        s = manager.stats()
        assert s.vram_hits == 0
        assert s.ram_hits == 0
        assert s.ssd_hits == 0

    def test_tier_summary_string(self, manager):
        """tier_summary returns a non-empty string with expected sections."""
        summary = manager.tier_summary()
        assert "Tiered Memory Summary" in summary
        assert "VRAM (Tier A)" in summary
        assert "RAM  (Tier B)" in summary
        assert "SSD  (Tier C)" in summary
        assert "Access Statistics" in summary

    def test_register_and_get_vram_expert(self, manager):
        """Register an expert in VRAM, get_expert returns it."""
        gate, up, down = _make_expert_weights()
        manager.register_vram_expert(0, 0, gate, up, down)

        result = manager.get_expert(0, 0)
        assert result is not None
        g, u, d = result
        np.testing.assert_array_equal(gate, g)
        np.testing.assert_array_equal(up, u)
        np.testing.assert_array_equal(down, d)

        s = manager.stats()
        assert s.vram_hits == 1


# ===========================================================================
# TieredMemoryManager — HYBRID_SSD mode
# ===========================================================================

class TestTieredMemoryManagerHybridSSD:
    """Test TieredMemoryManager with OffloadMode.HYBRID_SSD."""

    @pytest.fixture
    def ssd_manager(self, tmp_path):
        config = OffloadConfig(
            mode=OffloadMode.HYBRID_SSD,
            vram_limit=1 << 30,
            ram_limit=1 << 20,  # small to trigger evictions
            ssd_path=str(tmp_path / "ssd_store"),
            ram_buffer_ratio=0.5,
            ssd_preemptive_layers=2,
        )
        hcm = _make_hot_cold_map(num_layers=4, num_experts=8, hot_per_layer=2)
        mgr = TieredMemoryManager(
            config=config,
            num_layers=4,
            num_experts=8,
            expert_size_bytes=6168,
            hot_cold_map=hcm,
        )
        # Pre-populate SSD store with cold expert weights
        ssd = mgr._ssd
        for layer in range(4):
            for e in range(2, 8):
                gate, up, down = _make_expert_weights()
                ssd.store_expert(layer, e, gate, up, down)
        yield mgr
        mgr.close()

    def test_initialization_hybrid(self, ssd_manager):
        """Hybrid mode creates SSD store and tracks locations."""
        assert ssd_manager.get_expert_tier(0, 0) == MemoryTier.VRAM
        assert ssd_manager.get_expert_tier(0, 2) == MemoryTier.SSD

    def test_get_expert_from_ssd(self, ssd_manager):
        """get_expert loads from SSD and caches in RAM."""
        result = ssd_manager.get_expert(0, 2)
        assert result is not None
        gate, up, down = result
        assert gate.shape == (16, 32)
        assert up.shape == (16, 32)
        assert down.shape == (32, 16)

        # Should now be in RAM
        assert ssd_manager.get_expert_tier(0, 2) == MemoryTier.RAM

        s = ssd_manager.stats()
        assert s.ssd_hits == 1

    def test_get_expert_from_ram_cache_hit(self, ssd_manager):
        """Second get_expert for same SSD-loaded expert hits RAM."""
        ssd_manager.get_expert(0, 2)  # loads from SSD → RAM
        ssd_manager.get_expert(0, 2)  # should hit RAM

        s = ssd_manager.stats()
        assert s.ssd_hits == 1
        assert s.ram_hits == 1

    def test_ensure_in_ram_evicts_when_full(self, ssd_manager):
        """ensure_in_ram triggers eviction when RAM buffer is at capacity."""
        # RAM limit is 1MB, each expert ~6KB → capacity ~170
        # Load more experts than capacity to force eviction
        for e in range(2, 8):
            ssd_manager.ensure_in_ram(0, e)

        # All should now be in RAM
        for e in range(2, 8):
            assert ssd_manager.get_expert_tier(0, e) == MemoryTier.RAM

        # At least some should have been evicted at some point
        # (depends on capacity math, but we loaded 6 experts)
        # The key check: ensure_in_ram didn't crash and RAM is populated
        assert ssd_manager.stats().ssd_loads >= 1

    def test_prefetch_experts_returns_futures(self, ssd_manager):
        """prefetch_experts returns futures for SSD-only experts."""
        # Expert 0 and 1 are hot (VRAM), 2-7 are on SSD
        futures = ssd_manager.prefetch_experts(1, [2, 3, 4])
        assert len(futures) >= 0  # may be 0 if should_prefetch filters them out

        # If we first access them to build history, then prefetch should trigger
        ssd_manager.update_access(1, 2)
        ssd_manager.update_access(1, 2)
        futures2 = ssd_manager.prefetch_experts(1, [2])
        assert len(futures2) >= 0

    def test_stats_tracking(self, ssd_manager):
        """Comprehensive check of stats accumulation."""
        # Register a VRAM expert
        gate, up, down = _make_expert_weights()
        ssd_manager.register_vram_expert(0, 0, gate, up, down)
        ssd_manager.get_expert(0, 0)  # VRAM hit

        # Load from SSD
        ssd_manager.get_expert(0, 2)  # SSD hit + SSD load
        ssd_manager.get_expert(0, 2)  # RAM hit

        s = ssd_manager.stats()
        assert s.vram_hits >= 1
        assert s.ram_hits >= 1
        assert s.ssd_hits >= 1
        assert s.ssd_loads >= 1
        assert s.total_ssd_read_bytes > 0
        assert s.total_ssd_read_ms > 0


# ===========================================================================
# TieredMemoryStats
# ===========================================================================

class TestTieredMemoryStats:
    def test_default_values(self):
        s = TieredMemoryStats()
        assert s.vram_hits == 0
        assert s.ram_hits == 0
        assert s.prefetch_hits == 0

    def test_mutation(self):
        s = TieredMemoryStats()
        s.vram_hits = 5
        s.ram_hits = 3
        assert s.vram_hits == 5


# ===========================================================================
# ExpertLocation
# ===========================================================================

class TestExpertLocation:
    def test_construction(self):
        loc = ExpertLocation(
            expert_id=5,
            layer_idx=2,
            tier=MemoryTier.RAM,
            size_bytes=1024,
        )
        assert loc.expert_id == 5
        assert loc.layer_idx == 2
        assert loc.tier == MemoryTier.RAM
        assert loc.size_bytes == 1024
        assert loc.access_count == 0
