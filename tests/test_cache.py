"""Tests for VibeBlade KV Cache."""

import numpy as np
import pytest
from vibeblade.cache import KVCache


NUM_LAYERS = 4
NUM_HEADS = 8
HEAD_DIM = 16
MAX_SEQ = 64


@pytest.fixture
def cache():
    return KVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM, MAX_SEQ)


class TestKVCache:

    def test_cache_init_shape(self, cache):
        """Verify cache shape."""
        expected = (NUM_LAYERS, 2, NUM_HEADS, MAX_SEQ, HEAD_DIM)
        assert cache.cache.shape == expected
        assert cache.dtype == np.float16

    def test_cache_update_and_get(self, cache):
        """Insert K/V at position 0, retrieve and verify."""
        rng = np.random.default_rng(42)
        key = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        value = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        cache.update(0, key, value, position=0)
        k, v = cache.get(0)
        assert k.shape == (NUM_HEADS, 1, HEAD_DIM)
        assert v.shape == (NUM_HEADS, 1, HEAD_DIM)
        np.testing.assert_array_equal(k, key)
        np.testing.assert_array_equal(v, value)

    def test_cache_sequential_positions(self, cache):
        """Insert at positions 0, 1, 2, retrieve all."""
        rng = np.random.default_rng(99)
        keys, values = [], []
        for pos in range(3):
            k = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
            v = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
            keys.append(k)
            values.append(v)
            cache.update(0, k, v, position=pos)
        k_all, v_all = cache.get(0)
        assert k_all.shape == (NUM_HEADS, 3, HEAD_DIM)
        for i in range(3):
            np.testing.assert_array_equal(k_all[:, i, :], keys[i][:, 0, :])
            np.testing.assert_array_equal(v_all[:, i, :], values[i][:, 0, :])

    def test_cache_length_tracking(self, cache):
        """Length should track max position + 1."""
        assert cache.length == 0
        rng = np.random.default_rng(0)
        k = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        cache.update(0, k, v, position=0)
        assert cache.length == 1
        cache.update(0, k, v, position=4)
        assert cache.length == 5

    def test_cache_clear(self, cache):
        """After clear, length is 0 and get returns zeros."""
        rng = np.random.default_rng(0)
        k = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        cache.update(0, k, v, position=0)
        cache.clear()
        assert cache.length == 0
        k_out, v_out = cache.get(0, 0, 1)
        np.testing.assert_array_equal(k_out, np.zeros((NUM_HEADS, 1, HEAD_DIM), dtype=np.float16))
        np.testing.assert_array_equal(v_out, np.zeros((NUM_HEADS, 1, HEAD_DIM), dtype=np.float16))

    def test_cache_position_exceeds_max(self, cache):
        """Should raise IndexError."""
        rng = np.random.default_rng(0)
        k = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
        with pytest.raises(IndexError, match="exceeds max_seq_len"):
            cache.update(0, k, v, position=MAX_SEQ)

    def test_cache_memory_usage(self, cache):
        """Verify nbytes is correct."""
        expected_bytes = NUM_LAYERS * 2 * NUM_HEADS * MAX_SEQ * HEAD_DIM * np.dtype(np.float16).itemsize
        assert cache.memory_usage_bytes() == expected_bytes

    def test_cache_2d_key_input(self, cache):
        """Should handle (num_heads, head_dim) input (auto-expand)."""
        rng = np.random.default_rng(7)
        key_2d = rng.standard_normal((NUM_HEADS, HEAD_DIM)).astype(np.float16)
        value_2d = rng.standard_normal((NUM_HEADS, HEAD_DIM)).astype(np.float16)
        cache.update(0, key_2d, value_2d, position=0)
        k, v = cache.get(0)
        assert k.shape == (NUM_HEADS, 1, HEAD_DIM)
        assert v.shape == (NUM_HEADS, 1, HEAD_DIM)
        np.testing.assert_array_equal(k[:, 0, :], key_2d)
        np.testing.assert_array_equal(v[:, 0, :], value_2d)


class TestActivationBufferPool:
    """Tests for the MoE activation buffer pool."""

    def test_create_pool(self):
        """Pool creates correct number of buffers."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=4, buffer_size=1024)
        assert pool.total_count == 4
        assert pool.available_count == 4
        pool.close()

    def test_acquire_and_release(self):
        """Acquire reduces available count, release increases it."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=4, buffer_size=1024)
        buf = pool.acquire()
        assert buf is not None
        assert pool.available_count == 3
        assert buf.shape == pool.buffer_shape

        pool.release(buf)
        assert pool.available_count == 4
        pool.close()

    def test_acquire_exhausted_returns_none(self):
        """Acquiring from empty pool returns None."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=2, buffer_size=1024)
        b1 = pool.acquire()
        b2 = pool.acquire()
        assert pool.acquire() is None
        pool.release(b1)
        pool.release(b2)
        pool.close()

    def test_buffer_is_pinned_f32(self):
        """Buffers are float32 numpy arrays."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=1, buffer_size=4096)
        buf = pool.acquire()
        assert buf.dtype == np.float32
        assert len(buf) == 4096 // 4  # 1024 float32 elements
        pool.release(buf)
        pool.close()

    def test_release_zeros_buffer(self):
        """Released buffers are zeroed out."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=1, buffer_size=1024)
        buf = pool.acquire()
        buf[:5] = 999.0
        pool.release(buf)
        buf2 = pool.acquire()
        assert np.all(buf2 == 0.0)
        pool.close()

    def test_allocate_single_buffer(self):
        """Allocate a single standalone pinned buffer."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=1, buffer_size=1024)
        buf = pool.allocate_buffer(size_bytes=2048)
        assert buf.dtype == np.float32
        assert len(buf) == 512  # 2048 / 4
        pool.close()

    def test_stats(self):
        """Stats dict has correct keys and values."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=4, buffer_size=2048)
        s = pool.stats()
        assert s["total_buffers"] == 4
        assert s["available"] == 4
        assert s["in_use"] == 0
        assert s["buffer_bytes"] == 2048
        assert s["total_bytes"] == 8192
        pool.close()

    def test_repr(self):
        """Repr doesn't crash."""
        from vibeblade.memory import ActivationBufferPool
        pool = ActivationBufferPool(count=4, buffer_size=1024)
        r = repr(pool)
        assert "4/4" in r
        pool.close()
