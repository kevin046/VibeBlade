"""Tests for the PowerInfer 2 ctypes wrapper (vibeblade/powerinfer2.py).

Uses the real libllama.so built under llama.cpp/build — the same binary
the runtime uses. Skipped entirely if the library isn't available.
"""
import ctypes

import numpy as np
import pytest

import vibeblade.powerinfer2 as pi2

pytestmark = []


def _lib_available() -> bool:
    try:
        pi2._find_pi2_lib()
        return True
    except FileNotFoundError:
        return False


requires_lib = pytest.mark.skipif(not _lib_available(), reason="libllama.so not built")


def _make_engine(n_layers=2, hidden=64, neurons=256):
    eng = pi2.PowerInfer2Engine(n_layers=n_layers, hidden_dim=hidden, n_neurons=neurons)
    eng.configure(n_threads=4, hot_cluster_ratio=0.25)
    return eng


def _profile_structured(eng, layers=2, tokens=30, neurons=256, rng=None):
    """Neurons 0..63 fire strongly, the rest get weak noise."""
    rng = rng or np.random.default_rng(42)
    for _ in range(tokens):
        for li in range(layers):
            acts = rng.uniform(0, 0.05, neurons).astype(np.float32)
            acts[:64] = rng.uniform(0.5, 2.0, 64).astype(np.float32)
            pi2.profile(li, acts)


@requires_lib
def test_configure_and_finalize():
    eng = _make_engine()
    _profile_structured(eng)
    assert eng.finalize() is True
    pi2.set_enabled(True)
    assert pi2.is_enabled() is True
    # cleanup
    pi2.reset()


@requires_lib
def test_profile_then_finalize_without_profile_fails():
    eng = _make_engine()
    # No profiling data → C engine rejects finalize
    assert eng.finalize() is False
    pi2.reset()


@requires_lib
def test_compute_layer_matches_dense_on_hot_neurons():
    N_LAYERS, H, N = 2, 64, 256
    eng = _make_engine(n_layers=N_LAYERS, hidden=H, neurons=N)
    rng = np.random.default_rng(42)
    _profile_structured(eng, layers=N_LAYERS, neurons=N, rng=rng)
    assert eng.finalize() is True
    pi2.set_enabled(True)

    gate_w = rng.standard_normal((N, H)).astype(np.float32) * 0.05
    up_w = rng.standard_normal((N, H)).astype(np.float32) * 0.05
    down_w = rng.standard_normal((N, H)).astype(np.float32) * 0.05
    h = rng.standard_normal(H).astype(np.float32)
    out = np.zeros(H, dtype=np.float32)

    rc = eng.compute_layer(
        layer=0,
        hidden=h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output=out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_w=gate_w.ctypes.data_as(ctypes.c_void_p),
        up_w=up_w.ctypes.data_as(ctypes.c_void_p),
        down_w=down_w.ctypes.data_as(ctypes.c_void_p),
    )
    assert rc == 0
    assert np.count_nonzero(out) > 0

    stats = eng.stats()
    # Hot set = 64 neurons; the rest were skipped
    assert stats.hot_neurons_activated == 64
    assert stats.cold_neurons_activated == N - 64
    pi2.reset()


@requires_lib
def test_compute_layer_rejects_null_weights():
    """NULL weight pointers segfault the C engine — wrapper must refuse."""
    eng = _make_engine()
    _profile_structured(eng)
    eng.finalize()
    pi2.set_enabled(True)
    h = np.zeros(64, dtype=np.float32)
    out = np.zeros(256, dtype=np.float32)
    with pytest.raises(ValueError, match="weight pointers"):
        eng.compute_layer(
            layer=0,
            hidden=h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output=out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
    pi2.reset()


@requires_lib
def test_stats_structure():
    eng = _make_engine()
    _profile_structured(eng)
    eng.finalize()
    stats = eng.stats()
    # All fields populated with ints (0 pre-inference is fine)
    assert isinstance(stats.total_layers_computed, int)
    assert isinstance(stats.hot_neurons_activated, int)
    assert isinstance(stats.cold_neurons_activated, int)
    pi2.reset()
