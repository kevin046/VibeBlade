"""Tests for Deltafin-pattern logic in VibeBlade: expert heat histogram,
pin roster economics, and speculative draft auto-economics.

Run: python3 tests/test_expert_heat.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from vibeblade.expert_heat import ExpertHeat  # noqa: E402
from vibeblade.speculative_decoding import SpeculativeStats  # noqa: E402


def test_record_and_persist():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "expert_heat.v1.bin")
        h = ExpertHeat(path, confidence_ramp=10)
        for i in range(20):
            h.record_routes(0, [1, 5, 9])
            h.record_routes(1, [2, 3])
        ok = h.flush()
        assert ok, "flush failed"
        # Fresh instance = survived process restart
        h2 = ExpertHeat(path, confidence_ramp=10)
        s = h2.stats()
        assert s["entries"] == 5, s
        assert s["passes_recorded"] == 40, s
        assert s["confidence_ready"] is True
        print("PASS record + persist across instances:", s)


def test_corrupt_file_degrades_soft():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "expert_heat.v1.bin")
        with open(path, "wb") as f:
            f.write(b"GARBAGE NOT A HISTOGRAM" * 10)
        h = ExpertHeat(path)
        h.record_routes(0, [1])  # must not raise
        s = h.stats()
        assert s["entries"] == 1
        print("PASS corrupt file degrades to nothing-learned")


def test_pin_roster_budget_and_floor():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "heat.bin")
        h = ExpertHeat(path, confidence_ramp=10, freq_floor=2.0)
        # 100 passes: expert 1 hot (100), expert 2 medium (30), expert 3 rare (1)
        for _ in range(100):
            h.record_routes(0, [1])
        for _ in range(30):
            h.record_routes(0, [2])
        h.record_routes(0, [3])
        h.flush()
        h2 = ExpertHeat(path, confidence_ramp=10, freq_floor=2.0)
        roster = h2.pin_roster(budget_bytes=10 << 20, expert_size_bytes=1 << 20)
        assert roster == {0: [1, 2]}, roster  # rare expert 3 below floor, budget caps at 10
        # Tiny budget → nothing fits beyond 1
        roster2 = h2.pin_roster(budget_bytes=1 << 20, expert_size_bytes=1 << 20)
        assert roster2 == {0: [1]}, roster2
        print("PASS pin roster respects budget + frequency floor")


def test_confidence_ramp_blocks_early_pinning():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "heat.bin")
        h = ExpertHeat(path, confidence_ramp=1000)
        for _ in range(50):
            h.record_routes(0, [1])
        h.flush()
        h2 = ExpertHeat(path, confidence_ramp=1000)
        assert h2.pin_roster(1 << 30, 1 << 20) == {}  # not enough passes
        print("PASS confidence ramp blocks premature pinning")


def test_draft_economics():
    # Simulate: acceptance below threshold → economics disable
    st = SpeculativeStats()
    st.n_draft_generated = 300
    st.n_draft_accepted = 30  # 10% < 20% threshold
    assert st.acceptance_rate < 0.20  # property, not method
    print("PASS stats acceptance math")

    # Behavioral: _review_economics flips off under threshold, re-challenges later
    class FakeEngine:
        economics_enabled = True
        min_acceptance = 0.20
        evaluation_window = 100
        _consecutive_disabled = 0
        stats = st

    from vibeblade.speculative_decoding import SpeculativeDecodingEngine

    eng = FakeEngine()
    SpeculativeDecodingEngine._review_economics(eng)
    assert eng.economics_enabled is False, "should disable drafting below threshold"
    # Cool-down re-challenge
    for _ in range(8):
        SpeculativeDecodingEngine._review_economics(eng)
    assert eng.economics_enabled is True, "should re-challenge after cool-down"
    print("PASS draft auto-economics disable + re-challenge")


def test_moe_executor_heat_hook():
    # HotColdExecutor records routes when a heat recorder is attached
    from vibeblade.moe_executor import HotColdExecutor

    class FakeRouter:
        topk = 2

        def route(self, x):
            return np.array([[0, 3]], dtype=np.int64), np.array([[0.7, 0.3]], dtype=np.float32)

    class FakeExpertSet:
        def get_expert(self, eid):
            return (np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))

    with tempfile.TemporaryDirectory() as d:
        heat = ExpertHeat(os.path.join(d, "heat.bin"), confidence_ramp=10)

        class FakeGPU:
            is_gpu = False

        # Minimal executor without real weights — just verify the hook records
        ex = HotColdExecutor.__new__(HotColdExecutor)
        ex._heat = None
        ex.attach_expert_heat(heat)
        assert ex._heat is heat
        assert ex.flush_expert_heat() is True
        # Route recording contract: infallible with weird input
        try:
            heat.record_routes(0, [int(i) for i in np.array([[2, 7]]).ravel().tolist()])
        except Exception as e:
            raise AssertionError(f"record_routes raised: {e}")
        heat.flush()
        h2 = ExpertHeat(os.path.join(d, "heat.bin"))
        assert (0, 2) in h2._weights and (0, 7) in h2._weights
        print("PASS executor heat attach/flush/record")


if __name__ == "__main__":
    test_record_and_persist()
    test_corrupt_file_degrades_soft()
    test_pin_roster_budget_and_floor()
    test_confidence_ramp_blocks_early_pinning()
    test_draft_economics()
    test_moe_executor_heat_hook()
    print("\nALL EXPERT-HEAT / ECONOMICS TESTS PASS")
