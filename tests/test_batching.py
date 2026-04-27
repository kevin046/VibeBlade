"""Tests for VibeBlade SARATHI Batching."""

import numpy as np

from vibeblade.batching import ContinuousBatcher


class TestBatcherSubmit:
    def test_submit_returns_id(self):
        batcher = ContinuousBatcher(max_batch_size=4)
        rid = batcher.submit(np.array([1, 2, 3]), max_tokens=10)
        assert isinstance(rid, int)
        assert batcher.waiting_count == 1

    def test_submit_multiple(self):
        batcher = ContinuousBatcher(max_batch_size=4)
        ids = [batcher.submit(np.array([1, 2, 3]), max_tokens=10) for _ in range(3)]
        assert len(ids) == 3
        assert len(set(ids)) == 3  # unique IDs


class TestBatcherSchedule:
    def test_schedule_moves_waiting_to_active(self):
        batcher = ContinuousBatcher(max_batch_size=4, prefill_chunk_size=2)
        batcher.submit(np.array([1, 2, 3, 4, 5]), max_tokens=10)

        schedule = batcher.schedule()
        assert schedule["num_prefill"] == 1
        assert batcher.active_count == 1
        assert batcher.waiting_count == 0

    def test_chunked_prefill(self):
        batcher = ContinuousBatcher(max_batch_size=4, prefill_chunk_size=2)
        batcher.submit(np.array([1, 2, 3, 4, 5]), max_tokens=10)

        # First schedule: chunks 0-1
        s1 = batcher.schedule()
        assert s1["num_prefill"] == 1
        assert len(s1["prefill_requests"][0][1]) == 2

        # Second schedule: chunks 2-3
        s2 = batcher.schedule()
        assert s2["num_prefill"] == 1
        assert len(s2["prefill_requests"][0][1]) == 2

        # Third schedule: chunk 4 (last token), phase transitions to decode after
        s3 = batcher.schedule()
        assert s3["num_prefill"] == 1  # still categorized as prefill for this step
        assert len(s3["prefill_requests"][0][1]) == 1

        # Fourth schedule: now in decode phase
        s4 = batcher.schedule()
        assert s4["num_decode"] >= 1

    def test_max_batch_size(self):
        batcher = ContinuousBatcher(max_batch_size=2)
        for _ in range(5):
            batcher.submit(np.array([1, 2, 3]), max_tokens=10)

        batcher.schedule()
        assert batcher.active_count == 2
        assert batcher.waiting_count == 3


class TestBatcherDecode:
    def test_update_decode(self):
        batcher = ContinuousBatcher(max_batch_size=4, prefill_chunk_size=1)
        rid = batcher.submit(np.array([1]), max_tokens=3)

        # Prefill all tokens
        batcher.schedule()  # prefill token 1
        batcher.finish_prefill(rid)

        # Decode step
        logits = np.random.randn(100).astype(np.float32)
        batcher.update_decode(rid, next_token=42, logits=logits)

        schedule = batcher.schedule()
        assert schedule["num_decode"] == 1

    def test_finish_on_max_tokens(self):
        batcher = ContinuousBatcher(max_batch_size=4, prefill_chunk_size=1)
        rid = batcher.submit(np.array([1]), max_tokens=2)
        batcher.schedule()
        batcher.finish_prefill(rid)

        batcher.update_decode(rid, 42, np.zeros(100))
        batcher.update_decode(rid, 43, np.zeros(100))

        # Should be finished
        finished = batcher.collect_finished()
        assert len(finished) == 1
        assert batcher.active_count == 0


class TestBatcherCollect:
    def test_collect_finished(self):
        batcher = ContinuousBatcher(max_batch_size=4, prefill_chunk_size=1)
        rid = batcher.submit(np.array([1]), max_tokens=1)
        batcher.schedule()
        batcher.finish_prefill(rid)
        batcher.update_decode(rid, 2, np.zeros(100))  # EOS

        finished = batcher.collect_finished()
        assert len(finished) == 1
        assert finished[0].request_id == rid
