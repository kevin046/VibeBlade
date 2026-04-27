"""VibeBlade SARATHI — Continuous batching with chunked prefill.

Based on: SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills (2308.16369)

Key innovations:
1. Chunked prefill: Break large prefills into chunks that can be interleaved
   with ongoing decodes, reducing pipeline bubbles.
2. Continuous batching: No batch-wide synchronization. Requests enter/exit
   the batch independently (like Orca/vLLM).
3. Decode-maximal scheduling: Prefers decode requests to maximize GPU utilization
   since decode is memory-bandwidth bound and underutilizes compute.

Achieves up to 10× decode utilization and 1.33-1.91× end-to-end speedup.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class RequestPhase(enum.Enum):
    """Phase of a request in the batch."""
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class BatchRequest:
    """A single request in the continuous batch."""
    request_id: int
    token_ids: np.ndarray
    phase: RequestPhase = RequestPhase.WAITING
    # Prefill state
    prefill_pos: int = 0
    prefill_chunk_size: int = 512  # tokens per prefill chunk
    # Decode state
    generated_tokens: list[int] = field(default_factory=list)
    max_tokens: int = 256
    # Output
    output_logits: list[np.ndarray] = field(default_factory=list)
    finished: bool = False

    @property
    def is_prefill_done(self) -> bool:
        return self.prefill_pos >= len(self.token_ids)

    @property
    def decode_steps_remaining(self) -> int:
        return max(0, self.max_tokens - len(self.generated_tokens))


class ContinuousBatcher:
    """SARATHI-style continuous batching with chunked prefill.

    Unlike static batching where all requests in a batch must be in the
    same phase, continuous batching allows mixing prefill and decode requests.
    Chunked prefill breaks large prompts into chunks to reduce pipeline bubbles.

    Parameters
    ----------
    max_batch_size : int
        Maximum concurrent requests.
    prefill_chunk_size : int
        Tokens per prefill chunk (default 512).
    max_total_tokens : int
        Maximum total tokens across all active requests.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        prefill_chunk_size: int = 512,
        max_total_tokens: int = 8192,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self.max_total_tokens = max_total_tokens

        self._waiting: list[BatchRequest] = []
        self._active: list[BatchRequest] = []
        self._finished: list[BatchRequest] = []
        self._next_id = 0

    def submit(self, token_ids: np.ndarray, max_tokens: int = 256) -> int:
        """Submit a new request. Returns the request ID."""
        req = BatchRequest(
            request_id=self._next_id,
            token_ids=token_ids,
            max_tokens=max_tokens,
            prefill_chunk_size=self.prefill_chunk_size,
        )
        self._next_id += 1
        self._waiting.append(req)
        return req.request_id

    def schedule(self) -> dict[str, Any]:
        """Compute the next batch schedule.

        Returns a dict with:
            'prefill_requests': list of (request_id, token_chunk) tuples
            'decode_requests': list of (request_id,) tuples
            'num_prefill': int
            'num_decode': int
            'estimated_tokens': int
        """
        # Move waiting requests to active if space
        while (
            self._waiting
            and len(self._active) < self.max_batch_size
        ):
            req = self._waiting.pop(0)
            req.phase = RequestPhase.PREFILL
            req.prefill_pos = 0
            self._active.append(req)

        # Categorize active requests
        prefill_reqs: list[BatchRequest] = []
        decode_reqs: list[BatchRequest] = []

        for req in self._active:
            if req.phase == RequestPhase.PREFILL and not req.is_prefill_done:
                prefill_reqs.append(req)
            elif req.phase == RequestPhase.DECODE or req.is_prefill_done:
                if req.phase != RequestPhase.FINISHED:
                    req.phase = RequestPhase.DECODE
                if not req.finished:
                    decode_reqs.append(req)

        # SARATHI decode-maximal scheduling: prefer decodes for better utilization
        # Allocate remaining budget to prefills
        total_tokens = sum(len(r.generated_tokens) + 1 for r in decode_reqs)

        prefill_schedule: list[tuple[int, np.ndarray]] = []
        for req in prefill_reqs:
            chunk_end = min(req.prefill_pos + req.prefill_chunk_size, len(req.token_ids))
            chunk = req.token_ids[req.prefill_pos:chunk_end]
            token_count = chunk.size

            if total_tokens + token_count <= self.max_total_tokens:
                prefill_schedule.append((req.request_id, chunk))
                total_tokens += token_count
                req.prefill_pos = chunk_end
                if req.is_prefill_done:
                    req.phase = RequestPhase.DECODE

        decode_schedule: list[tuple[int]] = [(r.request_id,) for r in decode_reqs]

        return {
            "prefill_requests": prefill_schedule,
            "decode_requests": decode_schedule,
            "num_prefill": len(prefill_schedule),
            "num_decode": len(decode_schedule),
            "estimated_tokens": total_tokens,
        }

    def update_decode(
        self,
        request_id: int,
        next_token: int,
        logits: np.ndarray,
    ) -> None:
        """Update state after a decode step produces a token."""
        for req in self._active:
            if req.request_id == request_id:
                req.generated_tokens.append(next_token)
                req.output_logits.append(logits)
                if len(req.generated_tokens) >= req.max_tokens or next_token == 2:
                    req.finished = True
                    req.phase = RequestPhase.FINISHED
                return

    def finish_prefill(self, request_id: int) -> None:
        """Mark a request as finished prefilling."""
        for req in self._active:
            if req.request_id == request_id:
                req.phase = RequestPhase.DECODE
                return

    def collect_finished(self) -> list[BatchRequest]:
        """Remove and return all finished requests."""
        finished = [r for r in self._active if r.finished]
        self._active = [r for r in self._active if not r.finished]
        self._finished.extend(finished)
        return finished

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def finished_count(self) -> int:
        return len(self._finished)

    def __repr__(self) -> str:
        return (
            f"ContinuousBatcher(active={self.active_count}, "
            f"waiting={self.waiting_count}, done={self.finished_count})"
        )
