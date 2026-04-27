"""VibeBlade SARATHI — Chunked prefill scheduling for efficient LLM serving.

Based on: SARATHI: Efficient LLM Inference by Pipelining Prefill and Decoding Phases

Core insight: Traditional systems process all prefill tokens before any decode,
creating head-of-line blocking. SARATHI chunks prefill requests and interleaves
prefill chunks with decode iterations, achieving:
- 2.3x - 3.1x throughput improvement over vanilla prefill-then-decode
- Sub-100ms latency for decode requests even during active prefills
- Memory-efficient chunk sizes computed from KV cache budget

The chunk size is dynamically computed as:
    chunk_size = floor(available_kv_blocks * block_size / num_active_requests)

This ensures prefill progress without exceeding memory budget, while
decode requests are serviced every iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RequestPhase(Enum):
    """Phase of a request in the SARATHI scheduler."""
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"


@dataclass
class SarathiRequest:
    """A single request tracked by the SARATHI scheduler.

    Attributes
    ----------
    request_id : int or str
        Unique identifier for this request.
    total_prompt_tokens : int
        Total number of tokens in the prompt (for prefill).
    tokens_prefilled : int
        How many prompt tokens have been prefilled so far.
    tokens_decoded : int
        How many decode tokens have been generated.
    max_decode_tokens : int
        Maximum decode tokens to generate (stop condition).
    priority : float
        Request priority (higher = scheduled sooner). Default 1.0.
    arrival_time : float
        Monotonic timestamp when the request was created.
    phase : RequestPhase
        Current phase of the request.
    """
    request_id: int | str
    total_prompt_tokens: int
    tokens_prefilled: int = 0
    tokens_decoded: int = 0
    max_decode_tokens: int = 256
    priority: float = 1.0
    arrival_time: float = 0.0
    phase: RequestPhase = RequestPhase.WAITING

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.total_prompt_tokens - self.tokens_prefilled)

    @property
    def is_prefill_done(self) -> bool:
        return self.tokens_prefilled >= self.total_prompt_tokens

    @property
    def is_decode_done(self) -> bool:
        return self.tokens_decoded >= self.max_decode_tokens

    @property
    def is_done(self) -> bool:
        return self.phase == RequestPhase.COMPLETED


@dataclass
class SarathiConfig:
    """Configuration for the SARATHI scheduler.

    Attributes
    ----------
    max_batch_size : int
        Maximum concurrent requests in a batch.
    kv_cache_blocks : int
        Total number of KV cache blocks available.
    block_size : int
        Number of tokens per KV cache block (e.g., 16 for vLLM-style).
    max_num_batched_tokens : int
        Maximum total tokens processed per iteration.
    decode_ratio : float
        Fraction of iterations reserved for decode (vs prefill).
        Higher values prioritize decode latency (default 0.5).
    """
    max_batch_size: int = 32
    kv_cache_blocks: int = 1024
    block_size: int = 16
    max_num_batched_tokens: int = 2048
    decode_ratio: float = 0.5


class SarathiScheduler:
    """SARATHI chunked prefill scheduler.

    Interleaves prefill chunks with decode iterations to avoid head-of-line
    blocking. Prefill requests are chunked based on available KV cache budget,
    and decode requests are prioritized based on the decode_ratio.

    Scheduling algorithm per iteration:
    1. Compute chunk_size = floor(available_blocks * block_size / num_active)
    2. Reserve decode slots (decode_ratio fraction of batch)
    3. Fill remaining slots with prefill chunks
    4. Execute mixed batch
    5. Update request states

    Parameters
    ----------
    config : SarathiConfig or None
        Scheduler configuration. Uses defaults if None.
    """

    def __init__(self, config: SarathiConfig | None = None) -> None:
        self._config = config or SarathiConfig()
        self._waiting_queue: list[SarathiRequest] = []
        self._active_requests: list[SarathiRequest] = []
        self._completed_requests: list[SarathiRequest] = []
        self._next_id: int = 0

        # Statistics
        self._total_iterations: int = 0
        self._total_prefill_tokens: int = 0
        self._total_decode_tokens: int = 0
        self._total_scheduled_chunks: int = 0

    def add_request(
        self,
        prompt_tokens: int,
        max_decode_tokens: int = 256,
        priority: float = 1.0,
    ) -> SarathiRequest:
        """Add a new request to the waiting queue.

        Parameters
        ----------
        prompt_tokens : int
            Number of tokens in the prompt.
        max_decode_tokens : int
            Maximum decode tokens to generate.
        priority : float
            Request priority (higher = scheduled sooner).

        Returns
        -------
        SarathiRequest
            The created request object.
        """
        req = SarathiRequest(
            request_id=self._next_id,
            total_prompt_tokens=prompt_tokens,
            max_decode_tokens=max_decode_tokens,
            priority=priority,
            arrival_time=time.monotonic(),
        )
        self._next_id += 1
        self._waiting_queue.append(req)
        return req

    def schedule(self) -> dict[str, Any]:
        """Run one scheduling iteration.

        Computes chunk sizes, selects requests, and returns the execution
        plan for this iteration.

        Returns
        -------
        dict with keys:
            - 'prefill_chunks': list of (request_id, num_tokens) for prefill work
            - 'decode_requests': list of request_ids for decode work
            - 'chunk_size': int — computed chunk size for this iteration
            - 'available_blocks': int — KV blocks available
            - 'iteration': int — iteration counter
        """
        self._total_iterations += 1

        # Step 1: Move completed requests out
        self._active_requests = [
            r for r in self._active_requests if not r.is_done
        ]
        # Move to completed list for stats
        for req in list(self._active_requests):
            if req.is_done:
                self._completed_requests.append(req)

        self._active_requests = [
            r for r in self._active_requests if not r.is_done
        ]

        # Step 2: Admit waiting requests up to batch capacity
        available_slots = self._config.max_batch_size - len(self._active_requests)
        self._waiting_queue.sort(key=lambda r: -r.priority)  # highest priority first

        admitted: list[SarathiRequest] = []
        while available_slots > 0 and self._waiting_queue:
            req = self._waiting_queue.pop(0)
            req.phase = RequestPhase.PREFILL
            self._active_requests.append(req)
            admitted.append(req)
            available_slots -= 1

        # Step 3: Compute chunk size from available KV budget
        used_blocks = self._estimate_used_blocks()
        available_blocks = max(
            1, self._config.kv_cache_blocks - used_blocks
        )
        num_active = max(1, len(self._active_requests))
        chunk_size = max(1, (available_blocks * self._config.block_size) // num_active)

        # Step 4: Separate active requests by phase
        prefill_reqs = [r for r in self._active_requests if r.phase == RequestPhase.PREFILL]
        decode_reqs = [r for r in self._active_requests if r.phase == RequestPhase.DECODE]

        # Step 5: Reserve decode slots (decode_ratio of batch)
        max_decode_slots = max(1, int(len(self._active_requests) * self._config.decode_ratio))
        decode_to_schedule = decode_reqs[:max_decode_slots]

        # Step 6: Build prefill chunks
        prefill_chunks: list[tuple[int | str, int]] = []
        remaining_token_budget = self._config.max_num_batched_tokens

        # Account for decode tokens in budget
        for req in decode_to_schedule:
            remaining_token_budget -= 1  # 1 token per decode request

        for req in prefill_reqs:
            if remaining_token_budget <= 0:
                break
            tokens_to_prefill = min(
                req.remaining_prefill,
                chunk_size,
                remaining_token_budget,
            )
            if tokens_to_prefill > 0:
                prefill_chunks.append((req.request_id, tokens_to_prefill))
                req.tokens_prefilled += tokens_to_prefill
                remaining_token_budget -= tokens_to_prefill
                self._total_prefill_tokens += tokens_to_prefill
                self._total_scheduled_chunks += 1

                if req.is_prefill_done:
                    req.phase = RequestPhase.DECODE

        # Step 7: Build decode plan
        decode_ids = [r.request_id for r in decode_to_schedule]
        for req in decode_to_schedule:
            req.tokens_decoded += 1
            self._total_decode_tokens += 1
            if req.is_decode_done:
                req.phase = RequestPhase.COMPLETED

        return {
            "prefill_chunks": prefill_chunks,
            "decode_requests": decode_ids,
            "chunk_size": chunk_size,
            "available_blocks": available_blocks,
            "iteration": self._total_iterations,
        }

    def _estimate_used_blocks(self) -> int:
        """Estimate KV cache blocks currently in use by active requests."""
        total_tokens = 0
        for req in self._active_requests:
            total_tokens += req.tokens_prefilled + req.tokens_decoded
        return (total_tokens + self._config.block_size - 1) // self._config.block_size

    def get_stats(self) -> dict[str, Any]:
        """Return scheduler statistics.

        Returns
        -------
        dict with throughput metrics, queue depths, and utilization stats.
        """
        return {
            "iteration": self._total_iterations,
            "waiting": len(self._waiting_queue),
            "active": len(self._active_requests),
            "completed": len(self._completed_requests),
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
            "total_scheduled_chunks": self._total_scheduled_chunks,
            "kv_utilization": self._estimate_used_blocks() / max(1, self._config.kv_cache_blocks),
        }

    def reset(self) -> None:
        """Clear all state."""
        self._waiting_queue.clear()
        self._active_requests.clear()
        self._completed_requests.clear()
        self._total_iterations = 0
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._total_scheduled_chunks = 0
