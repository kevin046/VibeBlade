"""VibeBlade SageSched — Uncertainty-aware scheduling for LLM serving.

Based on: SageAttention / uncertainty-based scheduling research.

SageSched prioritizes requests based on their uncertainty score — a measure
of how "uncertain" the model is about its next token prediction. Requests
with higher uncertainty are scheduled first because they benefit more from
prompt context and compute, while low-uncertainty requests can be deferred
to fill remaining batch slots.

Uncertainty estimation:
    H(p) = -Σ p_i * log2(p_i)

Where p is the output probability distribution. Low entropy = confident
prediction, high entropy = uncertain.

Scheduling policy:
    priority_score = α * base_priority + β * uncertainty_score + γ * wait_penalty
    where:
        α = 1.0 (base priority weight)
        β = 2.0 (uncertainty weight — uncertainty is the primary signal)
        γ = 0.01 (wait time penalty to prevent starvation)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class SageRequest:
    """A request tracked by SageSched with uncertainty metadata.

    Attributes
    ----------
    request_id : int or str
        Unique identifier.
    total_prompt_tokens : int
        Prompt length.
    tokens_decoded : int
        Decode tokens generated so far.
    max_decode_tokens : int
        Maximum decode tokens.
    base_priority : float
        User-assigned priority (e.g., premium vs free tier).
    arrival_time : float
        Monotonic arrival timestamp.
    last_entropy : float
        Entropy of the last token's output distribution.
    avg_entropy : float
        Running average entropy over all decode steps.
    tokens_processed : int
        Number of decode steps processed (for running average).
    """
    request_id: int | str
    total_prompt_tokens: int = 128
    tokens_decoded: int = 0
    max_decode_tokens: int = 256
    base_priority: float = 1.0
    arrival_time: float = 0.0
    last_entropy: float = 0.0
    avg_entropy: float = 0.0
    tokens_processed: int = 0

    @property
    def is_done(self) -> bool:
        return self.tokens_decoded >= self.max_decode_tokens

    @property
    def wait_time(self) -> float:
        return time.monotonic() - self.arrival_time


@dataclass
class SageConfig:
    """Configuration for SageSched.

    Attributes
    ----------
    alpha : float
        Weight for base priority in scoring.
    beta : float
        Weight for uncertainty (entropy) in scoring. Higher = more
        uncertainty-driven scheduling.
    gamma : float
        Weight for wait time penalty. Prevents starvation.
    max_batch_size : int
        Maximum concurrent requests.
    entropy_window : int
        Window size for computing running average entropy.
    """
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.01
    max_batch_size: int = 32
    entropy_window: int = 10


def entropy_from_logits(logits: list[float] | tuple[float, ...]) -> float:
    """Compute Shannon entropy H(p) = -Σ p_i * log2(p_i) from logits.

    Applies softmax to convert logits to probabilities first.

    Parameters
    ----------
    logits : list of float
        Raw logit values from the model output.

    Returns
    -------
    float
        Entropy in bits (≥ 0). Higher values indicate more uncertainty.
    """
    if not logits:
        return 0.0
    # Numerically stable softmax
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    sum_exps = sum(exps)
    if sum_exps == 0:
        return 0.0

    h = 0.0
    for p in exps:
        prob = p / sum_exps
        if prob > 1e-10:
            h -= prob * math.log2(prob)
    return h


def entropy_from_probs(probs: list[float]) -> float:
    """Compute Shannon entropy directly from probabilities.

    Parameters
    ----------
    probs : list of float
        Probability distribution (should sum to ~1.0).

    Returns
    -------
    float
        Entropy in bits.
    """
    if not probs:
        return 0.0
    s = sum(probs)
    if s == 0:
        return 0.0
    h = 0.0
    for p in probs:
        norm_p = p / s
        if norm_p > 1e-10:
            h -= norm_p * math.log2(norm_p)
    return h


class SageSched:
    """Uncertainty-aware scheduler that prioritizes high-entropy requests.

    Key improvements over FIFO / priority scheduling:
    - Requests with uncertain predictions get scheduled first (they benefit
      more from compute resources)
    - Low-uncertainty "easy" requests are batched opportunistically
    - Wait time penalty prevents indefinite starvation
    - Running average entropy smooths noisy per-token measurements

    Parameters
    ----------
    config : SageConfig or None
        Scheduler configuration. Uses defaults if None.
    """

    def __init__(self, config: SageConfig | None = None) -> None:
        self._config = config or SageConfig()
        self._queue: list[SageRequest] = []
        self._active: list[SageRequest] = []
        self._completed: list[SageRequest] = []
        self._next_id: int = 0
        self._total_scheduled: int = 0

    def add_request(
        self,
        prompt_tokens: int = 128,
        max_decode_tokens: int = 256,
        base_priority: float = 1.0,
    ) -> SageRequest:
        """Add a new request to the scheduling queue.

        Parameters
        ----------
        prompt_tokens : int
            Number of tokens in the prompt.
        max_decode_tokens : int
            Maximum decode tokens to generate.
        base_priority : float
            User-assigned priority.

        Returns
        -------
        SageRequest
            The created request object.
        """
        req = SageRequest(
            request_id=self._next_id,
            total_prompt_tokens=prompt_tokens,
            max_decode_tokens=max_decode_tokens,
            base_priority=base_priority,
            arrival_time=time.monotonic(),
        )
        self._next_id += 1
        self._queue.append(req)
        return req

    def update_entropy(
        self,
        request_id: int | str,
        entropy: float | None = None,
        logits: list[float] | None = None,
    ) -> None:
        """Update the uncertainty (entropy) for a request.

        Either provide the entropy directly or logits (which will be
        converted to entropy internally).

        Parameters
        ----------
        request_id : int or str
            The request to update.
        entropy : float or None
            Direct entropy value in bits.
        logits : list of float or None
            Raw logits from the model (converted to entropy).
        """
        if entropy is None and logits is not None:
            entropy = entropy_from_logits(logits)
        if entropy is None:
            return

        # Find the request in active or queue
        req = self._find_request(request_id)
        if req is None:
            return

        req.last_entropy = entropy
        req.tokens_processed += 1

        # Running average with window
        window = min(req.tokens_processed, self._config.entropy_window)
        req.avg_entropy = (
            (req.avg_entropy * (window - 1) + entropy) / window
            if window > 0 else entropy
        )

    def _find_request(self, request_id: int | str) -> SageRequest | None:
        for req in self._active:
            if req.request_id == request_id:
                return req
        for req in self._queue:
            if req.request_id == request_id:
                return req
        return None

    def _compute_priority(self, req: SageRequest) -> float:
        """Compute the scheduling priority score for a request.

        score = α * base_priority + β * avg_entropy + γ * wait_time

        Higher score = scheduled sooner.
        """
        cfg = self._config
        return (
            cfg.alpha * req.base_priority
            + cfg.beta * req.avg_entropy
            + cfg.gamma * req.wait_time
        )

    def schedule(self) -> dict[str, Any]:
        """Run one scheduling iteration.

        Returns
        -------
        dict with:
            - 'scheduled_ids': list of request_ids scheduled this iteration
            - 'priorities': dict mapping request_id to priority score
            - 'avg_uncertainty': float — average entropy of scheduled requests
        """
        # Move completed requests
        self._active = [r for r in self._active if not r.is_done]
        for req in list(self._active):
            if req.is_done:
                self._completed.append(req)
        self._active = [r for r in self._active if not r.is_done]

        # Admit from queue (up to batch capacity)
        available_slots = self._config.max_batch_size - len(self._active)

        # Sort queue by priority score (descending)
        self._queue.sort(key=self._compute_priority, reverse=True)

        admitted: list[SageRequest] = []
        while available_slots > 0 and self._queue:
            req = self._queue.pop(0)
            self._active.append(req)
            admitted.append(req)
            available_slots -= 1

        # Compute priorities for all active requests
        priorities = {
            req.request_id: self._compute_priority(req)
            for req in self._active
        }

        # Sort active by priority for execution order
        sorted_active = sorted(
            self._active, key=lambda r: priorities[r.request_id], reverse=True
        )

        scheduled_ids = [r.request_id for r in sorted_active]
        self._total_scheduled += len(scheduled_ids)

        # Average uncertainty of scheduled requests
        avg_uncertainty = 0.0
        if sorted_active:
            avg_uncertainty = sum(r.avg_entropy for r in sorted_active) / len(sorted_active)

        return {
            "scheduled_ids": scheduled_ids,
            "priorities": priorities,
            "avg_uncertainty": avg_uncertainty,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return scheduler statistics."""
        return {
            "queue_length": len(self._queue),
            "active_count": len(self._active),
            "completed_count": len(self._completed),
            "total_scheduled": self._total_scheduled,
            "avg_active_entropy": (
                sum(r.avg_entropy for r in self._active) / max(1, len(self._active))
            ),
        }

    def reset(self) -> None:
        """Clear all state."""
        self._queue.clear()
        self._active.clear()
        self._completed.clear()
        self._total_scheduled = 0
