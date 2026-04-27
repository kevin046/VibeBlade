"""Markov-chain prediction oracle for MoE expert routing.

Predicts which experts will be needed at layer L+1 while the GPU is
still processing layer L's attention mechanism, enabling proactive
expert loading and reduced memory traffic.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


class ExpertOracle:
    """First-order (or higher-order) Markov transition table for expert routing prediction.

    Tracks which expert E_j tends to follow expert E_i at the next layer.
    Can also track cross-layer transitions (which layer-L experts predict
    which layer-L+1 experts).

    For order=1 the transition table is a dense ``(num_experts, num_experts)``
    matrix per layer.  For order>=2 a sparse nested-dict representation is
    used to avoid the combinatorial blow-up of higher-order tables.
    """

    def __init__(self, num_experts: int, order: int = 1) -> None:
        """Initialise the oracle.

        Args:
            num_experts: Total number of experts per layer.
            order: Markov chain order.
                   * 1 — next-expert depends only on the current expert.
                   * 2 — next-expert depends on the previous two experts, etc.
        """
        self.num_experts = num_experts
        self.order = order

        # Transition storage --------------------------------------------------
        if order == 1:
            # layer_idx → np.ndarray[num_experts, num_experts]
            self._transitions: dict[int, np.ndarray] = {}
        else:
            # layer_idx → {context_key → Counter of next-experts}
            self._transitions: dict[int, dict[tuple[int, ...], Counter]] = {}

        # History of observed experts per layer (needed for higher-order)
        self._history: dict[int, list[int]] = {}

        # Prediction tracking -------------------------------------------------
        self._last_prediction: dict[int, list[tuple[int, float]]] = {}
        self._correct_predictions: int = 0
        self._total_predictions: int = 0

    # --------------------------------------------------------------------- #
    # Core API
    # --------------------------------------------------------------------- #

    def observe(self, layer_idx: int, selected_experts: list[int]) -> None:
        """Record observed expert selections for a layer.

        Updates transition counts for observed transitions and evaluates
        the last prediction made for this layer (if any) to accumulate
        accuracy statistics.

        Args:
            layer_idx: Index of the layer that just finished execution.
            selected_experts: Expert IDs that were actually selected.
        """
        # Evaluate prediction accuracy before updating tables ----------------
        if layer_idx in self._last_prediction:
            predicted_ids = {eid for eid, _p in self._last_prediction[layer_idx]}
            actual_ids = set(selected_experts)
            if predicted_ids and actual_ids:
                self._correct_predictions += len(predicted_ids & actual_ids)
                self._total_predictions += len(predicted_ids | actual_ids)
            del self._last_prediction[layer_idx]

        # Update transition counts -------------------------------------------
        prev_experts = self._history.get(layer_idx - 1, [])
        cur_experts = list(selected_experts)

        if prev_experts:
            if self.order == 1:
                self._update_order1(layer_idx - 1, prev_experts, cur_experts)
            else:
                self._update_order_n(layer_idx - 1, prev_experts, cur_experts)

        self._history[layer_idx] = cur_experts

    def predict_next(
        self,
        layer_idx: int,
        current_experts: list[int],
        top_k: int = 4,
    ) -> list[tuple[int, float]]:
        """Predict experts for *layer_idx + 1* given current selections.

        For each current expert the corresponding transition row is looked
        up, and the rows are averaged to produce a merged probability
        distribution.  The top-*k* entries are returned.

        Args:
            layer_idx: Index of the layer currently being processed.
            current_experts: Expert IDs selected at *layer_idx*.
            top_k: Number of top predictions to return.

        Returns:
            List of ``(expert_id, probability)`` sorted descending by
            probability.
        """
        if not current_experts:
            return []

        if self.order == 1:
            probs = self._predict_order1(layer_idx, current_experts)
        else:
            probs = self._predict_order_n(layer_idx, current_experts)

        # Fallback: no data yet or all-zero distribution
        if probs is None:
            fallback = 1.0 / max(len(current_experts), 1)
            return [(eid, fallback) for eid in current_experts[:top_k]]

        if isinstance(probs, np.ndarray):
            if probs.sum() == 0.0:
                fallback = 1.0 / max(len(current_experts), 1)
                return [(eid, fallback) for eid in current_experts[:top_k]]
            sorted_indices = np.argsort(probs)[::-1][:top_k]
            return [(int(idx), float(probs[idx])) for idx in sorted_indices]
        else:
            # probs is a dict (higher-order case)
            if not probs:
                fallback = 1.0 / max(len(current_experts), 1)
                return [(eid, fallback) for eid in current_experts[:top_k]]
            sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            return sorted_items[:top_k]

    def transition_matrix(self, layer_idx: int) -> np.ndarray | None:
        """Return the row-normalised transition matrix for *layer_idx*.

        Only meaningful when ``order == 1``.  Returns *None* for
        higher-order models or if no data has been recorded for the
        requested layer.

        Args:
            layer_idx: The layer whose transition matrix to return.

        Returns:
            ``(num_experts, num_experts)`` array of transition
            probabilities, or *None*.
        """
        if self.order != 1:
            return None
        counts = self._transitions.get(layer_idx)
        if counts is None:
            return None
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        return counts / row_sums

    def accuracy(self) -> float:
        """Return the Jaccard-weighted prediction accuracy.

        Accuracy is computed as the ratio of correctly predicted experts
        to the total unique experts across all (predicted, actual) pairs
        that have been evaluated so far.

        Returns:
            Float in [0, 1].  Returns 0.0 if no evaluation has occurred.
        """
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def reset(self) -> None:
        """Clear all transition counts, history, and accuracy tracking."""
        if self.order == 1:
            self._transitions: dict[int, np.ndarray] = {}
        else:
            self._transitions: dict[int, dict[tuple[int, ...], Counter]] = {}
        self._history.clear()
        self._last_prediction.clear()
        self._correct_predictions = 0
        self._total_predictions = 0

    # --------------------------------------------------------------------- #
    # Internal helpers – order 1
    # --------------------------------------------------------------------- #

    def _ensure_order1_matrix(self, layer_idx: int) -> np.ndarray:
        """Lazily create a zero matrix for *layer_idx*."""
        if layer_idx not in self._transitions:
            self._transitions[layer_idx] = np.zeros(
                (self.num_experts, self.num_experts), dtype=np.float64
            )
        return self._transitions[layer_idx]

    def _update_order1(
        self,
        layer_idx: int,
        prev_experts: list[int],
        cur_experts: list[int],
    ) -> None:
        matrix = self._ensure_order1_matrix(layer_idx)
        for src in prev_experts:
            for dst in cur_experts:
                matrix[src, dst] += 1.0

    def _predict_order1(
        self,
        layer_idx: int,
        current_experts: list[int],
    ) -> np.ndarray | None:
        matrix = self._transitions.get(layer_idx)
        if matrix is None:
            return None
        rows = matrix[current_experts, :]
        return rows.mean(axis=0)

    # --------------------------------------------------------------------- #
    # Internal helpers – order >= 2
    # --------------------------------------------------------------------- #

    def _context_keys_for_order_n(
        self,
        prev_experts: list[int],
    ) -> list[tuple[int, ...]]:
        """Generate all ``order``-length context tuples from *prev_experts*.

        For order=2 with prev_experts=[0, 3, 1] the contexts are
        ``(0, 3)`` and ``(3, 1)`` — sliding windows of length ``order``.
        """
        keys: list[tuple[int, ...]] = []
        for start in range(len(prev_experts) - self.order + 1):
            keys.append(tuple(prev_experts[start : start + self.order]))
        return keys

    def _update_order_n(
        self,
        layer_idx: int,
        prev_experts: list[int],
        cur_experts: list[int],
    ) -> None:
        if layer_idx not in self._transitions:
            self._transitions[layer_idx] = {}
        tbl = self._transitions[layer_idx]
        for ctx in self._context_keys_for_order_n(prev_experts):
            if ctx not in tbl:
                tbl[ctx] = Counter()
            for dst in cur_experts:
                tbl[ctx][dst] += 1

    def _predict_order_n(
        self,
        layer_idx: int,
        current_experts: list[int],
    ) -> dict[int, float] | None:
        tbl = self._transitions.get(layer_idx)
        if tbl is None:
            return None
        contexts = self._context_keys_for_order_n(current_experts)
        if not contexts:
            return None
        merged: Counter = Counter()
        for ctx in contexts:
            counter = tbl.get(ctx)
            if counter is not None:
                merged.update(counter)
        total = sum(merged.values())
        if total == 0:
            return None
        return {k: v / total for k, v in merged.items()}


class PatternOracle:
    """Detects repeating expert access patterns across layers.

    Many MoE models exhibit periodic expert activation patterns.  This
    oracle detects *n*-gram patterns in the layer-wise expert selection
    sequence and uses them to predict upcoming expert activations.

    The pattern table is keyed by a tuple of ``frozenset`` objects —
    one per layer in the prefix — and maps to a ``Counter`` of which
    experts appeared in the following layer.
    """

    def __init__(self, num_experts: int, pattern_length: int = 4) -> None:
        """Initialise the pattern oracle.

        Args:
            num_experts: Total number of experts per layer.
            pattern_length: Length of n-grams to track (default 4 layers).
                The predictor uses the last ``pattern_length - 1`` layers
                to look up the most likely continuation.
        """
        self.num_experts = num_experts
        self.pattern_length = pattern_length

        # Ordered history of per-layer expert selections
        self._layer_history: list[frozenset[int]] = []

        # Pattern table: prefix tuple of frozensets → Counter of next experts
        self._pattern_table: dict[
            tuple[frozenset[int], ...], Counter[int]
        ] = {}

    # --------------------------------------------------------------------- #
    # Core API
    # --------------------------------------------------------------------- #

    def observe(self, layer_idx: int, selected_experts: list[int]) -> None:
        """Record expert selections and update the pattern frequency table.

        When enough history has been accumulated (≥ ``pattern_length``
        entries) every new observation causes a new pattern entry to be
        recorded in the frequency table.

        Args:
            layer_idx: Index of the layer that just finished (unused for
                the frequency table but recorded for potential diagnostics).
            selected_experts: Expert IDs that were actually selected.
        """
        current = frozenset(selected_experts)
        self._layer_history.append(current)

        # We need at least pattern_length entries to form a full n-gram
        if len(self._layer_history) >= self.pattern_length:
            prefix = tuple(
                self._layer_history[
                    -(self.pattern_length) : -1
                ]
            )
            if prefix not in self._pattern_table:
                self._pattern_table[prefix] = Counter()
            self._pattern_table[prefix].update(current)

    def predict(self, layer_idx: int, top_k: int = 4) -> list[tuple[int, float]]:
        """Predict experts for this layer based on recent pattern history.

        Looks up the most recent ``pattern_length - 1`` layer selections
        in the pattern table and returns the most likely continuation
        experts.

        Args:
            layer_idx: Index of the layer to predict for (unused —
                predictions are based on history alone).
            top_k: Number of top predictions to return.

        Returns:
            List of ``(expert_id, probability)`` sorted descending by
            probability.  Empty if no matching pattern prefix exists.
        """
        if len(self._layer_history) < self.pattern_length - 1:
            return []

        prefix = tuple(
            self._layer_history[-(self.pattern_length - 1) :]
        )
        counter = self._pattern_table.get(prefix)
        if counter is None:
            return []

        total = sum(counter.values())
        if total == 0:
            return []

        sorted_items = sorted(
            ((eid, count / total) for eid, count in counter.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return sorted_items[:top_k]

    def dominant_patterns(
        self, min_frequency: float = 0.1,
    ) -> list[tuple[tuple[frozenset[int], ...], float]]:
        """Return patterns that appear more than *min_frequency* of the time.

        Frequency is computed relative to the total number of recorded
        pattern observations (i.e. total n-gram count).

        Args:
            min_frequency: Minimum relative frequency threshold.

        Returns:
            List of ``(pattern_prefix, frequency)`` pairs, sorted
            descending by frequency.
        """
        if not self._pattern_table:
            return []

        # Total observations = sum of all counter sums
        total_obs = sum(sum(c.values()) for c in self._pattern_table.values())
        if total_obs == 0:
            return []

        result: list[tuple[tuple[frozenset[int], ...], float]] = []
        for prefix, counter in self._pattern_table.items():
            freq = sum(counter.values()) / total_obs
            if freq >= min_frequency:
                result.append((prefix, freq))

        return sorted(result, key=lambda kv: kv[1], reverse=True)

    def reset(self) -> None:
        """Clear all pattern data and history."""
        self._layer_history.clear()
        self._pattern_table.clear()
