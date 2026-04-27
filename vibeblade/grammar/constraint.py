"""GrammarConstraint — main integration point for grammar-constrained decoding.

Tracks a DFA state across autoregressive generation steps and produces
boolean masks over the tokenizer vocabulary at each step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional

import numpy as np

from .fsm import DFA, State

if TYPE_CHECKING:
    pass


class GrammarConstraint:
    """Vocabulary-level grammar constraint driven by a character-level DFA.

    Precomputes a *state → token-mask* cache so that per-step masking is
    O(1) after the first visit to each DFA state.

    Parameters
    ----------
    vocab : list[str]
        The tokenizer vocabulary as a list of decoded token strings.
    dfa : DFA
        A character-level deterministic finite automaton.
    eos_token_id : int, optional
        Index of the end-of-sequence token.  When no regular tokens are
        valid the EOS token is always allowed (prevents dead-ends).
    """

    def __init__(
        self,
        vocab: List[str],
        dfa: DFA,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.vocab = vocab
        self.dfa = dfa
        self.eos_token_id = eos_token_id
        self.vocab_size = len(vocab)

        # Current DFA state (frozen-set of NFA states from subset construction)
        self._current_state = dfa.start_state

        # Cache: DFA state -> np.ndarray[bool] (token mask)
        self._mask_cache: Dict[FrozenSet[State], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Factory class-methods
    # ------------------------------------------------------------------

    @classmethod
    def from_regex(cls, vocab: List[str], pattern: str, **kwargs) -> "GrammarConstraint":
        """Create a constraint from a regex pattern string.

        Parameters
        ----------
        vocab : list[str]
            Tokenizer vocabulary.
        pattern : str
            Regular expression pattern.
        """
        from .regex_grammar import regex_to_dfa

        dfa = regex_to_dfa(pattern)
        return cls(vocab, dfa, **kwargs)

    @classmethod
    def from_json_schema(cls, vocab: List[str], schema: dict, **kwargs) -> "GrammarConstraint":
        """Create a constraint from a JSON Schema.

        Parameters
        ----------
        vocab : list[str]
            Tokenizer vocabulary.
        schema : dict
            JSON Schema dictionary.
        """
        from .json_schema import JsonSchemaGrammar

        jg = JsonSchemaGrammar(schema)
        return cls(vocab, jg.dfa, **kwargs)

    @classmethod
    def from_ebnf(cls, vocab: List[str], grammar_str: str, start_rule: Optional[str] = None,
                  **kwargs) -> "GrammarConstraint":
        """Create a constraint from an EBNF/GBNF grammar string.

        Parameters
        ----------
        vocab : list[str]
            Tokenizer vocabulary.
        grammar_str : str
            EBNF grammar text.
        start_rule : str, optional
            Name of the start rule (first rule if *None*).
        """
        from .ebnf import EbnfGrammar

        eg = EbnfGrammar(grammar_str, start_rule=start_rule)
        return cls(vocab, eg.dfa, **kwargs)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def advance(self, token_str: str) -> None:
        """Update internal DFA state after a token has been selected.

        Parameters
        ----------
        token_str : str
            The decoded string of the chosen token.
        """
        for ch in token_str:
            nxt = self.dfa.transitions.get(self._current_state, {}).get(ch)
            if nxt is None:
                # Dead state — remain dead (no valid continuations possible)
                self._current_state = DFA._dead_state()
                return
            self._current_state = nxt

    def get_allowed_tokens(self) -> List[int]:
        """Return indices of tokens that are valid next steps."""
        mask = self.get_token_mask()
        return [int(i) for i in np.where(mask)[0]]

    def get_token_mask(self) -> np.ndarray:
        """Return a boolean mask (shape ``(vocab_size,)``) over the vocabulary.

        ``True`` entries indicate tokens whose characters all lead to valid
        DFA transitions from the current state *and* whose final state can
        still reach an accepting state.
        """
        if self._current_state in self._mask_cache:
            return self._mask_cache[self._current_state]

        mask = np.zeros(self.vocab_size, dtype=bool)
        dead = frozenset()  # sentinel dead state

        for idx, token_str in enumerate(self.vocab):
            if not token_str:
                # Empty token strings are always allowed (e.g. special tokens)
                mask[idx] = True
                continue

            state = self._current_state
            valid = True
            for ch in token_str:
                nxt = self.dfa.transitions.get(state, {}).get(ch)
                if nxt is None or nxt == dead:
                    valid = False
                    break
                state = nxt

            if valid and state != dead:
                # Only allow if this token's end state can still reach an
                # accepting state (avoids forced dead-ends).
                if self.dfa.is_accepting(state) or self.dfa.can_accept(state):
                    mask[idx] = True

        # Safety net: always allow EOS to prevent infinite loops
        # (only if no non-empty real tokens are valid)
        has_real_tokens = any(mask[i] and self.vocab[i] for i in range(self.vocab_size))
        if self.eos_token_id is not None and not has_real_tokens:
            mask[self.eos_token_id] = True

        self._mask_cache[self._current_state] = mask
        return mask

    def is_finished(self) -> bool:
        """Return ``True`` if the current DFA state is an accepting state.

        This means the generated text so far is a valid complete match.
        """
        return self.dfa.is_accepting(self._current_state)

    def reset(self) -> None:
        """Reset to the DFA start state."""
        self._current_state = self.dfa.start_state
        self._mask_cache.clear()
