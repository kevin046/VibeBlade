"""Finite State Machine primitives for grammar-constrained decoding.

Provides State, NFA, and DFA classes with Thompson's construction
(regex → NFA → DFA via subset construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    """Immutable, hashable DFA/NFA state identifier."""
    id: int

    def __repr__(self) -> str:
        return f"S{self.id}"


# ---------------------------------------------------------------------------
# NFA (Non-deterministic Finite Automaton)
# ---------------------------------------------------------------------------

class NFA:
    """NFA built via Thompson's construction.

    Uses *epsilon* (ε) transitions freely.  Supports conversion to a DFA
    via the standard subset-construction algorithm.
    """

    def __init__(self) -> None:
        self.state_counter: int = 0
        self.transitions: Dict[State, Dict[Optional[str], Set[State]]] = {}
        self.start_state: Optional[State] = None
        self.accept_states: Set[State] = set()

    def copy(self) -> "NFA":
        """Shallow copy (states are shared, but transition dict is fresh)."""
        nfa = NFA()
        nfa.state_counter = self.state_counter
        nfa.start_state = self.start_state
        nfa.accept_states = set(self.accept_states)
        for src, edges in self.transitions.items():
            nfa.transitions[src] = {k: set(v) for k, v in edges.items()}
        return nfa

    def renumber(self, offset: int) -> "NFA":
        """Return a new NFA with every state ID shifted by *offset*."""
        nfa = NFA()
        nfa.state_counter = self.state_counter + offset
        mapping: Dict[State, State] = {}
        for s in self.transitions:
            mapping[s] = State(s.id + offset)
        if self.start_state is not None:
            nfa.start_state = mapping[self.start_state]
        nfa.accept_states = {mapping[s] for s in self.accept_states}
        for src, edges in self.transitions.items():
            new_edges: Dict[Optional[str], Set[State]] = {}
            for char, dsts in edges.items():
                new_edges[char] = {mapping[d] for d in dsts}
            nfa.transitions[mapping[src]] = new_edges
        return nfa

    # -- state factory -------------------------------------------------------

    def new_state(self) -> State:
        s = State(self.state_counter)
        self.state_counter += 1
        self.transitions.setdefault(s, {})
        return s

    # -- transition helpers --------------------------------------------------

    def add_transition(self, src: State, char: Optional[str], dst: State) -> None:
        """Add a transition.  *char=None* means epsilon transition."""
        self.transitions.setdefault(src, {}).setdefault(char, set()).add(dst)

    def add_epsilon(self, src: State, dst: State) -> None:
        self.add_transition(src, None, dst)

    # -- Thompson construction primitives ------------------------------------

    @classmethod
    def from_char(cls, ch: str) -> "NFA":
        """NFA that accepts the single character *ch*."""
        nfa = cls()
        s0 = nfa.new_state()
        s1 = nfa.new_state()
        nfa.start_state = s0
        nfa.accept_states = {s1}
        nfa.add_transition(s0, ch, s1)
        return nfa

    @classmethod
    def from_charset(cls, chars: Set[str]) -> "NFA":
        """NFA that accepts any single character in *chars*."""
        nfa = cls()
        s0 = nfa.new_state()
        s1 = nfa.new_state()
        nfa.start_state = s0
        nfa.accept_states = {s1}
        for ch in chars:
            nfa.add_transition(s0, ch, s1)
        return nfa

    @classmethod
    def from_dot(cls) -> "NFA":
        """NFA that accepts any character (``.`` in regex)."""
        chars = set(chr(c) for c in range(32, 127)) | {"\n", "\r", "\t"}
        return cls.from_charset(chars)

    @classmethod
    def from_epsilon(cls) -> "NFA":
        """NFA that accepts the empty string."""
        nfa = cls()
        s = nfa.new_state()
        nfa.start_state = s
        nfa.accept_states = {s}
        return nfa

    @classmethod
    def concat(cls, a: "NFA", b: "NFA") -> "NFA":
        """Concatenation: a then b."""
        b_shifted = b.renumber(a.state_counter)
        nfa = NFA()
        nfa.state_counter = b_shifted.state_counter
        nfa.start_state = a.start_state
        nfa.accept_states = set(b_shifted.accept_states)
        # Merge transitions from a
        for src, edges in a.transitions.items():
            nfa.transitions[src] = {k: set(v) for k, v in edges.items()}
        # Merge transitions from b (renumbered)
        for src, edges in b_shifted.transitions.items():
            merged = nfa.transitions.setdefault(src, {})
            for char, dsts in edges.items():
                merged.setdefault(char, set()).update(dsts)
        # epsilon-link a's accept states to b's start state
        for s in a.accept_states:
            nfa.add_epsilon(s, b_shifted.start_state)
        return nfa

    @classmethod
    def alternate(cls, a: "NFA", b: "NFA") -> "NFA":
        """Alternation: a | b."""
        b_shifted = b.renumber(a.state_counter)
        nfa = NFA()
        s0 = nfa.new_state()  # id = next_id
        s1 = nfa.new_state()  # id = next_id + 1
        nfa.state_counter = s1.id + 1
        nfa.start_state = s0
        nfa.accept_states = {s1}
        # Merge a
        for src, edges in a.transitions.items():
            nfa.transitions[src] = {k: set(v) for k, v in edges.items()}
        # Merge b (renumbered)
        for src, edges in b_shifted.transitions.items():
            merged = nfa.transitions.setdefault(src, {})
            for char, dsts in edges.items():
                merged.setdefault(char, set()).update(dsts)
        # epsilon from new start to a.start and b.start
        nfa.add_epsilon(s0, a.start_state)
        nfa.add_epsilon(s0, b_shifted.start_state)
        # epsilon from a.accept and b.accept to new accept
        for s in a.accept_states:
            nfa.add_epsilon(s, s1)
        for s in b_shifted.accept_states:
            nfa.add_epsilon(s, s1)
        return nfa

    @classmethod
    def star(cls, a: "NFA") -> "NFA":
        """Kleene star: a*."""
        nfa = NFA()
        s0 = nfa.new_state()  # id = next_id
        s1 = nfa.new_state()  # id = next_id + 1
        nfa.state_counter = s1.id + 1
        nfa.start_state = s0
        nfa.accept_states = {s1}
        # Merge a
        for src, edges in a.transitions.items():
            nfa.transitions[src] = {k: set(v) for k, v in edges.items()}
        nfa.add_epsilon(s0, a.start_state)
        nfa.add_epsilon(s0, s1)
        for s in a.accept_states:
            nfa.add_epsilon(s, a.start_state)
            nfa.add_epsilon(s, s1)
        return nfa

    @classmethod
    def plus(cls, a: "NFA") -> "NFA":
        """One-or-more: a+  (equivalent to a a*)."""
        return cls.concat(a, cls.star(a))

    @classmethod
    def optional(cls, a: "NFA") -> "NFA":
        """Zero-or-one: a?."""
        return cls.alternate(a, cls.from_epsilon())

    # -- epsilon closure & subset construction -------------------------------

    def _epsilon_closure(self, states: Set[State]) -> Set[State]:
        """Compute the epsilon-closure of a set of NFA states."""
        stack = list(states)
        closure: Set[State] = set(states)
        while stack:
            s = stack.pop()
            for dst in self.transitions.get(s, {}).get(None, ()):
                if dst not in closure:
                    closure.add(dst)
                    stack.append(dst)
        return closure

    def to_dfa(self) -> "DFA":
        """Convert this NFA to a DFA via subset construction."""
        dfa = DFA()
        start_closure = frozenset(self._epsilon_closure({self.start_state}))
        dfa.start_state = start_closure
        dfa.states.add(start_closure)

        if self.accept_states & start_closure:
            dfa.accept_states.add(start_closure)

        worklist = [start_closure]
        while worklist:
            current = worklist.pop()
            # Collect all characters reachable from any state in current
            char_map: Dict[str, Set[State]] = {}
            for s in current:
                for char, dsts in self.transitions.get(s, {}).items():
                    if char is not None:  # skip epsilon
                        char_map.setdefault(char, set()).update(dsts)

            for char, dst_states in char_map.items():
                next_closure = frozenset(self._epsilon_closure(dst_states))
                dfa.alphabet.add(char)
                if next_closure not in dfa.states:
                    dfa.states.add(next_closure)
                    worklist.append(next_closure)
                    if self.accept_states & next_closure:
                        dfa.accept_states.add(next_closure)
                dfa.transitions.setdefault(current, {})[char] = next_closure

        return dfa


# ---------------------------------------------------------------------------
# DFA (Deterministic Finite Automaton)
# ---------------------------------------------------------------------------

class DFA:
    """Deterministic finite automaton over characters."""

    def __init__(self) -> None:
        self.states: Set[FrozenSet[State]] = set()
        self.alphabet: Set[str] = set()
        self.transitions: Dict[FrozenSet[State], Dict[str, FrozenSet[State]]] = {}
        self.start_state: Optional[FrozenSet[State]] = None
        self.accept_states: Set[FrozenSet[State]] = set()

    def transition(self, state, char: str):
        """Follow a transition, returning a dead sentinel if stuck."""
        dead = frozenset()
        if state == dead:
            return dead
        edges = self.transitions.get(state, {})
        return edges.get(char, dead)

    def is_accepting(self, state) -> bool:
        return state in self.accept_states

    def matches(self, text: str) -> bool:
        """Return True if *text* is accepted by this DFA."""
        state = self.start_state
        for ch in text:
            nxt = self.transitions.get(state, {}).get(ch)
            if nxt is None:
                return False
            state = nxt
        return state in self.accept_states

    def can_accept(self, state) -> bool:
        """Check if any path from *state* reaches an accepting state.

        BFS from *state*.  Used by GrammarConstraint to decide whether a
        partial match can still become valid.
        """
        visited: Set = set()
        queue = [state]
        while queue:
            s = queue.pop()
            if s in self.accept_states:
                return True
            if s in visited:
                continue
            visited.add(s)
            for char, nxt in self.transitions.get(s, {}).items():
                if nxt not in visited:
                    queue.append(nxt)
        return False

    @staticmethod
    def _dead_state() -> FrozenSet[State]:
        """Sentinel for the dead (sink) state."""
        return frozenset()
