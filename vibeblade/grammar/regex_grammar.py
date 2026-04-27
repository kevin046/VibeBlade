"""Regex-to-DFA compiler using Thompson's construction.

Parses a regex string into an NFA, then converts to a DFA.
Supports: ``.``, ``*``, ``+``, ``?``, ``|``, ``()``, ``[]`` character
classes, ``\\d``, ``\\w``, ``\\s``, and common escape sequences.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .fsm import DFA, NFA


# ---------------------------------------------------------------------------
# Character-set helpers
# ---------------------------------------------------------------------------

_DIGIT = set("0123456789")
_WORD = _DIGIT | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_SPACE = set(" \t\n\r\f\v")


def _parse_charset(s: str, i: int) -> Tuple[Set[str], int]:
    """Parse a ``[...]`` character class starting at position *i* (after ``[``).

    Returns (char_set, next_index).
    Handles ranges like ``a-z``, negation ``[^...]``, and escapes.
    """
    negate = False
    if i < len(s) and s[i] == "^":
        negate = True
        i += 1
    chars: Set[str] = set()
    first = True
    while i < len(s) and (first or s[i] != "]"):
        first = False
        if s[i] == "\\" and i + 1 < len(s):
            i += 1
            esc = s[i]
            if esc == "d":
                chars |= _DIGIT
            elif esc == "w":
                chars |= _WORD
            elif esc == "s":
                chars |= _SPACE
            else:
                chars.add(esc)
            i += 1
            # Check for range  (e.g. \d-z is unusual but we handle it simply)
            if i < len(s) and s[i] == "-" and i + 1 < len(s) and s[i + 1] != "]":
                i += 1
                hi_char = s[i]
                if s[i - 2] == "\\" and s[i - 2 - 1] == "\\":
                    # second-to-last was escape; already expanded above — skip
                    pass
                else:
                    lo = s[i - 2]  # character before '-'
                    if isinstance(lo, str) and len(lo) == 1:
                        for c in range(ord(lo), ord(hi_char) + 1):
                            chars.add(chr(c))
                i += 1
            continue
        ch = s[i]
        i += 1
        # range  a-z
        if i < len(s) and s[i] == "-" and i + 1 < len(s) and s[i + 1] != "]":
            i += 1
            hi = s[i]
            i += 1
            for c in range(ord(ch), ord(hi) + 1):
                chars.add(chr(c))
        else:
            chars.add(ch)
    # skip closing ]
    if i < len(s):
        i += 1  # skip ']'
    if negate:
        # all printable + common whitespace minus chars
        all_chars = set(chr(c) for c in range(32, 127)) | {"\n", "\r", "\t"}
        chars = all_chars - chars
    return chars, i


# ---------------------------------------------------------------------------
# Recursive-descent regex parser → NFA
# ---------------------------------------------------------------------------

class _RegexParser:
    """Recursive-descent parser: regex string → NFA (Thompson's)."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def advance(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    # ---- grammar ----------------------------------------------------------
    # expr   := term ('|' term)*
    # term   := factor+
    # factor := atom quant?
    # quant  := '*' | '+' | '?'
    # atom   := '(' expr ')' | '[' charset ']' | '.' | '\' esc | literal

    def parse(self) -> NFA:
        nfa = self._expr()
        return nfa

    def _expr(self) -> NFA:
        """Alternation."""
        nfa = self._term()
        while self.peek() == "|":
            self.advance()  # consume |
            right = self._term()
            nfa = NFA.alternate(nfa, right)
        return nfa

    def _term(self) -> NFA:
        """Concatenation of one or more factors."""
        parts: List[NFA] = []
        while self.peek() is not None and self.peek() not in ("|", ")"):
            parts.append(self._factor())
        if not parts:
            return NFA.from_epsilon()
        nfa = parts[0]
        for p in parts[1:]:
            nfa = NFA.concat(nfa, p)
        return nfa

    def _factor(self) -> NFA:
        """atom followed by optional quantifier."""
        nfa = self._atom()
        q = self.peek()
        if q == "*":
            self.advance()
            nfa = NFA.star(nfa)
        elif q == "+":
            self.advance()
            nfa = NFA.plus(nfa)
        elif q == "?":
            self.advance()
            nfa = NFA.optional(nfa)
        return nfa

    def _atom(self) -> NFA:
        ch = self.peek()
        if ch == "(":
            self.advance()
            # Handle non-capturing groups (?:...)
            if self.peek() == "?" and self.pos + 1 < len(self.pattern) and self.pattern[self.pos + 1] == ":":
                self.advance()  # consume ?
                self.advance()  # consume :
            nfa = self._expr()
            if self.peek() == ")":
                self.advance()
            return nfa
        if ch == "[":
            self.advance()  # consume [
            chars, self.pos = _parse_charset(self.pattern, self.pos)
            return NFA.from_charset(chars)
        if ch == ".":
            self.advance()
            return NFA.from_dot()
        if ch == "\\":
            self.advance()
            esc = self.advance()
            return self._escape_to_nfa(esc)
        # literal character
        self.advance()
        return NFA.from_char(ch)

    @staticmethod
    def _escape_to_nfa(esc: str) -> NFA:
        if esc == "d":
            return NFA.from_charset(_DIGIT)
        if esc == "w":
            return NFA.from_charset(_WORD)
        if esc == "s":
            return NFA.from_charset(_SPACE)
        # Treat everything else as literal (including \\, \", etc.)
        return NFA.from_char(esc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def regex_to_dfa(pattern: str) -> DFA:
    """Compile a regex pattern string into a character-level DFA.

    Args:
        pattern: regex pattern (supports ``.``, ``*``, ``+``, ``?``, ``|``,
                 ``()``, ``[]``, ``\\d``, ``\\w``, ``\\s``).

    Returns:
        A :class:`DFA` that accepts exactly the language described by *pattern*.
    """
    parser = _RegexParser(pattern)
    nfa = parser.parse()
    return nfa.to_dfa()


class RegexGrammar:
    """Convenience wrapper: compile a regex and expose the resulting DFA.

    Example::

        rg = RegexGrammar("[a-z]+")
        assert rg.dfa.matches("hello")
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.dfa = regex_to_dfa(pattern)
