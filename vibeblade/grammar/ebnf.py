"""EBNF (Extended Backus-Naur Form) parser and compiler.

Parses a GBNF-style grammar string (as used by llama.cpp) into an NFA/DFA
for character-level validation.

Supports: ``rule ::= production``, alternation ``|``, grouping ``()``,
repetition ``*`` ``+`` ``?``, optional ``[]``, literal strings ``"..."``,
character ranges ``[a-z]``, and escape sequences.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# EBNF Parser
# ---------------------------------------------------------------------------

class _EbnfParser:
    """Parse GBNF-style grammar into a dict of rule-name to regex string."""

    def __init__(self, grammar_str: str) -> None:
        self.grammar_str = grammar_str
        self.rules: Dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse all rules from the grammar string."""
        lines = self.grammar_str.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Remove trailing comments
            if "#" in line:
                line = line[: line.index("#")].strip()
            if "//" in line:
                line = line[: line.index("//")].strip()
            if not line:
                continue
            # Parse rule: name ::= production
            m = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*::=\s*(.*)', line)
            if m:
                name = m.group(1)
                body = m.group(2).strip()
                self.rules[name] = body
                # Handle multi-line rules (continued with | at start)
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith("|"):
                        self.rules[name] += " " + next_line
                        i += 1
                    else:
                        break

    def to_regex(self, start_rule: Optional[str] = None) -> str:
        """Convert the grammar to a regex string by expanding non-terminals.

        Args:
            start_rule: the rule to use as entry point (first rule if None).

        Returns:
            A regex pattern string.
        """
        if start_rule is None:
            if not self.rules:
                return ""
            start_rule = next(iter(self.rules))

        expanded = self._expand_rule(start_rule, set())
        return expanded

    def _expand_rule(self, name: str, visited: Set[str]) -> str:
        if name in visited:
            return ""  # prevent infinite recursion
        visited = visited | {name}  # use new set to avoid cross-branch pollution
        body = self.rules.get(name, "")
        if not body:
            return ""
        return self._expand_production(body, visited)

    def _expand_production(self, body: str, visited: Set[str]) -> str:
        """Expand a production body into a regex string."""
        tokens = self._tokenize(body)
        regex_parts = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            # Check for quantifier
            quant = ""
            if i + 1 < len(tokens) and tokens[i + 1] in ("*", "+", "?"):
                quant = tokens[i + 1]
                i += 2
            else:
                i += 1

            part = self._token_to_regex(tok, visited)
            if quant:
                if quant == "*":
                    part = "(?:" + part + ")*"
                elif quant == "+":
                    part = "(?:" + part + ")+"
                elif quant == "?":
                    part = "(?:" + part + ")?"
            regex_parts.append(part)

        return "".join(regex_parts)

    def _tokenize(self, body: str) -> List[str]:
        """Tokenize a production body into terminals, refs, quantifiers."""
        tokens: List[str] = []
        i = 0
        while i < len(body):
            ch = body[i]

            if ch in (" ", "\t", "\n", "\r"):
                i += 1
                continue

            # Alternation
            if ch == "|":
                tokens.append("|")
                i += 1
                continue

            # Grouping ()
            if ch == "(":
                depth = 1
                j = i + 1
                while j < len(body) and depth > 0:
                    if body[j] == "(":
                        depth += 1
                    elif body[j] == ")":
                        depth -= 1
                    j += 1
                tokens.append(body[i:j])
                i = j
                continue

            # Optional [] or character class
            if ch == "[":
                depth = 1
                j = i + 1
                while j < len(body) and depth > 0:
                    if body[j] == "[":
                        depth += 1
                    elif body[j] == "]":
                        depth -= 1
                    j += 1
                tokens.append(body[i:j])
                i = j
                continue

            # Literal string "..."
            if ch == '"':
                j = i + 1
                while j < len(body) and body[j] != '"':
                    if body[j] == "\\":
                        j += 1  # skip escaped char
                    j += 1
                tokens.append(body[i : j + 1])
                i = j + 1
                continue

            # Quantifiers
            if ch in ("*", "+", "?"):
                tokens.append(ch)
                i += 1
                continue

            # Rule reference (identifier)
            if ch.isalpha() or ch == "_":
                j = i
                while j < len(body) and (body[j].isalnum() or body[j] == "_"):
                    j += 1
                tokens.append(body[i:j])
                i = j
                continue

            # Escape
            if ch == "\\" and i + 1 < len(body):
                tokens.append(body[i : i + 2])
                i += 2
                continue

            # Other characters
            tokens.append(ch)
            i += 1

        return tokens

    def _token_to_regex(self, tok: str, visited: Set[str]) -> str:
        """Convert a single token to regex."""
        if tok == "|":
            return "|"
        if tok in ("*", "+", "?"):
            return tok
        if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
            # Literal string
            inner = tok[1:-1]
            return _escape_for_regex(inner)
        if tok.startswith("[") and tok.endswith("]") and len(tok) >= 2:
            inner = tok[1:-1]
            if self._is_ebnf_optional(inner):
                # EBNF optional: [a b c] -> (?:a b c)?
                expanded = self._expand_production(inner, visited)
                return "(?:" + expanded + ")?"
            else:
                # Character class — pass through to regex engine
                return tok
        if tok.startswith("(") and tok.endswith(")") and len(tok) >= 2:
            inner = tok[1:-1]
            expanded = self._expand_production(inner, visited)
            return "(?:" + expanded + ")"
        if tok.startswith("\\"):
            return tok
        # Rule reference
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tok):
            if tok in self.rules:
                return "(?:" + self._expand_rule(tok, visited) + ")"
            return tok
        # Bare character or other
        return _escape_for_regex(tok)

    def _is_ebnf_optional(self, inner: str) -> bool:
        """Heuristic: distinguish [a-z] charset from [a "b" c] optional group."""
        if ' "' in inner or '" ' in inner:
            return True
        if re.search(r'\s[a-zA-Z_]', inner):
            return True
        return False


def _escape_for_regex(s: str) -> str:
    """Escape a literal string for use in a regex pattern."""
    special = set(r'\.^$*+?()[]{}|')
    out: List[str] = []
    for ch in s:
        if ch in special:
            out.append("\\")
        out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EbnfGrammar:
    """Parse an EBNF/GBNF grammar string into a DFA.

    Example::

        grammar = '''root ::= "hello" " " name
        name ::= [a-z]+'''
        eg = EbnfGrammar(grammar)
        dfa = eg.dfa
    """

    def __init__(self, grammar_str: str, start_rule: Optional[str] = None) -> None:
        self.grammar_str = grammar_str
        parser = _EbnfParser(grammar_str)
        self.rules = parser.rules
        self.pattern = parser.to_regex(start_rule)
        from .regex_grammar import regex_to_dfa

        self.dfa = regex_to_dfa(self.pattern)
