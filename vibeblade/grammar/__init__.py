"""VibeBlade Grammar — Constrained decoding via finite-state machines.

Guarantee valid JSON, regex matches, or EBNF grammar outputs during
autoregressive generation by masking invalid tokens at each step.
"""

from .constraint import GrammarConstraint
from .json_schema import JsonSchemaGrammar
from .regex_grammar import RegexGrammar
from .ebnf import EbnfGrammar
from .fsm import DFA, NFA

__all__ = [
    "GrammarConstraint",
    "JsonSchemaGrammar",
    "RegexGrammar",
    "EbnfGrammar",
    "DFA",
    "NFA",
]
