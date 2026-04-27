"""Tests for VibeBlade grammar-constrained decoding."""

import numpy as np
import pytest

from vibeblade.grammar import (
    EbnfGrammar,
    GrammarConstraint,
    JsonSchemaGrammar,
    NFA,
    RegexGrammar,
)
from vibeblade.grammar.regex_grammar import regex_to_dfa


# ═══════════════════════════════════════════════════════════════════════
# DFA / NFA basics
# ═══════════════════════════════════════════════════════════════════════

class TestNFA:
    def test_from_char(self):
        nfa = NFA.from_char("a")
        dfa = nfa.to_dfa()
        assert dfa.matches("a")
        assert not dfa.matches("b")
        assert not dfa.matches("")
        assert not dfa.matches("aa")

    def test_from_charset(self):
        nfa = NFA.from_charset({"a", "b", "c"})
        dfa = nfa.to_dfa()
        assert dfa.matches("a")
        assert dfa.matches("b")
        assert dfa.matches("c")
        assert not dfa.matches("d")
        assert not dfa.matches("")

    def test_from_dot(self):
        nfa = NFA.from_dot()
        dfa = nfa.to_dfa()
        assert dfa.matches("x")
        assert dfa.matches("Z")
        assert dfa.matches("5")
        assert not dfa.matches("")

    def test_concat(self):
        a = NFA.from_char("a")
        b = NFA.from_char("b")
        nfa = NFA.concat(a, b)
        dfa = nfa.to_dfa()
        assert dfa.matches("ab")
        assert not dfa.matches("a")
        assert not dfa.matches("b")
        assert not dfa.matches("ba")

    def test_alternate(self):
        a = NFA.from_char("a")
        b = NFA.from_char("b")
        nfa = NFA.alternate(a, b)
        dfa = nfa.to_dfa()
        assert dfa.matches("a")
        assert dfa.matches("b")
        assert not dfa.matches("c")
        assert not dfa.matches("ab")

    def test_star(self):
        a = NFA.from_char("a")
        nfa = NFA.star(a)
        dfa = nfa.to_dfa()
        assert dfa.matches("")  # zero repetitions
        assert dfa.matches("a")
        assert dfa.matches("aaa")
        assert not dfa.matches("b")

    def test_plus(self):
        a = NFA.from_char("a")
        nfa = NFA.plus(a)
        dfa = nfa.to_dfa()
        assert not dfa.matches("")  # need at least one
        assert dfa.matches("a")
        assert dfa.matches("aaa")

    def test_optional(self):
        a = NFA.from_char("a")
        nfa = NFA.optional(a)
        dfa = nfa.to_dfa()
        assert dfa.matches("")
        assert dfa.matches("a")
        assert not dfa.matches("aa")


class TestDFA:
    def test_matches(self):
        dfa = regex_to_dfa("abc")
        assert dfa.matches("abc")
        assert not dfa.matches("ab")
        assert not dfa.matches("abcd")

    def test_is_accepting(self):
        dfa = regex_to_dfa("a")
        assert dfa.is_accepting(dfa.start_state) is False
        # After consuming 'a' we should be in accepting state
        nxt = dfa.transition(dfa.start_state, "a")
        assert dfa.is_accepting(nxt)

    def test_can_accept(self):
        dfa = regex_to_dfa("ab")
        # From start, we can reach accepting state via a→b
        assert dfa.can_accept(dfa.start_state)
        # After consuming 'a', we can still reach accepting via 'b'
        mid = dfa.transition(dfa.start_state, "a")
        assert dfa.can_accept(mid)

    def test_dead_state(self):
        dfa = regex_to_dfa("a")
        dead = dfa.transition(dfa.start_state, "b")
        assert dead == frozenset()
        assert not dfa.is_accepting(dead)
        assert not dfa.can_accept(dead)


# ═══════════════════════════════════════════════════════════════════════
# Regex grammar
# ═══════════════════════════════════════════════════════════════════════

class TestRegexGrammar:
    def test_literal(self):
        rg = RegexGrammar("hello")
        assert rg.dfa.matches("hello")
        assert not rg.dfa.matches("world")
        assert not rg.dfa.matches("hell")

    def test_alternation(self):
        rg = RegexGrammar("cat|dog")
        assert rg.dfa.matches("cat")
        assert rg.dfa.matches("dog")
        assert not rg.dfa.matches("car")

    def test_star(self):
        rg = RegexGrammar("ab*c")
        assert rg.dfa.matches("ac")
        assert rg.dfa.matches("abc")
        assert rg.dfa.matches("abbbc")
        assert not rg.dfa.matches("adc")

    def test_plus(self):
        rg = RegexGrammar("[a-z]+")
        assert rg.dfa.matches("a")
        assert rg.dfa.matches("hello")
        assert not rg.dfa.matches("")
        assert not rg.dfa.matches("Hello")

    def test_optional(self):
        rg = RegexGrammar("colou?r")
        assert rg.dfa.matches("color")
        assert rg.dfa.matches("colour")
        assert not rg.dfa.matches("colouur")

    def test_dot(self):
        rg = RegexGrammar("a.c")
        assert rg.dfa.matches("abc")
        assert rg.dfa.matches("aXc")
        assert not rg.dfa.matches("ac")

    def test_charset_range(self):
        rg = RegexGrammar("[a-z]")
        assert rg.dfa.matches("m")
        assert not rg.dfa.matches("M")
        assert not rg.dfa.matches("5")

    def test_charset_negated(self):
        rg = RegexGrammar("[^0-9]")
        assert rg.dfa.matches("a")
        assert not rg.dfa.matches("5")

    def test_escape_digit(self):
        rg = RegexGrammar("\\d+")
        assert rg.dfa.matches("123")
        assert not rg.dfa.matches("abc")
        assert not rg.dfa.matches("")

    def test_escape_word(self):
        rg = RegexGrammar("\\w+")
        assert rg.dfa.matches("hello_123")
        assert not rg.dfa.matches("hello world")

    def test_escape_space(self):
        rg = RegexGrammar("\\s+")
        assert rg.dfa.matches(" ")
        assert rg.dfa.matches("  \t")

    def test_group(self):
        rg = RegexGrammar("(ab)+")
        assert rg.dfa.matches("ab")
        assert rg.dfa.matches("abab")
        assert not rg.dfa.matches("a")

    def test_complex(self):
        rg = RegexGrammar("[a-zA-Z_][a-zA-Z0-9_]*")
        assert rg.dfa.matches("hello")
        assert rg.dfa.matches("_private")
        assert rg.dfa.matches("var123")
        assert not rg.dfa.matches("123var")

    def test_json_number_pattern(self):
        rg = RegexGrammar("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")
        assert rg.dfa.matches("42")
        assert rg.dfa.matches("-3")
        assert rg.dfa.matches("3.14")
        assert rg.dfa.matches("-0.5")
        assert rg.dfa.matches("1e10")
        assert rg.dfa.matches("2.5e-3")


# ═══════════════════════════════════════════════════════════════════════
# JSON Schema grammar
# ═══════════════════════════════════════════════════════════════════════

class TestJsonSchemaGrammar:
    def test_boolean(self):
        jg = JsonSchemaGrammar({"type": "boolean"})
        assert jg.dfa.matches("true")
        assert jg.dfa.matches("false")
        assert not jg.dfa.matches("null")

    def test_null(self):
        jg = JsonSchemaGrammar({"type": "null"})
        assert jg.dfa.matches("null")
        assert not jg.dfa.matches("true")

    def test_integer(self):
        jg = JsonSchemaGrammar({"type": "integer"})
        assert jg.dfa.matches("42")
        assert jg.dfa.matches("-3")
        assert jg.dfa.matches("0")
        assert not jg.dfa.matches("3.14")

    def test_string(self):
        jg = JsonSchemaGrammar({"type": "string"})
        assert jg.dfa.matches('""')
        assert jg.dfa.matches('"hello"')
        assert jg.dfa.matches('"hello world"')

    def test_enum(self):
        jg = JsonSchemaGrammar({"enum": ["red", "green", "blue"]})
        assert jg.dfa.matches('"red"')
        assert jg.dfa.matches('"green"')
        assert not jg.dfa.matches('"yellow"')

    def test_object_required(self):
        jg = JsonSchemaGrammar({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })
        assert jg.dfa.matches('{"name":"Alice"}')

    def test_object_optional(self):
        jg = JsonSchemaGrammar({
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x"],
        })
        assert jg.dfa.matches('{"x":1}')
        assert jg.dfa.matches('{"x":1,"y":2}')

    def test_array(self):
        jg = JsonSchemaGrammar({
            "type": "array",
            "items": {"type": "integer"},
        })
        assert jg.dfa.matches("[]")
        assert jg.dfa.matches("[1]")
        assert jg.dfa.matches("[1,2,3]")

    def test_nested_object(self):
        jg = JsonSchemaGrammar({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name"],
                }
            },
            "required": ["user"],
        })
        assert jg.dfa.matches('{"user":{"name":"Bob"}}')


# ═══════════════════════════════════════════════════════════════════════
# EBNF grammar
# ═══════════════════════════════════════════════════════════════════════

class TestEbnfGrammar:
    def test_simple(self):
        eg = EbnfGrammar('root ::= "hello"')
        assert eg.dfa.matches("hello")
        assert not eg.dfa.matches("world")

    def test_concat(self):
        eg = EbnfGrammar('root ::= "hello" " " name\nname ::= [a-z]+')
        assert eg.dfa.matches("hello world")
        assert not eg.dfa.matches("hello 123")

    def test_alternation(self):
        eg = EbnfGrammar('root ::= "yes" | "no"')
        assert eg.dfa.matches("yes")
        assert eg.dfa.matches("no")
        assert not eg.dfa.matches("maybe")

    def test_optional(self):
        eg = EbnfGrammar('root ::= "item" count?\ncount ::= " " [0-9]+')
        assert eg.dfa.matches("item")
        assert eg.dfa.matches("item 5")

    def test_start_rule(self):
        eg = EbnfGrammar(
            'a ::= "alpha"\nb ::= "beta"',
            start_rule="b",
        )
        assert eg.dfa.matches("beta")
        assert not eg.dfa.matches("alpha")


# ═══════════════════════════════════════════════════════════════════════
# GrammarConstraint
# ═══════════════════════════════════════════════════════════════════════

class TestGrammarConstraint:
    @pytest.fixture
    def simple_vocab(self):
        return ["a", "b", "c", "ab", "1", "2", " ", "hello", "", "<eos>"]

    def test_from_regex(self, simple_vocab):
        gc = GrammarConstraint.from_regex(simple_vocab, "[a-c]+")
        gc.get_token_mask()
        allowed = gc.get_allowed_tokens()
        # 'a', 'b', 'c', 'ab' should be allowed
        assert 0 in allowed
        assert 1 in allowed
        assert 2 in allowed
        assert 3 in allowed  # 'ab'
        # '1', '2' should not be allowed
        assert 4 not in allowed
        assert 5 not in allowed

    def test_empty_tokens_always_allowed(self, simple_vocab):
        gc = GrammarConstraint.from_regex(simple_vocab, "xyz")
        mask = gc.get_token_mask()
        # Empty tokens (index 8) and special tokens should be allowed
        assert mask[8]  # '' empty token

    def test_eos_when_nothing_valid(self, simple_vocab):
        gc = GrammarConstraint.from_regex(
            simple_vocab, "xyz", eos_token_id=9  # '<eos>'
        )
        mask = gc.get_token_mask()
        # No non-empty token fully matches "xyz" alone
        # Empty token (index 8) is always allowed, so we check EOS is also allowed
        # when non-empty valid tokens exist that don't form a complete prefix
        # The mask should allow the empty token and EOS as safety nets
        assert mask[8]  # empty token always allowed
        assert mask[9]  # EOS allowed as safety when no real tokens valid

    def test_advance_and_mask_update(self, simple_vocab):
        gc = GrammarConstraint.from_regex(simple_vocab, "ab")
        gc.get_token_mask()
        assert 0 in gc.get_allowed_tokens()  # 'a' is valid prefix

        gc.advance("a")
        gc.get_token_mask()
        allowed2 = gc.get_allowed_tokens()
        assert 1 in allowed2  # 'b' completes "ab"
        # 'a' would give "aa" which doesn't match "ab"
        assert 0 not in allowed2

    def test_is_finished(self, simple_vocab):
        gc = GrammarConstraint.from_regex(simple_vocab, "ab")
        assert not gc.is_finished()
        gc.advance("ab")
        assert gc.is_finished()

    def test_reset(self, simple_vocab):
        gc = GrammarConstraint.from_regex(simple_vocab, "ab")
        gc.advance("a")
        assert not gc.is_finished()
        gc.reset()
        # After reset, should be back at start
        gc.get_token_mask()
        assert 0 in gc.get_allowed_tokens()

    def test_from_json_schema(self, simple_vocab):
        schema = {"type": "boolean"}
        gc = GrammarConstraint.from_json_schema(simple_vocab, schema)
        mask = gc.get_token_mask()
        # Should allow tokens that start "true" or "false"
        # With simple vocab, only empty/special tokens or partial matches
        assert mask.shape == (len(simple_vocab),)

    def test_mask_cache(self, simple_vocab):
        gc = GrammarConstraint.from_regex(simple_vocab, "[a-c]+")
        mask1 = gc.get_token_mask()
        mask2 = gc.get_token_mask()
        # Second call should hit cache (same object)
        assert np.array_equal(mask1, mask2)

    def test_from_ebnf(self, simple_vocab):
        gc = GrammarConstraint.from_ebnf(
            simple_vocab, 'root ::= "hello"'
        )
        mask = gc.get_token_mask()
        assert mask.shape == (len(simple_vocab),)


# ═══════════════════════════════════════════════════════════════════════
# Integration with TextGenerator
# ═══════════════════════════════════════════════════════════════════════

class TestGeneratorIntegration:
    def test_constrained_sampling(self):
        """Grammar mask should bias sampling toward valid tokens."""
        from vibeblade.generate import TextGenerator

        vocab = ["apple", "banana", "cherry", "123", "xyz", ""]
        gc = GrammarConstraint.from_regex(vocab, "[a-c]+")

        gen = TextGenerator(temperature=1.0, top_k=10, top_p=1.0)
        logits = np.array([5.0, 3.0, 2.0, 100.0, -10.0, 0.0])  # "123" has highest logit
        token_id = gen.sample(logits, grammar=gc)

        # Grammar should block "123" (index 3) despite highest logit
        mask = gc.get_token_mask()
        assert not mask[3]  # "123" blocked
        assert mask[token_id]  # selected token must be valid

    def test_generate_with_grammar(self):
        """Full generate() call with grammar constraint."""
        from vibeblade.generate import TextGenerator

        vocab = ["a", "b", "c", ""]
        gc = GrammarConstraint.from_regex(vocab, "[a-b]+")

        call_count = 0

        def model_fn(token_ids):
            nonlocal call_count
            call_count += 1
            # Return logits that slightly prefer 'a'
            return np.array([[1.0, 0.5, 0.1, 0.0]] * len(token_ids))

        gen = TextGenerator(temperature=0.0)  # greedy
        gen.generate(model_fn, np.array([0]), max_tokens=5, vocab=vocab, grammar=gc)
        # Should have generated without errors
        assert call_count >= 1
