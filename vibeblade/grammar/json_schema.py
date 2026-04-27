"""JSON Schema → regex → DFA conversion.

Translates a JSON Schema dictionary into a regex pattern (as a string)
that matches any valid JSON conforming to the schema, then compiles it
to a DFA via :func:`~.regex_grammar.regex_to_dfa`.

Supports ``string``, ``number``, ``integer``, ``boolean``, ``null``,
``object`` (with ``properties``, ``required``), ``array`` (with ``items``),
``enum``, ``const``, and arbitrary nesting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional



# ---------------------------------------------------------------------------
# Regex fragments for JSON primitives
# ---------------------------------------------------------------------------

_JSON_STRING_FRAG = r'"([^"\\]|\\.)*"'
_JSON_INTEGER_FRAG = r'-?(0|[1-9][0-9]*)'
_JSON_NUMBER_FRAG = r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?'
_JSON_BOOL_FRAG = r'(true|false)'
_JSON_NULL_FRAG = r'null'
_WS = r'[ \t\n\r]*'


# ---------------------------------------------------------------------------
# Schema → regex conversion
# ---------------------------------------------------------------------------

def _escape_regex(s: str) -> str:
    """Escape regex meta-characters in a literal string."""
    special = r'\.^$*+?()[]{}|'
    out: List[str] = []
    for ch in s:
        if ch in special:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _json_value_to_regex(value: Any) -> str:
    """Convert a literal JSON value to a regex fragment."""
    if value is None:
        return _JSON_NULL_FRAG
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + _escape_regex(value) + '"'
    if isinstance(value, list):
        parts = [_json_value_to_regex(v) for v in value]
        return r'\[' + _WS + ("," + _WS).join(parts) + _WS + r'\]'
    if isinstance(value, dict):
        entries = []
        for k, v in value.items():
            entry = '"' + _escape_regex(k) + '":' + _WS + _json_value_to_regex(v)
            entries.append(entry)
        return r'\{' + _WS + ("," + _WS).join(entries) + _WS + r'\}'
    return _JSON_NULL_FRAG


def _schema_to_regex(schema: Dict[str, Any], defs: Optional[Dict] = None) -> str:
    """Recursively convert a JSON Schema to a regex pattern."""
    if defs is None:
        defs = schema.get("$defs", schema.get("definitions", {}))

    # Resolve $ref
    if "$ref" in schema:
        ref = schema["$ref"]
        parts = ref.lstrip("#/").split("/")
        resolved = defs
        for p in parts:
            if p in ("$defs", "definitions"):
                continue
            resolved = resolved[p]
        return _schema_to_regex(resolved, defs)

    # enum
    if "enum" in schema:
        alternatives = [_json_value_to_regex(v) for v in schema["enum"]]
        if not alternatives:
            return _JSON_NULL_FRAG
        return "(?:" + "|".join(alternatives) + ")"

    # const
    if "const" in schema:
        return _json_value_to_regex(schema["const"])

    # type
    type_ = schema.get("type", "string")
    if isinstance(type_, list):
        alternatives = [
            _schema_to_regex({**schema, "type": t}, defs) for t in type_
        ]
        return "(?:" + "|".join(alternatives) + ")"

    if type_ == "string":
        if "pattern" in schema:
            return r'"(' + schema["pattern"] + r')"'
        return _JSON_STRING_FRAG

    if type_ == "integer":
        return _JSON_INTEGER_FRAG

    if type_ == "number":
        return _JSON_NUMBER_FRAG

    if type_ == "boolean":
        return _JSON_BOOL_FRAG

    if type_ == "null":
        return _JSON_NULL_FRAG

    if type_ == "object":
        return _object_to_regex(schema, defs)

    if type_ == "array":
        return _array_to_regex(schema, defs)

    # Fallback: accept any JSON value
    return "(?:" + "|".join([
        _JSON_STRING_FRAG, _JSON_NUMBER_FRAG,
        _JSON_BOOL_FRAG, _JSON_NULL_FRAG,
    ]) + ")"


def _object_to_regex(schema: Dict[str, Any], defs: Optional[Dict]) -> str:
    """Convert an object schema to regex."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not props:
        return r'\{\}'

    # Sort: required first, then optional — deterministic output
    prop_names = sorted(props.keys(), key=lambda n: (0 if n in required else 1, n))

    required_entries: List[str] = []
    optional_entries: List[str] = []
    for name in prop_names:
        sub = _schema_to_regex(props[name], defs)
        entry = _WS + '"' + _escape_regex(name) + '":' + _WS + sub + _WS
        if name in required:
            required_entries.append(entry)
        else:
            optional_entries.append(entry)

    # Build: { required (,required)* (,optional)* }
    inner = ""
    if required_entries:
        inner += required_entries[0]
        for e in required_entries[1:]:
            inner += "," + e
    if optional_entries:
        for e in optional_entries:
            inner += "(," + e + ")?"

    return r'\{' + inner + r'\}'


def _array_to_regex(schema: Dict[str, Any], defs: Optional[Dict]) -> str:
    """Convert an array schema to regex."""
    items = schema.get("items", {})
    min_items = schema.get("minItems", 0)

    if isinstance(items, list):
        if not items:
            return r'\[\]'
        parts = [_schema_to_regex(it, defs) for it in items]
        inner = _WS + ("," + _WS).join(parts) + _WS
        return r'\[' + inner + r'\]'

    if isinstance(items, dict):
        sub = _schema_to_regex(items, defs)
        if min_items == 0:
            inner = _WS + "(?:" + sub + "(?:" + _WS + "," + _WS + sub + ")*)?"
        else:
            inner = _WS + sub + "(?:" + _WS + "," + _WS + sub + ")*"
        return r'\[' + inner + r'\]'

    return r'\[' + _WS + r'\]'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class JsonSchemaGrammar:
    """Compile a JSON Schema into a DFA for constrained decoding.

    Example::

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        jg = JsonSchemaGrammar(schema)
        dfa = jg.dfa
    """

    def __init__(self, schema: dict) -> None:
        from .regex_grammar import regex_to_dfa

        self.schema = schema
        self.pattern = _schema_to_regex(schema)
        self.dfa = regex_to_dfa(self.pattern)
