"""Pure-Python BPE tokenizer built from GGUF metadata.

Supports GPT-2 style BPE (tokenizer.ggml.model == 'gpt2') with Qwen pre-tokenization
(tokenizer.ggml.pre == 'qwen35'). Requires the `regex` module for Unicode properties.
"""

from __future__ import annotations

import regex
from dataclasses import dataclass, field


def _bytes_to_unicode() -> dict[int, str]:
    """Build GPT-2 byte-to-unicode mapping table.

    Maps each byte 0-255 to a unique Unicode character. Printable ASCII (33-126 except
    some) maps to itself; all others are shifted to avoid control characters.
    """
    # Bytes that map to themselves (printable, no whitespace except space handling)
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# Singleton — shared across all instances
_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


@dataclass
class GGUFTokenizer:
    """BPE tokenizer constructed from GGUF metadata fields."""

    tokens: list[str]                     # id -> token text (in byte-unicode space)
    merges: list[tuple[str, str]]         # merge rules in priority order
    eos_token_id: int = 248046
    bos_token_id: int | None = None
    add_bos: bool = False

    # Internal state (set in __post_init__)
    _pre_pattern: regex.Pattern = field(init=False, repr=False)
    _merge_ranks: dict = field(init=False, repr=False)
    _token_to_id: dict = field(init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Qwen pre-tokenization regex (same as Qwen2/Qwen2.5)
        self._pre_pattern = regex.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
            regex.UNICODE,
        )

        # Build merge rank lookup
        self._merge_ranks: dict[tuple[str, str], int] = {}
        for i, (a, b) in enumerate(self.merges):
            self._merge_ranks[(a, b)] = i

        # Build token -> id lookup
        self._token_to_id: dict[str, int] = {}
        for i, tok in enumerate(self.tokens):
            self._token_to_id[tok] = i

    def _text_to_byte_unicode(self, text: str) -> str:
        """Convert UTF-8 text to GPT-2 byte-unicode representation."""
        return "".join(_BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))

    def _bpe(self, piece_bytes: str) -> list[int]:
        """Apply BPE to a pre-tokenized piece (already in byte-unicode space)."""

        if piece_bytes in self._cache:
            return self._cache[piece_bytes]

        # Start with individual byte-unicode characters
        ids = []
        for ch in piece_bytes:
            if ch in self._token_to_id:
                ids.append(self._token_to_id[ch])
            else:
                # Unknown byte — skip (shouldn't happen with valid UTF-8)
                pass

        if not ids:
            self._cache[piece_bytes] = []
            return []

        # Apply merges greedily (lowest rank first)
        while len(ids) >= 2:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(ids) - 1):
                pair = (self.tokens[ids[i]], self.tokens[ids[i + 1]])
                rank = self._merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = i

            if best_pair is None or best_rank == float("inf"):
                break

            idx = best_pair
            merged = self.tokens[ids[idx]] + self.tokens[ids[idx + 1]]
            if merged in self._token_to_id:
                merged_id = self._token_to_id[merged]
                ids = ids[:idx] + [merged_id] + ids[idx + 2:]
            else:
                break

        self._cache[piece_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ids: list[int] = []

        if self.add_bos and self.bos_token_id is not None:
            ids.append(self.bos_token_id)

        # Pre-tokenize into pieces, then convert each to byte-unicode space
        pieces = self._pre_pattern.findall(text)
        for piece in pieces:
            byte_unicode = self._text_to_byte_unicode(piece)
            ids.extend(self._bpe(byte_unicode))

        return ids

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        """Decode token IDs to text."""
        byte_values: list[int] = []
        for tid in ids:
            if not (0 <= tid < len(self.tokens)):
                continue
            tok = self.tokens[tid]

            # Handle special tokens
            if tok.startswith("<|") and tok.endswith("|>"):
                if stop_at_eos and tid == self.eos_token_id:
                    break
                continue

            # Convert byte-unicode token text back to raw bytes
            for ch in tok:
                if ch in _UNICODE_TO_BYTE:
                    byte_values.append(_UNICODE_TO_BYTE[ch])
                else:
                    # Fallback: encode the char as-is UTF-8
                    byte_values.extend(ch.encode("utf-8"))

        return bytes(byte_values).decode("utf-8", errors="replace")

    @classmethod
    def from_gguf_metadata(cls, metadata: dict) -> "GGUFTokenizer":
        """Construct tokenizer from GGUF loader metadata dict."""
        tokens = metadata.get("tokenizer.ggml.tokens", [])
        merges_raw = metadata.get("tokenizer.ggml.merges", [])
        eos_id = metadata.get("tokenizer.ggml.eos_token_id", 248046)
        bos_id = metadata.get("tokenizer.ggml.bos_token_id", None)
        add_bos = metadata.get("tokenizer.ggml.add_bos_token", False)

        # Parse merge strings like "Ġ Ġ" -> ("Ġ", "Ġ")
        merges = []
        for m in merges_raw:
            parts = m.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

        return cls(
            tokens=tokens,
            merges=merges,
            eos_token_id=eos_id,
            bos_token_id=bos_id,
            add_bos=add_bos,
        )
