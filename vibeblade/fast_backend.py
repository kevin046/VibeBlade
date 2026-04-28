"""VibeBlade Fast Backend — C++ inference via VibeBladeFast.

Entire generate loop runs in C++:
  tokenize → prefill → [decode → sample] → detokenize

Python only handles CLI I/O and streaming print callbacks.
"""

from __future__ import annotations

from typing import Callable, Optional


def _load_native():
    """Lazily import the C++ native module."""
    import importlib

    for path in ["cpp/build", "."]:
        try:
            return importlib.import_module("_vibeblade_native")
        except ImportError:
            continue
    try:
        return importlib.import_module("vibeblade._vibeblade_native")
    except ImportError:
        raise ImportError(
            "VibeBladeFast C++ backend not found. "
            "Build it first: cd cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make"
        )


class FastModelWrapper:
    """Drop-in replacement for VibeBladeModel using the C++ fast backend.

    Loads GGUF files via mmap (instant load), runs entire generate loop in C++.
    Tokenization, sampling, and detokenization all happen in native code.
    """

    def __init__(self, model_path: str):
        self.path = str(model_path)
        self._native = _load_native()
        self._model = self._native.VibeBladeFast()
        self._model.load(model_path)

        cfg = self._model.config
        self.config = {
            "hidden_dim": cfg["hidden_dim"],
            "num_heads": cfg["n_heads"],
            "num_kv_heads": cfg["n_kv_heads"],
            "num_layers": cfg["n_layers"],
            "intermediate_dim": cfg["intermediate_dim"],
            "vocab_size": cfg["vocab_size"],
            "context_length": cfg["context_length"],
            "head_dim": cfg["head_dim"],
        }
        self.metadata = {"general.architecture": cfg["arch"]}
        self.is_moe = False
        self._moe_executor = None

    @property
    def position(self) -> int:
        return self._model.position

    def reset(self):
        """Reset KV cache and position."""
        self._model.reset()

    def generate(
        self,
        prompt: str = None,
        token_ids=None,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        stream: bool = True,
        on_token: Optional[Callable] = None,
    ) -> tuple:
        """Generate text from a prompt.

        The entire loop (tokenize → prefill → decode → sample → detokenize)
        runs in C++. Python only crosses the FFI boundary for the final result
        and optional streaming callbacks.

        Args:
            prompt: text prompt
            token_ids: pre-tokenized input (alternative to prompt)
            max_tokens: maximum new tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering
            top_p: nucleus (top-p) filtering
            stream: print tokens as they generate
            on_token: optional callback(token_id, text_piece)

        Returns:
            (generated_text, tokens_per_second) tuple
        """
        if prompt is None and token_ids is not None:
            # Pre-tokenized: decode through C++ detokenizer then use generate
            text = self._model.detokenize(token_ids)
            prompt = text

        if prompt is None:
            prompt = ""

        # Build streaming callback for C++ → Python
        cpp_callback = None
        if stream and on_token is None:
            # Default: print each piece as it arrives
            def cpp_callback(token_id, piece):
                print(piece, end="", flush=True)

        elif on_token is not None:
            def cpp_callback(token_id, piece):
                on_token(token_id, piece)

        # Single C++ call — everything runs native
        result = self._model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            on_token=cpp_callback,
        )

        if stream and on_token is None:
            print()

        return result.text, result.tokens_per_second
