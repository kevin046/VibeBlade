"""VibeBlade Fast Backend — C++ inference via VibeBladeFast.

Zero-copy GGUF mmap, inline dequantization, no Python in the decode loop.
Use this for maximum inference speed on quantized models.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Callable, Optional


def _load_native():
    """Lazily import the C++ native module."""
    import importlib
    # Try project build first, then installed package
    for path in ["cpp/build", "."]:
        try:
            mod = importlib.import_module("_vibeblade_native")
            return mod
        except ImportError:
            continue
    # Try from installed package
    try:
        return importlib.import_module("vibeblade._vibeblade_native")
    except ImportError:
        raise ImportError(
            "VibeBladeFast C++ backend not found. "
            "Build it first: cd cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make"
        )


class FastModelWrapper:
    """Drop-in replacement for VibeBladeModel using the C++ fast backend.

    Loads GGUF files via mmap (instant load), runs inference entirely in C++.
    Supports Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32 quantization.
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

        Args:
            prompt: text prompt
            token_ids: pre-tokenized input (alternative to prompt)
            max_tokens: maximum new tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering
            top_p: nucleus (top-p) filtering
            stream: print tokens as they generate
            on_token: optional callback(token_id, pos)

        Returns:
            (generated_text, tokens_per_second) tuple
        """
        import numpy as np

        if token_ids is None:
            if prompt is None:
                token_ids = [1]  # BOS
            else:
                # Simple byte tokenization (placeholder — real tokenizer later)
                token_ids = list(prompt.encode("utf-8")) + [1]

        # Prefill
        logits = self._model.prefill(token_ids)

        # Generate
        output_tokens = []
        gen_start = time.time()

        for i in range(max_tokens):
            token_id = self._sample(logits, temperature, top_k, top_p)
            output_tokens.append(token_id)

            if on_token is not None:
                on_token(token_id, i)
            elif stream:
                try:
                    print(chr(token_id), end="", flush=True)
                except (ValueError, OverflowError):
                    pass

            if token_id == 2:  # EOS
                break

            # Decode next token
            logits = self._model.decode(token_id)

        gen_elapsed = time.time() - gen_start
        tps = len(output_tokens) / max(gen_elapsed, 1e-6)

        if stream and on_token is None:
            print()

        # Decode output tokens to text
        try:
            text = "".join(chr(t) for t in output_tokens if 32 <= t < 127)
        except (ValueError, OverflowError):
            text = f"[{len(output_tokens)} tokens generated at {tps:.1f} t/s]"

        return text, tps

    @staticmethod
    def _sample(logits, temperature, top_k, top_p):
        """Sample a token from logits with temperature, top_k, and top_p."""
        import numpy as np

        logits = np.array(logits, dtype=np.float64)

        # Temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Greedy
        if temperature == 0:
            return int(np.argmax(logits))

        # Top-K
        if top_k > 0 and top_k < len(logits):
            top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -1e10)
            mask[top_k_idx] = logits[top_k_idx]
            logits = mask

        # Softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # Top-P (nucleus) sampling
        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, top_p) + 1
            top_idx = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[top_idx] = probs[top_idx]
            probs = mask / np.sum(mask)

        return int(np.random.choice(len(probs), p=probs))
