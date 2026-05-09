"""Neural draft head using a separate GGUF draft model.

This replaces NgramDraftHead with a learned draft model that predicts
the target model's next token distribution, significantly improving
acceptance rates and overall speedup from speculative decoding.

Usage:
    spec = SpeculativeBackend()
    spec.load("target.gguf")
    spec.set_draft_model("draft.gguf")  # enable neural drafting
    result = spec.generate(prompt, speculative=True)
"""

from __future__ import annotations

from .llama_backend import LlamaCppBackend


class NeuralDraftHead:
    """Neural draft model — a small separate GGUF model loaded via LlamaCppBackend.

    The draft model runs on the same CPU backend but with smaller weight
    matrix, so each forward pass is faster than the target model. It
    predicts what the target *would* output next, given the history.

    Parameters
    ----------
    draft_model_path : str
        Path to a smaller/faster GGUF model file (e.g., 100M–300M params).
    n_threads : int
        Number of threads for draft model inference.
    """

    def __init__(self, draft_model_path: str, n_threads: int = 4) -> None:
        self._draft_backend = LlamaCppBackend()
        self._draft_backend.load(draft_model_path, n_ctx=512, n_threads=n_threads)
        self._draft_backend._set_sampler(temperature=0.0)  # greedy draft
        # Vocab size not needed — generate_from_tokens handles it

    def draft(self, history: list[int], draft_max_override: int = 0) -> list[int]:
        """Generate up to `draft_max_override` tokens using the draft model.

        Runs the draft model autoregressively on the given token history,
        returning the next-token predictions as a draft list.

        Parameters
        ----------
        history : list[int]
            Token history (full sequence so far).
        draft_max_override : int
            Override maximum draft length (default uses instance max).

        Returns
        -------
        list[int]
            Draft token IDs (may be empty if draft model fails).
        """
        max_d = draft_max_override if draft_max_override > 0 else 8
        if len(history) < 1:
            return []

        # Use last N tokens as context (respect draft model's context)
        context = history[-256:]  # within draft model's n_ctx=512
        try:
            result = self._draft_backend.generate_from_tokens(
                context, max_tokens=max_d, temperature=0.0
            )
            return result.tokens[:max_d]
        except Exception:
            return []

    def free(self) -> None:
        if self._draft_backend is not None:
            self._draft_backend.free()
            self._draft_backend = None


# Alias for compatibility — SpeculativeBackend can swap draft head class
EAGLEDraftHead = NeuralDraftHead
