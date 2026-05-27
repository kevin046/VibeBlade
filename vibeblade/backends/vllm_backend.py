"""vLLM Target Backend.

Extends OpenAIHttpTargetBackend with vLLM-specific defaults.
"""

from __future__ import annotations

from typing import Optional

from .openai_http_backend import OpenAIHttpTargetBackend


class VllmTargetBackend(OpenAIHttpTargetBackend):
    """Target backend for vLLM server via OpenAI-compatible HTTP API.

    vLLM-specific defaults:
    - EOS token: varies by model (auto-detected)
    - Vocab size: varies by model
    - Supports /tokenize and /detokenize endpoints
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "model",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        eos_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
        )

    def name(self) -> str:
        return f"vllm({self._base_url}, model={self._model})"
