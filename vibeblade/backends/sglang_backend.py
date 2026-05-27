"""sglang Target Backend.

Extends OpenAIHttpTargetBackend with sglang-specific features:
- Uses /tokenize and /detokenize endpoints
- Auto-detects model info from /v1/models
- Sets correct Qwen3 EOS token and vocab size defaults
"""

from __future__ import annotations

from typing import Optional

from .openai_http_backend import OpenAIHttpTargetBackend


class SglangTargetBackend(OpenAIHttpTargetBackend):
    """Target backend for sglang server via OpenAI-compatible HTTP API.

    sglang-specific defaults:
    - EOS token: 151645 (Qwen3 series)
    - Vocab size: 248320 (Qwen3.6-27B)
    - Supports /tokenize and /detokenize endpoints
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "model",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            eos_token_id=151645,
            vocab_size=248320,
        )

    def name(self) -> str:
        return f"sglang({self._base_url}, model={self._model})"
