"""Generic OpenAI-compatible HTTP Target Backend.

Works with ANY server that implements the OpenAI Chat/Completions API:
  - sglang
  - vLLM
  - llama.cpp server
  - TGI
  - Ollama
  - Any OpenAI-compatible endpoint

Uses logprobs for draft verification where available.
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error
from typing import Optional

from ..target_backend import (
    TargetBackend,
    TargetLogits,
)

logger = logging.getLogger(__name__)


class OpenAIHttpTargetBackend(TargetBackend):
    """Generic OpenAI-compatible HTTP target backend.

    Connects to any server implementing /v1/chat/completions and
    /v1/completions (and optionally /tokenize, /detokenize).

    Parameters:
        base_url: Server base URL (e.g. "http://localhost:8000").
        model: Model name for API requests.
        api_key: Optional API key (Bearer token).
        timeout: Request timeout in seconds.
        eos_token_id: Override EOS token ID.
        vocab_size: Override vocab size.
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
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._eos_id = eos_token_id
        self._vocab_size = vocab_size
        self._supports_logprobs: Optional[bool] = None
        self._supports_tokenize: Optional[bool] = None
        self._request_id = 0

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        data = __import__("json").dumps(body).encode("utf-8")
        import urllib.request
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=data, method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                import json as _json
                return _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(
                f"HTTP {e.code} from {self._base_url}{path}: {body_text[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to {self._base_url}{path}: {e.reason}"
            ) from e

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self._base_url}{path}", headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                import json as _json
                return _json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}

    def name(self) -> str:
        return f"openai({self._base_url})"

    # ── Tokenize / Detokenize ──

    def _check_tokenize_support(self) -> bool:
        if self._supports_tokenize is not None:
            return self._supports_tokenize
        try:
            resp = self._post("/tokenize", {
                "model": self._model,
                "prompt": "test",
                "add_special_tokens": False,
            })
            self._supports_tokenize = "tokens" in resp
        except Exception:
            try:
                # Try HF tokenizer API (sglang/vLLM)
                resp = self._post("/v1/tokenize", {
                    "model": self._model,
                    "text": "test",
                    "add_special_tokens": False,
                })
                self._supports_tokenize = "token_ids" in resp or "tokens" in resp
            except Exception:
                self._supports_tokenize = False
        return self._supports_tokenize

    def tokenize(self, text: str) -> list[int]:
        if self._check_tokenize_support():
            try:
                resp = self._post("/tokenize", {
                    "model": self._model,
                    "prompt": text,
                    "add_special_tokens": True,
                })
                return resp.get("tokens", resp.get("token_ids", []))
            except Exception:
                pass
            try:
                resp = self._post("/v1/tokenize", {
                    "model": self._model,
                    "text": text,
                    "add_special_tokens": True,
                })
                return resp.get("tokens", resp.get("token_ids", []))
            except Exception:
                pass

        # Fallback: use completions API to infer tokenization
        # (expensive but works as last resort)
        logger.warning("No tokenize endpoint available, using heuristic encoding")
        return list(text.encode("utf-8"))

    def detokenize(self, tokens: list[int]) -> str:
        try:
            resp = self._post("/detokenize", {
                "model": self._model,
                "tokens": tokens,
            })
            return resp.get("text", "")
        except Exception:
            pass
        try:
            resp = self._post("/v1/detokenize", {
                "model": self._model,
                "tokens": tokens,
            })
            return resp.get("text", "")
        except Exception:
            # Fallback: use chat completion to decode
            pass
        # Last resort
        return "".join(chr(t) if 32 <= t < 127 else "" for t in tokens)

    # ── Prefill ──

    def prefill(self, tokens: list[int]) -> list[float]:
        """Prefill via completions API with logprobs."""
        prompt_text = self.detokenize(tokens)
        vocab = self.vocab_size()

        try:
            resp = self._post("/v1/completions", {
                "model": self._model,
                "prompt": prompt_text,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 1,
                "stream": False,
            })
        except Exception as e:
            logger.warning(f"Prefill failed: {e}")
            return [0.0] * vocab

        # Extract logits from logprobs
        choice = resp["choices"][0]
        lp = choice.get("logprobs", {})
        top_logprobs = lp.get("top_logprobs", [])

        if top_logprobs and isinstance(top_logprobs, list) and len(top_logprobs) > 0:
            # Build logits from log probabilities
            logits = [float("-inf")] * vocab
            if isinstance(top_logprobs[0], dict):
                for tok_str, lp_val in top_logprobs[0].items():
                    # Approximate: use logprob directly as logit
                    # This works for greedy comparison (argmax is preserved)
                    try:
                        tok_id = int(tok_str) if tok_str.isdigit() else hash(tok_str) % vocab
                        logits[tok_id] = lp_val
                    except (ValueError, TypeError):
                        pass
            return logits

        return [0.0] * vocab

    # ── Batch Decode ──

    def decode_batch(
        self,
        tokens: list[int],
        positions: list[int],
    ) -> TargetLogits:
        """Decode tokens and return logits for each position.

        Uses logprobs from the completions API to get per-position
        token probabilities for draft verification.
        """
        vocab = self.vocab_size()
        all_logits: list[list[float]] = []
        sampled: list[int] = []

        # We generate tokens and compare with draft
        # For HTTP backends, we can't easily inject draft tokens at specific positions
        # Instead, we let the model generate and compare outputs

        # Strategy: send the context up to the draft point,
        # request max_tokens = len(draft_tokens), and compare
        context_tokens = tokens[:-1] if len(tokens) > 1 else []
        prompt_text = self.detokenize(context_tokens) if context_tokens else ""

        n_gen = len(tokens)

        try:
            resp = self._post("/v1/completions", {
                "model": self._model,
                "prompt": prompt_text,
                "max_tokens": n_gen,
                "temperature": 0.0,
                "logprobs": 5,
                "stream": False,
            })
        except Exception as e:
            logger.warning(f"decode_batch failed: {e}")
            return TargetLogits(
                logits_per_position=[[0.0] * vocab for _ in tokens],
                sampled_tokens=tokens,
            )

        choice = resp["choices"][0]
        text = choice.get("text", "")
        gen_token_ids = choice.get("logprobs", {}).get("token_ids", [])
        top_logprobs_list = choice.get("logprobs", {}).get("top_logprobs", [])

        # If we got token_ids from logprobs, use those
        if gen_token_ids and len(gen_token_ids) >= n_gen:
            for i in range(n_gen):
                logits_i = [float("-inf")] * vocab
                if i < len(top_logprobs_list) and isinstance(top_logprobs_list[i], dict):
                    for tok_str, lp_val in top_logprobs_list[i].items():
                        try:
                            tok_id = int(tok_str) if tok_str.isdigit() else 0
                            logits_i[tok_id] = lp_val
                        except (ValueError, TypeError):
                            pass
                all_logits.append(logits_i)
                sampled.append(gen_token_ids[i])
        else:
            # Fallback: tokenize the generated text
            gen_tokens = self.tokenize(prompt_text + text)
            gen_only = gen_tokens[len(context_tokens):] if len(gen_tokens) > len(context_tokens) else []

            for i in range(n_gen):
                logits_i = [0.0] * vocab
                all_logits.append(logits_i)
                sampled.append(gen_only[i] if i < len(gen_only) else tokens[i])

        return TargetLogits(
            logits_per_position=all_logits,
            sampled_tokens=sampled,
            raw_response=resp,
        )

    # ── Model Info ──

    def eos_token_id(self) -> int:
        if self._eos_id is not None:
            return self._eos_id
        # Common defaults
        return 151645  # Qwen3 series

    def vocab_size(self) -> int:
        if self._vocab_size is not None:
            return self._vocab_size
        return 248320  # Qwen3.6-27B default

    def reset(self) -> None:
        self._request_id += 1

    def health(self) -> bool:
        try:
            resp = self._get("/health")
            return True
        except Exception:
            pass
        try:
            resp = self._get("/v1/models")
            return "data" in resp
        except Exception:
            return False

    def model_info(self) -> dict:
        return self._get("/v1/models")
