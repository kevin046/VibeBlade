"""Tests for the OpenAI-compatible server.

Uses FastAPI's TestClient (from httpx/starlette) with a mock model
so no real GGUF file is needed. CI runs without fastapi installed,
so these tests are skipped gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if fastapi not installed
pytest.importorskip("fastapi", reason="fastapi not installed")

from vibeblade.openai_server import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    ModelRegistry,
    _apply_stop,
    _estimate_tokens,
    _finish_reason,
    _format_chat_prompt,
    create_app,
)


# ── Mock Model ──


class MockModel:
    """Fake VibeBladeModel for testing endpoints."""

    def __init__(self, response_text: str = "Hello!"):
        self.response_text = response_text
        self.generate_call_count = 0

    def generate(self, token_ids, max_tokens=128, temperature=0.7,
                 top_p=0.9, on_token=None):
        self.generate_call_count += 1
        # Return a few dummy tokens
        n_prompt = len(token_ids)
        gen_tokens = np.array([72, 101, 108, 108, 111, 33], dtype=np.int32)  # "Hello!"
        result = np.concatenate([token_ids[:1], gen_tokens])  # minimal valid output
        stats = {
            "n_prompt": n_prompt,
            "n_generated": len(gen_tokens),
            "prefill_ms": 10.0,
            "decode_ms_per_token": 5.0,
            "total_ms": 40.0,
            "tokens_per_sec": 25.0,
        }
        return result, 25.0, stats

    def enable_sparse(self):
        pass

    def enable_minicache(self):
        pass

    def enable_paged_attn(self):
        pass


# ── Fixtures ──


@pytest.fixture
def mock_registry():
    reg = ModelRegistry()
    reg._models["test-model"] = MockModel()
    reg._models["another-model"] = MockModel("World!")
    return reg


@pytest.fixture
def client(mock_registry):
    from fastapi.testclient import TestClient

    app = create_app(model_id="test-model", registry=mock_registry)
    return TestClient(app)


# ── Unit tests ──


class TestFormatChatPrompt:
    def test_single_user_message(self):
        msgs = [ChatMessage(role="user", content="Hi")]
        result = _format_chat_prompt(msgs)
        assert "<|im_start|>user" in result
        assert "Hi<|im_end|>" in result
        assert "<|im_start|>assistant" in result

    def test_system_and_user(self):
        msgs = [
            ChatMessage(role="system", content="Be helpful."),
            ChatMessage(role="user", content="What is AI?"),
        ]
        result = _format_chat_prompt(msgs)
        assert "system" in result
        assert "Be helpful." in result
        assert "user" in result
        assert "What is AI?" in result
        assert result.index("system") < result.index("user")

    def test_multi_turn(self):
        msgs = [
            ChatMessage(role="user", content="Q1"),
            ChatMessage(role="assistant", content="A1"),
            ChatMessage(role="user", content="Q2"),
        ]
        result = _format_chat_prompt(msgs)
        assert result.count("<|im_start|>") == 4  # 3 messages + assistant prompt
        assert result.count("<|im_end|>") == 3


class TestApplyStop:
    def test_no_stop(self):
        assert _apply_stop("hello world", []) == "hello world"

    def test_single_stop(self):
        assert _apply_stop("hello\nworld", ["\n"]) == "hello"

    def test_multiple_stops(self):
        assert _apply_stop("hello END world", ["END", "STOP"]) == "hello "
        assert _apply_stop("hello STOP world", ["END", "STOP"]) == "hello "

    def test_stop_not_found(self):
        assert _apply_stop("hello world", ["xyz"]) == "hello world"


class TestFinishReason:
    def test_long_output(self):
        assert _finish_reason("x" * 200, max_tokens=200) == "length"

    def test_short_output(self):
        assert _finish_reason("hi", max_tokens=100) == "stop"


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_english(self):
        assert _estimate_tokens("Hello world") == max(1, 11 // 4)


class TestModelRegistry:
    def test_get_model(self, mock_registry):
        model = mock_registry.get("test-model")
        assert isinstance(model, MockModel)

    def test_get_missing(self, mock_registry):
        with pytest.raises(KeyError, match="not loaded"):
            mock_registry.get("nonexistent")

    def test_list_models(self, mock_registry):
        models = mock_registry.list_models()
        assert len(models) == 2
        assert models[0]["id"] in ("test-model", "another-model")
        assert models[0]["object"] == "model"

    def test_remove_model(self, mock_registry):
        assert mock_registry.remove("test-model") is True
        assert len(mock_registry.list_models()) == 1
        assert mock_registry.remove("test-model") is False


# ── Integration tests (HTTP) ──


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"


class TestModelsEndpoint:
    def test_list_models(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2

    def test_api_info(self, client):
        r = client.get("/v1/me")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "VibeBlade"
        assert len(data["models"]) == 2


class TestChatCompletionsEndpoint:
    def test_basic_chat(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "temperature": 0.5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]
        assert data["choices"][0]["finish_reason"] in ("stop", "length")
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0

    def test_default_model(self, client):
        """When model is empty, should use the default model_id."""
        r = client.post("/v1/chat/completions", json={
            "model": "",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        })
        assert r.status_code == 200

    def test_missing_model(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        })
        assert r.status_code == 404

    def test_empty_messages(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [],
        })
        assert r.status_code == 400

    def test_system_and_user_messages(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Explain AI"},
            ],
            "max_tokens": 10,
        })
        assert r.status_code == 200
        assert r.json()["object"] == "chat.completion"

    def test_streaming_chat(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        })
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        # Should contain SSE data
        body = r.text
        assert "data: " in body
        assert "[DONE]" in body

    def test_response_has_id(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        })
        data = r.json()
        assert data["id"].startswith("chatcmpl-")
        assert "created" in data


class TestCompletionsEndpoint:
    def test_basic_completion(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert data["usage"]["prompt_tokens"] > 0

    def test_echo_completion(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 5,
            "echo": True,
        })
        assert r.status_code == 200
        data = r.json()
        # With echo, text should start with the prompt
        assert data["choices"][0]["text"].startswith("Hello")

    def test_streaming_completion(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": True,
        })
        assert r.status_code == 200
        body = r.text
        assert "data: " in body
        assert "[DONE]" in body

    def test_empty_prompt(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "",
        })
        assert r.status_code == 400


class TestRequestModels:
    def test_chat_request_defaults(self):
        req = ChatCompletionRequest()
        assert req.model == ""
        assert req.messages == []
        assert req.max_tokens == 128
        assert req.temperature == 0.7
        assert req.stream is False

    def test_completion_request_defaults(self):
        req = CompletionRequest()
        assert req.prompt == ""
        assert req.echo is False
        assert req.stop == []
