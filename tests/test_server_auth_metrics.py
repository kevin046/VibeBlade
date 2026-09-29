
"""Tests for server-side auth + /metrics (added 2026-09-29)."""
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient

from vibeblade.openai_server import create_app


class _FakeStats:
    acceptance_rate = 0.5
    effective_speedup = 1.5
    draft_yield_rate = 0.4
    n_draft_accepted = 3
    n_draft_generated = 6

class _FakeResult:
    text = "Hello!"
    tokens = [1, 2, 3, 4, 5, 6]
    tokens_per_second = 25.0
    prompt_tokens = 2
    stop_reason = "eos"
    time_prefill = 0.01
    time_decode = 0.2
    time_total = 0.21

class _FakeEngine:
    stats = _FakeStats()
    def generate(self, prompt, **kwargs):
        return _FakeResult()
    def stream_generate(self, prompt, **kwargs):
        yield (1, "Hello")
        return _FakeResult()


APP_KEY = "test-key-12345"


def _client(auth=True):
    keys = {APP_KEY} if auth else None
    app = create_app(engine=_FakeEngine(), model_id="m", api_keys=keys)
    return TestClient(app)


class TestAuth:
    def test_missing_key_rejected(self):
        r = _client().get("/v1/models")
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "invalid_api_key"

    def test_wrong_key_rejected(self):
        r = _client().get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_valid_key_accepted(self):
        r = _client().get("/v1/models", headers={"Authorization": f"Bearer {APP_KEY}"})
        assert r.status_code == 200
        assert r.json()["object"] == "list"

    def test_health_is_public(self):
        assert _client().get("/health").status_code == 200

    def test_metrics_is_public(self):
        assert _client().get("/metrics").status_code == 200

    def test_no_auth_when_unconfigured(self):
        assert _client(auth=False).get("/v1/models").status_code == 200

    def test_request_id_header(self):
        r = _client().get("/v1/models", headers={"Authorization": f"Bearer {APP_KEY}"})
        assert "X-Request-Id" in r.headers

    def test_constant_time_compare_no_leak(self):
        # empty header must not 500
        r = _client().get("/v1/models", headers={"Authorization": ""})
        assert r.status_code == 401


class TestMetrics:
    def test_metrics_prometheus_format(self):
        c = _client()
        c.get("/v1/models", headers={"Authorization": f"Bearer {APP_KEY}"})
        body = c.get("/metrics").text
        assert "vibeblade_requests_total" in body
        assert "http_request_duration_seconds_count" in body
        assert "vibeblade_uptime_seconds" in body

    def test_metrics_can_be_disabled(self):
        app = create_app(engine=_FakeEngine(), api_keys=None, enable_metrics=False)
        assert TestClient(app).get("/metrics").status_code == 404

    def test_auth_failures_counted(self):
        c = _client()
        c.get("/v1/models")  # no key -> 401
        assert "vibeblade_auth_failures_total" in c.get("/metrics").text
