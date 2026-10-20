"""Tests for VibeBlade Fusion — the universal routing front."""

import asyncio
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/home/ubuntu/vibeblade")

from vibeblade.fusion import (
    EngineSpec, FusionRouter, FusionEngine, SpeedTier,
)

ok = 0


def fake_engine(models, latency=0.05, fail_first=0, supports_tools=True):
    """Spin up a local HTTP server impersonating an OpenAI engine."""
    state = {"calls": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/v1/models":
                body = json.dumps({"data": [{"id": m} for m in models]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            state["calls"] += 1
            if state["calls"] <= fail_first:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"{}")
                return
            import time
            time.sleep(latency)
            resp = {
                "id": "x", "object": "chat.completion", "created": 0,
                "model": req.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
            }
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, state


# 1. selection prefers fastest tier
r = FusionRouter()
r.register(EngineSpec(name="cpu-gguf", kind="gguf", models=["m1"], speed_tier=SpeedTier.LOCAL))
r.register(EngineSpec(name="gpu-sglang", kind="sglang", models=["m1"],
                      speed_tier=SpeedTier.TENSOR_CORE_LOCAL))
sel = r.select("m1")
assert sel.name == "gpu-sglang", sel.name
print("1. tier-based selection (tensor-core first): OK")
ok += 1

# 2. capability gates: tools / vision / context
r2 = FusionRouter()
r2.register(EngineSpec(name="no-tools", kind="gguf", models=["m1"],
                       supports_tools=False))
sel = r2.select("m1", needs_tools=True)
assert sel is None or sel.supports_tools
print("2. tools capability gate: OK")
ok += 1

r3 = FusionRouter()
r3.register(EngineSpec(name="small-ctx", kind="gguf", models=["m1"], max_context=8192))
assert r3.select("m1", context=100000) is None
print("3. context gate: OK")
ok += 1

# 4. live HTTP routing + fusion metadata
srv, st = fake_engine(["m1"], latency=0.02)
port = srv.server_address[1]
r4 = FusionRouter()
r4.register(EngineSpec(name="live", kind="openai",
                       base_url=f"http://127.0.0.1:{port}", models=["m1"]))
eng = FusionEngine(r4)
out = asyncio.run(eng.chat({"model": "m1",
                            "messages": [{"role": "user", "content": "hi"}]}))
assert out["choices"][0]["message"]["content"] == "hi"
assert out["vibeblade_fusion"]["engine"] == "live"
assert out["vibeblade_fusion"]["latency_s"] >= 0.0
print("4. live HTTP route + fusion metadata: OK")
ok += 1

# 5. automatic failover when the first engine 500s
srv_bad, st_bad = fake_engine(["m1"], fail_first=99)   # always 500
srv_ok, st_ok = fake_engine(["m1"])
r5 = FusionRouter()
r5.register(EngineSpec(name="broken", kind="openai",
                       base_url=f"http://127.0.0.1:{srv_bad.server_address[1]}",
                       models=["m1"], speed_tier=SpeedTier.TENSOR_CORE_LOCAL))
r5.register(EngineSpec(name="backup", kind="openai",
                       base_url=f"http://127.0.0.1:{srv_ok.server_address[1]}",
                       models=["m1"]))
out = asyncio.run(FusionEngine(r5).chat({"model": "m1", "messages": []}))
assert out["vibeblade_fusion"]["engine"] == "backup", out["vibeblade_fusion"]
assert st_ok["calls"] == 1
print("5. automatic failover on 500: OK")
ok += 1

# 6. health probe fills models from /v1/models
r6 = FusionRouter()
srv6, _ = fake_engine(["alpha", "beta"])
r6.register(EngineSpec(name="discover", kind="openai",
                       base_url=f"http://127.0.0.1:{srv6.server_address[1]}", models=[]))
spec = r6.engines[0]
healthy = asyncio.run(r6.probe(spec))
assert healthy and set(spec.models) == {"alpha", "beta"}
print("6. probe auto-discovers models: OK")
ok += 1

# 7. no engine available raises cleanly
try:
    asyncio.run(FusionEngine(FusionRouter()).chat({"model": "ghost", "messages": []}))
    raise SystemExit("should have raised")
except RuntimeError as e:
    assert "no engine serves" in str(e)
print("7. unknown model → clean error: OK")
ok += 1

for s in (srv, srv_bad, srv_ok, srv6):
    s.shutdown()

print(f"ALL {ok}/7 PASS")
