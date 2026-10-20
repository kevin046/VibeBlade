"""Offline unit test for vibeblade tool-calling functions.

Loads openai_server.py in isolation (bypasses package __init__ heavy imports).
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

SRC = Path("/home/ubuntu/vibeblade/vibeblade/openai_server.py")

# Minimal stubs so the module imports without fastapi/np deps
sys.modules.setdefault("fastapi", types.ModuleType("fastapi"))
srv = types.ModuleType("vibeblade.openai_server")
spec = importlib.util.spec_from_file_location("vibeblade.openai_server", SRC)
sys.modules["vibeblade.openai_server"] = srv
spec.loader.exec_module(srv)

OK = 0

tools = [{"type": "function", "function": {
    "name": "create_file", "description": "Create a file",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"},
                                  "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]

req = srv.ChatCompletionRequest(
    model="qwen",
    messages=[{"role": "user", "content": "Create result.txt with hi"}],
    tools=tools, tool_choice="auto")
prompt = srv._format_chat_prompt(req.messages, tools=req.tools)
assert "<tools>" in prompt and "create_file" in prompt, prompt[:200]
assert "<|im_start|>system" in prompt
print("1. prompt renders <tools> block: OK")
OK += 1

sim = ('<tool_call>\n{"name": "create_file", "arguments": '
       '{"path": "result.txt", "content": "hi"}}\n</tool_call>')
calls = srv._parse_tool_calls(sim)
assert calls, "no calls parsed"
assert calls[0]["function"]["name"] == "create_file"
assert json.loads(calls[0]["function"]["arguments"]) == {"path": "result.txt", "content": "hi"}
assert calls[0]["id"].startswith("call_") and calls[0]["type"] == "function"
print("2. parse <tool_call> -> OpenAI tool_calls: OK")
OK += 1

assert srv._strip_tool_blocks(sim) == ""
assert srv._strip_tool_blocks("<think>hmm</think>done") == "done"
print("3. strip tool/think blocks from content: OK")
OK += 1

req2 = srv.ChatCompletionRequest(messages=[
    {"role": "user", "content": "Create result.txt"},
    {"role": "assistant", "content": None, "tool_calls": calls},
    {"role": "tool", "tool_call_id": calls[0]["id"], "content": "file created"},
], tools=tools)
p2 = srv._format_chat_prompt(req2.messages, tools=req2.tools)
assert "<tool_response>" in p2, p2[-300:]
assert '"name": "create_file"' in p2
print("4. multi-round history (assistant tool_calls + tool response): OK")
OK += 1

# tool_choice="none" must not render tools
p3 = srv._format_chat_prompt(req.messages, tools=[])
assert "<tools>" not in p3
print("5. no-tools prompt stays plain ChatML: OK")
OK += 1

# malformed tool_call ignored gracefully
assert srv._parse_tool_calls("<tool_call>{broken json}</tool_call>") == []
print("6. malformed <tool_call> ignored: OK")
OK += 1

print(f"ALL {OK}/6 PASS")
