"""VibeBlade Backends — pluggable target model implementations.

Backend types:
  - sglang: Connects to a running sglang server via HTTP (OpenAI-compatible)
  - vllm: Connects to a running vLLM server via HTTP (OpenAI-compatible)
  - llama_cpp: Local GGUF inference via llama.cpp ctypes
  - openai: Generic OpenAI-compatible HTTP API
"""
