#!/usr/bin/env bash
# Run quality tests for all models
cd ~/VibeBlade
export PYTHONUNBUFFERED=1
export LLAMA_LOG=0

MODELS=(
  "models/llama-3.2-1b-q4_k_m.gguf:Llama-3.2-1B"
  "models/qwen2.5-3b-q4_k_m.gguf:Qwen2.5-3B"
  "models/qwen3.5-moe-0.87b-q4_k_s.gguf:Qwen3.5-MoE-0.87B"
  "models/phi-3.5-mini-q4_k_m.gguf:Phi-3.5-mini"
  "models/gemma-2-2b-q4_k_m.gguf:Gemma-2-2B"
)

for entry in "${MODELS[@]}"; do
  mpath="${entry%%:*}"
  mname="${entry##*:}"
  if [ ! -f "$mpath" ]; then
    echo "SKIP: $mpath not found"
    continue
  fi
  echo "==== Running $mname ===="
  python3 -u test_one.py "$mpath" "$mname" 2>/dev/null
  echo ""
done

echo "=== ALL COMPLETE ==="
