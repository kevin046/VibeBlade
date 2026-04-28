"""VibeBlade Chat — interactive REPL for chatting with local LLMs.

Pure standard CLI terminal, no external dependencies.

Usage:
 python -m vibeblade chat
 python -m vibeblade chat --config vibeblade.yaml
 python -m vibeblade chat --model model.gguf
"""

from __future__ import annotations

import os
import sys

# ── Terminal helpers (same as setup_wizard) ──────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

if os.name == "nt":
    try:
        os.system("")
    except Exception:
        BOLD = DIM = CYAN = GREEN = YELLOW = RED = RESET = ""


def _b(s):
    return f"{BOLD}{s}{RESET}"


def _c(s):
    return f"{CYAN}{s}{RESET}"


def _g(s):
    return f"{GREEN}{s}{RESET}"


def _y(s):
    return f"{YELLOW}{s}{RESET}"


def _r(s):
    return f"{RED}{s}{RESET}"


def _d(s):
    return f"{DIM}{s}{RESET}"


# ── Chat History ─────────────────────────────────────────────────────────────

class ChatHistory:
    """Manages conversation context for multi-turn chat."""

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.messages: list[dict[str, str]] = []

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        while len(self.messages) > self.max_turns * 2:
            self.messages.pop(0)

    def clear(self):
        self.messages.clear()

    def format_context(self) -> str:
        """Format conversation history into a single prompt string."""
        if not self.messages:
            return ""
        parts = []
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @property
    def turn_count(self) -> int:
        return len(self.messages)


def _approx_tokens(text: str) -> int:
    """Very rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ── Chat REPL ────────────────────────────────────────────────────────────────

def chat_loop(model_path: str, max_tokens: int = 512, temperature: float = 0.7,
              top_k: int = 50, top_p: float = 0.9, ctx_size: int = 2048):
    """Launch interactive chat REPL with a loaded model."""
    history = ChatHistory(max_turns=max(1, ctx_size // 64))
    response_count = 0

    # Print banner
    print()
    print(f" {_c(_b('VibeBlade Chat'))}")
    print(f" {_d('Model:')} {model_path}")
    print(f" {_d('Temperature:')} {temperature} | {_d('Max tokens:')} {max_tokens} | {_d('Context:')} {ctx_size}")
    print()
    print(f" {_d('Commands:')} /help /clear /reset /quit /undo")
    print(f" {_d('───────────────────────────────────────────')}")
    print()

    # Load model with progress
    import time
    load_start = time.time()
    last_tensor = [None]

    def _progress(name, done, total, loading=False):
        """Show loading progress — timer + tensor name."""
        elapsed = time.time() - load_start
        if loading:
            pct = done / max(total, 1)
            bar_len = 20
            filled = int(bar_len * pct)
            bar = _g("█" * filled) + _d("░" * (bar_len - filled))
            short_name = name.split(".")[-1][:24] if name else ""
            sys.stdout.write(
                f"\r {bar} {_b(f'{pct:5.1%}')} {_d(f'{elapsed:5.1f}s')} {short_name:<24}"
            )
            sys.stdout.flush()
            last_tensor[0] = name

    # Show immediate feedback that we're loading
    sys.stdout.write(f"\r {_d('░' * 20)} {_b('  0.0%')} {_d('   0.0s')} {'parsing header...':<24}")
    sys.stdout.flush()

    try:
        from . import VibeBladeModel
    except ImportError:
        from vibeblade import VibeBladeModel

    model = VibeBladeModel(model_path, progress_cb=_progress)

    elapsed = time.time() - load_start
    print(f"\r {_g('█' * 20)} {_b('100.0%')} {_d(f'{elapsed:5.1f}s')} {'done':<24} ")

    # MoE hot/cold split requires a pre-computed HotColdMap from profiling
    # Skip in chat mode — runs all experts on CPU by default
    if model.is_moe:
        print()
        print(" MoE detected — all experts on CPU (use 'run' for hot/cold split)")

    print()
    print(f" {_g('Model loaded. Start chatting!')}")
    print()

    while True:
        try:
            user_input = input(f" {_c(_b('You'))}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n {_y('Goodbye!')}")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            cmd0 = cmd[0] if cmd else ""

            if cmd0 in ("/quit", "/exit", "/q"):
                print(f" {_y('Goodbye!')}")
                break

            elif cmd0 == "/clear":
                history.clear()
                print(f" {_d('Chat history cleared.')}")
                continue

            elif cmd0 == "/reset":
                history.clear()
                print(f" {_d('Chat history and KV cache reset.')}")
                continue

            elif cmd0 == "/undo":
                if history.messages:
                    removed = history.messages.pop()
                    role = removed["role"]
                    snippet = removed["content"][:50]
                    msg = f"Undid: [{role}] {snippet}..."
                    print(f" {_d(msg)}")
                else:
                    print(f" {_d('Nothing to undo.')}")
                continue

            elif cmd0 == "/help":
                print()
                print(f" {_b('Available commands:')}")
                print(" /help - Show this help")
                print(" /clear - Clear chat history")
                print(" /reset - Reset chat and model state")
                print(" /undo - Remove last message")
                print(" /quit - Exit chat")
                print()
                continue

            elif cmd0 == "/tokens":
                info = f"Turns: {history.turn_count}, Response tokens cap: {max_tokens}"
                print(f" {_d(info)}")
                continue

            else:
                hint = f"Unknown command: {cmd0}. Type /help for available commands."
                print(f" {_y(hint)}")
                continue

        # Build prompt from history
        history.add_user(user_input)
        prompt = history.format_context()
        prompt_toks = _approx_tokens(prompt)

        # Generate response with timing
        gen_start = time.time()

        # Collect output tokens via streaming callback
        output_tokens = []

        def _on_token(token_id, pos):
            output_tokens.append(token_id)
            if stream:
                try:
                    print(chr(token_id), end="", flush=True)
                except (ValueError, OverflowError):
                    pass

        try:
            print(f" {_g(_b('Assistant'))}: ", end="", flush=True)
            stream = True
            result, tps = model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stream=True,
                on_token=_on_token,
            )
            gen_elapsed = time.time() - gen_start

            # LM Studio style: prompt tokens | gen time | t/s
            print()
            print(f" {_d(f'{prompt_toks} prompt | {gen_elapsed:.2f}s | {tps:.2f} t/s')}")

            history.add_assistant(result)
            response_count += 1

        except Exception as e:
            print(f"\n {_r('Error:')} {e}")
            if history.messages and history.messages[-1]["role"] == "user":
                history.messages.pop()
            continue

        # Print stats every 5 responses
        if response_count % 5 == 0 and model.is_moe and model._moe_executor is not None:
            stats = model._moe_executor.stats
            gpu_lat = f"{stats.gpu_latency_ms:.1f}ms"
            cpu_lat = f"{stats.cpu_latency_ms:.1f}ms"
            hit = f"{stats.hit_rate:.1%}"
            msg = f"MoE Stats: GPU hit rate {hit}, GPU {gpu_lat} / CPU {cpu_lat}"
            print(f" {_d('-- ' + msg + ' --')}")

    # Final stats
    if model.is_moe and model._moe_executor is not None:
        stats = model._moe_executor.stats
        print(f"\n {_b('Session Stats:')}")
        print(f" Turns: {history.turn_count // 2}")
        print(f" GPU hits: {stats.gpu_hits} | CPU falls: {stats.cpu_falls}")
        print(f" Hit rate: {stats.hit_rate:.1%}")