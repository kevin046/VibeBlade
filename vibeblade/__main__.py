"""VibeBlade CLI — python -m vibeblade [serve|bench|run|wizard]"""

from __future__ import annotations

import sys


def _check_for_updates() -> None:
    """Check if the local git repo is behind remote. If so, prompt to pull."""
    import subprocess

    # Only check once per day — store timestamp in a temp file
    import os
    from pathlib import Path

    stamp_file = Path.home() / ".vibeblade_update_check"
    now = int(os.environ.get("VIBEBlade_SKIP_UPDATE_CHECK", "0")) == 1
    if now:
        return
    try:
        if stamp_file.exists():
            age = (Path.stat(stamp_file).st_mtime - __import__("time").time())
            if age > -86400:  # checked within last 24h
                return
    except Exception:
        pass

    try:
        # Find the git repo root
        pkg_dir = Path(__file__).resolve().parent
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5, cwd=pkg_dir,
        )
        if result.returncode != 0:
            return
        repo_root = result.stdout.strip()

        # Fetch latest from remote (quiet, no output)
        subprocess.run(
            ["git", "fetch", "--quiet"],
            capture_output=True, timeout=15, cwd=repo_root,
        )

        # Compare local HEAD to remote tracking branch
        local = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_root,
        )
        if local.returncode != 0:
            return
        local_sha = local.stdout.strip()[:8]

        # Try origin/main first, then origin/master
        remote = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            capture_output=True, text=True, timeout=5, cwd=repo_root,
        )
        if remote.returncode != 0:
            remote = subprocess.run(
                ["git", "rev-parse", "origin/master"],
                capture_output=True, text=True, timeout=5, cwd=repo_root,
            )
        if remote.returncode != 0:
            return
        remote_sha = remote.stdout.strip()[:8]

        if local_sha != remote_sha:
            # Check if ahead, behind, or diverged
            count = subprocess.run(
                ["git", "rev-list", "--count", "--left-right", f"HEAD...{remote.stdout.strip()}"],
                capture_output=True, text=True, timeout=5, cwd=repo_root,
            )
            if count.returncode == 0:
                parts = count.stdout.strip().split("\t")
                ahead, behind = int(parts[0]), int(parts[1])
                if behind > 0 and ahead == 0:
                    print(f"\n  ⚡ Update available: you are {behind} commit(s) behind")
                    print("     Run: [bold cyan]git pull[/bold cyan] to update")
                elif behind > 0 and ahead > 0:
                    print(f"\n  ⚡ Branches diverged: {ahead} ahead, {behind} behind")
                    print("     Run: [bold cyan]git pull --rebase[/bold cyan] to update")

        # Write timestamp
        try:
            stamp_file.touch()
        except Exception:
            pass

    except Exception:
        pass  # silent — never block the CLI


def _print_help(exit_code: int = 0) -> None:
    """Unified help screen shared by both entry points."""
    from .ui import b, c, d, g, header, hr

    print(header())
    print()
    print(f" {b('Usage:')} vibeblade {'<command>'} {d('[options]')}")
    print()
    print(f" {b('Commands')}")
    rows = [
        ("wizard", "Interactive setup wizard", g("first run")),
        ("chat", "Interactive chat REPL with a loaded model", ""),
        ("serve", "Start OpenAI-compatible API server", ""),
        ("run", "Single-prompt inference with memory tiering", ""),
        ("bench", "Throughput benchmark suite", ""),
        ("tune", "Generate a hardware-tuned vibeblade.yaml", ""),
        ("pull", "Download models from HuggingFace", ""),
        ("models", "List locally available models", ""),
    ]
    width = max(len(cmd) for cmd, _, _ in rows)
    for cmd, desc, tag in rows:
        line = f"  {c(cmd.ljust(width))}  {d(desc)}"
        if tag:
            line += f"  {d('·')} {d(tag)}"
        print(line)
    print()
    print(f" {b('Options')}")
    print(f"  {c('-h, --help')}      Show this help message")
    print()
    print(f" {b('Examples')}")
    print(f"  {d('vibeblade wizard')}                          {d('# first-time setup')}")
    print(f"  {d('vibeblade tune --auto')}                     {d('# generate vibeblade.yaml')}")
    print(f"  {d('vibeblade serve --model model.gguf')}        {d('# OpenAI-compatible API')}")
    print(f"  {d('vibeblade chat --model model.gguf')}         {d('# interactive REPL')}")
    print(f"  {d('vibeblade <command> --help')}                {d('# per-command options')}")
    print()
    print(hr())
    print(f" {d('VibeDrift Inc. · vibedrift.com')}")
    sys.exit(exit_code)


def main():
    # Handle -h/--help before anything else
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help", "help"):
        _print_help(exit_code=0)

    if len(sys.argv) < 2:
        _print_help(exit_code=1)

    cmd = sys.argv[1]

    # Check for updates before running any command (once per 24h)
    _check_for_updates()

    if cmd == "serve":
        from .openai_server import main as serve_main
        sys.argv = sys.argv[1:]  # strip "serve" so argparse sees the rest
        serve_main()
    elif cmd == "dashboard":
        print("Dashboard is available in VibeBlade Pro.")
        print("Visit https://vibedrift.com for commercial licensing.")
        sys.exit(1)
    elif cmd == "bench":
        from .benchmark import main as bench_main
        sys.argv = sys.argv[1:]
        bench_main()
    elif cmd == "run":
        from ._cli_run import main as run_main
        sys.argv = sys.argv[1:]
        run_main()
    elif cmd == "browse":
        print("Model Browser is available in VibeBlade Pro.")
        print("Visit https://vibedrift.com for commercial licensing.")
        sys.exit(1)
    elif cmd == "wizard":
        from . import setup_wizard as _sw
        _sw.main()
    elif cmd in ("tune", "pull", "models"):
        # Newer subcommands live in cli.py's argparse tree — delegate so
        # python -m vibeblade and the `vibeblade` console script stay in sync.
        from .cli import main as cli_main

        cli_main(sys.argv[1:])
    elif cmd == "chat":
        from .chat import chat_loop
        import argparse

        parser = argparse.ArgumentParser(
            prog="vibeblade chat",
            description="Interactive chat with a local LLM",
        )
        parser.add_argument("--model", type=str, default=None, help="Path to .gguf file")
        parser.add_argument("--config", type=str, default="vibeblade.yaml", help="Path to vibeblade.yaml")
        parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
        parser.add_argument("--ctx-size", type=int, default=2048, help="Context window size in tokens")
        parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
        parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
        parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) filtering")
        parser.add_argument("--backend", type=str, default="auto",
                            choices=["auto", "fast", "numpy"],
                            help="Inference backend: 'fast' = C++ GGUF (fastest), 'numpy' = pure Python, 'auto' = try fast first")
        args = parser.parse_args(sys.argv[2:])

        # Resolve model path
        model_path = args.model
        if not model_path:
            # Try config file
            try:
                import yaml
                with open(args.config, "r") as f:
                    cfg = yaml.safe_load(f)
                model_path = cfg.get("model", "")
            except Exception:
                pass
        if not model_path:
            print("No model specified. Use --model or --config.", file=sys.stderr)
            print("  python -m vibeblade chat --model model.gguf", file=sys.stderr)
            print("  python -m vibeblade chat --config vibeblade.yaml", file=sys.stderr)
            sys.exit(1)

        chat_loop(
            model_path=model_path,
            max_tokens=args.max_tokens,
            ctx_size=args.ctx_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            backend=args.backend,
        )
    else:
        # Unknown command — suggest the closest real one (typo-friendly)
        import difflib

        known = ["wizard", "chat", "serve", "bench", "run", "browse",
                 "tune", "pull", "models", "dashboard"]
        close = difflib.get_close_matches(cmd, known, n=2, cutoff=0.5)
        from .ui import d, err
        print(err(f"unknown command: {cmd}"))
        if close:
            print(f"  {d('did you mean:')} {close[0]}?")
        print(f"  {d('run')} vibeblade --help {d('to see all commands')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
