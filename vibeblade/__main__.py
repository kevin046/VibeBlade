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


def main():
    # Handle -h/--help before anything else
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help", "help"):
        print("VibeBlade — Adaptive Memory Tiering for LLM Inference")
        print("Developed by VibeDrift Inc. — vibedrift.com")
        print()
        print("Usage: python -m vibeblade [command] [options]")
        print()
        print("Commands:")
        print("  wizard              Interactive setup wizard (recommended first run)")
        print("  chat                Interactive chat REPL with loaded model")
        print("  serve               Start OpenAI-compatible API server")
        print("  bench               Run performance benchmark suite")
        print("  run                 Run inference with memory tiering")
        print()
        print("Options:")
        print("  -h, --help          Show this help message")
        print()
        print("Examples:")
        print("  python -m vibeblade wizard          # First-time setup")
        print("  python -m vibeblade chat             # Chat with model")
        print("  python -m vibeblade chat --help      # Chat options")
        print("  python -m vibeblade serve --help     # Server options")
        print("  python -m vibeblade bench --help     # Benchmark options")
        print("  python -m vibeblade run --help       # Run options")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("VibeBlade — Adaptive Memory Tiering for LLM Inference")
        print("Developed by VibeDrift Inc. — vibedrift.com")
        print()
        print("Usage: python -m vibeblade [command] [options]")
        print()
        print("Commands:")
        print("  wizard              Interactive setup wizard (recommended first run)")
        print("  chat                Interactive chat REPL with loaded model")
        print("  serve               Start OpenAI-compatible API server")
        print("  bench               Run performance benchmark suite")
        print("  run                 Run inference with memory tiering")
        print()
        print("Examples:")
        print("  python -m vibeblade wizard")
        print("  python -m vibeblade chat")
        print("  python -m vibeblade chat --help")
        sys.exit(1)

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
        )
    else:
        print(f"Unknown command: {cmd}")
        print("Use 'serve', 'bench', 'run', or 'wizard'")
        sys.exit(1)


if __name__ == "__main__":
    main()
