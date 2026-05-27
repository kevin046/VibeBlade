"""
VibeBlade CLI — Unified command-line interface.

Subcommands:
  vibeblade serve   — Start speculative decoding API server
  vibeblade chat    — Launch ChatGPT-like web UI
  vibeblade bench   — Run throughput benchmarks

SR&ED: Unified CLI interface for experimental evaluation of inference
optimization strategies across multiple backend configurations.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vibeblade",
        description="VibeBlade — Universal Speculative Decoding Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serve   Start OpenAI-compatible speculative decoding API server
  chat    Launch ChatGPT-like web UI for interactive inference
  bench   Run throughput benchmarks

Examples:
  vibeblade serve --backend sglang --backend-url http://localhost:8000 \\
                  --model qwen3.6-27b-mtp --draft ngram

  vibeblade chat --backend-url http://localhost:8000 --port 8080

  vibeblade bench --backend-url http://localhost:8000 --concurrent 8
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── serve ─────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start speculative decoding API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vibeblade serve --backend sglang --backend-url http://localhost:8000 \\
                  --model qwen3.6-27b-mtp --draft ngram

  vibeblade serve --backend openai --backend-url http://localhost:8000 \\
                  --model my-model --draft eagle --draft-model draft.gguf
        """,
    )

    serve_parser.add_argument("--backend", default="openai",
                              choices=["sglang", "vllm", "llama_cpp", "openai"],
                              help="Target model backend type (default: openai)")
    serve_parser.add_argument("--backend-url", default="http://localhost:8000",
                              help="Target backend URL (default: http://localhost:8000)")
    serve_parser.add_argument("--model", required=True,
                              help="Model name at the target backend")
    serve_parser.add_argument("--api-key", default=None,
                              help="API key for target backend (if required)")

    serve_parser.add_argument("--draft", default="ngram",
                              choices=["ngram", "eagle", "dflash", "nextn", "none"],
                              help="Draft strategy (default: ngram)")
    serve_parser.add_argument("--max-draft", type=int, default=8,
                              help="Maximum draft tokens per step (default: 8)")
    serve_parser.add_argument("--draft-model", default=None,
                              help="Path to draft model (for EAGLE/DFlash)")
    serve_parser.add_argument("--draft-ngram-size", type=int, default=5,
                              help="N-gram context size (default: 5)")

    serve_parser.add_argument("--temperature", type=float, default=0.0,
                              help="Sampling temperature (default: 0.0 = greedy)")
    serve_parser.add_argument("--top-k", type=int, default=40,
                              help="Top-k filtering (default: 40)")
    serve_parser.add_argument("--top-p", type=float, default=0.95,
                              help="Top-p (nucleus) filtering (default: 0.95)")

    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--reload", action="store_true", help="Hot reload")

    # ── chat ──────────────────────────────────────────────────────
    chat_parser = subparsers.add_parser(
        "chat",
        help="Launch ChatGPT-like web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vibeblade chat --backend-url http://localhost:8000

  vibeblade chat --backend-url http://localhost:8000 --port 3000 --host 0.0.0.0
        """,
    )

    chat_parser.add_argument("--backend-url", default="http://localhost:8000",
                             help="Inference backend URL (default: http://localhost:8000)")
    chat_parser.add_argument("--host", default="0.0.0.0",
                             help="Host to bind (default: 0.0.0.0)")
    chat_parser.add_argument("--port", type=int, default=8080,
                             help="Port to bind (default: 8080)")
    chat_parser.add_argument("--reload", action="store_true",
                             help="Enable hot reload for development")

    # ── bench ─────────────────────────────────────────────────────
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run throughput benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vibeblade bench --backend-url http://localhost:8000

  vibeblade bench --backend-url http://localhost:8000 --concurrent 8 --max-tokens 512
        """,
    )

    bench_parser.add_argument("--backend-url", default="http://localhost:8000",
                              help="Inference backend URL (default: http://localhost:8000)")
    bench_parser.add_argument("--model", default="qwen3.6-27b-mtp",
                              help="Model name (default: qwen3.6-27b-mtp)")
    bench_parser.add_argument("--concurrent", type=int, default=1,
                              help="Number of concurrent requests (default: 1)")
    bench_parser.add_argument("--max-tokens", type=int, default=256,
                              help="Max tokens per request (default: 256)")
    bench_parser.add_argument("--rounds", type=int, default=3,
                              help="Number of benchmark rounds (default: 3)")

    # ── Parse and dispatch ────────────────────────────────────────
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "bench":
        _cmd_bench(args)


def _cmd_serve(args: argparse.Namespace) -> None:
    """Dispatch to the existing OpenAI-compatible server."""
    # Reuse the existing server entry point
    from .openai_server import main as serve_main

    # Build equivalent argv for the existing parser
    argv = [
        "--backend", args.backend,
        "--backend-url", args.backend_url,
        "--model", args.model,
        "--draft", args.draft,
        "--max-draft", str(args.max_draft),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--host", args.host,
        "--port", str(args.port),
        "--draft-ngram-size", str(args.draft_ngram_size),
    ]
    if args.api_key:
        argv.extend(["--api-key", args.api_key])
    if args.draft_model:
        argv.extend(["--draft-model", args.draft_model])
    if args.reload:
        argv.append("--reload")

    serve_main(argv)


def _cmd_chat(args: argparse.Namespace) -> None:
    """Launch the web UI server."""
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Inject backend URL into the web app's environment
    os.environ["VIBEBLADE_BACKEND_URL"] = args.backend_url

    try:
        import uvicorn
    except ImportError:
        print("  Error: uvicorn is required for the web UI.")
        print("  Install with: pip install uvicorn fastapi httpx")
        sys.exit(1)

    try:
        import httpx  # noqa: F401
    except ImportError:
        print("  Error: httpx is required for the web UI.")
        print("  Install with: pip install httpx")
        sys.exit(1)

    print("\n  VibeBlade Chat v1.0.0")
    print(f"  Backend: {args.backend_url}")
    print(f"  UI:      http://{args.host}:{args.port}")
    print()

    uvicorn.run(
        "vibeblade.web_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def _cmd_bench(args: argparse.Namespace) -> None:
    """Run throughput benchmarks against the inference backend."""
    import json
    import time
    import concurrent.futures

    try:
        import httpx
    except ImportError:
        print("  Error: httpx is required. Install with: pip install httpx")
        sys.exit(1)

    print("\n  VibeBlade Benchmark")
    print(f"  Backend:  {args.backend_url}")
    print(f"  Model:    {args.model}")
    print(f"  Workers:  {args.concurrent}")
    print(f"  Tokens:   {args.max_tokens}")
    print(f"  Rounds:   {args.rounds}")
    print()

    prompts = [
        "Write a Python function that computes the Fibonacci sequence using memoization. Include type hints and a docstring.",
        "Explain the difference between TCP and UDP in networking. When would you choose one over the other?",
        "What is speculative decoding in LLM inference? How does it improve throughput?",
    ]

    def run_request(prompt: str) -> dict:
        start = time.time()
        tokens = 0
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{args.backend_url}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                    "stream": True,
                    "reasoning": {"effort": "none"},
                },
                headers={"Content-Type": "application/json"},
            ) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line[6:].strip() not in ("[DONE]", ""):
                        try:
                            data = json.loads(line[6:])
                            if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                tokens += 1
                        except json.JSONDecodeError:
                            pass
        elapsed = time.time() - start
        return {"tokens": tokens, "elapsed": elapsed, "tok_s": tokens / max(elapsed, 0.001)}

    results = []
    for round_i in range(args.rounds):
        print(f"  Round {round_i + 1}/{args.rounds}...")
        round_results = []

        if args.concurrent == 1:
            for prompt in prompts:
                r = run_request(prompt)
                round_results.append(r)
                print(f"    {r['tokens']:4d} tokens in {r['elapsed']:5.1f}s  =  {r['tok_s']:.1f} tok/s")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
                futures = [executor.submit(run_request, p) for p in prompts * ((args.concurrent // len(prompts)) + 1)]
                for f in concurrent.futures.as_completed(futures):
                    r = f.result()
                    round_results.append(r)
                    print(f"    {r['tokens']:4d} tokens in {r['elapsed']:5.1f}s  =  {r['tok_s']:.1f} tok/s")

        total_tokens = sum(r["tokens"] for r in round_results)
        total_time = sum(r["elapsed"] for r in round_results)
        avg_tok_s = total_tokens / max(total_time, 0.001)
        results.append(avg_tok_s)
        print(f"    Aggregate: {total_tokens} tokens / {total_time:.1f}s = {avg_tok_s:.1f} tok/s\n")

    if results:
        print(f"  Average across rounds: {sum(results) / len(results):.1f} tok/s")
        print(f"  Best round:            {max(results):.1f} tok/s")


if __name__ == "__main__":
    main()
