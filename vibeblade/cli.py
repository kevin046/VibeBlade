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

    # ── pull ──────────────────────────────────────────────────────
    pull_parser = subparsers.add_parser(
        "pull",
        help="Download any compatible model from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vibeblade pull bartowski/Llama-3.2-3B-Instruct-GGUF

  vibeblade pull bartowski/Qwen2.5-7B-Instruct-GGUF --quant Q4_K_M

  vibeblade pull org/model --dir /mnt/nvme/models --include-safetensors
        """,
    )
    pull_parser.add_argument("repo", help="HuggingFace repo id (org/model)")
    pull_parser.add_argument("--quant", default=None,
                             help="Prefer files matching this quant substring (e.g. Q4_K_M)")
    pull_parser.add_argument("--dir", default="", help="Destination directory (default: ~/.vibeblade/models)")
    pull_parser.add_argument("--revision", default="main", help="HF revision/branch (default: main)")
    pull_parser.add_argument("--include-safetensors", action="store_true",
                             help="Also download safetensors weights (default: GGUF first)")

    # ── models ────────────────────────────────────────────────────
    models_parser = subparsers.add_parser(
        "models",
        help="List locally available models",
    )
    models_parser.add_argument("--verbose", "-v", action="store_true",
                               help="Show full paths and metadata")

    # ── tune ──────────────────────────────────────────────────────
    tune_parser = subparsers.add_parser(
        "tune",
        help="Generate a hardware-optimized vibeblade.yaml for local LLM hosting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  vibeblade tune --gpu 24GB --ram 128GB --ssd /mnt/nvme/vibeblade_cache \\
                  --model /models/KimiK3-Inst.Q4_K.gguf --output vibeblade.yaml

  vibeblade tune --auto --output vibeblade.yaml  # auto-detect hardware
        """,
    )
    tune_parser.add_argument("--gpu", type=str, help="GPU VRAM size (e.g. 24GB, 12GB)")
    tune_parser.add_argument("--ram", type=str, help="System RAM size (e.g. 128GB, 64GB)")
    tune_parser.add_argument("--ssd", type=str, help="Path to fast NVMe SSD for offload (required for HYBRID_SSD)")
    tune_parser.add_argument("--model", type=str, required=True, help="Path to GGUF model file")
    tune_parser.add_argument("--output", type=str, default="vibeblade.yaml", help="Output config file path")
    tune_parser.add_argument("--auto", action="store_true", help="Auto-detect hardware (Linux only)")

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
    elif args.command == "pull":
        _cmd_pull(args)
    elif args.command == "tune":
        _cmd_tune(args)
    elif args.command == "models":
        _cmd_models(args)


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


def _cmd_models(args: argparse.Namespace) -> None:
    """List locally available models from the VibeBlade model cache."""
    import os
    import time

    from .ui import d, header, ok, table

    print(header("local models"))
    print()

    model_dir = os.path.expanduser("~/.vibeblade/models")
    registry = os.path.join(model_dir, "models.json")

    entries: list[tuple[str, float, float, str]] = []
    if os.path.isfile(registry):
        try:
            import json

            with open(registry) as f:
                reg = json.load(f)
            for item in reg if isinstance(reg, list) else reg.get("models", []):
                path = item.get("path") or item.get("local_path") or ""
                name = item.get("name") or os.path.basename(path)
                size = item.get("size_bytes") or 0
                if not size and path and os.path.isfile(path):
                    size = os.path.getsize(path)
                added = item.get("added_at") or item.get("timestamp") or 0
                entries.append((name, float(size), float(added), path))
        except Exception:
            pass

    # Fall back to scanning the directory for GGUF files
    if not entries and os.path.isdir(model_dir):
        for fn in sorted(os.listdir(model_dir)):
            if fn.endswith(".gguf"):
                p = os.path.join(model_dir, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    sz = 0
                entries.append((fn, float(sz), os.path.getmtime(p), p))

    if not entries:
        print(f" {ok('no models yet')}  {d('— pull one with:')} vibeblade pull <hf-repo>")
        return

    rows = []
    for name, size, added, path in entries:
        when = time.strftime("%Y-%m-%d", time.localtime(added)) if added else d("—")
        rows.append([name, f"{size / (1024 ** 3):.1f} GB", when, d(path)])
    print(table(["model", "size", "added", "path"], rows))
    print()
    print(f" {d(len(entries))} model(s) · {d('serve one with:')} vibeblade serve --model <path>")


def _cmd_tune(args: argparse.Namespace) -> None:
    """Generate a hardware-optimized vibeblade.yaml for local LLM hosting."""
    import os

    import yaml

    from .auto_tune import tune_hardware

    args.model = os.path.abspath(args.model)
    if not os.path.exists(args.model) and not args.auto:
        print(f"  Error: model file not found: {args.model}")
        sys.exit(1)
    profile = tune_hardware(
        model_path=args.model,
        gpu_vram_str=args.gpu,
        ram_str=args.ram,
        ssd_path=args.ssd,
        auto=args.auto,
    )
    config = {
        "model": {
            "path": args.model,
            "n_ctx": profile.n_ctx,
            "n_batch": profile.n_batch,
        },
        "offload_strategy": {
            "mode": profile.mode.value,
            "vram_limit": profile.vram_limit,
            "ram_limit": profile.ram_limit,
            "hot_threshold": profile.hot_threshold,
            "ssd_path": profile.ssd_path,
            "ram_buffer_ratio": profile.ram_buffer_ratio,
            "ssd_preemptive_layers": profile.ssd_preemptive_layers,
            "layer_map": profile.layer_map,
        },
        "optimization": {
            "activation_sparsity": profile.activation_sparsity,
            "speculative_decoding": profile.speculative_decoding,
            "kv_quantization": profile.kv_quantization,
            "paged_attention": profile.paged_attention,
            "use_native_engine": profile.use_native_engine,
            "flash_attention": profile.flash_attention,
        },
    }
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    from .ui import b, c, d, g, kv, ok, panel, status

    mode_line = {
        "gpu": c("gpu — all weights in VRAM"),
        "cpu": c("cpu — all weights in system RAM"),
        "hybrid_ram": c("hybrid — hot layers in VRAM, rest in RAM"),
        "hybrid_ssd": c("hybrid_ssd — NVMe streaming for cold layers"),
    }.get(profile.mode.value, profile.mode.value)

    lines = [
        kv("mode", mode_line),
        kv("model size", f"{_model_size_gb(profile):.1f} GB"),
        kv("context", f"{profile.n_ctx:,} tokens"),
        kv("batch", str(profile.n_batch)),
        kv("hot threshold", f"{profile.hot_threshold:.2f}"),
    ]
    if profile.vram_limit:
        lines.append(kv("vram budget", f"{profile.vram_limit:.0f} GB"))
    lines.append(kv("ram budget", f"{profile.ram_limit:.0f} GB"))
    if profile.ssd_path:
        lines.append(kv("ssd cache", profile.ssd_path))
    flags = []
    if profile.activation_sparsity:
        flags.append("sparsity")
    if profile.speculative_decoding:
        flags.append("spec-decode")
    if profile.kv_quantization:
        flags.append("kv-quant")
    if profile.paged_attention:
        flags.append("paged-attn")
    if profile.flash_attention:
        flags.append("flash-attn")
    lines.append(kv("optimizations", d(" ".join(flags) or "none")))

    print(panel("tuned config", lines))
    print()
    print(f" {ok(f'saved to {args.output}')}")
    print(f" {d('To serve:')} vibeblade serve --config {args.output}")


def _model_size_gb(profile) -> float:
    """Best-effort model size for the tune summary panel."""
    import os

    try:
        return os.path.getsize(profile.model_path) / (1024 ** 3)
    except OSError:
        return 0.0


def _cmd_pull(args: argparse.Namespace) -> None:
    """Download any compatible model from HuggingFace and register it locally."""
    from vibeblade.model_pull import PullError, pull_model

    try:
        result = pull_model(
            args.repo,
            dest_dir=args.dir or "",
            pattern=args.quant,
            revision=args.revision,
            include_safetensors=args.include_safetensors,
        )
        print()
        print(result.summary())

        # Auto-optimize: run the tuner on the first downloaded weight file so
        # the model is served with its best profile immediately.
        try:
            from vibeblade.auto_tune import auto_tune

            weight_files = sorted(
                p for p in __import__("pathlib").Path(result.model_dir).iterdir()
                if p.suffix in (".gguf", ".safetensors")
            )
            if weight_files:
                profile = auto_tune(str(weight_files[0]))
                print("\nAuto-tune profile applied:")
                for key, val in vars(profile).items():
                    print(f"  {key}: {val}")
        except Exception as e:  # tuning is advisory — never fail the pull
            print(f"(auto-tune skipped: {e})")
    except PullError as e:
        print(f"pull failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
