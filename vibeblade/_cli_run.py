"""VibeBlade CLI: vibeblade run — inference with adaptive memory tiering.

Usage:
    vibeblade run minimax-m2.7 --vram 16 --ram 32 --allow-ssd
    vibeblade run minimax-m2.7 --vram 16 --ram 256
    vibeblade run minimax-m2.7 --config vibeblade.yaml
    vibeblade run minimax-m2.7 --profile-experts --prompts calibration.txt
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_gb(value: str) -> int:
    """Parse a GB value like '16' or '16GB' to bytes."""
    s = str(value).strip().upper()
    if s.endswith("GB"):
        return int(s[:-2]) * (1024**3)
    if s.endswith("MB"):
        return int(s[:-2]) * (1024**2)
    if s.endswith("TB"):
        return int(s[:-2]) * (1024**4)
    return int(s) * (1024**3)  # assume GB if bare number


def main():
    parser = argparse.ArgumentParser(
        prog="vibeblade run",
        description="Run inference with adaptive memory tiering (VRAM/RAM/SSD)",
    )
    parser.add_argument(
        "model",
        help="Model identifier (e.g., minimax-m2.7) or path to GGUF file",
    )
    parser.add_argument(
        "--vram",
        type=parse_gb,
        default="16GB",
        help="VRAM limit in GB (default: 16)",
    )
    parser.add_argument(
        "--ram",
        type=parse_gb,
        default=None,
        help="RAM limit in GB (default: auto-detect system RAM)",
    )
    parser.add_argument(
        "--allow-ssd",
        action="store_true",
        help="Enable HYBRID_SSD mode (offload cold experts to NVMe SSD)",
    )
    parser.add_argument(
        "--ssd-path",
        type=str,
        default="/mnt/nvme/vibeblade_cache",
        help="SSD cache directory for HYBRID_SSD mode (default: /mnt/nvme/vibeblade_cache)",
    )
    parser.add_argument(
        "--hot-threshold",
        type=float,
        default=0.15,
        help="Fraction of experts to keep hot in VRAM (default: 0.15 = top 15%%)",
    )
    parser.add_argument(
        "--ram-buffer-ratio",
        type=float,
        default=0.25,
        help="Fraction of RAM limit for medium-heat expert buffer (default: 0.25)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to vibeblade.yaml config file (overrides --vram/--ram flags)",
    )
    parser.add_argument(
        "--profile-experts",
        action="store_true",
        help="Run offline expert profiling to generate hot/cold map",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to calibration prompts file (for --profile-experts)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="CPU thread pool size for cold expert compute (default: os.cpu_count)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt text",
    )

    args = parser.parse_args()

    # Build config
    if args.config:
        from .config import load_config
        try:
            ts_config = load_config(args.config)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        from .config import OffloadMode, OffloadConfig, VibeBladeConfig

        if args.allow_ssd:
            mode = OffloadMode.HYBRID_SSD
        else:
            mode = OffloadMode.RAM_ONLY

        ram_limit = args.ram
        if ram_limit is None:
            # Auto-detect system RAM
            try:
                ram_limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except (ValueError, AttributeError):
                ram_limit = 32 * (1024**3)  # fallback 32GB

        offload = OffloadConfig(
            mode=mode,
            vram_limit=args.vram,
            ram_limit=ram_limit,
            hot_threshold=args.hot_threshold,
            ssd_path=args.ssd_path,
            ram_buffer_ratio=args.ram_buffer_ratio,
        )
        ts_config = VibeBladeConfig(offload_strategy=offload)

    # Print config summary
    off = ts_config.offload_strategy
    print(f"VibeBlade v{__import__('vibeblade').__version__} — Adaptive Memory Tiering")
    print("  Developed by VibeDrift Inc. — vibedrift.com")
    print(f"  Mode:         {off.mode.value}")
    print(f"  VRAM limit:   {off.vram_limit / (1024**3):.1f} GB")
    print(f"  RAM limit:    {off.ram_limit / (1024**3):.1f} GB")
    if off.mode.value == "HYBRID_SSD":
        print(f"  SSD path:     {off.ssd_path}")
        print(f"  RAM buffer:   {off.ram_buffer_ratio * 100:.0f}% of RAM")
    print(f"  Hot experts:  top {off.hot_threshold * 100:.0f}%")
    print()

    if args.profile_experts:
        print("Expert profiling mode — generating hot/cold expert map...")
        print("Note: This requires a loaded model and calibration data.")
        print("Use the Python API for full profiling:")
        print("  model = VibeBladeModel('model.gguf')")
        print("  model.profile_experts(prompts=[...])")
        print("  model.moe_executor.stats  # view results")
        print()
        print("Profiling config saved. Re-run without --profile-experts for inference.")
        return

    # Inference mode
    model_path = args.model
    if not os.path.isfile(model_path):
        # Try as model hub identifier
        print(f"Model path '{model_path}' not found locally.")
        print("Checking model hub...")
        try:
            from .model_hub import resolve_model_path
            model_path = resolve_model_path(model_path)
            print(f"Resolved to: {model_path}")
        except Exception as e:
            print(f"Error resolving model: {e}", file=sys.stderr)
            sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    prompt = args.prompt or "Hello, how are you?"
    print(f"Model:  {model_path}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print()

    # Load and run
    try:
        from . import VibeBladeModel

        model = VibeBladeModel(model_path)

        # Enable MoE hot/cold split if model is MoE
        if model.is_moe:
            print("MoE model detected — enabling hot/cold split...")
            model.enable_moe_hot_cold(
                hot_threshold=off.hot_threshold,
                cpu_threads=args.cpu_threads,
            )

            # If HYBRID_SSD, configure tiered memory
            if off.mode.value == "HYBRID_SSD" and hasattr(model, 'configure_tiered_memory'):
                model.configure_tiered_memory(
                    ssd_path=off.ssd_path,
                    ram_limit=off.ram_limit,
                    ram_buffer_ratio=off.ram_buffer_ratio,
                    preemptive_layers=getattr(off, 'ssd_preemptive_layers', 2),
                )
                print("Tiered memory: VRAM (hot) + RAM (active) + SSD (deep)")
            else:
                print("Memory mode: VRAM (hot) + RAM (cold)")

        # Generate
        print(f"Generating up to {args.max_new_tokens} tokens...\n")
        output = model.generate(prompt, max_new_tokens=args.max_new_tokens)
        print(output)
        print()

        # Print stats
        if model.is_moe and model.moe_executor is not None:
            stats = model.moe_executor.stats
            print("--- MoE Stats ---")
            print(f"  GPU hits:    {stats.gpu_hits}")
            print(f"  CPU falls:   {stats.cpu_falls}")
            print(f"  Hit rate:    {stats.hit_rate:.1%}")
            print(f"  GPU latency: {stats.gpu_latency_ms:.1f} ms")
            print(f"  CPU latency: {stats.cpu_latency_ms:.1f} ms")

    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
