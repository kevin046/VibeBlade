#!/usr/bin/env python3
"""Debug script for Qwen3.6 model tensor shapes and metadata.

Run: python scripts/debug_qwen_model.py /path/to/model.gguf
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibeblade.loader import GGUFLoader


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_qwen_model.py /path/to/model.gguf")
        sys.exit(1)

    path = sys.argv[1]

    def progress(name, done, total, loading=False):
        pass

    print(f"Loading header: {path}")
    loader = GGUFLoader(path, progress_cb=progress)
    meta = dict(loader.metadata)

    print("\n=== Architecture ===")
    arch = meta.get("general.architecture", "UNKNOWN")
    print(f"  general.architecture = {arch!r}")

    # Print all arch-specific metadata
    print("\n=== Model Config ===")
    for k, v in sorted(meta.items()):
        if k.startswith(arch + ".") or k.startswith("general."):
            if isinstance(v, (int, float, str, bool)):
                print(f"  {k} = {v}")
            elif isinstance(v, bytes):
                print(f"  {k} = <bytes {len(v)}>")
            else:
                print(f"  {k} = {type(v).__name__}({v})")

    # Print tensor shapes for block 0
    print("\n=== Block 0 Tensors ===")
    for info in loader.tensor_infos:
        name = info["name"]
        if ".0." in name or name.startswith("blk.0."):
            # Check if it's block 0
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "0" and i > 0:
                    print(f"  {name}: shape={info['shape']}, dtype={info.get('dtype', '?')}, "
                          f"nbytes={info.get('nbytes', '?')}")
                    break

    # Print shared tensors
    print("\n=== Shared Tensors ===")
    shared_names = {"token_embd.weight", "output.weight", "output_norm.weight"}
    for info in loader.tensor_infos:
        name = info["name"]
        if any(s in name for s in ["token_embd", "output_norm", "output.weight"]):
            print(f"  {name}: shape={info['shape']}, dtype={info.get('dtype', '?')}")

    # Check attention dimensions
    print("\n=== Attention Dimension Check ===")
    n_heads = meta.get(f"{arch}.attention.head_count") or meta.get("general.attention.head_count")
    n_kv_heads = meta.get(f"{arch}.attention.head_count_kv") or meta.get("general.attention.head_count_kv")
    hidden_dim = meta.get(f"{arch}.embedding_length") or meta.get("general.embedding_length")
    key_len = meta.get(f"{arch}.attention.key_length") or meta.get("general.attention.key_length")

    print(f"  n_heads = {n_heads}")
    print(f"  n_kv_heads = {n_kv_heads}")
    print(f"  hidden_dim = {hidden_dim}")
    print(f"  key_length = {key_len}")

    if n_heads and hidden_dim:
        head_dim = int(hidden_dim) // int(n_heads)
        print(f"  computed head_dim = {head_dim}")
        if n_kv_heads:
            qkv_rows = (int(n_heads) + 2 * int(n_kv_heads)) * head_dim
            print(f"  expected QKV rows = {qkv_rows}")

    # Find the actual QKV tensor
    print("\n=== QKV Tensor ===")
    for info in loader.tensor_infos:
        name = info["name"]
        if "qkv" in name.lower():
            print(f"  {name}: shape={info['shape']}, dtype={info.get('dtype', '?')}")

    # Find attn_q, attn_k, attn_v if they exist separately
    print("\n=== Attention Weights ===")
    for info in loader.tensor_infos:
        name = info["name"]
        if "attn" in name.lower() and ("q" in name.lower() or "k" in name.lower() or "v" in name.lower() or "gate" in name.lower()):
            print(f"  {name}: shape={info['shape']}, dtype={info.get('dtype', '?')}")

    loader.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
