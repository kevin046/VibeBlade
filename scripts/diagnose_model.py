#!/usr/bin/env python3
"""VibeBlade model diagnostic — print every tensor shape that matters."""

import sys
import os

# Add vibeblade to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vibeblade.loader import load_model

def diagnose(path: str):
    print(f"Loading {path} ...")
    result = load_model(path, lazy=False)  # eager load so we can see all shapes
    weights = result['weights']
    config = result['config']
    metadata = result.get('metadata', {})

    print("\n=== CONFIG ===")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n=== ARCHITECTURE METADATA ===")
    arch = metadata.get('general.architecture', 'unknown')
    print(f"  architecture: {arch}")
    arch_meta = {k: v for k, v in metadata.items() if k.startswith(arch + '.')}
    for k, v in sorted(arch_meta.items()):
        print(f"  {k}: {v}")

    print("\n=== MODEL TENSORS ===")
    # Get first block prefix
    blk_keys = sorted(k for k in weights if k.startswith('blk.') and k.endswith('.weight'))
    prefixes = sorted(set('.'.join(k.split('.')[:2]) for k in blk_keys))[:2]
    n_layers = len(set('.'.join(k.split('.')[:2]) for k in blk_keys))
    print(f"  Total layers: {n_layers}")
    print(f"  Sample prefixes: {prefixes}")

    # Print ALL tensor shapes for first block (dense check)
    first_block = prefixes[0] if prefixes else None
    if first_block:
        print(f"\n  === {first_block} full tensor list ===")
        block_keys = sorted(k for k in weights if k.startswith(first_block + '.'))
        for k in block_keys:
            v = weights[k]
            print(f"    {k}: {v.shape} dtype={v.dtype} nbytes={v.nbytes:,}")

    # Print MoE tensor shapes for all blocks
    moe_blocks = sorted(set('.'.join(k.split('.')[:2]) for k in weights
                             if 'ffn_gate_inp' in k or 'ffn_up_exps' in k))
    if moe_blocks:
        print(f"\n  === MoE blocks: {moe_blocks} ===")
        for bp in moe_blocks:
            moe_keys = sorted(k for k in weights if k.startswith(bp + '.') and
                            ('ffn_gate_inp' in k or 'ffn_up_exps' in k or
                             'ffn_down_exps' in k or 'ffn_gate_exps' in k or
                             'ffn_gate_shexp' in k or 'ffn_up_shexp' in k or
                             'ffn_down_shexp' in k))
            for k in moe_keys:
                v = weights[k]
                print(f"    {k}: {v.shape} dtype={v.dtype} nbytes={v.nbytes:,}")

    # Print shared tensors
    print(f"\n  === Shared tensors ===")
    shared_keys = ['token_embd.weight', 'output_norm.weight', 'output.weight']
    for k in shared_keys:
        if k in weights:
            v = weights[k]
            print(f"    {k}: {v.shape} dtype={v.dtype} nbytes={v.nbytes:,}")
        else:
            print(f"    {k}: NOT FOUND")

    # Dense FFN check for all blocks (non-MoE)
    dense_blocks = sorted(set('.'.join(k.split('.')[:2]) for k in weights
                               if 'ffn_gate.weight' in k and 'ffn_gate_inp' not in k
                               and 'ffn_gate_shexp' not in k and 'ffn_gate_exps' not in k
                               and k.startswith('blk.')))
    if dense_blocks:
        print(f"\n  === Dense FFN blocks: {dense_blocks} ===")
        for bp in dense_blocks[:2]:
            for suffix in ['ffn_norm.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight']:
                k = f"{bp}.{suffix}"
                if k in weights:
                    v = weights[k]
                    print(f"    {k}: {v.shape}")

    # Simulate forward pass shapes
    print("\n=== FORWARD PASS SHAPE SIMULATION ===")
    hidden_dim = config.get('embedding_length', 2048)
    n_heads = config.get('attention.head_count', 32)
    n_kv_heads = config.get('attention.head_count_kv', n_heads)
    head_dim_cfg = config.get('attention.key_length') or (hidden_dim // n_heads if n_heads else 64)
    n_layers = config.get('block_count', 0)
    intermediate_dim = config.get('feed_forward_length', 0)
    vocab_size = config.get('vocab_size', 0)
    seq_len = 21

    print(f"  hidden_dim={hidden_dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim_cfg}")
    print(f"  n_layers={n_layers}, intermediate_dim={intermediate_dim}, vocab_size={vocab_size}")
    print(f"  Simulating forward pass for seq_len={seq_len}:")

    x_seq = (seq_len, hidden_dim)
    print(f"  token_emb[{seq_len}] -> x: {x_seq}")

    # attn_norm
    if first_block:
        attn_norm_key = f"{first_block}.attn_norm.weight"
        if attn_norm_key in weights:
            print(f"  rms_norm(x, attn_norm): x={x_seq}, attn_norm={weights[attn_norm_key].shape} -> {x_seq}")

        # QKV
        q_key = f"{first_block}.attn_q.weight"
        k_key = f"{first_block}.attn_k.weight"
        v_key = f"{first_block}.attn_v.weight"
        if q_key in weights:
            q_w = weights[q_key]
            k_w = weights[k_key] if k_key in weights else None
            v_w = weights[v_key] if v_key in weights else None
            q_out = (seq_len, q_w.shape[0])
            k_out = (seq_len, k_w.shape[0]) if k_w is not None else "N/A"
            v_out = (seq_len, v_w.shape[0]) if v_w is not None else "N/A"
            print(f"  h @ q_w.T: {x_seq} @ {q_w.shape}.T -> {q_out}")
            print(f"  h @ k_w.T: {x_seq} @ {k_w.shape}.T -> {k_out}")
            print(f"  h @ v_w.T: {x_seq} @ {v_w.shape}.T -> {v_out}")

        # attn_output
        o_key = f"{first_block}.attn_output.weight"
        if o_key in weights:
            o_w = weights[o_key]
            q_out_dim = weights[q_key].shape[0] if q_key in weights else n_heads * head_dim_cfg
            attn_out = (seq_len, q_out_dim)
            print(f"  attn_out @ o_w.T: {attn_out} @ {o_w.shape}.T")

            # Check shape compatibility
            if attn_out[1] == o_w.shape[0]:
                print(f"    -> {attn_out[0], o_w.shape[1]} (OK)")
            else:
                print(f"    -> MISMATCH: attn_out dim={attn_out[1]} but o_w rows={o_w.shape[0]}")

        # FFN
        ffn_norm_key = f"{first_block}.ffn_norm.weight"
        if ffn_norm_key in weights:
            print(f"  ffn_norm: x={x_seq}, ffn_norm={weights[ffn_norm_key].shape}")

        # Check which FFN path would be taken
        gate_inp_key = f"{first_block}.ffn_gate_inp.weight"
        gate_key = f"{first_block}.ffn_gate.weight"
        if gate_inp_key in weights:
            print(f"  -> MoE path (ffn_gate_inp={weights[gate_inp_key].shape})")
            up_key = f"{first_block}.ffn_up_exps.weight"
            down_key = f"{first_block}.ffn_down_exps.weight"
            gate_exp_key = f"{first_block}.ffn_gate_exps.weight"
            if up_key in weights:
                print(f"     up_exps={weights[up_key].shape}")
            if down_key in weights:
                print(f"     down_exps={weights[down_key].shape}")
            if gate_exp_key in weights:
                print(f"     gate_exps={weights[gate_exp_key].shape}")
        elif gate_key in weights:
            print(f"  -> Dense path (gate={weights[gate_key].shape})")
            up_key = f"{first_block}.ffn_up.weight"
            down_key = f"{first_block}.ffn_down.weight"
            if up_key in weights:
                print(f"     up={weights[up_key].shape}")
            if down_key in weights:
                print(f"     down={weights[down_key].shape}")

    # Final projection
    if 'output.weight' in weights and 'output_norm.weight' in weights:
        o_w = weights['output.weight']
        print(f"  final rms_norm: {x_seq}")
        print(f"  logits: {x_seq} @ {o_w.shape}.T = ({x_seq[0]}, {o_w.shape[0]})")

if __name__ == '__main__':
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?')
    args = parser.parse_args()

    if args.model:
        diagnose(args.model)
    else:
        # Auto-find GGUF
        for pattern in ['models/**/*.gguf', '../models/**/*.gguf',
                       '**/*.gguf', '/tmp/**/*.gguf']:
            files = glob.glob(pattern, recursive=True)
            if files:
                print(f"Auto-found: {files[0]}")
                diagnose(files[0])
                break
        else:
            print("No GGUF model found. Pass path as argument.")
