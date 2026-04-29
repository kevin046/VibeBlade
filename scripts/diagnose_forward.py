#!/usr/bin/env python3
"""Step-by-step forward pass with shape validation — find exactly where the matmul fails."""

import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from vibeblade.loader import load_model
from vibeblade.transformer import rms_norm, silu, rope, attention


def validate_matmul(name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Do matmul with explicit shape validation. Returns result or raises with context."""
    a_k = a.shape[-1]
    b_k = b.shape[-2 if b.ndim > 1 else 0]
    expected_out = a.shape[:-1] + (b.shape[-1 if b.ndim > 1 else 0],)
    if a_k != b_k:
        raise ValueError(
            f"MATMUL SHAPE MISMATCH in [{name}]:\n"
            f"  A={a.shape} (K={a_k})  B={b.shape} (K={b_k})\n"
            f"  Expected output: {expected_out}\n"
            f"  {traceback.format_stack()[-3].strip()}"
        )
    return np.matmul(a, b)


def run_forward(weights: dict, config: dict):
    """Run a minimal forward pass layer by layer with strict shape validation."""

    # Extract config
    hidden_dim = config.get('embedding_length', 2048)
    n_heads = config.get('attention.head_count', 32)
    n_kv_heads = config.get('attention.head_count_kv', n_heads)
    head_dim = config.get('attention.key_length') or (hidden_dim // n_heads)
    n_layers = config.get('block_count', 0)
    vocab_size = config.get('vocab_size', 0)
    rms_eps = config.get('attention.layer_norm_rms_epsilon', 1e-5)

    seq_len = 5  # minimal sequence
    print(f"Config: hidden={hidden_dim}, n_heads={n_heads}, n_kv={n_kv_heads}, "
          f"head_dim={head_dim}, n_layers={n_layers}, vocab={vocab_size}")
    print(f"Running forward pass for seq_len={seq_len}...\n")

    # Token embedding
    token_emb = weights.get('token_embd.weight')
    if token_emb is None:
        raise KeyError("token_embd.weight NOT FOUND. Available keys: " +
                       str(sorted(k for k in weights if 'emb' in k.lower())))

    print(f"[OK] token_emb: {token_emb.shape}")

    # Build fake token IDs
    token_ids = np.arange(1, seq_len + 1, dtype=np.int64) % token_emb.shape[0]

    # Output norm
    output_norm_w = weights.get('output_norm.weight')
    if output_norm_w is None:
        raise KeyError("output_norm.weight NOT FOUND. Available keys: " +
                       str(sorted(k for k in weights if 'norm' in k.lower())))

    # Output projection
    output_w = weights.get('output.weight')
    if output_w is None:
        raise KeyError("output.weight NOT FOUND. Available keys: " +
                       str(sorted(k for k in weights if 'output' in k.lower())))

    print(f"[OK] output_norm: {output_norm_w.shape}")
    print(f"[OK] output: {output_w.shape}")

    # Simulate token embedding
    x = token_emb[token_ids]
    print(f"[OK] token_emb[{token_ids.min()}:{token_ids.max()}] -> x: {x.shape}")

    # RoPE cache
    max_seq = 2048
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    print(f"[OK] RoPE cache: cos={cos_cache.shape}, sin={sin_cache.shape}\n")

    # RoPE slices
    cos_slice = cos_cache[:seq_len]
    sin_slice = sin_cache[:seq_len]

    # Process only FIRST layer to pinpoint the crash
    layer_idx = 0
    prefix = f"blk.{layer_idx}"

    print(f"=== Processing {prefix} ===")

    # --- attn_norm ---
    attn_norm_key = f"{prefix}.attn_norm.weight"
    attn_norm_w = weights.get(attn_norm_key)
    if attn_norm_w is None:
        raise KeyError(f"{attn_norm_key} NOT FOUND. Blk keys: " +
                       str(sorted(k for k in weights if k.startswith(prefix))))
    print(f"[OK] attn_norm: {attn_norm_w.shape}")
    h = rms_norm(x, attn_norm_w, rms_eps)
    print(f"[OK] rms_norm -> h: {h.shape}")

    # --- QKV ---
    q_key = f"{prefix}.attn_q.weight"
    k_key = f"{prefix}.attn_k.weight"
    v_key = f"{prefix}.attn_v.weight"
    q_w = weights.get(q_key)
    k_w = weights.get(k_key)
    v_w = weights.get(v_key)

    if q_w is None:
        raise KeyError(f"{q_key} NOT FOUND. Available QKV: " +
                       str(sorted(k for k in weights if 'attn_q' in k or 'attn_k' in k or 'attn_v' in k)))
    print(f"[OK] q_w: {q_w.shape}")
    print(f"[OK] k_w: {k_w.shape if k_w is not None else 'MISSING'}")
    print(f"[OK] v_w: {v_w.shape if v_w is not None else 'MISSING'}")

    # Check if K/V exist or if they're aliased to Q
    if k_w is None:
        k_w = weights.get(q_key)  # same tensor if aliased
    if v_w is None:
        v_w = weights.get(q_key)

    q = validate_matmul(f"{prefix}.attn_q", h, q_w.T)
    k = validate_matmul(f"{prefix}.attn_k", h, k_w.T)
    v = validate_matmul(f"{prefix}.attn_v", h, v_w.T)
    print(f"[OK] Q: {q.shape}, K: {k.shape}, V: {v.shape}")

    # Check Q/K dimensions are compatible with attention
    expected_q_dim = n_heads * head_dim
    if q.shape[-1] != expected_q_dim:
        print(f"[WARN] Q dim {q.shape[-1]} != expected {expected_q_dim}. "
              f"n_heads={n_heads}, head_dim={head_dim}")
    expected_k_dim = n_kv_heads * head_dim
    if k.shape[-1] != expected_k_dim:
        print(f"[WARN] K dim {k.shape[-1]} != expected {expected_k_dim}. "
              f"n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    # RoPE
    q_r = rope(q.reshape(seq_len, n_heads, head_dim), cos_slice, sin_slice).reshape(seq_len, -1)
    k_r = rope(k.reshape(seq_len, n_kv_heads, head_dim), cos_slice, sin_slice).reshape(seq_len, -1)
    print(f"[OK] RoPE: q_r={q_r.shape}, k_r={k_r.shape}")

    # Store in cache
    kv_k = k_r.reshape(n_kv_heads, seq_len, head_dim)
    kv_v = v.reshape(n_kv_heads, seq_len, head_dim)
    print(f"[OK] KV cache: k={kv_k.shape}, v={kv_v.shape}")

    # Attention
    attn_out = attention(q_r, k_r, v, n_heads, n_kv_heads)
    print(f"[OK] attention -> attn_out: {attn_out.shape}")

    # --- Output projection ---
    o_key = f"{prefix}.attn_output.weight"
    o_w = weights.get(o_key)
    if o_w is None:
        # Try alias
        alt_keys = [k for k in weights if k.startswith(prefix) and 'output' in k]
        print(f"[WARN] {o_key} NOT FOUND. Found: {alt_keys}")
        raise KeyError(f"{o_key} NOT FOUND")

    print(f"[OK] attn_output: {o_w.shape}")
    # Validate o_w shape: should be (hidden_dim, hidden_dim) = (2048, 2048) or (hidden_dim, n_heads*head_dim)
    expected_o_rows = hidden_dim
    if o_w.shape[0] != expected_o_rows:
        print(f"[WARN] attn_output rows {o_w.shape[0]} != hidden_dim {expected_o_rows}")
        print(f"       This will cause matmul mismatch!")

    attn_out2 = validate_matmul(f"{prefix}.attn_output", attn_out, o_w.T)
    print(f"[OK] attn_output matmul -> {attn_out2.shape}")

    x = x + attn_out2
    print(f"[OK] residual x: {x.shape}")

    # --- FFN ---
    ffn_norm_key = f"{prefix}.ffn_norm.weight"
    ffn_norm_w = weights.get(ffn_norm_key)
    if ffn_norm_w is None:
        raise KeyError(f"{ffn_norm_key} NOT FOUND")
    print(f"[OK] ffn_norm: {ffn_norm_w.shape}")

    h_ffn = rms_norm(x, ffn_norm_w, rms_eps)
    print(f"[OK] ffn rms_norm -> h_ffn: {h_ffn.shape}")

    # Check which FFN path
    gate_inp_key = f"{prefix}.ffn_gate_inp.weight"
    has_moe = gate_inp_key in weights

    if has_moe:
        print(f"[MoE] Found gate_inp={gate_inp_key}")
        up_key = f"{prefix}.ffn_up_exps.weight"
        down_key = f"{prefix}.ffn_down_exps.weight"
        gate_exp_key = f"{prefix}.ffn_gate_exps.weight"
        up_w = weights.get(up_key)
        down_w = weights.get(down_key)
        gate_exp_w = weights.get(gate_exp_key)
        print(f"[MoE] up_exps={up_w.shape if up_w is not None else 'MISSING'}")
        print(f"[MoE] down_exps={down_w.shape if down_w is not None else 'MISSING'}")
        print(f"[MoE] gate_exps={gate_exp_w.shape if gate_exp_w is not None else 'MISSING'}")

        if up_w is None or down_w is None:
            raise KeyError(f"MoE weights MISSING! up={up_key}, down={down_key}")
    else:
        gate_key = f"{prefix}.ffn_gate.weight"
        up_key = f"{prefix}.ffn_up.weight"
        down_key = f"{prefix}.ffn_down.weight"
        gate_w = weights.get(gate_key)
        up_w = weights.get(up_key)
        down_w = weights.get(down_key)

        if gate_w is None:
            # Find what gate-like weights exist
            gate_like = sorted(k for k in weights if k.startswith(prefix) and
                             ('gate' in k or 'ffn' in k))
            print(f"[DENSE] {gate_key} NOT FOUND. Available FFN keys: {gate_like}")
            raise KeyError(f"{gate_key} NOT FOUND")

        print(f"[DENSE] gate={gate_w.shape}, up={up_w.shape if up_w is not None else 'MISSING'}, "
              f"down={down_w.shape if down_w is not None else 'MISSING'}")

        if up_w is None or down_w is None:
            raise KeyError(f"Dense FFN weights MISSING!")

        # Dense FFN: gate, up, down
        gate = validate_matmul(f"{prefix}.ffn_gate", h_ffn, gate_w.T)
        up = validate_matmul(f"{prefix}.ffn_up", h_ffn, up_w.T)
        print(f"[OK] gate: {gate.shape}, up: {up.shape}")
        hidden = silu(gate) * up
        print(f"[OK] silu*up -> hidden: {hidden.shape}")

        # Check hidden vs down weight compatibility
        expected_down_in = hidden.shape[-1]
        actual_down_in = down_w.shape[-2]
        if expected_down_in != actual_down_in:
            print(f"[WARN] down weight: expected input dim {expected_down_in}, "
                  f"but down_w rows={down_w.shape[-2]}")
            print(f"       This will cause matmul failure!")

        ffn_out = validate_matmul(f"{prefix}.ffn_down", hidden, down_w.T)
        print(f"[OK] ffn_down -> ffn_out: {ffn_out.shape}")

        x = x + ffn_out
        print(f"[OK] FFN residual x: {x.shape}")

    print(f"\n[SUCCESS] Layer {layer_idx} forward pass completed without errors!")
    print(f"x final shape: {x.shape} (should be (seq={seq_len}, hidden={hidden_dim}))")

    if x.shape == (seq_len, hidden_dim):
        print("[OK] Shape is correct!")
    else:
        print(f"[ERROR] Shape is WRONG! Expected ({seq_len}, {hidden_dim})")

    return x


if __name__ == '__main__':
    import glob
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('model', nargs='?')
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        for pattern in ['models/**/*.gguf', '../models/**/*.gguf', '**/*.gguf']:
            files = glob.glob(pattern, recursive=True)
            if files:
                model_path = files[0]
                break

    if not model_path or not os.path.exists(model_path):
        print("Usage: python diagnose_forward.py /path/to/model.gguf")
        print("No model found.")
        sys.exit(1)

    print(f"Loading model: {model_path}\n")
    try:
        result = load_model(model_path, lazy=False)
        weights = result['weights']
        config = result['config']
        print(f"Loaded {len(weights)} tensors.\n")

        out = run_forward(weights, config)

    except Exception as e:
        print(f"\n[CRASH] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
