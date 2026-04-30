#!/usr/bin/env python3
"""Pre-process F16 GGUF model: apply Hadamard H4/2 to weight columns.

Math: For weight W (out_features, in_features), rotate columns by H4/2:
  W_rot = W @ H4/2   (each group of 4 columns mixed)

At runtime, apply H4/2 to activations:
  y = W_rot @ (H4/2 @ x) = W @ H4/2 @ H4/2 @ x = W @ I @ x = W @ x

This is mathematically exact in F32. The quantization step F16 -> Q4_K
introduces rounding error, but it's a SINGLE quantization pass (same as
normal), just with better rounding because H4 spreads outliers before
quantization blocks are formed.

Usage:
  python rotorquant_preprocess.py input_f16.gguf output_f16_rotated.gguf
"""

import sys
import struct
import numpy as np

# Hadamard H4 / 2 = H4 * 0.5
# H4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
# H4/2 * H4/2 = I (self-inverse)
H4_OVER_2 = np.array([
    [ 0.5,  0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5, -0.5],
    [ 0.5,  0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5,  0.5],
], dtype=np.float32)


def hadamard4_rotate_rows(data: np.ndarray) -> np.ndarray:
    """Apply H4/2 to groups of 4 rows (input features).

    In GGUF/ggml, weight shape is [ne0, ne1] = [input_features, output_features].
    ggml_mul_mat(W, x) computes W^T @ x, so rows = input features.
    
    We LEFT-multiply groups of 4 rows by H4/2 to mix input features.
    At runtime, we apply H4/2 to activations x:
      (H4/2 · W)^T · (H4/2 · x) = W^T · (H4/2)(H4/2) · x = W^T · x
    """
    in_features, out_features = data.shape
    # Pad in_features to multiple of 4
    n_groups = (in_features + 3) // 4
    padded_in = n_groups * 4

    if padded_in != in_features:
        padded = np.zeros((padded_in, out_features), dtype=np.float32)
        padded[:in_features, :] = data
        data = padded

    # Reshape to (n_groups, 4, out_features) and apply H4/2
    # H4/2 @ each (4, out_features) group → new (4, out_features)
    grouped = data.reshape(n_groups, 4, out_features)
    # np.matmul broadcasts batch dims: (4,4) @ (n_groups, 4, out_features) fails
    # Need: (1, 4, 4) @ (n_groups, 4, out_features) -> (n_groups, 4, out_features)
    rotated = np.matmul(H4_OVER_2[np.newaxis, :, :], grouped)

    result = rotated.reshape(padded_in, out_features)
    return result[:in_features, :]


def read_gguf_tensor_data(reader, tensor_info):
    """Read raw tensor data from GGUF, return as numpy array."""
    import gguf

    # Get the tensor data offset
    data_offset = tensor_info.data_offset
    n_bytes = tensor_info.n_bytes

    with open(reader._path, 'rb') as f:
        f.seek(data_offset)
        raw = f.read(n_bytes)

    return raw


def get_tensor_shape_for_columns(tensor_name: str) -> bool:
    """Return True if this tensor should have its columns rotated.

    Weight tensors in llama.cpp have shape (out_features, in_features) 
    in row-major order, where in_features is the last dimension.
    We want to rotate groups of columns (input features).
    
    Skip 1D tensors (biases, norms, etc.) and embedding/output tensors
    that have different semantics.
    """
    # Rotate all 2D+ weight tensors
    # Skip: attention norm, ffn norm, output norm (1D)
    # Skip: token embedding (conceptually 2D but rotating input features
    #        of embeddings would require also rotating the input tokens)
    
    skip_patterns = [
        'norm', 'bias', 'output_norm', 'attn_norm', 'ffn_norm',
    ]
    # token_embd.weight: shape (vocab, dim) — rotating columns changes
    # token semantics (embedding lookup by ID). Skip it.
    force_skip = ['token_embd.weight']
    if tensor_name in force_skip:
        return False
    for pat in skip_patterns:
        if pat in tensor_name.lower():
            return False
    
    return True


def preprocess_model(input_path: str, output_path: str):
    """Read F16 GGUF, rotate weight columns by H4/2, write new GGUF."""
    import gguf

    print(f"Reading {input_path}...")
    reader = gguf.GGUFReader(input_path)

    # Collect all tensor info
    tensors = []
    for t in reader.tensors:
        tensors.append({
            'name': t.name,
            'shape': t.shape,
            'n_bytes': t.n_bytes,
            'data_offset': t.data_offset,
            'tensor_type': t.tensor_type,
        })

    print(f"Found {len(tensors)} tensors")
    for t in tensors:
        print(f"  {t['name']:50s} shape={t['shape']} type={t['tensor_type']} size={t['n_bytes']}")

    # Copy the file first (GGUF header + metadata)
    # Then patch the tensor data in place
    import shutil
    shutil.copy2(input_path, output_path)

    # Track total bytes processed
    total_rotated = 0

    with open(output_path, 'r+b') as f:
        for t in tensors:
            name = t['name']
            shape = t['shape']
            n_bytes = t['n_bytes']
            offset = t['data_offset']

            if not get_tensor_shape_for_columns(name):
                print(f"  SKIP {name}")
                continue

            if len(shape) < 2:
                print(f"  SKIP (1D) {name}")
                continue

            # GGUF data layout: data[j*ne0 + i0] = tensor[i0, i1]
            # ne0 is CONTIGUOUS (input features for mul_mat)
            # ne1 is STRIDE (output features for mul_mat)
            # 
            # ggml_mul_mat computes: C[i1] = sum_{i0} W[i0, i1] * x[i0]
            # We want to mix groups of 4 input features (i0 = ne0 dimension)
            # 
            # In numpy: reshape as (ne1, ne0), then right-multiply each row's
            # groups of 4 elements by (H4/2)^T = H4/2 (symmetric)
            in_features = shape[0]   # ne0 (contiguous, input features)
            out_features = 1
            for s in shape[1:]:
                out_features *= s   # ne1 (stride, output features)

            # Check it's F16 (GGUF type 1)
            if t['tensor_type'] != 1:  # F16
                print(f"  SKIP (type={t['tensor_type']}, not F16) {name}")
                continue

            if in_features % 4 != 0:
                print(f"  SKIP (ne0={in_features} not div by 4) {name}")
                continue

            # Read F16 data
            f.seek(offset)
            raw = f.read(n_bytes)
            data = np.frombuffer(raw, dtype=np.float16).astype(np.float32)

            print(f"  ROTATE {name:50s} (ne0={in_features}, ne1={out_features})...", end=' ')

            # Reshape as (ne1, ne0) — each row = one output feature
            data_2d = data.reshape(out_features, in_features)
            n_groups = in_features // 4
            # Reshape to (out_features, n_groups, 4) and right-multiply by H4/2
            grouped = data_2d.reshape(out_features, n_groups, 4)
            rotated = np.matmul(grouped, H4_OVER_2.T)  # (out_features, n_groups, 4) @ (4, 4)
            rotated_flat = rotated.reshape(-1)

            # Convert back to F16 and write
            f.seek(offset)
            f.write(rotated_flat.astype(np.float16).tobytes())

            total_rotated += 1
            print("done")

    print(f"\nRotated {total_rotated} tensors")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input_f16.gguf output_f16_rotated.gguf")
        sys.exit(1)

    preprocess_model(sys.argv[1], sys.argv[2])
