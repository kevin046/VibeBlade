#!/usr/bin/env python3
"""Debug GGUF data layout and verify rotation round-trip."""

import numpy as np
import gguf

path = "/home/ubuntu/VibeBlade/models/qwen2.5-0.5b-instruct-fp16.gguf"
reader = gguf.GGUFReader(path)

H4_OVER_2 = np.array([
    [ 0.5,  0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5, -0.5],
    [ 0.5,  0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5,  0.5],
], dtype=np.float32)

# Pick a small tensor for testing
for t in reader.tensors:
    if t.name == 'blk.0.attn_q.weight':
        shape = t.shape  # [ne0, ne1]
        n_bytes = t.n_bytes
        offset = t.data_offset
        ne0, ne1 = shape[0], shape[1]
        print(f"Tensor: {t.name}")
        print(f"  GGUF shape: ne0={ne0}, ne1={ne1}")
        print(f"  n_bytes={n_bytes}, offset={offset}")
        
        # Read raw data
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = f.read(n_bytes)
        data = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        print(f"  data shape: {data.shape} ({len(data)} elements)")
        print(f"  Expected: {ne0 * ne1} elements")
        print(f"  First 8 values: {data[:8]}")
        print(f"  Values at [0, ne0]: {data[0:4]}")
        
        # GGUF layout: data[i1 * ne0 + i0] = tensor[i0, i1]
        # So ne0 is contiguous. In numpy reshape: (ne1, ne0) = (out, in) for row-major
        
        # Let's verify by trying both interpretations
        # Interpretation A: data as (ne0, ne1) = what we're doing now
        A = data.reshape(ne0, ne1)
        print(f"\n  Interpretation A: reshape({ne0}, {ne1})")
        print(f"    A[0,:4] = {A[0,:4]}")
        print(f"    A[1,:4] = {A[1,:4]}")
        
        # Interpretation B: data as (ne1, ne0)
        B = data.reshape(ne1, ne0)
        print(f"\n  Interpretation B: reshape({ne1}, {ne0})")
        print(f"    B[0,:4] = {B[0,:4]}")
        print(f"    B[1,:4] = {B[1,:4]}")
        
        # Test: what are the "columns" of the weight matrix?
        # For attn_q.weight [896, 896], this maps hidden→Q (896→896)
        # In ggml_mul_mat: result[j] = sum_i W[i,j] * x[i]
        # Where W[i,j] = data[j * ne0 + i] (i=input, j=output)
        # So data[j*896 + i] = weight from input i to output j
        # data[0:896] = column 0 of the weight matrix = weights from all inputs to output 0
        # data[896:1792] = column 1 = weights from all inputs to output 1
        
        print(f"\n  GGUF layout analysis:")
        print(f"    data[0] = W[0,0] = weight(input=0, output=0)")
        print(f"    data[895] = W[895,0] = weight(input=895, output=0)")
        print(f"    data[896] = W[0,1] = weight(input=0, output=1)")
        print(f"    So data[j*ne0 + i] = W[i,j]")
        print(f"    This means ne0 is CONTIGUOUS (input features)")
        print(f"    And ne1 is the STRIDE dimension (output features)")
        
        # For rotation: we want to mix groups of 4 INPUT features
        # Input features are indexed by i0 (ne0), stored contiguously
        # Groups of 4 input features: {i0, i0+1, i0+2, i0+3}
        # For each output j: W_rot[i0', j] = sum_k (H/2)[i0',k] * W[k,j]
        
        # In data layout: W[i,j] = data[j*ne0 + i]
        # W_rot[i',j] = sum_k (H/2)[i',k] * data[j*ne0 + k]
        # data_rot[j*ne0 + i'] = sum_k (H/2)[i',k] * data[j*ne0 + k]
        
        # This is: for each contiguous block of ne0 elements (= one column of W),
        # apply H/2 to groups of 4 consecutive elements within the block.
        
        # In numpy with data reshaped as (ne1, ne0) = (ne1 blocks of ne0 elements):
        # Each row = one column of W = ne0 input weights for one output
        # Apply H/2 to groups of 4 elements within each row
        # This is RIGHT-multiplication: row @ (H/2)^T, but only within groups of 4
        
        print(f"\n  CORRECT rotation approach:")
        print(f"    Reshape as ({ne1}, {ne0}) — each row is one output feature")
        print(f"    Mix groups of 4 elements (input features) within each row")
        print(f"    This is RIGHT-multiply by H4/2 within non-overlapping groups of 4")
        
        # Implement correct rotation
        data_2d = data.reshape(ne1, ne0)  # (ne1, ne0)
        n_groups = ne0 // 4
        # Reshape to (ne1, n_groups, 4) and right-multiply each (4,) group by (H4/2)^T
        grouped = data_2d.reshape(ne1, n_groups, 4)
        # (ne1, n_groups, 4) @ (4, 4) = (ne1, n_groups, 4)
        rotated = np.matmul(grouped, H4_OVER_2.T)
        rotated_flat = rotated.reshape(-1)
        
        # Verify round-trip: apply H4/2 twice should give identity
        rotated2 = np.matmul(rotated.reshape(ne1, n_groups, 4), H4_OVER_2.T)
        rotated2_flat = rotated2.reshape(-1)
        
        diff = np.max(np.abs(data - rotated2_flat))
        print(f"\n  Round-trip check: max|data - H4/2(H4/2(data))| = {diff:.2e}")
        if diff < 1e-5:
            print(f"  ✓ H4/2 is self-inverse — rotation is correct!")
        else:
            print(f"  ✗ H4/2 is NOT self-inverse — BUG in rotation code!")
        
        # Compare with the approach used in preprocessing (wrong)
        data_2d_wrong = data.reshape(ne0, ne1)  # WRONG reshape
        n_groups_wrong = ne0 // 4
        grouped_wrong = data_2d_wrong.reshape(n_groups_wrong, 4, ne1)
        rotated_wrong = np.matmul(H4_OVER_2[np.newaxis, :, :], grouped_wrong)
        rotated_wrong_flat = rotated_wrong.reshape(-1)
        
        same = np.allclose(rotated_flat, rotated_wrong_flat)
        print(f"\n  Are the two approaches the same? {same}")
        if not same:
            print(f"  Max diff: {np.max(np.abs(rotated_flat - rotated_wrong_flat)):.4f}")
            print(f"  → The current preprocessing code uses the WRONG reshape!")
        
        break
