// TurboStack Vulkan Kernels — GLSL compute shaders for cross-platform GPU inference
// Compiled to SPIR-V at build time via glslangValidator or glslc

// ─────────────────────────────────────────────
// RotorQuant: 4-bit weight unpack + SO(4) rotation
// ─────────────────────────────────────────────

layout(local_size_x = 256) in;
layout(binding = 0) buffer PackedWeights { uint8_t packed_weights[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) buffer Rotor { float rotor[]; };
layout(binding = 3) uniform Params { uint n; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= n) return;

    uint byte_idx = gid / 2u;
    uint8_t byte_val = packed_weights[byte_idx];
    float val = (gid & 1u) == 1u
        ? float(byte_val & 0x0Fu)
        : float((byte_val >> 4u) & 0x0Fu);

    uint group_start = (gid / 4u) * 4u;
    output_data[gid] = val * rotor[group_start + gid % 4u];
}

// ─────────────────────────────────────────────
// dReLU: TurboSparse activation sparsification
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer Input { float input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) uniform Params { uint n; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= n) return;
    output_data[gid] = max(input_data[gid], 0.0);
}

// ─────────────────────────────────────────────
// SiLU: Sigmoid Linear Unit (x * sigmoid(x))
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer Input { float input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) uniform Params { uint n; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= n) return;
    float x = input_data[gid];
    output_data[gid] = x * (1.0 / (1.0 + exp(-x)));
}

// ─────────────────────────────────────────────
// RMSNorm: Root Mean Square Layer Normalization
// Vectorized (one workgroup per row)
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer Input { float input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) buffer Weight { float weight_data[]; };
layout(binding = 3) uniform Params { float eps; uint dim; };

shared float shared_sum[256];

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint block_dim = gl_WorkGroupSize.x;

    float sum_sq = 0.0;
    uint row_off = row * dim;

    // Each thread processes a chunk of the row
    for (uint i = lid; i < dim; i += block_dim) {
        float v = input_data[row_off + i];
        sum_sq += v * v;
    }

    // Reduce within workgroup
    shared_sum[lid] = sum_sq;
    barrier();
    for (uint s = block_dim / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        barrier();
    }

    float rms = 1.0 / sqrt(shared_sum[0] / float(dim) + eps);

    // Apply normalization
    for (uint i = lid; i < dim; i += block_dim) {
        output_data[row_off + i] = input_data[row_off + i] * rms * weight_data[i];
    }
}

// ─────────────────────────────────────────────
// Matrix Multiply: C = A × B (tiled)
// A: (M, K), B: (K, N), C: (M, N)
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) buffer MatA { float A[]; };
layout(binding = 1) buffer MatB { float B[]; };
layout(binding = 2) buffer MatC { float C[]; };
layout(binding = 3) uniform Params { uint M; uint K; uint N; };

shared float tile_A[16][16];
shared float tile_B[16][16];

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint lid_x = gl_LocalInvocationID.x;
    uint lid_y = gl_LocalInvocationID.y;

    float acc = 0.0;

    for (uint t = 0u; t < (K + 15u) / 16u; t++) {
        // Load tile A
        uint a_col = t * 16u + lid_y;
        if (row < M && a_col < K)
            tile_A[lid_x][lid_y] = A[row * K + a_col];
        else
            tile_A[lid_x][lid_y] = 0.0;

        // Load tile B
        uint b_row = t * 16u + lid_x;
        if (b_row < K && col < N)
            tile_B[lid_x][lid_y] = B[b_row * N + col];
        else
            tile_B[lid_x][lid_y] = 0.0;

        barrier();

        // Compute partial dot product
        for (uint i = 0u; i < 16u; i++) {
            acc += tile_A[lid_x][i] * tile_B[i][lid_y];
        }

        barrier();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ─────────────────────────────────────────────
// Softmax (for attention scores)
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer Input { float input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) uniform Params { uint seq_len; uint num_heads; };

shared float shared_max[256];
shared float shared_sum[256];

void main() {
    uint head = gl_GlobalInvocationID.x / gl_WorkGroupSize.x;
    uint lid  = gl_LocalInvocationID.x;
    uint wgs  = gl_WorkGroupSize.x;

    if (head >= num_heads) return;

    uint off = head * seq_len * seq_len;

    // Each thread processes a row of the attention matrix
    uint row_start = off + lid * seq_len;

    // Find max
    float max_val = input_data[row_start];
    for (uint i = 1u; i < seq_len; i++) {
        max_val = max(max_val, input_data[row_start + i]);
    }
    shared_max[lid] = max_val;
    barrier();

    // Exp and sum
    float sum = 0.0;
    for (uint i = 0u; i < seq_len; i++) {
        output_data[row_start + i] = exp(input_data[row_start + i] - max_val);
        sum += output_data[row_start + i];
    }
    shared_sum[lid] = sum;
    barrier();

    // Normalize
    for (uint i = 0u; i < seq_len; i++) {
        output_data[row_start + i] /= sum;
    }
}

// ─────────────────────────────────────────────
// Rotary Position Embedding (RoPE)
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer Q { float q_data[]; };
layout(binding = 1) buffer K { float k_data[]; };
layout(binding = 2) buffer CosTable { float cos_table[]; };
layout(binding = 3) buffer SinTable { float sin_table[]; };
layout(binding = 4) uniform Params {
    uint head_dim;
    uint num_heads;
    uint position;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= num_heads * head_dim) return;

    uint head = gid / head_dim;
    uint d    = gid % head_dim;
    uint half = head_dim / 2u;

    if (d >= half) return;

    uint off = head * head_dim;
    float cos_val = cos_table[position * half + d];
    float sin_val = sin_table[position * half + d];

    float q0 = q_data[off + d];
    float q1 = q_data[off + d + half];
    q_data[off + d]        = q0 * cos_val - q1 * sin_val;
    q_data[off + d + half] = q1 * cos_val + q0 * sin_val;

    float k0 = k_data[off + d];
    float k1 = k_data[off + d + half];
    k_data[off + d]        = k0 * cos_val - k1 * sin_val;
    k_data[off + d + half] = k1 * cos_val + k0 * sin_val;
}

// ─────────────────────────────────────────────
// Scale + Add (residual connections)
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer A { float a_data[]; };
layout(binding = 1) buffer B { float b_data[]; };
layout(binding = 2) buffer Output { float output_data[]; };
layout(binding = 3) uniform Params { float scale; uint n; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= n) return;
    output_data[gid] = a_data[gid] + scale * b_data[gid];
}

// ─────────────────────────────────────────────
// Predict activations (neuron mask for PowerInfer)
// ─────────────────────────────────────────────

#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer Activations { float activations[]; };
layout(binding = 1) buffer Mask { uint8_t mask_data[]; };
layout(binding = 2) uniform Params { float threshold; uint n; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= n) return;
    mask_data[gid] = activations[gid] > threshold ? 1u : 0u;
}
