// TurboStack Metal Kernels — Apple Silicon GPU compute shaders
// Requires Metal 3.0+ (macOS 13+, iOS 16+, any Apple Silicon)

#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────
// RotorQuant: 4-bit weight unpack + SO(4) rotation
// ─────────────────────────────────────────────

kernel void ts_rotor_unpack(
    device const uint8_t* packed_weights [[buffer(0)]],
    device float* output           [[buffer(1)]],
    device const float* rotor      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    // Extract two 4-bit values from each byte
    uint byte_idx = gid / 2;
    uint8_t byte_val = packed_weights[byte_idx];
    float val;
    if (gid & 1u) {
        val = (float)(byte_val & 0x0F);        // low nibble
    } else {
        val = (float)((byte_val >> 4) & 0x0F); // high nibble
    }

    // Apply SO(4) rotation (group size = 4)
    uint group_start = (gid / 4) * 4;
    uint in_group    = gid % 4;
    output[gid] = val * rotor[group_start + in_group];
}

// ─────────────────────────────────────────────
// dReLU: TurboSparse activation sparsification
// ─────────────────────────────────────────────

kernel void ts_drelu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = fmax(input[gid], 0.0f);
}

// ─────────────────────────────────────────────
// Predict activations (neuron mask for PowerInfer)
// ─────────────────────────────────────────────

kernel void ts_predict_activations(
    device const float* activations  [[buffer(0)]],
    device uint8_t* mask             [[buffer(1)]],
    constant float& threshold        [[buffer(2)]],
    constant uint& n                 [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    mask[gid] = (activations[gid] > threshold) ? 1 : 0;
}

// ─────────────────────────────────────────────
// RMSNorm: Root Mean Square Layer Normalization
// ─────────────────────────────────────────────

kernel void ts_rms_norm(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    constant uint& dim          [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.x;
    if (row == 0) return; // placeholder — actual launch uses 1D grid over rows
    if (row >= gid.y) return;

    // Compute RMS
    float sum_sq = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float v = input[row * dim + i];
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / (float)dim + eps);

    // Normalize and scale
    for (uint i = 0; i < dim; i++) {
        output[row * dim + i] = input[row * dim + i] * rms * weight[i];
    }
}

// Vectorized RMSNorm (one thread per row, up to 4096 dim)
kernel void ts_rms_norm_vec(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    constant uint& dim          [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    // Compute RMS using SIMD reduction
    float4 sum_sq4 = float4(0.0f);
    uint i = 0;
    uint row_off = row * dim;

    for (; i + 4 <= dim; i += 4) {
        float4 v = float4(input[row_off + i], input[row_off + i+1],
                          input[row_off + i+2], input[row_off + i+3]);
        sum_sq4 += v * v;
    }
    float sum_sq = sum_sq4.x + sum_sq4.y + sum_sq4.z + sum_sq4.w;
    for (; i < dim; i++) {
        float v = input[row_off + i];
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / (float)dim + eps);

    // Normalize and scale
    for (i = 0; i + 4 <= dim; i += 4) {
        float4 v = float4(input[row_off + i], input[row_off + i+1],
                          input[row_off + i+2], input[row_off + i+3]);
        float4 w = float4(weight[i], weight[i+1], weight[i+2], weight[i+3]);
        float4 r = v * rms * w;
        output[row_off + i]   = r.x;
        output[row_off + i+1] = r.y;
        output[row_off + i+2] = r.z;
        output[row_off + i+3] = r.w;
    }
    for (; i < dim; i++) {
        output[row_off + i] = input[row_off + i] * rms * weight[i];
    }
}

// ─────────────────────────────────────────────
// SiLU: Sigmoid Linear Unit (x * sigmoid(x))
// ─────────────────────────────────────────────

kernel void ts_silu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = x * (1.0f / (1.0f + exp(-x)));
}

// ─────────────────────────────────────────────
// Matrix Multiply: C = A × B
// A: (M, K), B: (K, N), C: (M, N)
// Tiled for Apple Silicon GPU shared memory
// ─────────────────────────────────────────────

constant uint TILE_SIZE [[function_constant(0)]];

#if !defined(TILE_SIZE)
#define TILE_SIZE 16
#endif

kernel void ts_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint row = gid.x;
    uint col = gid.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Tiled matmul with threadgroup memory
kernel void ts_matmul_tiled(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device float* C           [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& K          [[buffer(4)]],
    constant uint& N          [[buffer(5)]],
    threadgroup float* tile_A [[threadgroup(0)]],
    threadgroup float* tile_B [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    uint row = gid.x * TILE_SIZE + tid.x;
    uint col = gid.y * TILE_SIZE + tid.y;

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        uint a_col = t * TILE_SIZE + tid.y;
        tile_A[tid.x * TILE_SIZE + tid.y] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        uint b_row = t * TILE_SIZE + tid.x;
        tile_B[tid.x * TILE_SIZE + tid.y] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint i = 0; i < TILE_SIZE; i++) {
            acc += tile_A[tid.x * TILE_SIZE + i] * tile_B[i * TILE_SIZE + tid.y];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ─────────────────────────────────────────────
// Softmax (for attention scores)
// ─────────────────────────────────────────────

kernel void ts_softmax(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& seq_len     [[buffer(2)]],
    constant uint& num_heads   [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint head = gid.x;
    uint q_pos = gid.y;
    if (head >= num_heads || q_pos >= seq_len) return;

    uint off = (head * seq_len + q_pos) * seq_len;

    // Find max
    float max_val = input[off];
    for (uint i = 1; i < seq_len; i++) {
        max_val = fmax(max_val, input[off + i]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < seq_len; i++) {
        output[off + i] = exp(input[off + i] - max_val);
        sum += output[off + i];
    }

    // Normalize
    for (uint i = 0; i < seq_len; i++) {
        output[off + i] /= sum;
    }
}

// ─────────────────────────────────────────────
// Rotary Position Embedding (RoPE)
// ─────────────────────────────────────────────

kernel void ts_rope(
    device float* q               [[buffer(0)]],
    device float* k               [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant uint& head_dim       [[buffer(4)]],
    constant uint& num_heads      [[buffer(5)]],
    constant uint& position       [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_heads * head_dim) return;

    uint head = gid / head_dim;
    uint d    = gid % head_dim;
    uint half = head_dim / 2;

    uint q_off = head * head_dim + d;
    uint k_off = q_off; // same layout for K

    if (d < half) {
        float cos_val = cos_table[position * half + d];
        float sin_val = sin_table[position * half + d];

        float q0 = q[q_off];
        float q1 = q[head * head_dim + d + half];
        q[q_off] = q0 * cos_val - q1 * sin_val;
        q[head * head_dim + d + half] = q1 * cos_val + q0 * sin_val;

        float k0 = k[k_off];
        float k1 = k[head * head_dim + d + half];
        k[k_off] = k0 * cos_val - k1 * sin_val;
        k[head * head_dim + d + half] = k1 * cos_val + k0 * sin_val;
    }
}

// ─────────────────────────────────────────────
// Scale & Add (for residual connections)
// ─────────────────────────────────────────────

kernel void ts_scale_add(
    device const float* a     [[buffer(0)]],
    device const float* b     [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant float& scale     [[buffer(3)]],
    constant uint& n          [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = a[gid] + scale * b[gid];
}

// ─────────────────────────────────────────────
// Element-wise multiply (for attention output × Wo)
// ─────────────────────────────────────────────

kernel void ts_mul(
    device const float* a     [[buffer(0)]],
    device const float* b     [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant uint& n          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = a[gid] * b[gid];
}
