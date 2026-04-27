#pragma once
// VibeBlade Native Kernels — cross-platform SIMD-optimized inference primitives
//
// Layout convention: row-major (C order), matching NumPy
// FP16 uses IEEE 754 half (binary-compatible with numpy.float16 via uint16_t)

#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace vibeblade {

// ──────────────────────────── GEMM ────────────────────────────
// C = alpha * A @ B + beta * C
// A: (M, K), B: (K, N), C: (M, N) — all row-major float16
// Operates in FP32 internally for numerical stability.
void gemm_f16(const uint16_t* A, const uint16_t* B, uint16_t* C,
              int M, int K, int N,
              float alpha = 1.0f, float beta = 0.0f);

// ──────────────────────────── RMSNorm ─────────────────────────
// out = x * weight / sqrt(mean(x^2) + eps)
// x: (rows, D), weight: (D,), out: (rows, D) — all float16
void rms_norm(const uint16_t* x, const uint16_t* weight, uint16_t* out,
              int rows, int D, float eps = 1e-5f);

// ──────────────────────── Activations ─────────────────────────
// SiLU: out = x * sigmoid(x)  — element-wise
// x: (N,), out: (N,) — float16
void silu_f16(const uint16_t* x, uint16_t* out, int N);

// Fused SiLU + element-wise multiply: out = silu(a) * b
// a: (N,), b: (N,), out: (N,) — float16 (SwiGLU gate)
void silu_mul_f16(const uint16_t* a, const uint16_t* b, uint16_t* out, int N);

// ───────────────────── Quantization ───────────────────────────
// Quantize to 2-bit: packs 4 values per byte
// x: (S, D) row-major float16
// axis=0 → per-token (reduce over D): scale_out shape (S,), min_out shape (S,)
// axis=1 → per-channel (reduce over S): scale_out shape (D,), min_out shape (D,)
// Returns number of output bytes (= S*D/4)
int quantize_2bit(const uint16_t* x, uint8_t* packed, int S, int D,
                  int axis, float* scale_out, float* min_out);

// Dequantize from 2-bit packed
// packed: (S*D/4 bytes), scale/min as above
void dequantize_2bit(const uint8_t* packed, uint16_t* out, int S, int D,
                     int axis, const float* scale, const float* min_val);

// Quantize to 4-bit: packs 2 values per byte
int quantize_4bit(const uint16_t* x, uint8_t* packed, int S, int D,
                  int axis, float* scale_out, float* min_out);

// Dequantize from 4-bit packed
void dequantize_4bit(const uint8_t* packed, uint16_t* out, int S, int D,
                     int axis, const float* scale, const float* min_val);

// Quantize to 8-bit (symmetric): out[i] = round(x[i] / scale)
// Returns scale (scalar)
float quantize_8bit_sym(const uint16_t* x, int8_t* out, int N, float max_abs);

// Dequantize from 8-bit symmetric
void dequantize_8bit_sym(const int8_t* x, uint16_t* out, int N, float scale);

// ──────────────────────── Attention ───────────────────────────
// Fused SDPA: O = softmax(Q @ K^T / scale) @ V
// Uses online softmax — O(M*N) memory, not O(M*N*d)
// Q: (M, d), K: (N, d), V: (N, d), O: (M, d) — all float16
void fused_sdpa(const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                uint16_t* O, int M, int N, int d, float scale);

// ──────────────────────── RoPE ────────────────────────────────
// Apply rotary positional embeddings in-place
// x: (seq_len, num_heads, head_dim) — float16
// freqs: (seq_len, head_dim/2) — float32 cos/sin pairs interleaved [cos, sin, cos, sin, ...]
void apply_rope(uint16_t* x, const float* freqs, int seq_len, int num_heads, int head_dim);

// ──────────────────────── Softmax ─────────────────────────────
// Row-wise softmax in-place
// x: (rows, cols) — float32
void softmax_f32(float* x, int rows, int cols);

}  // namespace vibeblade
