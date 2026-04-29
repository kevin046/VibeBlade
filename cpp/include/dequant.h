#pragma once
// VibeBlade dequantization kernels — row-wise block dequant.
// Each function dequantizes a row of quantized data into fp32 output.

#include "ggml_types.h"
#include <cstddef>

namespace vibeblade {

// Row-wise dequantization: dequantize n_values from quantized row → fp32 out
void dequantize_row(const void* row, float* out, int64_t n_values, ggml_type type);

// Per-type dequant functions
void dequantize_row_q4_0(const void* row, float* out, int64_t n);
void dequantize_row_q4_1(const void* row, float* out, int64_t n);
void dequantize_row_q5_0(const void* row, float* out, int64_t n);
void dequantize_row_q5_1(const void* row, float* out, int64_t n);
void dequantize_row_q8_0(const void* row, float* out, int64_t n);
void dequantize_row_q4_K(const void* row, float* out, int64_t n);
void dequantize_row_q5_K(const void* row, float* out, int64_t n);
void dequantize_row_q6_K(const void* row, float* out, int64_t n);

// Matrix-vector multiply with inline dequantization:
// GGUF stores weights as (K, N) — K rows of N elements.
// Computes: out[n] = sum_k(x[k] * W[k][n]) for n = 0..N-1
// scratch: caller-provided buffer of at least N floats for row dequantization.
void gemv_dequant(
 const float* x,    // (K,) input vector
 const void* weights, // quantized weight matrix in GGUF (K, N) layout
 float* out,        // (N,) output
 int64_t K,         // input dimension (= GGUF rows)
 int64_t N,         // output dimension (= GGUF cols)
 ggml_type wtype,   // quantization type of weights
 float* scratch     // (N,) temp buffer for row dequantization
);

// Multi-threaded gemv_dequant — falls back to single-threaded for now.
// TODO: per-thread partial outputs for proper MT with GGUF layout.
void gemv_dequant_mt(
 const float* x,
 const void* weights,
 float* out,
 int64_t K,
 int64_t N,
 ggml_type wtype,
 int n_threads,     // 0 = auto-detect
 float* scratch     // (N,) temp buffer
);

} // namespace vibeblade