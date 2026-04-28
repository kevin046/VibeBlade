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
// out[j] = sum_i(x[i] * dequant(weight_row(j)[i]))  for each output row j
// x: (1, K) fp32, weight: (N, K) quantized, out: (N,) fp32
// This is the hottest function in decode — dequant directly into dot product.
void gemv_dequant(
    const float* x,          // (K,) input vector
    const void* weights,     // quantized weight matrix, row-major
    float* out,              // (N,) output
    int64_t K,               // input dimension
    int64_t N,               // output dimension
    ggml_type wtype          // quantization type of weights
);

}  // namespace vibeblade
