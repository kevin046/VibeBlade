#pragma once
// FP16 ↔ FP32 conversion utilities for VibeBlade native kernels.
// Provides vectorized (AVX-512/AVX2) and scalar paths.

#include <cstdint>

namespace vibeblade {

// Convert a single IEEE 754 fp16 (as uint16_t) to float32
float f16_to_f32(uint16_t h);

// Convert a single float32 to IEEE 754 fp16 (as uint16_t)
uint16_t f32_to_f16(float fv);

// Batch convert fp16 → fp32 (vectorized when SIMD available)
void f16_to_f32_batch(const uint16_t* src, float* dst, int n);

// Batch convert fp32 → fp16 (vectorized when SIMD available)
void f32_to_f16_batch(const float* src, uint16_t* dst, int n);

}  // namespace vibeblade
