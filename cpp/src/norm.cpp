#include "kernels.h"
#include "fp16_compat.h"
#include <new>
#include <cmath>
#include <cstring>

#ifdef TS_AVX512F
#include <immintrin.h>
#elif defined(TS_AVX2)
#include <immintrin.h>
#endif

namespace vibeblade {

// Reuse conversion from gemm.cpp — declare extern

// ════════════════════════════════════════════════════════════════
//  RMSNorm:  out = x * weight / sqrt(mean(x^2) + eps)
//  x: (rows, D), weight: (D,), out: (rows, D)
// ════════════════════════════════════════════════════════════════

void rms_norm(const uint16_t* x, const uint16_t* weight, uint16_t* out,
              int rows, int D, float eps) {

    float* x_f32 = (float*)aligned_alloc(64, D * sizeof(float));
    float* w_f32 = (float*)aligned_alloc(64, D * sizeof(float));
    float* o_f32 = (float*)aligned_alloc(64, D * sizeof(float));
    if (!x_f32 || !w_f32 || !o_f32) { free(x_f32); free(w_f32); free(o_f32); throw std::bad_alloc(); }

    // Pre-convert weight to fp32 (constant across rows)
    f16_to_f32_batch(weight, w_f32, D);

    for (int r = 0; r < rows; r++) {
        const uint16_t* xrow = x + r * D;
        uint16_t* orow = out + r * D;

        // Convert row to fp32
        f16_to_f32_batch(xrow, x_f32, D);

        // Compute mean of squares
        float ss = 0.0f;
#ifdef TS_AVX512F
        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            __m512 v = _mm512_loadu_ps(x_f32 + i);
            acc = _mm512_fmadd_ps(v, v, acc);
        }
        ss = _mm512_reduce_add_ps(acc);
        for (; i < D; i++) ss += x_f32[i] * x_f32[i];
#elif defined(TS_AVX2)
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= D; i += 8) {
            __m256 v = _mm256_loadu_ps(x_f32 + i);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        ss = acc[0] + acc[1] + acc[2] + acc[3] +
              acc[4] + acc[5] + acc[6] + acc[7];
        for (; i < D; i++) ss += x_f32[i] * x_f32[i];
#else
        for (int i = 0; i < D; i++) ss += x_f32[i] * x_f32[i];
#endif
        float rms = 1.0f / sqrtf(ss / D + eps);

        // Normalize and scale by weight
        float inv_rms = rms;  // 1/rms for multiply, we want x / rms * w
        // Actually rms = 1/sqrt(...) so we already have the inverse
#ifdef TS_AVX512F
        __m512 v_inv = _mm512_set1_ps(inv_rms);
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            __m512 vx = _mm512_loadu_ps(x_f32 + i);
            __m512 vw = _mm512_loadu_ps(w_f32 + i);
            __m512 vo = _mm512_mul_ps(_mm512_mul_ps(vx, v_inv), vw);
            _mm512_storeu_ps(o_f32 + i, vo);
        }
        for (; i < D; i++) o_f32[i] = x_f32[i] * inv_rms * w_f32[i];
#elif defined(TS_AVX2)
        __m256 v_inv = _mm256_set1_ps(inv_rms);
        int i = 0;
        for (; i + 8 <= D; i += 8) {
            __m256 vx = _mm256_loadu_ps(x_f32 + i);
            __m256 vw = _mm256_loadu_ps(w_f32 + i);
            __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, v_inv), vw);
            _mm256_storeu_ps(o_f32 + i, vo);
        }
        for (; i < D; i++) o_f32[i] = x_f32[i] * inv_rms * w_f32[i];
#else
        for (int i = 0; i < D; i++) o_f32[i] = x_f32[i] * inv_rms * w_f32[i];
#endif

        f32_to_f16_batch(o_f32, orow, D);
    }

    free(x_f32);
    free(w_f32);
    free(o_f32);
}

}  // namespace vibeblade
