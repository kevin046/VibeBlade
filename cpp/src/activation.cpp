#include "kernels.h"
#include "fp16_compat.h"
#include <new>
#include <cmath>

#ifdef TS_AVX512F
#include <immintrin.h>
#elif defined(TS_AVX2)
#include <immintrin.h>
#endif

namespace vibeblade {


// ════════════════════════════════════════════════════════════════
//  SiLU:  out = x * sigmoid(x)
// ════════════════════════════════════════════════════════════════

void silu_f16(const uint16_t* x, uint16_t* out, int N) {
    float* x_f32 = (float*)malloc((size_t)N * sizeof(float));
    float* o_f32 = (float*)malloc((size_t)N * sizeof(float));
    if (!x_f32 || !o_f32) { free(x_f32); free(o_f32); throw std::bad_alloc(); }
    f16_to_f32_batch(x, x_f32, N);

    for (int i = 0; i < N; i++) {
        float sig = 1.0f / (1.0f + expf(-x_f32[i]));
        o_f32[i] = x_f32[i] * sig;
    }

    f32_to_f16_batch(o_f32, out, N);
    free(x_f32);
    free(o_f32);
}

// ════════════════════════════════════════════════════════════════
//  Fused SiLU + multiply:  out = silu(a) * b
//  Used for SwiGLU: silu(x @ W_gate) * (x @ W_up)
// ════════════════════════════════════════════════════════════════

void silu_mul_f16(const uint16_t* a, const uint16_t* b, uint16_t* out, int N) {
    float* a_f32 = (float*)malloc((size_t)N * sizeof(float));
    float* b_f32 = (float*)malloc((size_t)N * sizeof(float));
    float* o_f32 = (float*)malloc((size_t)N * sizeof(float));
    if (!a_f32 || !b_f32 || !o_f32) { free(a_f32); free(b_f32); free(o_f32); throw std::bad_alloc(); }
    f16_to_f32_batch(a, a_f32, N);
    f16_to_f32_batch(b, b_f32, N);

#ifdef TS_AVX512F
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 va = _mm512_loadu_ps(a_f32 + i);
        // sigmoid approximation for AVX-512: use expf path
        // For now, scalar sigmoid per element within vector
        alignas(64) float sig[16];
        for (int j = 0; j < 16; j++)
            sig[j] = 1.0f / (1.0f + expf(-a_f32[i + j]));
        __m512 vs = _mm512_loadu_ps(sig);
        __m512 vb = _mm512_loadu_ps(b_f32 + i);
        __m512 vo = _mm512_mul_ps(_mm512_mul_ps(va, vs), vb);
        _mm512_storeu_ps(o_f32 + i, vo);
    }
    for (; i < N; i++) {
        float sig = 1.0f / (1.0f + expf(-a_f32[i]));
        o_f32[i] = a_f32[i] * sig * b_f32[i];
    }
#elif defined(TS_AVX2)
    int i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 va = _mm256_loadu_ps(a_f32 + i);
        alignas(32) float sig[8];
        for (int j = 0; j < 8; j++)
            sig[j] = 1.0f / (1.0f + expf(-a_f32[i + j]));
        __m256 vs = _mm256_loadu_ps(sig);
        __m256 vb = _mm256_loadu_ps(b_f32 + i);
        __m256 vo = _mm256_mul_ps(_mm256_mul_ps(va, vs), vb);
        _mm256_storeu_ps(o_f32 + i, vo);
    }
    for (; i < N; i++) {
        float sig = 1.0f / (1.0f + expf(-a_f32[i]));
        o_f32[i] = a_f32[i] * sig * b_f32[i];
    }
#else
    for (int i = 0; i < N; i++) {
        float sig = 1.0f / (1.0f + expf(-a_f32[i]));
        o_f32[i] = a_f32[i] * sig * b_f32[i];
    }
#endif

    f32_to_f16_batch(o_f32, out, N);
    free(a_f32);
    free(b_f32);
    free(o_f32);
}

}  // namespace vibeblade
