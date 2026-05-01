#include "kernels.h"
#include "fp16_compat.h"
#include "win_compat.h"
#include <new>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace vibeblade {


// ════════════════════════════════════════════════════════════════
//  Fused SDPA:  O = softmax(Q @ K^T / scale) @ V
//  Q: (M, d), K: (N, d), V: (N, d), O: (M, d)
//
//  Uses online softmax algorithm (flash-attention style):
//    For each query row, iterate over K/V rows and maintain
//    running max and sum -- O(d) memory per query row, not O(N*d).
// ════════════════════════════════════════════════════════════════

void fused_sdpa(const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                uint16_t* O, int M, int N, int d, float scale) {

    float* q_row = (float*)aligned_alloc(64, d * sizeof(float));
    float* k_row = (float*)aligned_alloc(64, d * sizeof(float));
    float* v_row = (float*)aligned_alloc(64, d * sizeof(float));
    float* o_row = (float*)aligned_alloc(64, d * sizeof(float));
    float* K_f32 = (float*)aligned_alloc(64, (size_t)N * d * sizeof(float));
    float* V_f32 = (float*)aligned_alloc(64, (size_t)N * d * sizeof(float));
    if (!q_row || !k_row || !v_row || !o_row || !K_f32 || !V_f32) {
        aligned_free(q_row); aligned_free(k_row); aligned_free(v_row);
        aligned_free(o_row); aligned_free(K_f32); aligned_free(V_f32);
        throw std::bad_alloc();
    }
    f16_to_f32_batch(K, K_f32, N * d);
    f16_to_f32_batch(V, V_f32, N * d);

    for (int m = 0; m < M; m++) {
        // Load query row
        f16_to_f32_batch(Q + m * d, q_row, d);

        // Online softmax state
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        memset(o_row, 0, d * sizeof(float));

        for (int n = 0; n < N; n++) {
            const float* k = K_f32 + n * d;
            const float* v = V_f32 + n * d;

            // Compute dot(Q[m], K[n]) * scale
            float dot = 0.0f;
            int i;
#ifdef TS_AVX512F
            {
                __m512 acc_v = _mm512_setzero_ps();
                i = 0;
                for (; i + 16 <= d; i += 16) {
                    __m512 vq = _mm512_loadu_ps(q_row + i);
                    __m512 vk = _mm512_loadu_ps(k + i);
                    acc_v = _mm512_fmadd_ps(vq, vk, acc_v);
                }
                dot = _mm512_reduce_add_ps(acc_v);
                for (; i < d; i++) dot += q_row[i] * k[i];
            }
#elif defined(TS_AVX2)
            {
                __m256 acc_v = _mm256_setzero_ps();
                i = 0;
                for (; i + 8 <= d; i += 8) {
                    __m256 vq = _mm256_loadu_ps(q_row + i);
                    __m256 vk = _mm256_loadu_ps(k + i);
                    acc_v = _mm256_fmadd_ps(vq, vk, acc_v);
                }
                dot = acc_v[0] + acc_v[1] + acc_v[2] + acc_v[3] +
                      acc_v[4] + acc_v[5] + acc_v[6] + acc_v[7];
                for (; i < d; i++) dot += q_row[i] * k[i];
            }
#else
            for (i = 0; i < d; i++) dot += q_row[i] * k[i];
#endif
            float attn = dot * scale;

            // Online softmax update
            float new_max = std::max(row_max, attn);
            float exp_diff = expf(row_max - new_max);
            float exp_attn = expf(attn - new_max);

            // Rescale running accumulator
            if (row_max != -INFINITY) {
                for (i = 0; i < d; i++) o_row[i] *= exp_diff;
            }

            // Accumulate V weighted by attention
            for (i = 0; i < d; i++) o_row[i] += exp_attn * v[i];

            row_sum = row_sum * exp_diff + exp_attn;
            row_max = new_max;
        }

        // Normalize
        if (row_sum > 0.0f) {
            float inv_sum = 1.0f / row_sum;
            for (i = 0; i < d; i++) o_row[i] *= inv_sum;
        }

        // Store output
        f32_to_f16_batch(o_row, O + m * d, d);
    }

    aligned_free(q_row);
    aligned_free(k_row);
    aligned_free(v_row);
    aligned_free(o_row);
    aligned_free(K_f32);
    aligned_free(V_f32);
}

}  // namespace vibeblade
