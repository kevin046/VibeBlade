#include "kernels.h"
#include "fp16_compat.h"
#include <new>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef TS_AVX512F
#include <immintrin.h>
#endif

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  GEMM:  C = alpha * A @ B + beta * C
//  A: (M, K), B: (K, N), C: (M, N)  row-major, float16 storage
//  Strategy: convert rows of A to fp32, dot-product against columns of B
// ════════════════════════════════════════════════════════════════

void gemm_f16(const uint16_t* A, const uint16_t* B, uint16_t* C,
              int M, int K, int N,
              float alpha, float beta) {

    // Allocate fp32 buffers for a row of A and a row of C
    float* a_row = (float*)aligned_alloc(64, K * sizeof(float));
    float* b_col = (float*)aligned_alloc(64, K * sizeof(float));
    float* c_row = (float*)aligned_alloc(64, N * sizeof(float));
    if (!a_row || !b_col || !c_row) { free(a_row); free(b_col); free(c_row); throw std::bad_alloc(); }

    for (int m = 0; m < M; m++) {
        // Convert row m of A to fp32
        f16_to_f32_batch(A + m * K, a_row, K);

        // Initialize c_row
        if (beta != 0.0f) {
            f16_to_f32_batch(C + m * N, c_row, N);
            for (int j = 0; j < N; j++) c_row[j] *= beta;
        } else {
            memset(c_row, 0, N * sizeof(float));
        }

        for (int n = 0; n < N; n++) {
            // Load column n of B (stride = N) into b_col
            for (int k = 0; k < K; k++) {
                b_col[k] = f16_to_f32(B[k * N + n]);
            }

            // Dot product
            float sum = 0.0f;
#ifdef TS_AVX512F
            int k = 0;
            __m512 acc = _mm512_setzero_ps();
            for (; k + 16 <= K; k += 16) {
                __m512 va = _mm512_loadu_ps(a_row + k);
                __m512 vb = _mm512_loadu_ps(b_col + k);
                acc = _mm512_fmadd_ps(va, vb, acc);
            }
            sum = _mm512_reduce_add_ps(acc);
            for (; k < K; k++) sum += a_row[k] * b_col[k];
#elif defined(TS_AVX2)
            int k = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; k + 8 <= K; k += 8) {
                __m256 va = _mm256_loadu_ps(a_row + k);
                __m256 vb = _mm256_loadu_ps(b_col + k);
                acc = _mm256_fmadd_ps(va, vb, acc);
            }
            sum = acc[0] + acc[1] + acc[2] + acc[3] +
                  acc[4] + acc[5] + acc[6] + acc[7];
            for (; k < K; k++) sum += a_row[k] * b_col[k];
#else
            for (int k = 0; k < K; k++) sum += a_row[k] * b_col[k];
#endif
            c_row[n] += alpha * sum;
        }

        // Convert c_row back to fp16
        f32_to_f16_batch(c_row, C + m * N, N);
    }

    free(a_row);
    free(b_col);
    free(c_row);
}

}  // namespace vibeblade
