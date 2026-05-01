#include "kernels.h"
#include "fp16_compat.h"
#include "win_compat.h"
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
    if (!a_row || !b_col || !c_row) { aligned_free(a_row); aligned_free(b_col); aligned_free(c_row); throw std::bad_alloc(); }

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
            for (int kk = 0; kk < K; kk++) {
                b_col[kk] = f16_to_f32(B[kk * N + n]);
            }

            // Dot product
            float sum = 0.0f;
#ifdef TS_AVX512F
            {
                int kk = 0;
                __m512 acc_v = _mm512_setzero_ps();
                for (; kk + 16 <= K; kk += 16) {
                    __m512 va = _mm512_loadu_ps(a_row + kk);
                    __m512 vb = _mm512_loadu_ps(b_col + kk);
                    acc_v = _mm512_fmadd_ps(va, vb, acc_v);
                }
                sum = _mm512_reduce_add_ps(acc_v);
                for (; kk < K; kk++) sum += a_row[kk] * b_col[kk];
            }
#elif defined(TS_AVX2)
            {
                int kk = 0;
                __m256 acc_v = _mm256_setzero_ps();
                for (; kk + 8 <= K; kk += 8) {
                    __m256 va = _mm256_loadu_ps(a_row + kk);
                    __m256 vb = _mm256_loadu_ps(b_col + kk);
                    acc_v = _mm256_fmadd_ps(va, vb, acc_v);
                }
                sum = acc_v[0] + acc_v[1] + acc_v[2] + acc_v[3] +
                      acc_v[4] + acc_v[5] + acc_v[6] + acc_v[7];
                for (; kk < K; kk++) sum += a_row[kk] * b_col[kk];
            }
#else
            for (int kk = 0; kk < K; kk++) sum += a_row[kk] * b_col[kk];
#endif
            c_row[n] += alpha * sum;
        }

        // Convert c_row back to fp16
        f32_to_f16_batch(c_row, C + m * N, N);
    }

    aligned_free(a_row);
    aligned_free(b_col);
    aligned_free(c_row);
}

}  // namespace vibeblade
