#include "kernels.h"
#include <cmath>
#include <cfloat>
#include <algorithm>

#ifdef TS_AVX512F
#include <immintrin.h>
#elif defined(TS_AVX2)
#include <immintrin.h>
#endif

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  Row-wise softmax in-place on fp32 data
//  x: (rows, cols) — float32, modified in-place
// ════════════════════════════════════════════════════════════════

void softmax_f32(float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = x + r * cols;

        // ── Find max ──
        float row_max = -FLT_MAX;
        {
#ifdef TS_AVX512F
            __m512 vmax = _mm512_set1_ps(-FLT_MAX);
            int i = 0;
            for (; i + 16 <= cols; i += 16) {
                __m512 v = _mm512_loadu_ps(row + i);
                vmax = _mm512_max_ps(vmax, v);
            }
            row_max = _mm512_reduce_max_ps(vmax);
            for (; i < cols; i++) row_max = std::max(row_max, row[i]);
#elif defined(TS_AVX2)
            __m256 vmax = _mm256_set1_ps(-FLT_MAX);
            int i = 0;
            for (; i + 8 <= cols; i += 8) {
                __m256 v = _mm256_loadu_ps(row + i);
                vmax = _mm256_max_ps(vmax, v);
            }
            // horizontal max
            __m128 hi = _mm256_extractf128_ps(vmax, 1);
            __m128 lo = _mm256_castps256_ps128(vmax);
            __m128 m = _mm_max_ps(lo, hi);
            m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
            m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
            row_max = _mm_cvtss_f32(m);
            for (; i < cols; i++) row_max = std::max(row_max, row[i]);
#else
            for (int i = 0; i < cols; i++) row_max = std::max(row_max, row[i]);
#endif
        }

        // ── exp(x - max) and accumulate sum ──
        float row_sum = 0.0f;
        {
#ifdef TS_AVX512F
            __m512 vsum = _mm512_setzero_ps();
            __m512 vrmax = _mm512_set1_ps(row_max);
            int i = 0;
            for (; i + 16 <= cols; i += 16) {
                __m512 v = _mm512_exp_ps(_mm512_sub_ps(_mm512_loadu_ps(row + i), vrmax));
                _mm512_storeu_ps(row + i, v);
                vsum = _mm512_add_ps(vsum, v);
            }
            row_sum = _mm512_reduce_add_ps(vsum);
            for (; i < cols; i++) {
                row[i] = expf(row[i] - row_max);
                row_sum += row[i];
            }
#elif defined(TS_AVX2)
            __m256 vsum = _mm256_setzero_ps();
            __m256 vrmax = _mm256_set1_ps(row_max);
            int i = 0;
            for (; i + 8 <= cols; i += 8) {
                __m256 v = _mm256_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + i), vrmax));
                _mm256_storeu_ps(row + i, v);
                vsum = _mm256_add_ps(vsum, v);
            }
            // horizontal sum
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            row_sum = _mm_cvtss_f32(s);
            for (; i < cols; i++) {
                row[i] = expf(row[i] - row_max);
                row_sum += row[i];
            }
#else
            for (int i = 0; i < cols; i++) {
                row[i] = expf(row[i] - row_max);
                row_sum += row[i];
            }
#endif
        }

        // ── Divide by sum ──
        float inv_sum = 1.0f / row_sum;
        {
#ifdef TS_AVX512F
            __m512 vinv = _mm512_set1_ps(inv_sum);
            int i = 0;
            for (; i + 16 <= cols; i += 16) {
                __m512 v = _mm512_loadu_ps(row + i);
                _mm512_storeu_ps(row + i, _mm512_mul_ps(v, vinv));
            }
            for (; i < cols; i++) row[i] *= inv_sum;
#elif defined(TS_AVX2)
            __m256 vinv = _mm256_set1_ps(inv_sum);
            int i = 0;
            for (; i + 8 <= cols; i += 8) {
                __m256 v = _mm256_loadu_ps(row + i);
                _mm256_storeu_ps(row + i, _mm256_mul_ps(v, vinv));
            }
            for (; i < cols; i++) row[i] *= inv_sum;
#else
            for (int i = 0; i < cols; i++) row[i] *= inv_sum;
#endif
        }
    }
}

}  // namespace vibeblade
