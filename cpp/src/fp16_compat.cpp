#include "fp16_compat.h"
#include <cmath>
#include <cstring>

#ifdef TS_AVX512F
#include <immintrin.h>
#elif defined(TS_AVX2)
#include <immintrin.h>
#endif

namespace vibeblade {

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(frac & 0x400)) { frac <<= 1; exp--; }
            frac &= 0x3ff;
            f = (sign << 31) | ((exp + 127 - 15 + 1) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (frac << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float ret;
    memcpy(&ret, &f, 4);
    return ret;
}

uint16_t f32_to_f16(float fv) {
    uint32_t f;
    memcpy(&f, &fv, 4);
    uint32_t sign = (f >> 31) & 1;
    int32_t  exp  = (f >> 23) & 0xff;
    uint32_t frac = f & 0x7fffff;
    if (exp == 0) return sign << 15;
    if (exp == 255) return (sign << 15) | 0x7c00 | (frac >> 13);
    int32_t new_exp = exp - 127 + 15;
    if (new_exp <= 0) return sign << 15;
    if (new_exp >= 31) return (sign << 15) | 0x7c00;
    return static_cast<uint16_t>((sign << 15) | (new_exp << 10) | (frac >> 13));
}

void f16_to_f32_batch(const uint16_t* src, float* dst, int n) {
#ifdef TS_AVX512F
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(src + i)));
        _mm512_storeu_ps(dst + i, v);
    }
    for (; i < n; i++) dst[i] = f16_to_f32(src[i]);
#elif defined(TS_AVX2)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(src + i)));
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; i++) dst[i] = f16_to_f32(src[i]);
#else
    for (int i = 0; i < n; i++) dst[i] = f16_to_f32(src[i]);
#endif
}

void f32_to_f16_batch(const float* src, uint16_t* dst, int n) {
#ifdef TS_AVX512F
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), h);
    }
    for (; i < n; i++) dst[i] = f32_to_f16(src[i]);
#elif defined(TS_AVX2)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), h);
    }
    for (; i < n; i++) dst[i] = f32_to_f16(src[i]);
#else
    for (int i = 0; i < n; i++) dst[i] = f32_to_f16(src[i]);
#endif
}

}  // namespace vibeblade
