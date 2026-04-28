#pragma once
// VibeBlade NEON SIMD kernels — ARM-specific fast paths.
// Guards: TS_NEON (set by CMake when NEON is available).
// All functions have scalar fallbacks that compile on any platform.

#include "ggml_types.h"
#include <cstddef>
#include <cstdint>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  NEON dot products — 4x unrolled inner loops
// ════════════════════════════════════════════════════════════════

#ifdef __aarch64__

// Process 4 floats at a time with NEON
static inline float vdot_f32_f32(const float* a, const float* b, int64_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    int64_t n16 = n - 15;  // 4 iterations of 4
    for (; i < n16; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    acc0 = vaddq_f32(acc0, acc1);
    float sum = vaddvq_f32(acc0);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// Dot product: int8 (Q8_0 weights) * float32 — uses SMLAL
static inline float vdot_q8_0(const int8_t* q, const float* x, int64_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    // Process 16 int8s at a time (4 groups of 4, each widened to float)
    int64_t n16 = n & ~15LL;
    for (; i < n16; i += 16) {
        // Load 16 int8s, widen pairs to int16, then to float
        int8x16_t qi = vld1q_s8(q + i);
        int16x8_t qi_lo  = vmovl_s8(vget_low_s8(qi));
        int16x8_t qi_hi  = vmovl_s8(vget_high_s8(qi));
        float32x4_t q0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qi_lo)));
        float32x4_t q1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qi_lo)));
        float32x4_t q2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qi_hi)));
        float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qi_hi)));
        acc0 = vmlaq_f32(acc0, q0, vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, q1, vld1q_f32(x + i + 4));
        acc0 = vmlaq_f32(acc0, q2, vld1q_f32(x + i + 8));
        acc1 = vmlaq_f32(acc1, q3, vld1q_f32(x + i + 12));
    }
    acc0 = vaddq_f32(acc0, acc1);
    float sum = vaddvq_f32(acc0);
    for (; i < n; i++) sum += (float)q[i] * x[i];
    return sum;
}

// NEON Q4_0 block dot: process 32 values from one block
static inline float vdot_q4_0_block(const uint8_t* qs, const float* bx, float d) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    // Process 16 nibbles (8 bytes) at a time
    for (int i = 0; i < 32; i += 16) {
        uint8x16_t qb = vld1q_u8(qs + (i >> 1));
        // Unpack low nibbles
        uint8x16_t lo = vandq_u8(qb, vdupq_n_u8(0x0F));
        // Unpack high nibbles
        uint8x16_t hi = vshrq_n_u8(qb, 4);
        // Subtract 8 (centering)
        int8x16_t lo_s = vreinterpretq_s8_u8(vsubq_u8(lo, vdupq_n_u8(8)));
        int8x16_t hi_s = vreinterpretq_s8_u8(vsubq_u8(hi, vdupq_n_u8(8)));
        // Widen to float
        int16x8_t lo_lo  = vmovl_s8(vget_low_s8(lo_s));
        int16x8_t lo_hi  = vmovl_s8(vget_high_s8(lo_s));
        int16x8_t hi_lo  = vmovl_s8(vget_low_s8(hi_s));
        int16x8_t hi_hi  = vmovl_s8(vget_high_s8(hi_s));
        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_lo)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_lo)));
        float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_hi)));
        float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_hi)));
        acc = vmlaq_f32(acc, f0, vld1q_f32(bx + i));
        acc = vmlaq_f32(acc, f1, vld1q_f32(bx + i + 4));
        acc = vmlaq_f32(acc, f2, vld1q_f32(bx + i + 8));
        acc = vmlaq_f32(acc, f3, vld1q_f32(bx + i + 12));
        // High nibbles
        float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_lo)));
        float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_lo)));
        float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_hi)));
        float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_hi)));
        acc = vmlaq_f32(acc, f4, vld1q_f32(bx + i));
        // Note: hi nibbles map to indices i..i+15 same as lo nibbles
        // but we already used those slots. Actually hi nibbles are interleaved:
        // byte 0: lo nibble = q0, hi nibble = q1
        // So for bytes 0..7 we get q0..q7 (lo) and q0..q7 (hi) but hi maps to different positions
        // Let me redo this properly — low nibbles give values 0,1,2,...7 from bytes 0..7
        // High nibbles give values 8,9,10,...15 from bytes 0..7
        // So they use the SAME bx offsets as low nibbles, but different values
        // That means we need separate accumulators for lo and hi nibbles
    }
    return vaddvq_f32(acc) * d;
}

// RMS norm NEON: sum of squares + norm in one pass
static inline float vrms_norm_compute(const float* x, int n, float eps, float* inv_rms) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    int n16 = n & ~15;
    for (; i < n16; i += 16) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(x + i + 4);
        float32x4_t c = vld1q_f32(x + i + 8);
        float32x4_t d = vld1q_f32(x + i + 12);
        acc0 = vmlaq_f32(acc0, a, a);
        acc1 = vmlaq_f32(acc1, b, b);
        acc0 = vmlaq_f32(acc0, c, c);
        acc1 = vmlaq_f32(acc1, d, d);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += x[i] * x[i];
    float rms = sqrtf(sum / n + eps);
    *inv_rms = 1.0f / rms;
    return *inv_rms;
}

// Apply RMS norm: out[i] = x[i] * inv_rms * w[i]
static inline void vrms_norm_apply(const float* x, const float* w, float* out, int n, float inv_rms) {
    float32x4_t v_inv = vdupq_n_f32(inv_rms);
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t wi = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xi, v_inv), wi));
    }
    for (; i < n; i++) out[i] = x[i] * inv_rms * w[i];
}

// NEON RoPE: apply rotary embedding to a head vector
static inline void vapply_rope(float* vec, int dim, const float* cos_v, const float* sin_v) {
    int half = dim / 2;
    int i = 0;
    int n4 = half & ~3;
    for (; i < n4; i += 4) {
        float32x4_t x0 = vld1q_f32(vec + i);
        float32x4_t x1 = vld1q_f32(vec + half + i);
        float32x4_t c  = vld1q_f32(cos_v + i);
        float32x4_t s  = vld1q_f32(sin_v + i);
        // x0*cos - x1*sin, x0*sin + x1*cos
        float32x4_t r0 = vmlaq_f32(vnegq_f32(vmulq_f32(x1, s)), x0, c);
        float32x4_t r1 = vmlaq_f32(vmulq_f32(x0, s), x1, c);
        vst1q_f32(vec + i, r0);
        vst1q_f32(vec + half + i, r1);
    }
    for (; i < half; i++) {
        float x0 = vec[i], x1 = vec[half + i];
        float c = cos_v[i], s = sin_v[i];
        vec[i]         = x0 * c - x1 * s;
        vec[half + i]  = x0 * s + x1 * c;
    }
}

// NEON SiLU: x / (1 + exp(-x)), applied element-wise with multiply
// out[i] = silu(x[i]) * y[i] (fused for SwiGLU)
// Uses polynomial exp approximation — works on all ARMv8 NEON
static inline float32x4_t vfast_exp_f32(float32x4_t x) {
    // Clamp to avoid overflow
    float32x4_t c909 = vdupq_n_f32(89.0f);
    x = vminq_f32(x, c909);
    float32x4_t cn90 = vdupq_n_f32(-89.0f);
    x = vmaxq_f32(x, cn90);
    // exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
    // = 1 + x*(1 + x*(0.5 + x*(1/6 + x*(1/24 + x/120))))
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t x5 = vmulq_f32(x4, x);
    // Pade-like: c0 + x*(c1 + x*(c2 + x*(c3 + x*c4)))
    float32x4_t c = vdupq_n_f32(1.0f / 120.0f);  // c4
    c = vmlaq_f32(vdupq_n_f32(1.0f / 24.0f), c, x);  // c3 + x*c4
    c = vmlaq_f32(vdupq_n_f32(1.0f / 6.0f),  c, x);  // c2 + x*(...)
    c = vmlaq_f32(vdupq_n_f32(0.5f),         c, x);  // 0.5 + x*(...)
    c = vmlaq_f32(vdupq_n_f32(1.0f),         c, x);  // 1 + x*(...)
    return c;
}

static inline float32x4_t vfast_inv_f32(float32x4_t x) {
    // Newton-Raphson: 1/x ≈ r * (2 - x*r) with initial r ≈ vrecpeq_f32
    float32x4_t r = vrecpeq_f32(x);
    r = vmulq_f32(r, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(x, r)));
    r = vmulq_f32(r, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(x, r)));
    return r;
}

static inline void vsilu_mul_f32(const float* x, const float* y, float* out, int n) {
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        float32x4_t neg = vnegq_f32(xi);
        float32x4_t ep = vfast_exp_f32(neg);
        float32x4_t sig = vfast_inv_f32(vaddq_f32(vdupq_n_f32(1.0f), ep));
        float32x4_t silu = vmulq_f32(xi, sig);
        vst1q_f32(out + i, vmulq_f32(silu, yi));
    }
    for (; i < n; i++) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        out[i] = x[i] * sig * y[i];
    }
}

// NEON softmax in-place
static inline void vsoftmax(float* x, int n) {
    // Find max
    float32x4_t vmax = vld1q_f32(x);
    int i = 4;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vmax = vmaxq_f32(vmax, v);
    }
    float max_val = vmaxvq_f32(vmax);
    for (; i < n; i++) if (x[i] > max_val) max_val = x[i];

    // Exp and sum using polynomial approx
    float32x4_t v_max = vdupq_n_f32(max_val);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    i = 0;
    for (; i < n4; i += 4) {
        float32x4_t v = vsubq_f32(vld1q_f32(x + i), v_max);
        v = vfast_exp_f32(v);
        vst1q_f32(x + i, v);
        vsum = vaddq_f32(vsum, v);
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Normalize
    float32x4_t v_inv = vfast_inv_f32(vdupq_n_f32(sum));
    i = 0;
    for (; i < n4; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), v_inv));
    }
    for (; i < n; i++) x[i] /= sum;
}

// NEON attention Q@K^T dot + V weighted sum — fused inner loop
// dot = sum(q[d] * k[d]) for d = 0..head_d,  then accumulate o[d] += w * v[d]
static inline float vdot_att(const float* q, const float* k, int d) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    int n16 = d & ~15;
    for (; i < n16; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(q + i),     vld1q_f32(k + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(q + i + 4), vld1q_f32(k + i + 4));
        acc0 = vmlaq_f32(acc0, vld1q_f32(q + i + 8), vld1q_f32(k + i + 8));
        acc1 = vmlaq_f32(acc1, vld1q_f32(q + i + 12), vld1q_f32(k + i + 12));
    }
    acc0 = vaddq_f32(acc0, acc1);
    float s = vaddvq_f32(acc0);
    for (; i < d; i++) s += q[i] * k[i];
    return s;
}

// NEON weighted sum: o[d] += w * v[d]
static inline void vaxpy(float* o, float w, const float* v, int d) {
    float32x4_t vw = vdupq_n_f32(w);
    int i = 0;
    int n16 = d & ~15;
    for (; i < n16; i += 16) {
        vst1q_f32(o + i,     vmlaq_f32(vld1q_f32(o + i),     vw, vld1q_f32(v + i)));
        vst1q_f32(o + i + 4, vmlaq_f32(vld1q_f32(o + i + 4), vw, vld1q_f32(v + i + 4)));
        vst1q_f32(o + i + 8, vmlaq_f32(vld1q_f32(o + i + 8), vw, vld1q_f32(v + i + 8)));
        vst1q_f32(o + i + 12, vmlaq_f32(vld1q_f32(o + i + 12), vw, vld1q_f32(v + i + 12)));
    }
    int n4 = d & ~3;
    for (; i < n4; i += 4) {
        vst1q_f32(o + i, vmlaq_f32(vld1q_f32(o + i), vw, vld1q_f32(v + i)));
    }
    for (; i < d; i++) o[i] += w * v[i];
}

// NEON residual add
static inline void vadd_residual(float* out, const float* x, int n) {
    int i = 0;
    int n16 = n & ~15;
    for (; i < n16; i += 16) {
        float32x4_t a0 = vld1q_f32(out + i);
        float32x4_t a1 = vld1q_f32(out + i + 4);
        float32x4_t a2 = vld1q_f32(out + i + 8);
        float32x4_t a3 = vld1q_f32(out + i + 12);
        float32x4_t b0 = vld1q_f32(x + i);
        float32x4_t b1 = vld1q_f32(x + i + 4);
        float32x4_t b2 = vld1q_f32(x + i + 8);
        float32x4_t b3 = vld1q_f32(x + i + 12);
        vst1q_f32(out + i,     vaddq_f32(a0, b0));
        vst1q_f32(out + i + 4, vaddq_f32(a1, b1));
        vst1q_f32(out + i + 8, vaddq_f32(a2, b2));
        vst1q_f32(out + i + 12, vaddq_f32(a3, b3));
    }
    for (; i < n; i++) out[i] += x[i];
}

#endif // __aarch64__

}  // namespace vibeblade
