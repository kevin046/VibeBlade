#include "kernels.h"
#include "fp16_compat.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <new>

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  2-bit Quantization
// ════════════════════════════════════════════════════════════════

int quantize_2bit(const uint16_t* x, uint8_t* packed, int S, int D,
                  int axis, float* scale_out, float* min_out) {
    // FIX #7: Validate axis
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 or 1");

    // FIX #1: Use size_t to prevent integer overflow in size calculations
    size_t total = (size_t)S * (size_t)D;
    float* x_f32 = (float*)malloc(total * sizeof(float));
    // FIX #6: NULL check
    if (!x_f32) throw std::bad_alloc();
    f16_to_f32_batch(x, x_f32, total);

    size_t out_bytes = (total + 3) / 4;
    memset(packed, 0, out_bytes);

    if (axis == 1) {
        // Per-channel: each channel (column) gets its own scale/min
        // Reduce over rows → scale_out[D], min_out[D]
        for (int d = 0; d < D; d++) {
            float mn = x_f32[d], mx = x_f32[d];
            for (int s = 1; s < S; s++) {
                float v = x_f32[(size_t)s * D + d];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            float range = std::max(mx - mn, 1e-8f);
            scale_out[d] = range / 3.0f;
            min_out[d] = mn;
        }
        // Quantize
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                float val = (x_f32[(size_t)s * D + d] - min_out[d]) / scale_out[d];
                int q = std::min(3, std::max(0, (int)std::roundf(val)));
                size_t idx = (size_t)s * D + d;
                packed[idx / 4] |= (q & 0x3) << ((idx % 4) * 2);
            }
        }
    } else {
        // Per-token (axis=0): each row gets its own scale/min
        // scale_out[S], min_out[S]
        for (int s = 0; s < S; s++) {
            float mn = x_f32[(size_t)s * D], mx = x_f32[(size_t)s * D];
            for (int d = 1; d < D; d++) {
                float v = x_f32[(size_t)s * D + d];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            float range = std::max(mx - mn, 1e-8f);
            scale_out[s] = range / 3.0f;
            min_out[s] = mn;
        }
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                float val = (x_f32[(size_t)s * D + d] - min_out[s]) / scale_out[s];
                int q = std::min(3, std::max(0, (int)std::roundf(val)));
                size_t idx = (size_t)s * D + d;
                packed[idx / 4] |= (q & 0x3) << ((idx % 4) * 2);
            }
        }
    }

    free(x_f32);
    return (int)out_bytes;
}

void dequantize_2bit(const uint8_t* packed, uint16_t* out, int S, int D,
                     int axis, const float* scale, const float* min_val) {
    // FIX #7: Validate axis
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 or 1");

    size_t total = (size_t)S * (size_t)D;
    float* o_f32 = (float*)malloc(total * sizeof(float));
    if (!o_f32) throw std::bad_alloc();

    if (axis == 1) {
        // Per-channel: scale[D], min[D]
        for (size_t s = 0; s < (size_t)S; s++) {
            for (int d = 0; d < D; d++) {
                size_t idx = s * D + d;
                int q = (packed[idx / 4] >> ((idx % 4) * 2)) & 0x3;
                o_f32[idx] = (float)q * scale[d] + min_val[d];
            }
        }
    } else {
        // Per-token: scale[S], min[S]
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                size_t idx = (size_t)s * D + d;
                int q = (packed[idx / 4] >> ((idx % 4) * 2)) & 0x3;
                o_f32[idx] = (float)q * scale[s] + min_val[s];
            }
        }
    }

    f32_to_f16_batch(o_f32, out, total);
    free(o_f32);
}

// ════════════════════════════════════════════════════════════════
//  4-bit Quantization
// ════════════════════════════════════════════════════════════════

int quantize_4bit(const uint16_t* x, uint8_t* packed, int S, int D,
                  int axis, float* scale_out, float* min_out) {
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 or 1");

    size_t total = (size_t)S * (size_t)D;
    float* x_f32 = (float*)malloc(total * sizeof(float));
    if (!x_f32) throw std::bad_alloc();
    f16_to_f32_batch(x, x_f32, total);

    size_t out_bytes = (total + 1) / 2;
    memset(packed, 0, out_bytes);

    if (axis == 1) {
        for (int d = 0; d < D; d++) {
            float mn = x_f32[d], mx = x_f32[d];
            for (int s = 1; s < S; s++) {
                float v = x_f32[(size_t)s * D + d];
                mn = std::min(mn, v); mx = std::max(mx, v);
            }
            float range = std::max(mx - mn, 1e-8f);
            scale_out[d] = range / 15.0f;
            min_out[d] = mn;
        }
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                float val = (x_f32[(size_t)s * D + d] - min_out[d]) / scale_out[d];
                int q = std::min(15, std::max(0, (int)std::roundf(val)));
                size_t idx = (size_t)s * D + d;
                if (idx % 2 == 0)
                    packed[idx / 2] = (q & 0xF);
                else
                    packed[idx / 2] |= ((q & 0xF) << 4);
            }
        }
    } else {
        for (int s = 0; s < S; s++) {
            float mn = x_f32[(size_t)s * D], mx = x_f32[(size_t)s * D];
            for (int d = 1; d < D; d++) {
                float v = x_f32[(size_t)s * D + d];
                mn = std::min(mn, v); mx = std::max(mx, v);
            }
            float range = std::max(mx - mn, 1e-8f);
            scale_out[s] = range / 15.0f;
            min_out[s] = mn;
        }
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                float val = (x_f32[(size_t)s * D + d] - min_out[s]) / scale_out[s];
                int q = std::min(15, std::max(0, (int)std::roundf(val)));
                size_t idx = (size_t)s * D + d;
                if (idx % 2 == 0)
                    packed[idx / 2] = (q & 0xF);
                else
                    packed[idx / 2] |= ((q & 0xF) << 4);
            }
        }
    }

    free(x_f32);
    return (int)out_bytes;
}

void dequantize_4bit(const uint8_t* packed, uint16_t* out, int S, int D,
                     int axis, const float* scale, const float* min_val) {
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 or 1");

    size_t total = (size_t)S * (size_t)D;
    float* o_f32 = (float*)malloc(total * sizeof(float));
    if (!o_f32) throw std::bad_alloc();

    if (axis == 1) {
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                size_t idx = (size_t)s * D + d;
                int q = (idx % 2 == 0)
                    ? (packed[idx / 2] & 0xF)
                    : ((packed[idx / 2] >> 4) & 0xF);
                o_f32[idx] = (float)q * scale[d] + min_val[d];
            }
        }
    } else {
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                size_t idx = (size_t)s * D + d;
                int q = (idx % 2 == 0)
                    ? (packed[idx / 2] & 0xF)
                    : ((packed[idx / 2] >> 4) & 0xF);
                o_f32[idx] = (float)q * scale[s] + min_val[s];
            }
        }
    }

    f32_to_f16_batch(o_f32, out, total);
    free(o_f32);
}

// ════════════════════════════════════════════════════════════════
//  8-bit Symmetric Quantization
// ════════════════════════════════════════════════════════════════

float quantize_8bit_sym(const uint16_t* x, int8_t* out, int N, float max_abs) {
    float* x_f32 = (float*)malloc((size_t)N * sizeof(float));
    if (!x_f32) throw std::bad_alloc();
    f16_to_f32_batch(x, x_f32, N);

    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;

    for (int i = 0; i < N; i++) {
        float v = x_f32[i] / scale;
        out[i] = (int8_t)std::min(127, std::max(-127, (int)std::roundf(v)));
    }

    free(x_f32);
    return scale;
}

void dequantize_8bit_sym(const int8_t* x, uint16_t* out, int N, float scale) {
    float* o_f32 = (float*)malloc((size_t)N * sizeof(float));
    if (!o_f32) throw std::bad_alloc();
    for (int i = 0; i < N; i++) o_f32[i] = (float)x[i] * scale;
    f32_to_f16_batch(o_f32, out, N);
    free(o_f32);
}

}  // namespace vibeblade
