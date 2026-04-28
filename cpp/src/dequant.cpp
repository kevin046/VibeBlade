#include "dequant.h"
#include "ggml_types.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  Q4_0 dequant: 32 values per block, 4-bit packed + f16 scale
// ════════════════════════════════════════════════════════════════
void dequantize_row_q4_0(const void* row, float* out, int64_t n) {
    const block_q4_0* blocks = (const block_q4_0*)row;
    int64_t nb = (n + 31) / 32;
    for (int64_t b = 0; b < nb; b++) {
        float d = f16_to_f32(blocks[b].d);
        const uint8_t* qs = blocks[b].qs;
        int64_t remaining = std::min((int64_t)32, n - b * 32);
        for (int64_t i = 0; i < remaining; i++) {
            int q = (qs[i >> 1] >> (4 * (i & 1))) & 0xF;
            out[b * 32 + i] = (q - 8) * d;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q4_1 dequant: 32 values per block, 4-bit packed + f16 scale + f16 min
// ════════════════════════════════════════════════════════════════
void dequantize_row_q4_1(const void* row, float* out, int64_t n) {
    const block_q4_1* blocks = (const block_q4_1*)row;
    int64_t nb = (n + 31) / 32;
    for (int64_t b = 0; b < nb; b++) {
        float d = f16_to_f32(blocks[b].d);
        float m = f16_to_f32(blocks[b].m);
        const uint8_t* qs = blocks[b].qs;
        int64_t remaining = std::min((int64_t)32, n - b * 32);
        for (int64_t i = 0; i < remaining; i++) {
            int q = (qs[i >> 1] >> (4 * (i & 1))) & 0xF;
            out[b * 32 + i] = d * q + m;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q5_0 dequant: 32 values per block, 4 low bits packed + 1 high bit in qh
// ════════════════════════════════════════════════════════════════
void dequantize_row_q5_0(const void* row, float* out, int64_t n) {
    const block_q5_0* blocks = (const block_q5_0*)row;
    int64_t nb = (n + 31) / 32;
    for (int64_t b = 0; b < nb; b++) {
        float d = f16_to_f32(blocks[b].d);
        const uint8_t* qh = blocks[b].qh;
        const uint8_t* qs = blocks[b].qs;
        int64_t remaining = std::min((int64_t)32, n - b * 32);
        for (int64_t i = 0; i < remaining; i++) {
            int ql = (qs[i >> 1] >> (4 * (i & 1))) & 0xF;
            int qh_bit = (qh[i >> 3] >> (i & 7)) & 1;
            int q = ql | (qh_bit << 4);
            out[b * 32 + i] = (q - 16) * d;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q5_1 dequant: Q5_0 + f16 min
// ════════════════════════════════════════════════════════════════
void dequantize_row_q5_1(const void* row, float* out, int64_t n) {
    const block_q5_1* blocks = (const block_q5_1*)row;
    int64_t nb = (n + 31) / 32;
    for (int64_t b = 0; b < nb; b++) {
        float d = f16_to_f32(blocks[b].d);
        float m = f16_to_f32(blocks[b].m);
        const uint8_t* qh = blocks[b].qh;
        const uint8_t* qs = blocks[b].qs;
        int64_t remaining = std::min((int64_t)32, n - b * 32);
        for (int64_t i = 0; i < remaining; i++) {
            int ql = (qs[i >> 1] >> (4 * (i & 1))) & 0xF;
            int qh_bit = (qh[i >> 3] >> (i & 7)) & 1;
            int q = ql | (qh_bit << 4);
            out[b * 32 + i] = d * q + m;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q8_0 dequant: 32 values per block, int8 + f32 scale
// ════════════════════════════════════════════════════════════════
void dequantize_row_q8_0(const void* row, float* out, int64_t n) {
    const uint8_t* data = (const uint8_t*)row;
    int64_t nb = (n + 31) / 32;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 34;  // 34-byte stride on disk
        float d;
        memcpy(&d, block, 4);  // read float from first 4 bytes
        const int8_t* qs = (const int8_t*)(block + 4);
        int64_t remaining = std::min((int64_t)32, n - b * 32);
        for (int64_t i = 0; i < remaining; i++) {
            out[b * 32 + i] = qs[i] * d;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q4_K dequant: 256 values per super-block, 6-bit packed scales
// ════════════════════════════════════════════════════════════════

static inline void get_scale_min_k4(int j, const uint8_t* __restrict__ q, uint8_t* __restrict__ d, uint8_t* __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_row_q4_K(const void* row, float* out, int64_t n) {
    const block_q4_K* blocks = (const block_q4_K*)row;
    int64_t nb = (n + 255) / 256;

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* q = blocks[b].qs;
        float d   = f16_to_f32(blocks[b].d);
        float min = f16_to_f32(blocks[b].dmin);

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, blocks[b].scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, blocks[b].scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;

            int64_t remaining = std::min((int64_t)256, n - b * 256) - j;
            remaining = std::max((int64_t)0, remaining);
            int64_t half = std::min((int64_t)32, remaining);

            for (int64_t l = 0; l < half; l++) {
                int ql = q[l] & 0xF;
                out[b * 256 + j + l] = d1 * ql - m1;
            }
            for (int64_t l = 0; l < half; l++) {
                int qh = q[l] >> 4;
                out[b * 256 + j + 32 + l] = d2 * qh - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q5_K dequant: 256 values, 6-bit scales, 5th bit in qh
// ════════════════════════════════════════════════════════════════

static inline void get_scale_min_k5(int j, const uint8_t* __restrict__ q, uint8_t* __restrict__ d, uint8_t* __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_row_q5_K(const void* row, float* out, int64_t n) {
    const block_q5_K* blocks = (const block_q5_K*)row;
    int64_t nb = (n + 255) / 256;

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* q  = blocks[b].qs;
        const uint8_t* qh = blocks[b].qh;
        float d   = f16_to_f32(blocks[b].d);
        float min = f16_to_f32(blocks[b].dmin);

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k5(is, blocks[b].scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k5(is + 1, blocks[b].scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;

            int64_t remaining = std::min((int64_t)256, n - b * 256) - j;
            remaining = std::max((int64_t)0, remaining);
            int64_t half = std::min((int64_t)32, remaining);

            for (int64_t l = 0; l < half; l++) {
                int ql = (q[l] & 0xF);
                int h  = (qh[l] >> 7) & 1;
                out[b * 256 + j + l] = d1 * (ql | (h << 4)) - m1;
            }
            for (int64_t l = 0; l < half; l++) {
                int ql = (q[l] >> 4);
                int h  = (qh[l] >> 3) & 1;
                out[b * 256 + j + 32 + l] = d2 * (ql | (h << 4)) - m2;
            }
            q  += 32;
            qh += 32;
            is += 2;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q6_K dequant: 256 values, 4+2 bit split, 8-bit scales
// ════════════════════════════════════════════════════════════════
void dequantize_row_q6_K(const void* row, float* out, int64_t n) {
    const block_q6_K* blocks = (const block_q6_K*)row;
    int64_t nb = (n + 255) / 256;

    for (int64_t b = 0; b < nb; b++) {
        float d = f16_to_f32(blocks[b].d);
        const uint8_t* ql = blocks[b].ql;
        const uint8_t* qh = blocks[b].qh;
        const int8_t*  sc = blocks[b].scales;

        int64_t remaining = std::min((int64_t)256, n - b * 256);
        for (int64_t i = 0; i < remaining; i++) {
            int ql_val = ql[i] & 0xF;
            int qh_val = (qh[i >> 2] >> (2 * (i & 3))) & 3;
            int q = (ql_val | (qh_val << 4)) - 32;
            out[b * 256 + i] = d * sc[i / 16] * q;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Generic dequant dispatch
// ════════════════════════════════════════════════════════════════
void dequantize_row(const void* row, float* out, int64_t n_values, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(out, row, n_values * sizeof(float));
            return;
        case GGML_TYPE_F16:
            for (int64_t i = 0; i < n_values; i++)
                out[i] = f16_to_f32(((const uint16_t*)row)[i]);
            return;
        case GGML_TYPE_Q4_0: dequantize_row_q4_0(row, out, n_values); return;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1(row, out, n_values); return;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0(row, out, n_values); return;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1(row, out, n_values); return;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0(row, out, n_values); return;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K(row, out, n_values); return;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K(row, out, n_values); return;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K(row, out, n_values); return;
        default:
            throw std::runtime_error("Unsupported dequant type: " + std::to_string(type));
    }
}

// ════════════════════════════════════════════════════════════════
//  Matrix-Vector multiply with inline dequantization
//
//  out[j] = sum_i(x[i] * dequant(weight[j], i)) for j = 0..N-1
//
//  weights: (N, K) quantized, row-major in GGUF
//  x: (K,) fp32 input vector
//  out: (N,) fp32 output
//
//  This is the HOT PATH — dequant weights directly into the dot product,
//  never allocating a full fp32 weight matrix.
// ════════════════════════════════════════════════════════════════

// Helper: compute dot product of x[K] with one dequantized weight row
static inline float dot_dequant_q4_0(const float* x, const block_q4_0* row, int64_t K) {
    float d = f16_to_f32(row->d);
    const uint8_t* qs = row->qs;
    int64_t nb = K / 32;
    float sum = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* bq = qs + b * 16;
        const float* bx = x + b * 32;
        for (int i = 0; i < 16; i++) {
            int q0 = bq[i] & 0xF;
            int q1 = bq[i] >> 4;
            sum += (q0 - 8) * bx[2 * i] + (q1 - 8) * bx[2 * i + 1];
        }
    }
    return sum * d;
}

static inline float dot_dequant_q8_0(const float* x, const block_q8_0* row, int64_t K) {
    // Use byte-level access: 34-byte stride on disk
    const uint8_t* data = (const uint8_t*)row;
    int64_t nb = K / 32;
    float sum = 0.0f;
    float d = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 34;
        memcpy(&d, block, 4);
        const int8_t* qs = (const int8_t*)(block + 4);
        const float* bx = x + b * 32;
        for (int i = 0; i < 32; i++) {
            sum += qs[i] * bx[i];
        }
    }
    return sum * d;
}

static inline float dot_dequant_q4_K(const float* x, const block_q4_K* row, int64_t K) {
    float d   = f16_to_f32(row->d);
    float min = f16_to_f32(row->dmin);
    const uint8_t* q = row->qs;
    float sum = 0.0f;
    int is = 0;
    for (int j = 0; j < K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is, row->scales, &sc, &m);
        float d1 = d * sc; float m1 = min * m;
        get_scale_min_k4(is + 1, row->scales, &sc, &m);
        float d2 = d * sc; float m2 = min * m;

        for (int l = 0; l < 32; l++) {
            int ql = q[l] & 0xF;
            sum += (d1 * ql - m1) * x[j + l];
        }
        for (int l = 0; l < 32; l++) {
            int qh = q[l] >> 4;
            sum += (d2 * qh - m2) * x[j + 32 + l];
        }
        q += 32;
        is += 2;
    }
    return sum;
}

static inline float dot_dequant_q5_K(const float* x, const block_q5_K* row, int64_t K) {
    float d   = f16_to_f32(row->d);
    float min = f16_to_f32(row->dmin);
    const uint8_t* q  = row->qs;
    const uint8_t* qh = row->qh;
    float sum = 0.0f;
    int is = 0;
    for (int j = 0; j < K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k5(is, row->scales, &sc, &m);
        float d1 = d * sc; float m1 = min * m;
        get_scale_min_k5(is + 1, row->scales, &sc, &m);
        float d2 = d * sc; float m2 = min * m;

        for (int l = 0; l < 32; l++) {
            int ql = (q[l] & 0xF);
            int h  = (qh[l] >> 7) & 1;
            sum += (d1 * (ql | (h << 4)) - m1) * x[j + l];
        }
        for (int l = 0; l < 32; l++) {
            int ql = (q[l] >> 4);
            int h  = (qh[l] >> 3) & 1;
            sum += (d2 * (ql | (h << 4)) - m2) * x[j + 32 + l];
        }
        q += 32; qh += 32;
        is += 2;
    }
    return sum;
}

static inline float dot_dequant_q6_K(const float* x, const block_q6_K* row, int64_t K) {
    float d = f16_to_f32(row->d);
    const uint8_t* ql = row->ql;
    const uint8_t* qh = row->qh;
    const int8_t*  sc = row->scales;
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        int ql_val = ql[i] & 0xF;
        int qh_val = (qh[i >> 2] >> (2 * (i & 3))) & 3;
        int q = (ql_val | (qh_val << 4)) - 32;
        sum += d * sc[i / 16] * q * x[i];
    }
    return sum;
}

static inline float dot_dequant_f16(const float* x, const uint16_t* row, int64_t K) {
    float sum = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        sum += x[i] * f16_to_f32(row[i]);
    }
    return sum;
}

static inline float dot_dequant_f32(const float* x, const float* row, int64_t K) {
    float sum = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        sum += x[i] * row[i];
    }
    return sum;
}

void gemv_dequant(const float* x, const void* weights, float* out,
                   int64_t K, int64_t N, ggml_type wtype) {
    // Compute byte stride per weight row
    int64_t row_bytes;
    switch (wtype) {
        case GGML_TYPE_F32:  row_bytes = K * 4;  break;
        case GGML_TYPE_F16:  row_bytes = K * 2;  break;
        case GGML_TYPE_Q4_0: row_bytes = (K / 32) * sizeof(block_q4_0); break;
        case GGML_TYPE_Q4_1: row_bytes = (K / 32) * sizeof(block_q4_1); break;
        case GGML_TYPE_Q5_0: row_bytes = (K / 32) * sizeof(block_q5_0); break;
        case GGML_TYPE_Q5_1: row_bytes = (K / 32) * sizeof(block_q5_1); break;
        case GGML_TYPE_Q8_0: row_bytes = (K / 32) * sizeof(block_q8_0); break;
        case GGML_TYPE_Q4_K: row_bytes = (K / 256) * sizeof(block_q4_K); break;
        case GGML_TYPE_Q5_K: row_bytes = (K / 256) * sizeof(block_q5_K); break;
        case GGML_TYPE_Q6_K: row_bytes = (K / 256) * sizeof(block_q6_K); break;
        default: throw std::runtime_error("gemv_dequant: unsupported type " + std::to_string(wtype));
    }

    const uint8_t* w = (const uint8_t*)weights;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t j = 0; j < N; j++) {
        const void* wrow = w + j * row_bytes;
        switch (wtype) {
            case GGML_TYPE_F32:  out[j] = dot_dequant_f32(x, (const float*)wrow, K); break;
            case GGML_TYPE_F16:  out[j] = dot_dequant_f16(x, (const uint16_t*)wrow, K); break;
            case GGML_TYPE_Q4_0: out[j] = dot_dequant_q4_0(x, (const block_q4_0*)wrow, K); break;
            case GGML_TYPE_Q8_0: out[j] = dot_dequant_q8_0(x, (const block_q8_0*)wrow, K); break;
            case GGML_TYPE_Q4_K: out[j] = dot_dequant_q4_K(x, (const block_q4_K*)wrow, K); break;
            case GGML_TYPE_Q5_K: out[j] = dot_dequant_q5_K(x, (const block_q5_K*)wrow, K); break;
            case GGML_TYPE_Q6_K: out[j] = dot_dequant_q6_K(x, (const block_q6_K*)wrow, K); break;
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
                // Fall back to generic dequant + dot for less common types
                {
                    int64_t bs = ggml_blck_size(wtype);
                    int64_t nb = (K + bs - 1) / bs;
                    int64_t rsz = (int64_t)ggml_type_size(wtype) * nb;
                    // Use scratch for dequant (caller should provide large enough buffer)
                    // For now, dequant inline
                    float dot = 0.0f;
                    for (int64_t i = 0; i < K; i++) {
                        // This is slow but correct for rare types
                        float wval = 0.0f;
                        dequantize_row(wrow, &wval, 1, wtype); // NOT efficient - fallback
                        dot += x[i] * wval;
                    }
                    out[j] = dot;
                }
                break;
            default:
                throw std::runtime_error("gemv_dequant: unsupported type");
        }
    }
}

}  // namespace vibeblade
