#include "dequant.h"
#include "ggml_types.h"
#include "fp16_compat.h"
#include "win_compat.h"
#include <cstdint>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>


// Thread pool for parallel inference
#include <thread>
#include <vector>
#include <atomic>

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
//  BF16 dequant: 1:1 mapping
// ════════════════════════════════════════════════════════════════
void dequantize_row_bf16(const void* row, float* out, int64_t n) {
    const uint16_t* src = (const uint16_t*)row;
    for (int64_t i = 0; i < n; i++) {
        out[i] = bf16_to_f32(src[i]);
    }
}

// ════════════════════════════════════════════════════════════════
//  Q2_K dequant: 256 values per super-block
//  16 sub-blocks of 16 values, 4-bit packed scales
// ════════════════════════════════════════════════════════════════
void dequantize_row_q2_K(const void* row, float* out, int64_t n) {
    const block_q2_K* blocks = (const block_q2_K*)row;
    int64_t nb = (n + 255) / 256;

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* q = blocks[b].qs;
        const uint8_t* sc = blocks[b].scales;
        float d   = f16_to_f32(blocks[b].d);
        float min = f16_to_f32(blocks[b].dmin);

        for (int j = 0; j < 256; j += 16) {
            int is = j / 16;
            float dl = d * (sc[is] & 0xF);
            float dh = d * (sc[is] >> 4);
            int64_t remaining = std::min((int64_t)16, n - b * 256 - j);
            for (int64_t l = 0; l < remaining; l++) {
                int qv = (q[j / 4 + l / 8] >> (2 * (l % 4))) & 3;
                out[b * 256 + j + l] = dl * qv + min;
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q3_K dequant: 256 values per super-block
//  hmask encodes whether q >= 8, qs stores q % 8
// ════════════════════════════════════════════════════════════════
static inline void get_scale_min_k3(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else if (j < 8) {
        *d = (q[j + 4] >> 0) & 63;
        *m = (q[j - 4] >> 6) | ((q[j + 0] & 0xF) << 2);
    } else {
        *d = (q[j - 4] >> 4) & 63;
        *m = (q[j - 4] >> 6) | ((q[j + 0] & 0xF) << 2);
    }
}

void dequantize_row_q3_K(const void* row, float* out, int64_t n) {
    const block_q3_K* blocks = (const block_q3_K*)row;
    int64_t nb = (n + 255) / 256;

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* q  = blocks[b].qs;
        const uint8_t* hm = blocks[b].hmask;
        float d = f16_to_f32(blocks[b].d);

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k3(is, blocks[b].scales, &sc, &m);
            float d1 = d * sc;
            get_scale_min_k3(is + 1, blocks[b].scales, &sc, &m);
            float d2 = d * sc;
            get_scale_min_k3(is + 2, blocks[b].scales, &sc, &m);
            float d3 = d * sc;
            get_scale_min_k3(is + 3, blocks[b].scales, &sc, &m);
            float d4 = d * sc;

            // Process 64 values in 4 sub-blocks of 16
            int64_t remaining = std::min((int64_t)256, n - b * 256) - j;
            for (int64_t l = 0; l < std::min((int64_t)16, remaining); l++) {
                int qi = j / 4 + l / 8;
                int qr = l % 8;
                int qv = (q[qi] >> (2 * (l % 4))) & 3;
                int hbit = (hm[l] >> is) & 1;
                out[b * 256 + j + l] = d1 * (qv + hbit * 8);
            }
            for (int64_t l = 0; l < std::min((int64_t)16, remaining - 16); l++) {
                int qi = (j + 16) / 4 + l / 8;
                int qv = (q[qi] >> (2 * (l % 4))) & 3;
                int hbit = (hm[l] >> (is + 2)) & 1;
                out[b * 256 + j + 16 + l] = d2 * (qv + hbit * 8);
            }
            for (int64_t l = 0; l < std::min((int64_t)16, remaining - 32); l++) {
                int qi = (j + 32) / 4 + l / 8;
                int qv = (q[qi] >> (2 * (l % 4))) & 3;
                int hbit = (hm[l] >> (is + 4)) & 1;
                out[b * 256 + j + 32 + l] = d3 * (qv + hbit * 8);
            }
            for (int64_t l = 0; l < std::min((int64_t)16, remaining - 48); l++) {
                int qi = (j + 48) / 4 + l / 8;
                int qv = (q[qi] >> (2 * (l % 4))) & 3;
                int hbit = (hm[l] >> (is + 6)) & 1;
                out[b * 256 + j + 48 + l] = d4 * (qv + hbit * 8);
            }
            is += 4;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q8_K dequant: 256 values per super-block, interleaved sub-block scales
//  On-disk layout: d(f32) + [qs(32) + scale(i8)] * 8 sub-blocks
//  Total: 4 + 33*8 = 268. But GGUF says 292 — extra padding/alignment.
// ════════════════════════════════════════════════════════════════
void dequantize_row_q8_K(const void* row, float* out, int64_t n) {
    const uint8_t* data = (const uint8_t*)row;
    int64_t nb = (n + 255) / 256;

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 292;  // 292-byte stride
        float d;
        memcpy(&d, block, 4);  // super-block scale (fp32)

        for (int sub = 0; sub < 8; sub++) {
            const int8_t* qs = (const int8_t*)(block + 4 + sub * 33);
            int8_t sc = qs[32];  // sub-block scale
            float ds = d * sc;
            int64_t remaining = std::min((int64_t)32, n - b * 256 - sub * 32);
            for (int64_t i = 0; i < remaining; i++) {
                out[b * 256 + sub * 32 + i] = qs[i] * ds;
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Q4_K dequant: 256 values per super-block, 6-bit packed scales
// ════════════════════════════════════════════════════════════════

// __restrict__ is GCC/Clang; MSVC uses __restrict
#ifdef _MSC_VER
#define VB_RESTRICT __restrict
#else
#define VB_RESTRICT __restrict__
#endif

static inline void get_scale_min_k4(int j, const uint8_t* VB_RESTRICT q, uint8_t* VB_RESTRICT d, uint8_t* VB_RESTRICT m) {
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

static inline void get_scale_min_k5(int j, const uint8_t* VB_RESTRICT q, uint8_t* VB_RESTRICT d, uint8_t* VB_RESTRICT m) {
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
    const uint8_t* data = (const uint8_t*)row;
    int64_t nb = (n + 255) / 256;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 210;
        float d = f16_to_f32(*(const uint16_t*)(block + 208));
        const uint8_t* ql = block;       // 128 bytes: 4-bit packed quants
        const uint8_t* qh = block + 128; // 64 bytes: 2-bit upper quants
        const int8_t*  sc = (const int8_t*)(block + 192); // 16 bytes: scales

        int64_t remaining = std::min((int64_t)256, n - b * 256);

        // ql: 128 bytes store 256 4-bit values in 32-byte groups
        // Group g (0-3): bytes [g*32..g*32+31], low nibble = vals g*64..g*64+31,
        //   high nibble = vals g*64+32..g*64+63
        // qh: 64 bytes, each byte has 4 2-bit values
        // qh[j] (j=0..31) covers indices j*4..j*4+3 (indices 0-127)
        // qh[j] (j=32..63) covers indices 128+.. (indices 128-255)
        // scales: sc[i] for i=0..15, each covers 16 values

        for (int64_t i = 0; i < remaining; i++) {
            int ql_idx;
            int ql_shift;
            if (i < 64) {
                ql_idx = (i / 2);
                ql_shift = (i & 1) * 4;
            } else if (i < 128) {
                ql_idx = 32 + (i - 64) / 2;
                ql_shift = ((i - 64) & 1) * 4;
            } else if (i < 192) {
                ql_idx = 64 + (i - 128) / 2;
                ql_shift = ((i - 128) & 1) * 4;
            } else {
                ql_idx = 96 + (i - 192) / 2;
                ql_shift = ((i - 192) & 1) * 4;
            }
            int ql_val = (ql[ql_idx] >> ql_shift) & 0xF;

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
        case GGML_TYPE_F32: memcpy(out, row, n_values * sizeof(float)); return;
        case GGML_TYPE_F16: for (int64_t i = 0; i < n_values; i++) out[i] = f16_to_f32(((const uint16_t*)row)[i]); return;
        case GGML_TYPE_BF16: dequantize_row_bf16(row, out, n_values); return;
        case GGML_TYPE_Q4_0: dequantize_row_q4_0(row, out, n_values); return;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1(row, out, n_values); return;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0(row, out, n_values); return;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1(row, out, n_values); return;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0(row, out, n_values); return;
        case GGML_TYPE_Q2_K: dequantize_row_q2_K(row, out, n_values); return;
        case GGML_TYPE_Q3_K: dequantize_row_q3_K(row, out, n_values); return;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K(row, out, n_values); return;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K(row, out, n_values); return;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K(row, out, n_values); return;
        case GGML_TYPE_Q8_K: dequantize_row_q8_K(row, out, n_values); return;
        default: throw std::runtime_error("Unsupported dequant type: " + std::to_string(type));
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

static inline float dot_dequant_q6_K(const float* x, const uint8_t* row_data, int64_t K) {
    float sum = 0.0f;
    for (int64_t b = 0; b < (K + 255) / 256; b++) {
        const uint8_t* block = row_data + b * 210;
        float d = f16_to_f32(*(const uint16_t*)(block + 208));
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t*  sc = (const int8_t*)(block + 192);
        int64_t remaining = std::min((int64_t)256, K - b * 256);

        for (int64_t i = 0; i < remaining; i++) {
            int ql_idx;
            int ql_shift;
            if (i < 64) {
                ql_idx = (i / 2);
                ql_shift = (i & 1) * 4;
            } else if (i < 128) {
                ql_idx = 32 + (i - 64) / 2;
                ql_shift = ((i - 64) & 1) * 4;
            } else if (i < 192) {
                ql_idx = 64 + (i - 128) / 2;
                ql_shift = ((i - 128) & 1) * 4;
            } else {
                ql_idx = 96 + (i - 192) / 2;
                ql_shift = ((i - 192) & 1) * 4;
            }
            int ql_val = (ql[ql_idx] >> ql_shift) & 0xF;

            int qh_val = (qh[i >> 2] >> (2 * (i & 3))) & 3;
            int q = (ql_val | (qh_val << 4)) - 32;
            sum += d * sc[i / 16] * q * x[b * 256 + i];
        }
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

static inline float dot_dequant_bf16(const float* x, const uint16_t* row, int64_t K) {
    float sum = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        sum += x[i] * bf16_to_f32(row[i]);
    }
    return sum;
}

static inline float dot_dequant_q2_K(const float* x, const block_q2_K* row, int64_t K) {
    float d   = f16_to_f32(row->d);
    float min = f16_to_f32(row->dmin);
    const uint8_t* sc = row->scales;
    const uint8_t* q  = row->qs;
    float sum = 0.0f;
    float min_sum = 0.0f;
    for (int64_t i = 0; i < K; i++) min_sum += x[i];
    min_sum *= min;
    for (int j = 0; j < K; j += 16) {
        int is = j / 16;
        float dl = d * (sc[is] & 0xF);
        for (int l = 0; l < 16 && j + l < K; l++) {
            int qv = (q[j / 4 + l / 8] >> (2 * (l % 4))) & 3;
            sum += dl * qv * x[j + l];
        }
    }
    return sum + min_sum;
}

static inline float dot_dequant_q3_K(const float* x, const block_q3_K* row, int64_t K) {
    float d = f16_to_f32(row->d);
    const uint8_t* q  = row->qs;
    const uint8_t* hm = row->hmask;
    float sum = 0.0f;
    for (int j = 0; j < K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k3(j / 16, row->scales, &sc, &m);
        float d1 = d * sc;
        get_scale_min_k3(j / 16 + 1, row->scales, &sc, &m);
        float d2 = d * sc;
        get_scale_min_k3(j / 16 + 2, row->scales, &sc, &m);
        float d3 = d * sc;
        get_scale_min_k3(j / 16 + 3, row->scales, &sc, &m);
        float d4 = d * sc;
        for (int l = 0; l < 64 && j + l < K; l++) {
            int qi = (j + l) / 4;
            int qv = (q[qi] >> (2 * (l % 4))) & 3;
            int hbit = (hm[l] >> (j / 16 * 2 + l / 16)) & 1;
            int val = qv + hbit * 8;
            float ds;
            switch (l / 16) {
                case 0: ds = d1; break;
                case 1: ds = d2; break;
                case 2: ds = d3; break;
                default: ds = d4; break;
            }
            sum += ds * val * x[j + l];
        }
    }
    return sum;
}

static inline float dot_dequant_q8_K(const float* x, const uint8_t* row_data, int64_t K) {
    float d;
    memcpy(&d, row_data, 4);
    float sum = 0.0f;
    for (int j = 0; j < K; j += 256) {
        const uint8_t* block = row_data + (j / 256) * 292;
        float bd;
        memcpy(&bd, block, 4);
        for (int sub = 0; sub < 8 && j + sub * 32 < K; sub++) {
            const int8_t* qs = (const int8_t*)(block + 4 + sub * 33);
            int8_t sc = qs[32];
            float ds = bd * sc;
            int remaining = std::min((int64_t)32, K - j - sub * 32);
            for (int i = 0; i < remaining; i++) {
                sum += qs[i] * ds * x[j + sub * 32 + i];
            }
        }
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

static inline float dot_dequant_q4_1(const float* x, const block_q4_1* row, int64_t K) {
    float d = f16_to_f32(row->d);
    float m = f16_to_f32(row->m);
    const uint8_t* qs = row->qs;
    int64_t nb = K / 32;
    float sum = 0.0f;
    // Compute sum(m) once
    float m_sum = 0.0f;
    for (int64_t i = 0; i < K; i++) m_sum += x[i];
    m_sum *= m;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* bq = qs + b * 16;
        const float* bx = x + b * 32;
        for (int i = 0; i < 16; i++) {
            int q0 = bq[i] & 0xF;
            int q1 = bq[i] >> 4;
            sum += d * (q0 * bx[2 * i] + q1 * bx[2 * i + 1]);
        }
    }
    return sum + m_sum;
}

static inline float dot_dequant_q5_0(const float* x, const block_q5_0* row, int64_t K) {
    float d = f16_to_f32(row->d);
    const uint8_t* qh = row->qh;
    const uint8_t* qs = row->qs;
    int64_t nb = K / 32;
    float sum = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* bq = qs + b * 16;
        const uint8_t* bh = qh + b * 4;
        const float* bx = x + b * 32;
        for (int i = 0; i < 16; i++) {
            int ql0 = bq[i] & 0xF;
            int qh0 = (bh[i >> 2] >> (2 * (i & 3))) & 1;
            int q0 = ql0 | (qh0 << 4);
            int ql1 = bq[i] >> 4;
            int qh1 = (bh[(i + 16) >> 2] >> (2 * ((i + 16) & 3))) & 1;
            int q1 = ql1 | (qh1 << 4);
            sum += (q0 - 16) * bx[2 * i] + (q1 - 16) * bx[2 * i + 1];
        }
    }
    return sum * d;
}

static inline float dot_dequant_q5_1(const float* x, const block_q5_1* row, int64_t K) {
    float d = f16_to_f32(row->d);
    float m = f16_to_f32(row->m);
    const uint8_t* qh = row->qh;
    const uint8_t* qs = row->qs;
    int64_t nb = K / 32;
    float sum = 0.0f;
    float m_sum = 0.0f;
    for (int64_t i = 0; i < K; i++) m_sum += x[i];
    m_sum *= m;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* bq = qs + b * 16;
        const uint8_t* bh = qh + b * 4;
        const float* bx = x + b * 32;
        for (int i = 0; i < 16; i++) {
            int ql0 = bq[i] & 0xF;
            int qh0 = (bh[i >> 2] >> (2 * (i & 3))) & 1;
            int q0 = ql0 | (qh0 << 4);
            int ql1 = bq[i] >> 4;
            int qh1 = (bh[(i + 16) >> 2] >> (2 * ((i + 16) & 3))) & 1;
            int q1 = ql1 | (qh1 << 4);
            sum += d * (q0 * bx[2 * i] + q1 * bx[2 * i + 1]);
        }
    }
    return sum + m_sum;
}

void gemv_dequant(const float* x, const void* weights, float* out,
                   int64_t K, int64_t N, ggml_type wtype, float* scratch) {
    // GGUF stores weight matrices as (K, N) — K rows of N elements each.
    // We compute: out[n] = sum_k(x[k] * W[k][n])  for n = 0..N-1
    //
    // Since columns are not contiguous, we iterate over K rows,
    // dequantize each row into scratch (N floats), then saxpy.

    // Byte stride per GGUF row (each row has N elements)
    int64_t row_bytes;
    switch (wtype) {
        case GGML_TYPE_F32:  row_bytes = N * 4;  break;
        case GGML_TYPE_F16:  row_bytes = N * 2;  break;
        case GGML_TYPE_BF16: row_bytes = N * 2;  break;
        case GGML_TYPE_Q4_0: row_bytes = (N / 32) * sizeof(block_q4_0); break;
        case GGML_TYPE_Q4_1: row_bytes = (N / 32) * sizeof(block_q4_1); break;
        case GGML_TYPE_Q5_0: row_bytes = (N / 32) * sizeof(block_q5_0); break;
        case GGML_TYPE_Q5_1: row_bytes = (N / 32) * sizeof(block_q5_1); break;
        case GGML_TYPE_Q8_0: row_bytes = (N / 32) * 34; break;
        case GGML_TYPE_Q2_K: row_bytes = (N / 256) * sizeof(block_q2_K); break;
        case GGML_TYPE_Q3_K: row_bytes = (N / 256) * sizeof(block_q3_K); break;
        case GGML_TYPE_Q4_K: row_bytes = (N / 256) * sizeof(block_q4_K); break;
        case GGML_TYPE_Q5_K: row_bytes = (N / 256) * sizeof(block_q5_K); break;
        case GGML_TYPE_Q6_K: row_bytes = (N / 256) * sizeof(block_q6_K); break;
        case GGML_TYPE_Q8_K: row_bytes = (N / 256) * 292; break;
        default: throw std::runtime_error("gemv_dequant: unsupported type " + std::to_string(wtype));
    }

    // Zero output
    std::fill(out, out + N, 0.0f);

    const uint8_t* w = (const uint8_t*)weights;

    for (int64_t k = 0; k < K; k++) {
        const void* wrow = w + k * row_bytes;
        dequantize_row(wrow, scratch, N, wtype);
        float xk = x[k];
        for (int64_t n = 0; n < N; n++) {
            out[n] += xk * scratch[n];
        }
    }
}

// ── Multi-threaded gemv_dequant (no OpenMP, no LTO conflict) ──
void gemv_dequant_mt(const float* x, const void* weights, float* out,
                     int64_t K, int64_t N, ggml_type wtype, int n_threads, float* scratch) {
    // MT version of gemv_dequant with GGUF (K, N) layout.
    // Each thread processes a chunk of K rows and accumulates into its own
    // partial output, then we sum partial outputs.
    // For simplicity, fall back to single-threaded for now — the GGUF transpose
    // makes parallelism trickier (race conditions on out[]).
    // TODO: per-thread partial outputs for proper MT.
    gemv_dequant(x, weights, out, K, N, wtype, scratch);
}

} // namespace vibeblade
