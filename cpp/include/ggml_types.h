#pragma once
// VibeBlade GGML type definitions — mirrors llama.cpp ggml-common.h subset
// Supports the quantization types used in popular GGUF models.

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace vibeblade {

enum ggml_type : int32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_BF16    = 30,
};

// ── Block size (number of values per quantization block) ──
inline int ggml_blck_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 1;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q4_1: return 32;
        case GGML_TYPE_Q5_0: return 32;
        case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q8_1: return 32;
        case GGML_TYPE_Q2_K: return 256;
        case GGML_TYPE_Q3_K: return 256;
        case GGML_TYPE_Q4_K: return 256;
        case GGML_TYPE_Q5_K: return 256;
        case GGML_TYPE_Q6_K: return 256;
        case GGML_TYPE_Q8_K: return 256;
        default: return 1;
    }
}

// ── Type size in bytes per block ──
inline size_t ggml_type_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 40;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q8_K: return 292;
        default: return 4;
    }
}

// ── Bytes per value (type_size / blck_size) ──
inline float ggml_type_bpw(ggml_type t) {
    return (float)ggml_type_size(t) / (float)ggml_blck_size(t);
}

// ── Block structs for quantized types ──

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;       // delta (scale)
    uint8_t  qs[16];  // nibbles / quants (4-bit packed, 32 values)
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size");

struct block_q4_1 {
    uint16_t d;       // delta
    uint16_t m;       // min
    uint8_t  qs[16];  // nibbles / quants
};
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 size");

struct block_q5_0 {
    uint16_t d;       // delta
    uint8_t  qh[4];   // 5th bit of quants (32 values)
    uint8_t  qs[16];  // nibbles / quants (4 low bits)
};
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 size");

struct block_q5_1 {
    uint16_t d;       // delta
    uint16_t m;       // min
    uint8_t  qh[4];   // 5th bit
    uint8_t  qs[16];  // nibbles
};
static_assert(sizeof(block_q5_1) == 24, "block_q5_1 size");

// Q8_0: on-disk format is 34 bytes (4 byte float scale + 32 int8_t).
// C struct is 36 bytes due to float alignment, so we compute stride as 34.
struct block_q8_0 {
    float    d;       // delta (4 bytes)
    int8_t   qs[32];  // quants (-128 to 127)
};
// On-disk stride is 34, struct size is 36 due to padding — use ggml_type_size(Q8_0)=34 for indexing.

// K-quant blocks (super-block = 256 values)
struct block_q4_K {
    uint16_t d;       // super-block scale (fp16)
    uint16_t dmin;    // super-block minimum (fp16)
    uint8_t  scales[12]; // 6-bit packed scales for 8 sub-blocks
    uint8_t  qs[128]; // 4-bit packed quants (256 values)
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size");

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];  // 5th bit of quants
    uint8_t  qs[128];
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K size");

struct block_q6_K {
    uint8_t  ql[128]; // quants, lower 4 bits
    uint8_t  qh[64];  // quants, upper 2 bits
    int8_t   scales[16]; // 8-bit scales for 16 sub-blocks
    uint16_t d;       // super-block scale (fp16)
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size");

// Q2_K: 256 values per super-block
struct block_q2_K {
    uint8_t  scales[16]; // 4-bit packed scales for 16 sub-blocks
    uint8_t  qs[64];     // 2-bit packed quants (256 values)
    uint16_t d;          // super-block scale (fp16)
    uint16_t dmin;       // super-block minimum (fp16)
};
static_assert(sizeof(block_q2_K) == 84, "block_q2_K size");

// Q3_K: 256 values per super-block
struct block_q3_K {
    uint8_t  hmask[32];  // 1-bit: is q >= 8?
    uint8_t  qs[64];     // 2-bit packed quants (256 values), values 0-3
    uint8_t  scales[12]; // 6-bit packed scales for 8 sub-blocks
    uint16_t d;          // super-block scale (fp16)
};
static_assert(sizeof(block_q3_K) == 110, "block_q3_K size");

// Q8_K: 256 values per super-block
struct block_q8_K {
    float    d;           // super-block scale (fp32)
    int8_t   qs[256];     // quants (-128 to 127)
    // NOTE: on-disk, scales are interleaved after every 32 quants.
    // Layout: d(4) + [qs(32) + scale(1)] * 8 = 4 + 33*8 = 268.
    // But actual on-disk is 292 bytes — scales are stored separately.
};
// Q8_K on-disk is 292 bytes. Struct above doesn't match layout;
// use byte-level access with 292-byte stride (ggml_type_size).

#pragma pack(pop)  // Restore default alignment after packed block structs

// ── BF16 ↔ F32 ──
inline float bf16_to_f32(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float ret;
    memcpy(&ret, &f, 4);
    return ret;
}

inline uint16_t f32_to_bf16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);
    // Round to nearest even
    uint32_t lsb = (f >> 16) & 1;
    uint32_t bias = 0x7FFF + lsb;
    return (uint16_t)((f + bias) >> 16);
}

// ── F16 -> F32: use fp16_compat.h implementation ──
// (declaration is in fp16_compat.h, included by most .cpp files that need it)

// ── Total tensor data size in bytes ──
inline size_t tensor_nbytes(ggml_type type, int64_t n_values) {
    int blck = ggml_blck_size(type);
    int bsz  = (int)ggml_type_size(type);
    return ((n_values + blck - 1) / blck) * bsz;
}

}  // namespace vibeblade
