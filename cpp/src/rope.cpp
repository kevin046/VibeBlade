#include "kernels.h"
#include "fp16_compat.h"
#include <new>
#include <cmath>
#include <cstring>

namespace vibeblade {


// ════════════════════════════════════════════════════════════════
//  RoPE:  Apply rotary positional embeddings in-place
//  x: (seq_len, num_heads, head_dim) — float16
//  freqs: (seq_len, head_dim/2) — float32 interleaved [cos,sin,...]
// ════════════════════════════════════════════════════════════════

void apply_rope(uint16_t* x, const float* freqs, int seq_len, int num_heads, int head_dim) {
    int half_d = head_dim / 2;
    int stride = num_heads * head_dim;  // elements per sequence position

    float* x_f32 = (float*)malloc((size_t)stride * sizeof(float));
    if (!x_f32) throw std::bad_alloc();

    for (int s = 0; s < seq_len; s++) {
        f16_to_f32_batch(x + s * stride, x_f32, stride);

        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < half_d; i++) {
                int fi = s * half_d + i;
                float cos_val = freqs[fi * 2];
                float sin_val = freqs[fi * 2 + 1];

                int base = h * head_dim;
                float x0 = x_f32[base + i];
                float x1 = x_f32[base + half_d + i];

                x_f32[base + i]         = x0 * cos_val - x1 * sin_val;
                x_f32[base + half_d + i] = x0 * sin_val + x1 * cos_val;
            }
        }

        f32_to_f16_batch(x_f32, x + s * stride, stride);
    }

    free(x_f32);
}

}  // namespace vibeblade
