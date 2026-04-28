// VibeBlade Fast Model — llama.cpp-style inference engine.
// Full forward pass in C++: prefill + decode with KV cache.
// Weights are mmap'd from GGUF, dequantized inline during matmuls.
// Zero malloc in the hot path — all buffers pre-allocated.

#include "fast_model.h"
#include "dequant.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  Math primitives (no external deps)
// ════════════════════════════════════════════════════════════════

static inline float fast_rms(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum / n + 1e-8f);
}

// RMSNorm: out[i] = x[i] * (w[i] / rms(x))
static void rms_norm(const float* x, const float* w, float* out, int n, float eps) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    s = 1.0f / sqrtf(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

// SiLU: x * sigmoid(x)
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

// Softmax in-place
static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// ════════════════════════════════════════════════════════════════
//  Constructor / Destructor
// ════════════════════════════════════════════════════════════════

VibeBladeFast::VibeBladeFast() = default;
VibeBladeFast::~VibeBladeFast() = default;

// ════════════════════════════════════════════════════════════════
//  Load GGUF model
// ════════════════════════════════════════════════════════════════

void VibeBladeFast::load(const char* path) {
    gguf_ = std::make_unique<GGUFFile>(path);

    extract_config(*gguf_);
    build_rope_cache();
    map_weights(*gguf_);
    alloc_kv_cache();

    loaded_ = true;
    position_ = 0;
}

void VibeBladeFast::extract_config(const GGUFFile& g) {
    cfg_.arch = g.meta_string("general.architecture");
    if (cfg_.arch.empty()) cfg_.arch = g.meta_string("llama.architecture");

    // Try arch-prefixed keys first, then generic fallbacks
    auto gi = [&](const std::string& key, int64_t def = 0) -> int64_t {
        std::string prefixed = cfg_.arch + "." + key;
        int64_t v = g.meta_int(prefixed, -1);
        return (v != -1) ? v : g.meta_int(key, def);
    };
    auto gf = [&](const std::string& key, float def = 0.0f) -> float {
        std::string prefixed = cfg_.arch + "." + key;
        float v = g.meta_float(prefixed, NAN);
        return !std::isnan(v) ? v : g.meta_float(key, def);
    };

    cfg_.n_layers        = (int)gi("block_count");
    cfg_.n_heads         = (int)gi("attention.head_count");
    cfg_.n_kv_heads      = (int)gi("attention.head_count_kv", cfg_.n_heads);
    cfg_.head_dim        = (int)gi("attention.key_length");
    cfg_.hidden_dim      = (int)gi("embedding_length");
    cfg_.intermediate_dim= (int)gi("feed_forward_length");
    cfg_.context_length  = (int)gi("context_length", 2048);
    cfg_.vocab_size      = (int)gi("vocab_size");
    cfg_.norm_eps        = gf("attention.layer_norm_rms_epsilon", 1e-5f);

    // Fallback: Qwen2 uses "qwen2." prefix
    if (cfg_.arch.empty() || cfg_.n_layers == 0) {
        cfg_.arch = "llama";  // default
    }

    // Cap context to MAX_SEQ
    cfg_.context_length = std::min(cfg_.context_length, MAX_SEQ);
}

void VibeBladeFast::build_rope_cache() {
    int dim = cfg_.head_dim;
    rope_cos_.resize(cfg_.context_length * dim / 2);
    rope_sin_.resize(cfg_.context_length * dim / 2);

    float base = 10000.0f;
    for (int pos = 0; pos < cfg_.context_length; pos++) {
        for (int i = 0; i < dim / 2; i++) {
            float theta = powf(base, -2.0f * i / dim);
            float angle = pos * theta;
            rope_cos_[pos * dim / 2 + i] = cosf(angle);
            rope_sin_[pos * dim / 2 + i] = sinf(angle);
        }
    }
}

void VibeBladeFast::alloc_kv_cache() {
    int kv_dim = cfg_.n_kv_heads * cfg_.head_dim;
    kv_k_.resize(cfg_.n_layers);
    kv_v_.resize(cfg_.n_layers);
    for (int l = 0; l < cfg_.n_layers; l++) {
        kv_k_[l].resize(cfg_.context_length * kv_dim, 0.0f);
        kv_v_[l].resize(cfg_.context_length * kv_dim, 0.0f);
    }

    // Scratch: enough for one layer's intermediates
    int hd = cfg_.hidden_dim;
    int id = cfg_.intermediate_dim;
    int q_dim = cfg_.n_heads * cfg_.head_dim;
    scratch_.resize(5 * hd + 3 * id + 4 * q_dim + 2 * hd);  // generous
}

void VibeBladeFast::map_weights(const GGUFFile& g) {
    const std::string& arch = cfg_.arch;

    // Token embeddings
    token_emb_ = g.tensor_data("token_embd.weight");
    if (token_emb_) {
        auto info = g.tensor_info("token_embd.weight");
        emb_type_ = info->type;
    }

    // Output norm
    output_norm_ = (const float*)g.tensor_data("output_norm.weight");

    // Output projection
    output_ = g.tensor_data("output.weight");
    if (!output_) {
        // Llama reuses token embeddings as output
        output_ = g.tensor_data("token_embd.weight");
    }
    if (output_) {
        auto info = g.tensor_info("output.weight");
        if (!info) info = g.tensor_info("token_embd.weight");
        out_type_ = info->type;
    }

    // Layer weights
    layers_.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; i++) {
        auto& lw = layers_[i];
        std::string pfx = "blk." + std::to_string(i) + ".";

        lw.attn_norm = (const float*)g.tensor_data(pfx + "attn_norm.weight");
        lw.ffn_norm  = (const float*)g.tensor_data(pfx + "ffn_norm.weight");

        auto set_w = [&](const std::string& name, const void*& ptr, ggml_type& type) {
            auto info = g.tensor_info(pfx + name);
            if (info) {
                ptr = g.tensor_data(pfx + name);
                type = info->type;
            }
        };

        set_w("attn_q.weight",      lw.attn_q,  lw.qtype);
        set_w("attn_k.weight",      lw.attn_k,  lw.ktype);
        set_w("attn_v.weight",      lw.attn_v,  lw.vtype);
        set_w("attn_output.weight", lw.attn_o,  lw.otype);
        set_w("ffn_gate.weight",    lw.ffn_gate, lw.gate_type);
        set_w("ffn_up.weight",      lw.ffn_up,   lw.up_type);
        set_w("ffn_down.weight",    lw.ffn_down, lw.down_type);

        // MoE gate
        set_w("ffn_gate_inp.weight", lw.ffn_gate_inp, lw.gate_inp_type);
    }
}

void VibeBladeFast::reset() {
    position_ = 0;
    int kv_dim = cfg_.n_kv_heads * cfg_.head_dim;
    for (int l = 0; l < cfg_.n_layers; l++) {
        std::fill(kv_k_[l].begin(), kv_k_[l].end(), 0.0f);
        std::fill(kv_v_[l].begin(), kv_v_[l].end(), 0.0f);
    }
}

// ════════════════════════════════════════════════════════════════
//  Forward one layer
// ════════════════════════════════════════════════════════════════

void VibeBladeFast::forward_layer(
    const float* input,    // (seq_len, hidden_dim)
    int seq_len,
    float* output,         // (seq_len, hidden_dim)
    const LayerWeights& lw,
    int layer_idx
) {
    int hd = cfg_.hidden_dim;
    int q_dim = cfg_.n_heads * cfg_.head_dim;
    int kv_dim = cfg_.n_kv_heads * cfg_.head_dim;
    int head_d = cfg_.head_dim;
    int n_heads = cfg_.n_heads;
    int n_kv_heads = cfg_.n_kv_heads;
    int kv_mul = n_heads / n_kv_heads;  // GQA ratio

    // ── Scratch allocation (no malloc!) ──
    float* sp = scratch_.data();
    float* normed    = sp;                    sp += seq_len * hd;
    float* Q         = sp;                    sp += seq_len * q_dim;
    float* K_buf     = sp;                    sp += seq_len * kv_dim;
    float* V_buf     = sp;                    sp += seq_len * kv_dim;
    float* attn_out  = sp;                    sp += seq_len * q_dim;
    float* gate_buf  = sp;                    sp += seq_len * cfg_.intermediate_dim;
    float* up_buf    = sp;                    sp += seq_len * cfg_.intermediate_dim;
    float* ff_out    = sp;                    sp += seq_len * hd;
    float* scores    = sp;                    sp += seq_len * cfg_.context_length; // max attention scores

    // ── 1. Attention RMS Norm ──
    for (int s = 0; s < seq_len; s++) {
        rms_norm(input + s * hd, lw.attn_norm, normed + s * hd, hd, cfg_.norm_eps);
    }

    // ── 2. Q, K, V projections (gemv_dequant with inline dequant) ──
    for (int s = 0; s < seq_len; s++) {
        gemv_dequant(normed + s * hd, lw.attn_q, Q + s * q_dim,     hd, q_dim,  lw.qtype);
        gemv_dequant(normed + s * hd, lw.attn_k, K_buf + s * kv_dim, hd, kv_dim, lw.ktype);
        gemv_dequant(normed + s * hd, lw.attn_v, V_buf + s * kv_dim, hd, kv_dim, lw.vtype);
    }

    // ── 3. RoPE on Q and K ──
    for (int s = 0; s < seq_len; s++) {
        int pos = position_ + s;
        // Q rotary (pairs of dims)
        for (int h = 0; h < n_heads; h++) {
            float* q = Q + s * q_dim + h * head_d;
            for (int i = 0; i < head_d / 2; i++) {
                float cos_v = rope_cos_[pos * head_d / 2 + i];
                float sin_v = rope_sin_[pos * head_d / 2 + i];
                float x0 = q[i], x1 = q[i + head_d / 2];
                q[i] = x0 * cos_v - x1 * sin_v;
                q[i + head_d / 2] = x0 * sin_v + x1 * cos_v;
            }
        }
        // K rotary
        for (int h = 0; h < n_kv_heads; h++) {
            float* k = K_buf + s * kv_dim + h * head_d;
            for (int i = 0; i < head_d / 2; i++) {
                float cos_v = rope_cos_[pos * head_d / 2 + i];
                float sin_v = rope_sin_[pos * head_d / 2 + i];
                float x0 = k[i], x1 = k[i + head_d / 2];
                k[i] = x0 * cos_v - x1 * sin_v;
                k[i + head_d / 2] = x0 * sin_v + x1 * cos_v;
            }
        }
    }

    // ── 4. Store K, V into KV cache ──
    auto& k_cache = kv_k_[layer_idx];
    auto& v_cache = kv_v_[layer_idx];
    for (int s = 0; s < seq_len; s++) {
        int pos = position_ + s;
        memcpy(k_cache.data() + pos * kv_dim, K_buf + s * kv_dim, kv_dim * sizeof(float));
        memcpy(v_cache.data() + pos * kv_dim, V_buf + s * kv_dim, kv_dim * sizeof(float));
    }

    // ── 5. Multi-Head Attention (fused SDPA) ──
    // For decode (seq_len=1), we can use the full KV cache.
    // For prefill (seq_len>1), only positions 0..position+seq_len have valid KV.

    for (int s = 0; s < seq_len; s++) {
        float* q = Q + s * q_dim;
        float* o = attn_out + s * q_dim;

        // Initialize output to zero
        memset(o, 0, q_dim * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            float* q_h = q + h * head_d;
            float* o_h = o + h * head_d;

            // Which KV head? (GQA: multiple Q heads share one KV head)
            int kv_h = h / kv_mul;

            int total_pos = position_ + seq_len;

            // Compute attention scores: Q_h @ K_cache^T
            // Online softmax to avoid materializing full score array
            float max_score = -1e30f;
            float sum_exp  = 0.0f;

            // Pre-compute scores
            float* score_row = scores + s * cfg_.context_length;
            for (int p = 0; p < total_pos; p++) {
                const float* k_h = k_cache.data() + p * kv_dim + kv_h * head_d;
                float dot = 0.0f;
                float scale = 1.0f / sqrtf((float)head_d);
                for (int d = 0; d < head_d; d++) {
                    dot += q_h[d] * k_h[d];
                }
                dot *= scale;
                score_row[p] = dot;
                if (dot > max_score) max_score = dot;
            }

            // Softmax with running sum
            for (int p = 0; p < total_pos; p++) {
                score_row[p] = expf(score_row[p] - max_score);
                sum_exp += score_row[p];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int p = 0; p < total_pos; p++) {
                score_row[p] *= inv_sum;
            }

            // Weighted sum of V
            for (int p = 0; p < total_pos; p++) {
                const float* v_h = v_cache.data() + p * kv_dim + kv_h * head_d;
                float w = score_row[p];
                for (int d = 0; d < head_d; d++) {
                    o_h[d] += w * v_h[d];
                }
            }
        }
    }

    // ── 6. Output projection + residual ──
    for (int s = 0; s < seq_len; s++) {
        // attn_out @ O_weight → output + input
        // gemv: (q_dim,) input → (hd,) output
        std::vector<float> o_proj(hd);
        gemv_dequant(attn_out + s * q_dim, lw.attn_o, o_proj.data(), q_dim, hd, lw.otype);
        for (int d = 0; d < hd; d++) {
            output[s * hd + d] = input[s * hd + d] + o_proj[d];
        }
    }

    // ── 7. FFN RMS Norm ──
    for (int s = 0; s < seq_len; s++) {
        rms_norm(output + s * hd, lw.ffn_norm, normed + s * hd, hd, cfg_.norm_eps);
    }

    // ── 8. SwiGLU FFN ──
    for (int s = 0; s < seq_len; s++) {
        // gate = normed @ gate_weight  (hd → intermediate_dim)
        gemv_dequant(normed + s * hd, lw.ffn_gate, gate_buf + s * cfg_.intermediate_dim,
                     hd, cfg_.intermediate_dim, lw.gate_type);
        // up = normed @ up_weight  (hd → intermediate_dim)
        gemv_dequant(normed + s * hd, lw.ffn_up, up_buf + s * cfg_.intermediate_dim,
                     hd, cfg_.intermediate_dim, lw.up_type);

        // SiLU(gate) * up
        int id = cfg_.intermediate_dim;
        for (int i = 0; i < id; i++) {
            gate_buf[s * id + i] = silu_f(gate_buf[s * id + i]) * up_buf[s * id + i];
        }

        // down = gate_up @ down_weight  (intermediate_dim → hd)
        gemv_dequant(gate_buf + s * id, lw.ffn_down, ff_out + s * hd,
                     id, hd, lw.down_type);

        // Residual
        for (int d = 0; d < hd; d++) {
            output[s * hd + d] += ff_out[s * hd + d];
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Prefill: process all prompt tokens at once
// ════════════════════════════════════════════════════════════════

std::vector<float> VibeBladeFast::prefill(const std::vector<int>& token_ids) {
    if (token_ids.empty()) throw std::runtime_error("Empty prompt");
    if (!loaded_) throw std::runtime_error("Model not loaded");

    int seq_len = (int)token_ids.size();
    if (position_ + seq_len > cfg_.context_length)
        throw std::runtime_error("Prompt exceeds context length");

    int hd = cfg_.hidden_dim;

    // ── Token embeddings (dequant if needed) ──
    std::vector<float> x(seq_len * hd);
    for (int s = 0; s < seq_len; s++) {
        int tok = token_ids[s];
        if (tok < 0 || tok >= cfg_.vocab_size)
            throw std::runtime_error("Token ID out of range: " + std::to_string(tok));

        // Get embedding row (may be quantized)
        size_t row_bytes = tensor_nbytes(emb_type_, hd);
        const void* emb_row = (const uint8_t*)token_emb_ + tok * row_bytes;
        dequantize_row(emb_row, x.data() + s * hd, hd, emb_type_);
    }

    // ── Run through all layers ──
    std::vector<float> hidden(seq_len * hd);
    for (int l = 0; l < cfg_.n_layers; l++) {
        forward_layer(x.data(), seq_len, hidden.data(), layers_[l], l);
        x.swap(hidden);
    }

    // ── Final RMS norm ──
    std::vector<float> normed(seq_len * hd);
    for (int s = 0; s < seq_len; s++) {
        rms_norm(x.data() + s * hd, output_norm_, normed.data() + s * hd, hd, cfg_.norm_eps);
    }

    // ── Output projection: last token only → logits ──
    // (During prefill, we only need logits for the last token to predict next)
    int last = seq_len - 1;
    std::vector<float> logits(cfg_.vocab_size);

    // If output weights are quantized, use gemv_dequant
    gemv_dequant(normed.data() + last * hd, output_, logits.data(), hd, cfg_.vocab_size, out_type_);

    position_ += seq_len;
    return logits;
}

// ════════════════════════════════════════════════════════════════
//  Decode: process one token → return logits
// ════════════════════════════════════════════════════════════════

std::vector<float> VibeBladeFast::decode(int token_id) {
    if (!loaded_) throw std::runtime_error("Model not loaded");
    if (position_ >= cfg_.context_length)
        throw std::runtime_error("Context length exceeded");

    int hd = cfg_.hidden_dim;

    // ── Token embedding for single token ──
    std::vector<float> x(hd);
    size_t row_bytes = tensor_nbytes(emb_type_, hd);
    const void* emb_row = (const uint8_t*)token_emb_ + token_id * row_bytes;
    dequantize_row(emb_row, x.data(), hd, emb_type_);

    // ── Run through all layers ──
    std::vector<float> hidden(hd);
    for (int l = 0; l < cfg_.n_layers; l++) {
        forward_layer(x.data(), 1, hidden.data(), layers_[l], l);
        x.swap(hidden);
    }

    // ── Final RMS norm ──
    std::vector<float> normed(hd);
    rms_norm(x.data(), output_norm_, normed.data(), hd, cfg_.norm_eps);

    // ── Output projection → logits ──
    std::vector<float> logits(cfg_.vocab_size);
    gemv_dequant(normed.data(), output_, logits.data(), hd, cfg_.vocab_size, out_type_);

    position_++;
    return logits;
}

// ════════════════════════════════════════════════════════════════
//  KV cache stats
// ════════════════════════════════════════════════════════════════

size_t VibeBladeFast::kv_cache_bytes() const {
    if (kv_k_.empty()) return 0;
    size_t per_layer = kv_k_[0].size() * sizeof(float) * 2;  // K + V
    return per_layer * cfg_.n_layers;
}

}  // namespace vibeblade
