// VibeBlade Fast Model — llama.cpp-style inference engine.
// Full forward pass in C++: prefill + decode with KV cache.
// Weights are mmap'd from GGUF, dequantized inline during matmuls.
// Zero malloc in the hot path — all buffers pre-allocated.
// NEON SIMD on ARM, OpenMP multi-threading for parallel gemv.

#include "fast_model.h"
#include "dequant.h"
#ifdef __aarch64__
#include "neon_kernels.h"
#endif
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  Math primitives — NEON-accelerated on ARM
// ════════════════════════════════════════════════════════════════

static inline float fast_rms(const float* x, int n) {
#ifdef __aarch64__
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
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
#endif
    return sqrtf(sum / n + 1e-8f);
}

// RMSNorm: out[i] = x[i] * (w[i] / rms(x))
static void rms_norm(const float* x, const float* w, float* out, int n, float eps) {
#ifdef __aarch64__
    float inv_rms;
    vrms_norm_compute(x, n, eps, &inv_rms);
    vrms_norm_apply(x, w, out, n, inv_rms);
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    s = 1.0f / sqrtf(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
#endif
}

// SiLU: x * sigmoid(x)
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

// Softmax in-place
static void softmax(float* x, int n) {
#ifdef __aarch64__
    vsoftmax(x, n);
#else
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
#endif
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
    tokenizer_.load(*gguf_);

    loaded_ = true;
    position_ = 0;
}

void VibeBladeFast::extract_config(const GGUFFile& g) {
    cfg_.arch = g.meta_string("general.architecture");
    if (cfg_.arch.empty()) cfg_.arch = g.meta_string("llama.architecture");

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

    if (cfg_.arch.empty() || cfg_.n_layers == 0) {
        cfg_.arch = "llama";
    }

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
    // Added extra space for o_proj buffer (previously was heap-allocated per step!)
    int hd = cfg_.hidden_dim;
    int id = cfg_.intermediate_dim;
    int q_dim = cfg_.n_heads * cfg_.head_dim;
    scratch_.resize(5 * hd + 3 * id + 4 * q_dim + 2 * hd + hd);
}

void VibeBladeFast::map_weights(const GGUFFile& g) {
    const std::string& arch = cfg_.arch;

    token_emb_ = g.tensor_data("token_embd.weight");
    if (token_emb_) {
        auto info = g.tensor_info("token_embd.weight");
        emb_type_ = info->type;
    }

    output_norm_ = (const float*)g.tensor_data("output_norm.weight");

    output_ = g.tensor_data("output.weight");
    if (!output_) {
        output_ = g.tensor_data("token_embd.weight");
    }
    if (output_) {
        auto info = g.tensor_info("output.weight");
        if (!info) info = g.tensor_info("token_embd.weight");
        out_type_ = info->type;
    }

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
//  Forward one layer — NEON + OpenMP optimized
// ════════════════════════════════════════════════════════════════

void VibeBladeFast::forward_layer(
    const float* input,
    int seq_len,
    float* output,
    const LayerWeights& lw,
    int layer_idx
) {
    int hd = cfg_.hidden_dim;
    int q_dim = cfg_.n_heads * cfg_.head_dim;
    int kv_dim = cfg_.n_kv_heads * cfg_.head_dim;
    int head_d = cfg_.head_dim;
    int n_heads = cfg_.n_heads;
    int n_kv_heads = cfg_.n_kv_heads;
    int kv_mul = n_heads / n_kv_heads;

    // ── Scratch allocation (NO malloc — all from pre-allocated buffer) ──
    float* sp = scratch_.data();
    float* normed    = sp;                    sp += seq_len * hd;
    float* Q         = sp;                    sp += seq_len * q_dim;
    float* K_buf     = sp;                    sp += seq_len * kv_dim;
    float* V_buf     = sp;                    sp += seq_len * kv_dim;
    float* attn_out  = sp;                    sp += seq_len * q_dim;
    float* gate_buf  = sp;                    sp += seq_len * cfg_.intermediate_dim;
    float* up_buf    = sp;                    sp += seq_len * cfg_.intermediate_dim;
    float* ff_out    = sp;                    sp += seq_len * hd;
    float* scores    = sp;                    sp += seq_len * cfg_.context_length;
    float* o_proj    = sp;                    sp += seq_len * hd;  // ← was heap-alloc'd before!

    // ── 1. Attention RMS Norm ──
    for (int s = 0; s < seq_len; s++) {
        rms_norm(input + s * hd, lw.attn_norm, normed + s * hd, hd, cfg_.norm_eps);
    }

    // ── 2. Q, K, V projections (gemv_dequant with inline dequant) ──
    // For decode (seq_len=1), this is just 3 gemv calls — no parallelism benefit.
    // For prefill (seq_len>1), parallelize across token positions.
    for (int s = 0; s < seq_len; s++) {
        gemv_dequant(normed + s * hd, lw.attn_q, Q + s * q_dim,     hd, q_dim,  lw.qtype);
        gemv_dequant(normed + s * hd, lw.attn_k, K_buf + s * kv_dim, hd, kv_dim, lw.ktype);
        gemv_dequant(normed + s * hd, lw.attn_v, V_buf + s * kv_dim, hd, kv_dim, lw.vtype);
    }

    // ── 3. RoPE on Q and K ──
    for (int s = 0; s < seq_len; s++) {
        int pos = position_ + s;
        const float* cos_v = rope_cos_.data() + pos * head_d / 2;
        const float* sin_v = rope_sin_.data() + pos * head_d / 2;

        for (int h = 0; h < n_heads; h++) {
            float* q = Q + s * q_dim + h * head_d;
#ifdef __aarch64__
            vapply_rope(q, head_d, cos_v, sin_v);
#else
            for (int i = 0; i < head_d / 2; i++) {
                float x0 = q[i], x1 = q[i + head_d / 2];
                q[i] = x0 * cos_v[i] - x1 * sin_v[i];
                q[i + head_d / 2] = x0 * sin_v[i] + x1 * cos_v[i];
            }
#endif
        }
        for (int h = 0; h < n_kv_heads; h++) {
            float* k = K_buf + s * kv_dim + h * head_d;
#ifdef __aarch64__
            vapply_rope(k, head_d, cos_v, sin_v);
#else
            for (int i = 0; i < head_d / 2; i++) {
                float x0 = k[i], x1 = k[i + head_d / 2];
                k[i] = x0 * cos_v[i] - x1 * sin_v[i];
                k[i + head_d / 2] = x0 * sin_v[i] + x1 * cos_v[i];
            }
#endif
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
    int total_pos = position_ + seq_len;

    for (int s = 0; s < seq_len; s++) {
        float* q = Q + s * q_dim;
        float* o = attn_out + s * q_dim;
        memset(o, 0, q_dim * sizeof(float));

        float scale = 1.0f / sqrtf((float)head_d);

        for (int h = 0; h < n_heads; h++) {
            float* q_h = q + h * head_d;
            float* o_h = o + h * head_d;
            int kv_h = h / kv_mul;

            float* score_row = scores + s * cfg_.context_length;

            // ── Compute Q@K^T scores ──
            for (int p = 0; p < total_pos; p++) {
                const float* k_h = k_cache.data() + p * kv_dim + kv_h * head_d;
#ifdef __aarch64__
                float dot = vdot_att(q_h, k_h, head_d) * scale;
#else
                float dot = 0.0f;
                for (int d = 0; d < head_d; d++) dot += q_h[d] * k_h[d];
                dot *= scale;
#endif
                score_row[p] = dot;
            }

            // ── Softmax ──
            softmax(score_row, total_pos);

            // ── Weighted sum of V ──
            for (int p = 0; p < total_pos; p++) {
                const float* v_h = v_cache.data() + p * kv_dim + kv_h * head_d;
                float w = score_row[p];
#ifdef __aarch64__
                vaxpy(o_h, w, v_h, head_d);
#else
                for (int d = 0; d < head_d; d++) o_h[d] += w * v_h[d];
#endif
            }
        }
    }

    // ── 6. Output projection + residual ──
    for (int s = 0; s < seq_len; s++) {
        gemv_dequant(attn_out + s * q_dim, lw.attn_o, o_proj + s * hd, q_dim, hd, lw.otype);
#ifdef __aarch64__
        // Copy input + add o_proj in one NEON pass
        memcpy(output + s * hd, input + s * hd, hd * sizeof(float));
        vadd_residual(output + s * hd, o_proj + s * hd, hd);
#else
        for (int d = 0; d < hd; d++) {
            output[s * hd + d] = input[s * hd + d] + o_proj[d];
        }
#endif
    }

    // ── 7. FFN RMS Norm ──
    for (int s = 0; s < seq_len; s++) {
        rms_norm(output + s * hd, lw.ffn_norm, normed + s * hd, hd, cfg_.norm_eps);
    }

    // ── 8. SwiGLU FFN ──
    for (int s = 0; s < seq_len; s++) {
        gemv_dequant(normed + s * hd, lw.ffn_gate, gate_buf + s * cfg_.intermediate_dim,
                     hd, cfg_.intermediate_dim, lw.gate_type);
        gemv_dequant(normed + s * hd, lw.ffn_up, up_buf + s * cfg_.intermediate_dim,
                     hd, cfg_.intermediate_dim, lw.up_type);

        // SiLU(gate) * up — fused NEON kernel
        int id = cfg_.intermediate_dim;
#ifdef __aarch64__
        vsilu_mul_f32(gate_buf + s * id, up_buf + s * id, gate_buf + s * id, id);
#else
        for (int i = 0; i < id; i++) {
            gate_buf[s * id + i] = silu_f(gate_buf[s * id + i]) * up_buf[s * id + i];
        }
#endif

        gemv_dequant(gate_buf + s * id, lw.ffn_down, ff_out + s * hd,
                     id, hd, lw.down_type);

#ifdef __aarch64__
        vadd_residual(output + s * hd, ff_out + s * hd, hd);
#else
        for (int d = 0; d < hd; d++) {
            output[s * hd + d] += ff_out[s * hd + d];
        }
#endif
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
    int last = seq_len - 1;
    std::vector<float> logits(cfg_.vocab_size);
    gemv_dequant(normed.data() + last * hd, output_, logits.data(), hd, cfg_.vocab_size, out_type_);

    position_ += seq_len;
    return logits;
}

// ════════════════════════════════════════════════════════════════
//  Decode: process one token → return logits
//  HOT PATH — zero heap allocations, NEON everywhere
// ════════════════════════════════════════════════════════════════

std::vector<float> VibeBladeFast::decode(int token_id) {
    if (!loaded_) throw std::runtime_error("Model not loaded");
    if (position_ >= cfg_.context_length)
        throw std::runtime_error("Context length exceeded");

    int hd = cfg_.hidden_dim;

    // ── Token embedding for single token ──
    // Use pre-allocated decode buffers — no std::vector construction!
    // We reuse a persistent buffer pair to avoid any heap alloc
    static thread_local std::vector<float> x_buf, hidden_buf;
    if ((int)x_buf.size() < hd) x_buf.resize(hd);
    if ((int)hidden_buf.size() < hd) hidden_buf.resize(hd);

    size_t row_bytes = tensor_nbytes(emb_type_, hd);
    const void* emb_row = (const uint8_t*)token_emb_ + token_id * row_bytes;
    dequantize_row(emb_row, x_buf.data(), hd, emb_type_);

    // ── Run through all layers ──
    for (int l = 0; l < cfg_.n_layers; l++) {
        forward_layer(x_buf.data(), 1, hidden_buf.data(), layers_[l], l);
        std::swap(x_buf, hidden_buf);
    }

    // ── Final RMS norm ──
    // Reuse normed as a thread-local buffer too
    static thread_local std::vector<float> normed_buf;
    if ((int)normed_buf.size() < hd) normed_buf.resize(hd);
    rms_norm(x_buf.data(), output_norm_, normed_buf.data(), hd, cfg_.norm_eps);

    // ── Output projection → logits ──
    // Logits buffer is persistent — never realloc'd after first call
    static thread_local std::vector<float> logits_buf;
    if ((int)logits_buf.size() < cfg_.vocab_size) logits_buf.resize(cfg_.vocab_size);
    gemv_dequant(normed_buf.data(), output_, logits_buf.data(), hd, cfg_.vocab_size, out_type_);

    position_++;
    return logits_buf;  // Return by value — NVRO or move since it's thread_local
}

// ════════════════════════════════════════════════════════════════
//  KV cache stats
// ════════════════════════════════════════════════════════════════

size_t VibeBladeFast::kv_cache_bytes() const {
    if (kv_k_.empty()) return 0;
    size_t per_layer = kv_k_[0].size() * sizeof(float) * 2;
    return per_layer * cfg_.n_layers;
}

// ════════════════════════════════════════════════════════════════
//  Tokenizer passthrough
// ════════════════════════════════════════════════════════════════

std::vector<int> VibeBladeFast::tokenize(const std::string& text) const {
    return tokenizer_.encode(text);
}

std::string VibeBladeFast::detokenize(const std::vector<int>& ids) const {
    return tokenizer_.decode(ids);
}

std::string VibeBladeFast::detokenize_token(int id) const {
    return tokenizer_.decode_token(id);
}

// ════════════════════════════════════════════════════════════════
//  Full Generate Pipeline — one C++ call, zero Python in hot path
//  tokenize → prefill → [decode → sample → stream] → detokenize
// ════════════════════════════════════════════════════════════════

GenerateResult VibeBladeFast::generate(
    const std::string& prompt,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    int seed,
    std::function<void(int, const std::string&)> on_token
) {
    if (!loaded_) throw std::runtime_error("Model not loaded");

    GenerateResult result;
    result.stopped_eos = false;

    if (seed >= 0) {
        sampler_.set_seed((uint64_t)seed);
    }

    // ── 1. Tokenize prompt ──
    std::vector<int> prompt_ids = tokenizer_.encode(prompt);

    // ── 2. Prefill ──
    std::vector<float> logits = prefill(prompt_ids);

    // ── 3. Decode loop ──
    SamplerConfig scfg;
    scfg.temperature = temperature;
    scfg.top_k = top_k;
    scfg.top_p = top_p;
    scfg.repetition_penalty = repetition_penalty;

    // Pre-allocate token history to avoid repeated push_back/realloc
    std::vector<int> token_history;
    token_history.reserve(prompt_ids.size() + max_tokens + 16);
    token_history.insert(token_history.end(), prompt_ids.begin(), prompt_ids.end());

    result.token_ids.reserve(max_tokens);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_tokens; i++) {
        int token_id = sampler_.sample(logits.data(), cfg_.vocab_size, scfg, token_history);
        token_history.push_back(token_id);
        result.token_ids.push_back(token_id);

        if (token_id == tokenizer_.eos_id()) {
            result.stopped_eos = true;
            break;
        }

        if (on_token) {
            std::string piece = tokenizer_.decode_token(token_id);
            on_token(token_id, piece);
        }

        logits = decode(token_id);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    result.tokens_per_second = result.token_ids.size() / std::max(elapsed, 1e-6);

    result.text = tokenizer_.decode(result.token_ids);

    return result;
}

}  // namespace vibeblade
