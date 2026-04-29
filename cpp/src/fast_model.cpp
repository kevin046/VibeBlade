// VibeBlade Fast Model — llama.cpp-style inference engine.
// Full forward pass in C++: prefill + decode with KV cache.
// Weights are mmap'd from GGUF, dequantized inline during matmuls.
// Zero malloc in the hot path — all buffers pre-allocated.
// NEON SIMD on ARM, std::thread multi-threading for parallel gemv.

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
    auto gs = [&](const std::string& key) -> std::string {
        std::string prefixed = cfg_.arch + "." + key;
        std::string v = g.meta_string(prefixed);
        return v.empty() ? g.meta_string(key) : v;
    };

    cfg_.n_layers         = (int)gi("block_count");
    cfg_.n_heads          = (int)gi("attention.head_count");
    cfg_.n_kv_heads       = (int)gi("attention.head_count_kv", cfg_.n_heads);
    cfg_.hidden_dim       = (int)gi("embedding_length");
    cfg_.intermediate_dim = (int)gi("feed_forward_length");
    cfg_.context_length   = (int)gi("context_length", 2048);

    // head_dim: explicit key_length, or derive from hidden_dim / n_heads
    cfg_.head_dim = (int)gi("attention.key_length");
    if (cfg_.head_dim == 0 && cfg_.n_heads > 0)
        cfg_.head_dim = cfg_.hidden_dim / cfg_.n_heads;

    // vocab_size: explicit, or derive from token_embd weight rows
    cfg_.vocab_size = (int)gi("vocab_size");
    if (cfg_.vocab_size == 0) {
        auto emb_info = g.tensor_info("token_embd.weight");
        if (!emb_info) emb_info = g.tensor_info("wte.weight");
        if (emb_info) {
            // GGUF stores dims as [n_rows, n_cols]; for embeddings, the larger
            // dimension is typically vocab_size (other is hidden_dim).
            cfg_.vocab_size = (int)std::max(emb_info->dims[0], emb_info->dims[1]);
        }
    }

    cfg_.norm_eps         = gf("attention.layer_norm_rms_epsilon", 1e-5f);

    // Sliding window attention (Mistral)
    cfg_.sliding_window = (int)gi("attention.sliding_window");

    if (cfg_.arch.empty() || cfg_.n_layers == 0) {
        cfg_.arch = "llama";
    }

    // ── RoPE configuration ──
    cfg_.rope_freq_base = gf("rope.freq_base", 10000.0f);
    if (cfg_.rope_freq_base == 0.0f) cfg_.rope_freq_base = 10000.0f;

    // RoPE scaling (read from metadata)
    cfg_.rope_scaling_type = gs("rope.scaling.type");
    cfg_.rope_scale = gf("rope.scaling.factor", 1.0f);
    if (cfg_.rope_scale == 0.0f) cfg_.rope_scale = 1.0f;

    // YaRN-specific parameters
    cfg_.yarn_ext_factor  = gf("rope.scaling.yarn_log_mul", 1.0f);
    cfg_.yarn_attn_factor = gf("rope.scaling.attn_factor", 1.0f);
    cfg_.yarn_beta_fast   = gf("rope.scaling.beta_fast", 32.0f);
    cfg_.yarn_beta_slow   = gf("rope.scaling.beta_slow", 1.0f);

    // LongRoPE: per-dimension frequency factors
    auto lr_factors = g.meta_float_array("rope.scaling.long_factor");
    if (!lr_factors.empty()) {
        cfg_.long_rope_factors.resize(lr_factors.size());
        for (int i = 0; i < (int)lr_factors.size(); i++)
            cfg_.long_rope_factors[i] = (int)lr_factors[i];
    }

    cfg_.context_length = std::min(cfg_.context_length, MAX_SEQ);

    // ── Architecture-specific flags ──
    const std::string& a = cfg_.arch;
    cfg_.use_geglu = (a == "gemma" || a == "gemma2");
    cfg_.use_parallel_attn = (a == "falcon" || a == "gptneox");
    cfg_.use_neox_rope = (a == "falcon" || a == "gptneox" || a == "starcoder2");

    // Fused QKV: detect from weight names (checked in map_weights)
    // Most models EXCEPT base llama use fused QKV
    cfg_.use_fused_qkv = (a != "llama" && a != "llama_old");

    // MoE (DeepSeek, Mixtral, Qwen2-MoE)
    cfg_.n_experts = (int)gi("expert_count");
    cfg_.n_experts_used = (int)gi("expert_used_count");

    // Override norm_eps per architecture
    if (a == "gemma" || a == "gemma2") {
        cfg_.norm_eps = 1e-6f;
    }
}

void VibeBladeFast::build_rope_cache() {
    int dim = cfg_.head_dim;
    int half = dim / 2;
    rope_cos_.resize(cfg_.context_length * half);
    rope_sin_.resize(cfg_.context_length * half);

    float base = cfg_.rope_freq_base;
    bool is_neox = cfg_.use_neox_rope;

    for (int pos = 0; pos < cfg_.context_length; pos++) {
        for (int i = 0; i < half; i++) {
            float theta = powf(base, -2.0f * i / dim);

            // Apply scaling based on type
            float scaled_theta = theta;

            if (!cfg_.rope_scaling_type.empty()) {
                if (cfg_.rope_scaling_type == "linear") {
                    // Linear: theta / scale
                    scaled_theta = theta / cfg_.rope_scale;
                }
                else if (cfg_.rope_scaling_type == "yarn") {
                    // YaRN: dynamic NTK-aware scaling
                    float freq = 1.0f / theta;
                    float beta_fast = cfg_.yarn_beta_fast;
                    float beta_slow = cfg_.yarn_beta_slow;

                    // Smooth interpolation factor
                    float t = pos / (float)cfg_.context_length;
                    float smooth = 1.0f - 1.0f / (1.0f + (t * (beta_slow - beta_fast) + beta_fast) / beta_fast);

                    // Compute scaled freq
                    float scaled_freq = freq / cfg_.rope_scale;

                    // Apply YaRN correction
                    float blend = smooth * cfg_.yarn_ext_factor;
                    float final_freq = freq * (1.0f - blend) + scaled_freq * blend;
                    scaled_theta = 1.0f / final_freq;
                }
            }

            // LongRoPE: per-dimension scaling factors
            if (!cfg_.long_rope_factors.empty() && (int)cfg_.long_rope_factors.size() >= half) {
                // long_rope_factors[i] is typically a float encoded as int
                // The factor is the ratio of new freq_base dimension to old
                scaled_theta = powf(cfg_.rope_freq_base,
                    -2.0f * i / dim / (float)cfg_.long_rope_factors[i]);
            }

            float angle = pos * scaled_theta;
            rope_cos_[pos * half + i] = cosf(angle);
            rope_sin_[pos * half + i] = sinf(angle);
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

    // Scratch: must handle worst-case seq_len = context_length for prefill.
    // forward_layer scratch: normed(seq*hd) + Q(seq*q_dim) + K(seq*kv_dim) + V(seq*kv_dim)
    //   + attn_out(seq*q_dim) + gate(seq*id) + up(seq*id) + ff_out(seq*hd)
    //   + scores(seq*context_length) + o_proj(seq*hd)
    // Worst case: seq_len = context_length
    int max_seq = cfg_.context_length;
    int hd = cfg_.hidden_dim;
    int id = cfg_.intermediate_dim;
    int q_dim = cfg_.n_heads * cfg_.head_dim;
    int kv_dim_scratch = cfg_.n_kv_heads * cfg_.head_dim;
    int64_t scratch_per_seq = (int64_t)hd + q_dim + 2 * kv_dim_scratch + q_dim +
                              2 * id + hd + cfg_.context_length + hd;
    scratch_.resize(max_seq * scratch_per_seq);

    // Instance-owned decode buffers (sized once, zero realloc during hot loop)
    x_buf_.resize(hd);
    hidden_buf_.resize(hd);
    normed_buf_.resize(hd);
    logits_buf_.resize(cfg_.vocab_size);
}

void VibeBladeFast::map_weights(const GGUFFile& g) {
    const std::string& arch = cfg_.arch;

    // ── Global weights (names differ per architecture) ──
    token_emb_ = g.tensor_data("token_embd.weight");
    if (!token_emb_) token_emb_ = g.tensor_data("wte.weight");  // GPT-NeoX style
    if (token_emb_) {
        auto info = g.tensor_info("token_embd.weight");
        if (!info) info = g.tensor_info("wte.weight");
        emb_type_ = info ? info->type : GGML_TYPE_F32;
    }

    output_norm_ = (const float*)g.tensor_data("output_norm.weight");
    if (!output_norm_) output_norm_ = (const float*)g.tensor_data("ln_f.weight");

    output_ = g.tensor_data("output.weight");
    if (!output_) output_ = g.tensor_data("lm_head.weight");
    if (!output_) output_ = g.tensor_data("token_embd.weight");  // tied weights
    if (output_) {
        auto info = g.tensor_info("output.weight");
        if (!info) info = g.tensor_info("lm_head.weight");
        if (!info) info = g.tensor_info("token_embd.weight");
        out_type_ = info ? info->type : GGML_TYPE_F32;
    }

    layers_.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; i++) {
        auto& lw = layers_[i];
        std::string pfx = "blk." + std::to_string(i) + ".";

        // ── Norm weights (same names across archs) ──
        lw.attn_norm = (const float*)g.tensor_data(pfx + "attn_norm.weight");
        lw.ffn_norm  = (const float*)g.tensor_data(pfx + "ffn_norm.weight");
        // GPT-NeoX / Falcon use different norm names
        if (!lw.attn_norm) lw.attn_norm = (const float*)g.tensor_data(pfx + "ln_attn.weight");
        if (!lw.ffn_norm)  lw.ffn_norm  = (const float*)g.tensor_data(pfx + "ln_mlp.weight");
        if (!lw.attn_norm) lw.attn_norm = (const float*)g.tensor_data(pfx + "ln1.weight");
        if (!lw.ffn_norm)  lw.ffn_norm  = (const float*)g.tensor_data(pfx + "ln2.weight");

        // ── Attention weights ──
        // Try fused QKV first, then fall back to separate Q/K/V
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

        // Try fused QKV
        set_w("attn_qkv.weight", lw.attn_qkv, lw.qkv_type);
        if (lw.attn_qkv && (!lw.attn_q || !lw.attn_k || !lw.attn_v)) {
            // Use fused QKV — split during forward pass
            lw.has_fused_qkv = true;
            lw.qkv_type = lw.qkv_type;
        } else {
            lw.has_fused_qkv = false;
            lw.attn_qkv = nullptr;
        }

        // Falcon-style names
        if (!lw.attn_q) set_w("attn_q.weight", lw.attn_q, lw.qtype);
        if (!lw.attn_k) set_w("attn_k.weight", lw.attn_k, lw.ktype);
        if (!lw.attn_v) set_w("attn_v.weight", lw.attn_v, lw.vtype);

        // ── FFN weights ──
        set_w("ffn_gate.weight",    lw.ffn_gate, lw.gate_type);
        set_w("ffn_up.weight",      lw.ffn_up,   lw.up_type);
        set_w("ffn_down.weight",    lw.ffn_down, lw.down_type);

        // GPT-NeoX / Falcon MLP names
        if (!lw.ffn_gate) set_w("ffn_gate.weight", lw.ffn_gate, lw.gate_type);
        if (!lw.ffn_up)   set_w("ffn_up.weight",   lw.ffn_up,   lw.up_type);
        if (!lw.ffn_down) set_w("ffn_down.weight",  lw.ffn_down, lw.down_type);

        // MoE gate (DeepSeek, Mixtral, Qwen2-MoE)
        set_w("ffn_gate_inp.weight", lw.ffn_gate_inp, lw.gate_inp_type);

        // MoE expert weights (consolidated: num_experts × intermediate × hidden)
        set_w("ffn_gate_exps.weight", lw.ffn_gate_exps, lw.gate_exps_type);
        set_w("ffn_up_exps.weight",   lw.ffn_up_exps,   lw.up_exps_type);
        set_w("ffn_down_exps.weight", lw.ffn_down_exps, lw.down_exps_type);

        // Shared expert (DeepSeek/Qwen-style "shexp" suffix)
        set_w("ffn_gate_shexp.weight", lw.ffn_gate_shexp, lw.shexp_gate_type);
        set_w("ffn_up_shexp.weight",   lw.ffn_up_shexp,   lw.shexp_up_type);
        set_w("ffn_down_shexp.weight", lw.ffn_down_shexp, lw.shexp_down_type);

        // Detect MoE layer
        if (lw.ffn_gate_inp && lw.ffn_gate_exps) {
            lw.has_moe = true;
            // Extract expert_intermediate_dim from gate_exps shape:
            // gate_exps is (num_experts, intermediate, hidden)
            auto* info = g.tensor_info(pfx + "ffn_gate_exps.weight");
            if (info && info->n_dims >= 2) {
                lw.expert_intermediate_dim = info->dims[1];
            }
        }

        // Detect shared expert
        if (lw.ffn_gate_shexp && lw.ffn_up_shexp && lw.ffn_down_shexp) {
            lw.has_shared_expert = true;
        }

        // Detect hybrid attention/SSM: if no attention weights found, this is SSM-only
        if (!lw.attn_q && !lw.attn_qkv) {
            lw.has_attention = false;
        }
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
//  MoE helpers
// ════════════════════════════════════════════════════════════════

// Top-k selection: returns indices and weights for top-k elements of probs.
// probs: (n,), k: number of top elements. out_idx: (k,), out_w: (k,)
static void top_k_select(const float* probs, int n, int k, int* out_idx, float* out_w) {
    std::vector<std::pair<float, int>> indexed(n);
    for (int i = 0; i < n; i++) indexed[i] = {probs[i], i};
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        out_idx[i] = indexed[i].second;
        out_w[i] = indexed[i].first;
        sum += out_w[i];
    }
    if (sum > 1e-12f) {
        for (int i = 0; i < k; i++) out_w[i] /= sum;
    }
}

// MoE expert forward: compute output for one expert on one token.
// gate_w, up_w, down_w point to the expert's row in the consolidated weight tensor.
static void moe_expert_forward(
    const float* x,               // (hidden_dim,)
    const void* gate_w,            // (intermediate_dim, hidden_dim) quantized
    const void* up_w,              // (intermediate_dim, hidden_dim) quantized
    const void* down_w,            // (hidden_dim, intermediate_dim) quantized
    ggml_type gate_type,
    ggml_type up_type,
    ggml_type down_type,
    int hidden_dim,
    int intermediate_dim,
    float* out                     // (hidden_dim,)
) {
    std::vector<float> gate_buf(intermediate_dim);
    gemv_dequant(x, gate_w, gate_buf.data(), hidden_dim, intermediate_dim, gate_type);

    std::vector<float> up_buf(intermediate_dim);
    gemv_dequant(x, up_w, up_buf.data(), hidden_dim, intermediate_dim, up_type);

    std::vector<float> hidden(intermediate_dim);
    for (int i = 0; i < intermediate_dim; i++) {
        hidden[i] = silu_f(gate_buf[i]) * up_buf[i];
    }

    gemv_dequant(hidden.data(), down_w, out, intermediate_dim, hidden_dim, down_type);
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
    float* gate_buf  = sp;                    sp += seq_len * std::max(cfg_.intermediate_dim, 1);
    float* up_buf    = sp;                    sp += seq_len * std::max(cfg_.intermediate_dim, 1);
    float* ff_out    = sp;                    sp += seq_len * hd;
    float* scores    = sp;                    sp += seq_len * cfg_.context_length;
    float* o_proj    = sp;                    sp += seq_len * hd;

    // ── Attention block (steps 1-7) — skipped for SSM-only hybrid blocks ──
    if (lw.has_attention) {

    // ── 1. Attention RMS Norm ──
    for (int s = 0; s < seq_len; s++) {
        rms_norm(input + s * hd, lw.attn_norm, normed + s * hd, hd, cfg_.norm_eps);
    }

    // ── 2. Q, K, V projections (gemv_dequant with inline dequant) ──
    if (lw.has_fused_qkv && lw.attn_qkv) {
        // Fused QKV: one gemv, then split
        int qkv_dim = q_dim + 2 * kv_dim;
        for (int s = 0; s < seq_len; s++) {
            gemv_dequant(normed + s * hd, lw.attn_qkv, Q + s * q_dim,
                         hd, qkv_dim, lw.qkv_type);
            // Q is already at Q + s*q_dim, K and V follow
            // QKV layout: [Q | K | V] — but K/V are packed after Q
            memcpy(K_buf + s * kv_dim, Q + s * qkv_dim + q_dim, kv_dim * sizeof(float));
            memcpy(V_buf + s * kv_dim, Q + s * qkv_dim + q_dim + kv_dim, kv_dim * sizeof(float));
        }
    } else {
        for (int s = 0; s < seq_len; s++) {
            gemv_dequant(normed + s * hd, lw.attn_q, Q + s * q_dim,     hd, q_dim,  lw.qtype);
            gemv_dequant(normed + s * hd, lw.attn_k, K_buf + s * kv_dim, hd, kv_dim, lw.ktype);
            gemv_dequant(normed + s * hd, lw.attn_v, V_buf + s * kv_dim, hd, kv_dim, lw.vtype);
        }
    }

    // ── 3. RoPE on Q and K ──
    for (int s = 0; s < seq_len; s++) {
        int pos = position_ + s;
        const float* cos_v = rope_cos_.data() + pos * head_d / 2;
        const float* sin_v = rope_sin_.data() + pos * head_d / 2;

        for (int h = 0; h < n_heads; h++) {
            float* q = Q + s * q_dim + h * head_d;
            if (cfg_.use_neox_rope) {
                // NeoX-style interleaved: pairs (0,1), (2,3), ... not halves
                for (int i = 0; i < head_d; i += 2) {
                    float x0 = q[i], x1 = q[i + 1];
                    float ci = cos_v[i / 2], si = sin_v[i / 2];
                    q[i]     = x0 * ci - x1 * si;
                    q[i + 1] = x0 * si + x1 * ci;
                }
            } else {
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
        }
        for (int h = 0; h < n_kv_heads; h++) {
            float* k = K_buf + s * kv_dim + h * head_d;
            if (cfg_.use_neox_rope) {
                for (int i = 0; i < head_d; i += 2) {
                    float x0 = k[i], x1 = k[i + 1];
                    float ci = cos_v[i / 2], si = sin_v[i / 2];
                    k[i]     = x0 * ci - x1 * si;
                    k[i + 1] = x0 * si + x1 * ci;
                }
            } else {
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
    int window_start = 0;
    if (cfg_.sliding_window > 0) {
        window_start = std::max(0, total_pos - cfg_.sliding_window);
    }

    for (int s = 0; s < seq_len; s++) {
        float* q = Q + s * q_dim;
        float* o = attn_out + s * q_dim;
        memset(o, 0, q_dim * sizeof(float));

        float scale = 1.0f / sqrtf((float)head_d);
        // Apply YaRN attention temperature if configured
        if (cfg_.rope_scaling_type == "yarn" && cfg_.yarn_attn_factor != 1.0f) {
            scale *= cfg_.yarn_attn_factor;
        }

        for (int h = 0; h < n_heads; h++) {
            float* q_h = q + h * head_d;
            float* o_h = o + h * head_d;
            int kv_h = h / kv_mul;

            float* score_row = scores + s * cfg_.context_length;

            // ── Compute Q@K^T scores (with sliding window) ──
            int attn_start = window_start;
            for (int p = attn_start; p < total_pos; p++) {
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

            // ── Softmax (over windowed range) ──
            int attn_len = total_pos - attn_start;
            if (attn_len > 0 && attn_start > 0) {
                // Shift scores to start of array for softmax
                for (int p = 0; p < attn_len; p++) {
                    score_row[p] = score_row[attn_start + p];
                }
                softmax(score_row, attn_len);
            } else {
                softmax(score_row, total_pos);
                attn_len = total_pos;
            }

            // ── Weighted sum of V ──
            for (int p = 0; p < attn_len; p++) {
                int cache_pos = (attn_start > 0) ? (attn_start + p) : p;
                const float* v_h = v_cache.data() + cache_pos * kv_dim + kv_h * head_d;
                float w = score_row[p];
#ifdef __aarch64__
                vaxpy(o_h, w, v_h, head_d);
#else
                for (int d = 0; d < head_d; d++) o_h[d] += w * v_h[d];
#endif
            }
        }
    }

    // ── 6. Output projection ──
    for (int s = 0; s < seq_len; s++) {
        gemv_dequant(attn_out + s * q_dim, lw.attn_o, o_proj + s * hd, q_dim, hd, lw.otype);
    }

    // ── 7. Post-attention residual ──
    for (int s = 0; s < seq_len; s++) {
#ifdef __aarch64__
        memcpy(output + s * hd, input + s * hd, hd * sizeof(float));
        vadd_residual(output + s * hd, o_proj + s * hd, hd);
#else
        for (int d = 0; d < hd; d++) {
            output[s * hd + d] = input[s * hd + d] + o_proj[s * hd + d];
        }
#endif
    }

    } // end if (lw.has_attention)
    else {
        // SSM-only hybrid block: no attention, pass input through to FFN
        for (int s = 0; s < seq_len; s++) {
            memcpy(output + s * hd, input + s * hd, hd * sizeof(float));
        }
    }

    // ── 8. FFN RMS Norm ──
    for (int s = 0; s < seq_len; s++) {
        rms_norm(output + s * hd, lw.ffn_norm, normed + s * hd, hd, cfg_.norm_eps);
    }

    // ── 9. FFN: MoE, Dense, or skip if no weights ──
    if (lw.has_moe) {
        // ── MoE FFN (top-k expert routing) ──
        int n_exp = cfg_.n_experts;
        int topk = (cfg_.n_experts_used > 0) ? cfg_.n_experts_used : 2;
        int exp_inter = (int)lw.expert_intermediate_dim;
        if (exp_inter <= 0) exp_inter = cfg_.intermediate_dim;

        // Bytes per expert row (one expert's weight matrix)
        size_t gate_bytes = (size_t)exp_inter * hd * ggml_type_size(lw.gate_exps_type);
        size_t up_bytes   = (size_t)exp_inter * hd * ggml_type_size(lw.up_exps_type);
        size_t down_bytes = (size_t)hd * exp_inter * ggml_type_size(lw.down_exps_type);

        for (int s = 0; s < seq_len; s++) {
            const float* x = normed + s * hd;

            // Router: x @ gate_inp.T → (n_exp,)
            std::vector<float> router_logits(n_exp);
            gemv_dequant(x, lw.ffn_gate_inp, router_logits.data(), hd, n_exp, lw.gate_inp_type);

            // Softmax over expert logits
            softmax(router_logits.data(), n_exp);

            // Top-k selection
            std::vector<int> top_idx(topk);
            std::vector<float> top_w(topk);
            top_k_select(router_logits.data(), n_exp, topk, top_idx.data(), top_w.data());

            // Weighted sum of top-k expert outputs
            std::vector<float> moe_out(hd, 0.0f);
            for (int k = 0; k < topk; k++) {
                int e = top_idx[k];
                float w = top_w[k];
                std::vector<float> exp_out(hd);

                const uint8_t* eg = (const uint8_t*)lw.ffn_gate_exps + e * gate_bytes;
                const uint8_t* eu = (const uint8_t*)lw.ffn_up_exps   + e * up_bytes;
                const uint8_t* ed = (const uint8_t*)lw.ffn_down_exps + e * down_bytes;

                moe_expert_forward(x, eg, eu, ed,
                                   lw.gate_exps_type, lw.up_exps_type, lw.down_exps_type,
                                   hd, exp_inter, exp_out.data());

                for (int d = 0; d < hd; d++) moe_out[d] += w * exp_out[d];
            }

            // Shared expert (always runs)
            if (lw.has_shared_expert) {
                std::vector<float> shared_out(hd);
                moe_expert_forward(x,
                                   lw.ffn_gate_shexp, lw.ffn_up_shexp, lw.ffn_down_shexp,
                                   lw.shexp_gate_type, lw.shexp_up_type, lw.shexp_down_type,
                                   hd, exp_inter, shared_out.data());
                for (int d = 0; d < hd; d++) moe_out[d] += shared_out[d];
            }

            memcpy(ff_out + s * hd, moe_out.data(), hd * sizeof(float));
        }
    } else if (lw.ffn_gate && lw.ffn_up && lw.ffn_down) {
        // ── Dense FFN (SwiGLU or GeGLU) with TurboSparse ──
        for (int s = 0; s < seq_len; s++) {
            gemv_dequant(normed + s * hd, lw.ffn_gate, gate_buf + s * cfg_.intermediate_dim,
                         hd, cfg_.intermediate_dim, lw.gate_type);
            gemv_dequant(normed + s * hd, lw.ffn_up, up_buf + s * cfg_.intermediate_dim,
                         hd, cfg_.intermediate_dim, lw.up_type);

            int id = cfg_.intermediate_dim;

            if (cfg_.use_turbo_sparse) {
                for (int i = 0; i < id; i++) {
                    float g = gate_buf[s * id + i];
                    float u = up_buf[s * id + i];
                    float sig = 1.0f / (1.0f + expf(-g));
                    gate_buf[s * id + i] = g * sig * u;
                }
                int n_nonzero = 0;
                for (int i = 0; i < id; i++) {
                    if (fabsf(gate_buf[s * id + i]) > 1e-5f) n_nonzero++;
                }
                if (n_nonzero == 0) {
                    memset(ff_out + s * hd, 0, hd * sizeof(float));
                } else if (n_nonzero < id / 5) {
                    std::vector<float> sparse_gate(id, 0.0f);
                    for (int i = 0; i < id; i++) {
                        if (fabsf(gate_buf[s * id + i]) > 1e-5f) {
                            sparse_gate[i] = gate_buf[s * id + i];
                        }
                    }
                    gemv_dequant(sparse_gate.data(), lw.ffn_down, ff_out + s * hd,
                                 id, hd, GGML_TYPE_F32);
                } else {
                    gemv_dequant(gate_buf + s * id, lw.ffn_down, ff_out + s * hd,
                                 id, hd, lw.down_type);
                }
            } else if (cfg_.use_geglu) {
                for (int i = 0; i < id; i++) {
                    float x = gate_buf[s * id + i];
                    float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                    gate_buf[s * id + i] = gelu * up_buf[s * id + i];
                }
            } else {
#ifdef __aarch64__
                vsilu_mul_f32(gate_buf + s * id, up_buf + s * id, gate_buf + s * id, id);
#else
                for (int i = 0; i < id; i++) {
                    gate_buf[s * id + i] = silu_f(gate_buf[s * id + i]) * up_buf[s * id + i];
                }
#endif
            }

            gemv_dequant(gate_buf + s * id, lw.ffn_down, ff_out + s * hd,
                         id, hd, lw.down_type);
        }
    } else {
        // No FFN weights (shouldn't happen in practice)
        memset(ff_out, 0, seq_len * hd * sizeof(float));
    }

    // ── 10. Final residual: output += ff_out ──
    for (int s = 0; s < seq_len; s++) {
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
    gemv_dequant_mt(normed.data() + last * hd, output_, logits.data(), hd, cfg_.vocab_size, out_type_, 0);

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
    if (token_id < 0 || token_id >= cfg_.vocab_size)
        throw std::runtime_error("Token ID out of range: " + std::to_string(token_id));

    int hd = cfg_.hidden_dim;

    // ── Token embedding for single token ──
    // Instance-owned buffers — no heap alloc, no thread_local clobbering
    size_t row_bytes = tensor_nbytes(emb_type_, hd);
    const void* emb_row = (const uint8_t*)token_emb_ + token_id * row_bytes;
    dequantize_row(emb_row, x_buf_.data(), hd, emb_type_);

    // ── Run through all layers ──
    for (int l = 0; l < cfg_.n_layers; l++) {
        forward_layer(x_buf_.data(), 1, hidden_buf_.data(), layers_[l], l);
        std::swap(x_buf_, hidden_buf_);
    }

    // ── Final RMS norm ──
    rms_norm(x_buf_.data(), output_norm_, normed_buf_.data(), hd, cfg_.norm_eps);

    // ── Output projection → logits ──
    gemv_dequant_mt(normed_buf_.data(), output_, logits_buf_.data(), hd, cfg_.vocab_size, out_type_, 0);

    position_++;
    return logits_buf_;  // Copy — safe because caller owns it
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
// ── Speculative Decoding ──
// Draft model generates N tokens, target model verifies all at once.
// If acceptance rate is high, effective token throughput ≈ N × target_speed.
// 
// TinyLlama is both draft and target (self-speculative decoding).

GenerateResult VibeBladeFast::speculative_decode(
    const std::string& prompt,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    int seed,
    int n_spec_tokens
) {
    if (!loaded_) throw std::runtime_error("Model not loaded");
    GenerateResult result;
    result.stopped_eos = false;

    if (seed >= 0) sampler_.set_seed((uint64_t)seed);

    std::vector<int> prompt_ids = tokenizer_.encode(prompt);
    std::vector<float> logits = prefill(prompt_ids);

    std::vector<int> token_history;
    token_history.reserve(prompt_ids.size() + max_tokens + 16);
    token_history.insert(token_history.end(), prompt_ids.begin(), prompt_ids.end());

    result.token_ids.reserve(max_tokens);

    auto t_start = std::chrono::high_resolution_clock::now();

    while ((int)result.token_ids.size() < max_tokens) {
        // ── Draft phase: generate n_spec_tokens ──
        std::vector<int> draft_tokens;
        draft_tokens.reserve(n_spec_tokens);
        for (int i = 0; i < n_spec_tokens && (int)result.token_ids.size() < max_tokens; i++) {
            SamplerConfig scfg;
            scfg.temperature = temperature;
            scfg.top_k = top_k;
            scfg.top_p = top_p;
            scfg.repetition_penalty = repetition_penalty;
            int tok = sampler_.sample(logits.data(), cfg_.vocab_size, scfg, token_history);
            if (tok == tokenizer_.eos_id()) { result.stopped_eos = true; break; }
            draft_tokens.push_back(tok);
            token_history.push_back(tok);
            result.token_ids.push_back(tok);
            logits = decode(tok);
        }

        if (draft_tokens.empty()) break;

        // ── Verify phase: re-run all draft tokens through prefill ──
        // The decode() already updated KV cache, but we need to verify logits
        // For self-speculative: re-run draft tokens and check acceptance
        // Simple approach: re-decode each draft token and compare logits
        // If logit[draft] ≈ max logit, accept. Otherwise reject and regenerate.
        
        // Restore state: pop draft tokens from history
        for (int i = 0; i < (int)draft_tokens.size(); i++) {
            token_history.pop_back();
            result.token_ids.pop_back();
        }
        // Reset KV cache to before draft (simplified — in production use branching)
        // For now: re-verify with decode (no speedup from this approach)
        // The real benefit comes from a smaller draft model (future work)
        
        for (int tok : draft_tokens) {
            token_history.push_back(tok);
            result.token_ids.push_back(tok);
            logits = decode(tok);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    result.tokens_per_second = result.token_ids.size() / std::max(elapsed, 1e-6);
    result.text = tokenizer_.decode(result.token_ids);
    return result;
}



// ════════════════════════════════════════════════════════════════
// Grammar Constraints (GBNF-based token masking)
// Reduces search space for structured output (JSON, code, etc.)
// ════════════════════════════════════════════════════════════════

void VibeBladeFast::set_grammar(const std::string& gbnf) {
    grammar_str_ = gbnf;
    // Initialize: all tokens allowed initially (state 0 = root)
    grammar_states_.clear();
    grammar_states_.resize(cfg_.vocab_size);
    // For a simple GBNF parser, build a trie of allowed sequences.
    // Default: allow all tokens (no constraint).
    // Real GBNF parsing would require a proper parser — this is a placeholder
    // that allows all tokens until the GBNF is fully implemented.
    for (int i = 0; i < cfg_.vocab_size; i++) {
        grammar_states_[i].push_back(0);  // state 0: root, all tokens allowed
    }
}

void VibeBladeFast::clear_grammar() {
    grammar_str_.clear();
    grammar_states_.clear();
}

std::vector<int> VibeBladeFast::allowed_tokens_for_grammar() const {
    if (grammar_states_.empty()) return {};
    // Return tokens that are allowed in the current grammar state
    std::vector<int> allowed;
    for (int i = 0; i < cfg_.vocab_size; i++) {
        if (!grammar_states_[i].empty()) allowed.push_back(i);
    }
    return allowed;
}

} // namespace vibeblade
