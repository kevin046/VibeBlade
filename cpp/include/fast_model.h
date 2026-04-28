#pragma once
// VibeBlade Fast Model — llama.cpp-style inference engine.
// Full generate loop in C++: tokenize → prefill → decode → sample → detokenize.
// Python calls generate() once and gets the full output string.

#include "gguf.h"
#include "ggml_types.h"
#include "tokenizer.h"
#include "sampler.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace vibeblade {

struct FastConfig {
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int hidden_dim = 0;
    int intermediate_dim = 0;
    int vocab_size = 0;
    int context_length = 2048;
    float norm_eps = 1e-5f;
    std::string arch = "llama";

    // RoPE configuration
    float rope_freq_base = 10000.0f;
    float rope_scale = 1.0f;           // context-length scaling factor
    std::string rope_scaling_type;     // "linear", "yarn", "longrope", ""
    float yarn_ext_factor = 1.0f;      // YaRN extrapolation factor
    float yarn_attn_factor = 1.0f;     // YaRN attention temperature factor
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;
    std::vector<float> yarn_orig_ctx;  // YaRN original context lengths (for longrope)
    std::vector<int> long_rope_factors;// per-dimension freq scaling (longrope)

    // Architecture-specific flags
    bool use_geglu = false;            // Gemma uses GeGLU instead of SwiGLU
    bool use_parallel_attn = false;    // Falcon: attention + FFN in parallel
    bool use_fused_qkv = false;        // Many models fuse Q/K/V projections
    bool use_neox_rope = false;        // NeoX-style interleaved RoPE (GPT-NeoX, Falcon)
    int n_experts = 0;                 // MoE: number of experts (0 = dense)
    int n_experts_used = 0;            // MoE: active experts per token
    int sliding_window = 0;            // Mistral-style sliding window attention
};

struct LayerWeights {
    const void* attn_q = nullptr;      ggml_type qtype = GGML_TYPE_F32;
    const void* attn_k = nullptr;      ggml_type ktype = GGML_TYPE_F32;
    const void* attn_v = nullptr;      ggml_type vtype = GGML_TYPE_F32;
    const void* attn_o = nullptr;      ggml_type otype = GGML_TYPE_F32;
    const void* ffn_gate = nullptr;    ggml_type gate_type = GGML_TYPE_F32;
    const void* ffn_up = nullptr;      ggml_type up_type = GGML_TYPE_F32;
    const void* ffn_down = nullptr;    ggml_type down_type = GGML_TYPE_F32;
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    // MoE (optional)
    const void* ffn_gate_inp = nullptr;  ggml_type gate_inp_type = GGML_TYPE_F32;
    // Fused QKV support
    const void* attn_qkv = nullptr;     ggml_type qkv_type = GGML_TYPE_F32;
    bool has_fused_qkv = false;
};

struct GenerateResult {
    std::string text;           // full generated text
    std::vector<int> token_ids; // generated token IDs
    double tokens_per_second;   // decode speed
    bool stopped_eos;           // true if stopped by EOS token
};

class VibeBladeFast {
public:
    VibeBladeFast();
    ~VibeBladeFast();

    // Load GGUF model (mmaps the file, parses metadata, maps weights, loads tokenizer)
    void load(const char* path);

    // ── Full generate pipeline (one C++ call, zero Python in hot path) ──
    GenerateResult generate(
        const std::string& prompt,        // text prompt
        int max_tokens = 128,             // max new tokens
        float temperature = 1.0f,
        int top_k = 50,
        float top_p = 0.9f,
        float repetition_penalty = 1.0f,
        int seed = -1,                    // -1 = random
        std::function<void(int, const std::string&)> on_token = nullptr  // streaming callback
    );

    // ── Individual steps (for advanced use / Python loop fallback) ──
    std::vector<float> prefill(const std::vector<int>& token_ids);
    std::vector<float> decode(int token_id);
    void reset();

    // ── Tokenizer ──
    std::vector<int> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int>& ids) const;
    std::string detokenize_token(int id) const;

    // ── State ──
    int position() const { return position_; }
    const FastConfig& config() const { return cfg_; }
    size_t kv_cache_bytes() const;
    int eos_id() const { return tokenizer_.eos_id(); }
    int bos_id() const { return tokenizer_.bos_id(); }

    // ── Sampler config ──
    Sampler& sampler() { return sampler_; }

private:
    void extract_config(const GGUFFile& gguf);
    void map_weights(const GGUFFile& gguf);
    void build_rope_cache();
    void alloc_kv_cache();

    // Forward one layer (used by both prefill and decode)
    void forward_layer(
        const float* input,
        int seq_len,
        float* output,
        const LayerWeights& lw,
        int layer_idx
    );

    // Scratch buffers (reused every decode step, no malloc)
    std::vector<float> scratch_;
    std::vector<float> rope_cos_, rope_sin_;

    // KV cache: per-layer, (n_heads or n_kv_heads, max_seq, head_dim)
    std::vector<std::vector<float>> kv_k_, kv_v_;

    // Instance-owned decode buffers (NOT thread_local — avoids cross-instance clobbering).
    // Sized once during alloc_kv_cache(), never realloc'd during decode.
    std::vector<float> x_buf_, hidden_buf_, normed_buf_, logits_buf_;

    FastConfig cfg_;
    std::vector<LayerWeights> layers_;

    // Global weights
    const void* token_emb_ = nullptr;   ggml_type emb_type_ = GGML_TYPE_F32;
    const void* output_ = nullptr;      ggml_type out_type_ = GGML_TYPE_F32;
    const float* output_norm_ = nullptr;

    // GGUF file handle (owns the mmap)
    std::unique_ptr<GGUFFile> gguf_;

    // Tokenizer & Sampler
    Tokenizer tokenizer_;
    Sampler sampler_;

    int position_ = 0;
    bool loaded_ = false;

    static constexpr int MAX_LAYERS = 256;
    static constexpr int MAX_SEQ = 8192;
};

}  // namespace vibeblade
