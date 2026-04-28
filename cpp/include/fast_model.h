#pragma once
// VibeBlade Fast Model — llama.cpp-style inference engine.
// Full decode loop in C++ with mmap'd weights and inline dequant.
// Python calls one function per token; everything else is C++.

#include "gguf.h"
#include "ggml_types.h"
#include <vector>
#include <string>
#include <memory>

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

class VibeBladeFast {
public:
    VibeBladeFast();
    ~VibeBladeFast();

    // Load GGUF model (mmaps the file, parses metadata, maps weights)
    void load(const char* path);

    // Prefill: process all prompt tokens at once → returns logits (vocab_size,)
    std::vector<float> prefill(const std::vector<int>& token_ids);

    // Decode: process one token → returns logits (vocab_size,)
    // KV cache is maintained internally across calls.
    std::vector<float> decode(int token_id);

    // Reset KV cache (start new conversation)
    void reset();

    // Current position in the sequence
    int position() const { return position_; }

    // Model config
    const FastConfig& config() const { return cfg_; }

    // KV cache stats
    size_t kv_cache_bytes() const;

private:
    void extract_config(const GGUFFile& gguf);
    void map_weights(const GGUFFile& gguf);
    void build_rope_cache();
    void alloc_kv_cache();

    // Forward one layer (used by both prefill and decode)
    void forward_layer(
        const float* input,       // (seq, hidden_dim) for prefill, (1, hidden_dim) for decode
        int seq_len,              // 1 for decode
        float* output,            // (seq, hidden_dim)
        const LayerWeights& lw,
        int layer_idx
    );

    // Scratch buffers (reused every decode step, no malloc)
    std::vector<float> scratch_;   // general scratch
    std::vector<float> rope_cos_, rope_sin_;

    // KV cache: per-layer, (n_heads or n_kv_heads, max_seq, head_dim)
    std::vector<std::vector<float>> kv_k_, kv_v_;

    FastConfig cfg_;
    std::vector<LayerWeights> layers_;

    // Global weights
    const void* token_emb_ = nullptr;   ggml_type emb_type_ = GGML_TYPE_F32;
    const void* output_ = nullptr;      ggml_type out_type_ = GGML_TYPE_F32;
    const float* output_norm_ = nullptr;

    // GGUF file handle (owns the mmap)
    std::unique_ptr<GGUFFile> gguf_;

    int position_ = 0;
    bool loaded_ = false;

    static constexpr int MAX_LAYERS = 256;
    static constexpr int MAX_SEQ = 8192;
};

}  // namespace vibeblade
