#pragma once
// VibeBlade CUDA Backend — manages GPU resources for VibeBladeFast.
// Handles weight upload, buffer allocation, and kernel dispatch.
//
// Usage: VibeBladeFast::load() calls CudaBackend::init() if CUDA is available.
// All forward-pass operations then route to GPU kernels automatically.

#ifdef VIBEBLADE_USE_CUDA

#include "cuda_kernels.h"
#include "ggml_types.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace vibeblade {

struct FastConfig;
struct LayerWeights;
namespace cuda {

// ════════════════════════════════════════════════════════════════
// CudaBackend: manages GPU state for one model instance
// ════════════════════════════════════════════════════════════════

class CudaBackend {
public:
    CudaBackend();
    ~CudaBackend();

    bool is_initialized() const { return initialized_; }

    // Initialize with model config (allocates GPU buffers)
    void init(const FastConfig& cfg);

    // Upload model weights to GPU (call once after load)
    void upload_weights(
        const void* token_emb, ggml_type emb_type,
        const float* output_norm,
        const void* output_w, ggml_type out_type,
        const std::vector<LayerWeights>& layers,
        int hidden_dim, int vocab_size
    );

    // Reset KV cache on GPU
    void reset();

    // ── Forward pass operations (GPU) ──

    // Token embedding lookup: ids → (seq_len, hidden_dim) fp32
    void embedding_lookup(
        const int* host_token_ids, int seq_len,
        float* d_out
    );

    // RMSNorm
    void rms_norm(
        const float* d_x, const float* d_weight, float* d_out,
        int rows, int D, float eps
    );

    // GEMV with inline dequant
    void gemv_dequant(
        const float* d_x, const void* d_weights, ggml_type type,
        float* d_y, int K, int M
    );

    // RoPE
    void apply_rope(
        float* d_QK, int seq_len, int num_heads, int head_dim,
        bool neox_style
    );

    // SDPA attention
    void fused_sdpa(
        const float* d_Q, const float* d_K, const float* d_V,
        float* d_O, int M, int N, int d, float scale
    );

    // SiLU + multiply (SwiGLU)
    void silu_mul(const float* d_a, const float* d_b, float* d_out, int N);

    // Residual add
    void residual_add(float* d_out, const float* d_a, const float* d_b, int N);

    // Softmax
    void softmax(float* d_x, int rows, int cols);

    // MoE top-k routing
    void moe_top_k(
        const float* d_router_logits,
        int* d_top_indices, float* d_top_weights,
        int n_experts, int k
    );

    // ── KV Cache on GPU ──
    void store_kv(int layer, int pos, const float* d_K, const float* d_V, int kv_dim, int seq_len);
    void get_kv(int layer, float** d_K_cache, float** d_V_cache);

    // ── Device buffers ──
    CudaStream& stream() { return stream_; }

    // Scratch buffer for dequant (reused across calls)
    float* dequant_scratch() { return d_dequant_scratch_; }
    size_t dequant_scratch_size() const { return dequant_scratch_bytes_; }

    // Get layer weight pointers on GPU
    struct LayerGPUWeights {
        DeviceBuffer q, k, v, o;          // attention weights (quantized)
        DeviceBuffer qkv;                 // fused QKV
        DeviceBuffer gate, up, down;      // FFN weights (quantized)
        DeviceBuffer gate_inp;            // MoE router
        DeviceBuffer gate_exps, up_exps, down_exps;  // MoE expert weights
        DeviceBuffer gate_shexp, up_shexp, down_shexp; // shared expert
        DeviceBuffer attn_norm, ffn_norm;  // norm weights (fp32)
        bool has_fused_qkv = false;
        bool has_moe = false;
        bool has_shared_expert = false;
        bool has_attention = true;
    };

    const std::vector<LayerGPUWeights>& layer_weights() const { return layer_gpu_; }

    // Global weights on GPU
    const DeviceBuffer& token_emb_gpu() const { return d_token_emb_; }
    const DeviceBuffer& output_norm_gpu() const { return d_output_norm_; }
    const DeviceBuffer& output_w_gpu() const { return d_output_w_; }
    ggml_type emb_type_gpu() const { return emb_type_; }
    ggml_type out_type_gpu() const { return out_type_; }

 // RoPE cache on GPU
 const float* rope_cos_gpu() const { return static_cast<const float*>(d_rope_cos_.data()); }
 const float* rope_sin_gpu() const { return static_cast<const float*>(d_rope_sin_.data()); }

private:
 bool initialized_ = false;
 CudaStream stream_;

 // Global weights
    DeviceBuffer d_token_emb_;
    DeviceBuffer d_output_norm_;  // fp32
    DeviceBuffer d_output_w_;
    ggml_type emb_type_ = GGML_TYPE_F32;
    ggml_type out_type_ = GGML_TYPE_F32;

    // Per-layer GPU weights
    std::vector<LayerGPUWeights> layer_gpu_;

    // KV cache on GPU
    std::vector<DeviceBuffer> d_kv_k_;
    std::vector<DeviceBuffer> d_kv_v_;

    // RoPE cache
    DeviceBuffer d_rope_cos_;
    DeviceBuffer d_rope_sin_;

    // Scratch buffers
    float* d_dequant_scratch_ = nullptr;
    size_t dequant_scratch_bytes_ = 0;

    // Working buffers for forward pass (reused)
    DeviceBuffer d_x_buf_;       // (hidden_dim,)
    DeviceBuffer d_hidden_buf_;  // (hidden_dim,)
    DeviceBuffer d_normed_buf_;  // (hidden_dim,)
    DeviceBuffer d_logits_buf_;  // (vocab_size,)

    // Additional per‑token workspace
    DeviceBuffer d_Q_buf_;          // (q_dim,)
    DeviceBuffer d_K_buf_;          // (kv_dim,)
    DeviceBuffer d_V_buf_;          // (kv_dim,)
    DeviceBuffer d_attn_out_buf_;   // (q_dim,)
    DeviceBuffer d_gate_buf_;       // (intermediate_dim,)
    DeviceBuffer d_up_buf_;         // (intermediate_dim,)
    DeviceBuffer d_ff_out_buf_;     // (hidden_dim,)
    DeviceBuffer d_scores_buf_;     // (context_length,)
    DeviceBuffer d_o_proj_buf_;     // (hidden_dim,)
    DeviceBuffer d_qkv_buf_;        // (q_dim + 2*kv_dim,)
    DeviceBuffer d_router_logits_;  // (n_experts,)
    DeviceBuffer d_top_indices_;    // (n_experts_used,)
    DeviceBuffer d_top_weights_;    // (n_experts_used,)

 // Config
 int hidden_dim_ = 0;
 int vocab_size_ = 0;
 int n_layers_ = 0;
 int n_heads_ = 0;
 int n_kv_heads_ = 0;
 int head_dim_ = 0;
 int intermediate_dim_ = 0;
 int context_length_ = 0;
 int n_experts_ = 0;

    // ── Methods ──
    void alloc_kv_cache(int n_layers, int context_length, int kv_dim);
    void upload_rope_cache(const float* host_cos, const float* host_sin, size_t size);
    void forward_layer(
        int layer_idx,
        const float* h_input, int seq_len, float* h_output,
        const LayerWeights& lw, const FastConfig& cfg
    );
};

} // namespace cuda
} // namespace vibeblade

#endif // VIBEBLADE_USE_CUDA
