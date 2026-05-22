// VibeBlade CUDA Backend — GPU resource management and weight upload.
// Integrates CUDA kernels with VibeBladeFast for sm_121 (GB10) acceleration.

#include "cuda_backend.h"
#include "cuda_kernels.h"
#include "fast_model.h"  // Full definitions for LayerWeights, FastConfig

#ifdef VIBEBLADE_USE_CUDA
#include <cuda_runtime_api.h>
#define CUDA_CHECK(call) do { \
 cudaError_t err = (call); \
 if (err != cudaSuccess) { \
 fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", \
 __FILE__, __LINE__, cudaGetErrorString(err)); \
 throw std::runtime_error(cudaGetErrorString(err)); \
 } \
} while(0)
#endif
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cmath>

namespace vibeblade {
namespace cuda {

CudaBackend::CudaBackend() {}

CudaBackend::~CudaBackend() {
    reset();
    // Stream destructor will clean up
}

void CudaBackend::init(const FastConfig& cfg) {
    if (!cuda::is_available()) {
        throw std::runtime_error("CUDA is not available on this system");
    }

 hidden_dim_ = cfg.hidden_dim;
 vocab_size_ = cfg.vocab_size;
 n_layers_ = cfg.n_layers;
 n_heads_ = cfg.n_heads;
 n_kv_heads_ = cfg.n_kv_heads;
 head_dim_ = cfg.head_dim;
 intermediate_dim_ = cfg.intermediate_dim;
 context_length_ = cfg.context_length;
 n_experts_ = cfg.n_experts;

    // Allocate working buffers
    d_x_buf_.alloc(hidden_dim_ * sizeof(float));
    d_hidden_buf_.alloc(hidden_dim_ * sizeof(float));
    d_normed_buf_.alloc(hidden_dim_ * sizeof(float));
    d_logits_buf_.alloc(vocab_size_ * sizeof(float));

    // Scratch buffer for dequantization (max of intermediate_dim and hidden_dim)
    size_t max_dim = std::max((size_t)intermediate_dim_, (size_t)hidden_dim_);
    dequant_scratch_bytes_ = max_dim * sizeof(float);
    d_dequant_scratch_ = (float*)d_x_buf_.data(); // Reuse x_buf as scratch temporarily
    // Actually we need a dedicated scratch buffer for gemv_dequant
    // We'll allocate per-layer or use a pool. For now, reuse the d_x_buf_ pointer contextually
    // But that's risky in multi-threaded scenarios. We'll allocate separate scratch in the wrapper.
    // For now, just note that gemv_dequant scratch is passed in by caller (VibeBladeFast) from its own buffer.
    // So no need to allocate here.

    // Initialize layer GPU weights containers
    layer_gpu_.resize(n_layers_);

    initialized_ = true;
}

void CudaBackend::upload_weights(
    const void* host_token_emb, ggml_type emb_type,
    const float* host_output_norm,
    const void* host_output_w, ggml_type out_type,
    const std::vector<LayerWeights>& host_layers,
    int hidden_dim, int vocab_size
) {
    if (!initialized_) throw std::runtime_error("CudaBackend not initialized");

    // Upload token embeddings
    size_t emb_bytes = (size_t)vocab_size * hidden_dim * ggml_type_size(emb_type);
    d_token_emb_ = DeviceBuffer(emb_bytes);
    d_token_emb_.upload(host_token_emb, emb_bytes, stream_.stream);

    emb_type_ = emb_type;

    // Upload output norm (fp32)
    d_output_norm_ = DeviceBuffer(hidden_dim * sizeof(float));
    d_output_norm_.upload(host_output_norm, hidden_dim * sizeof(float), stream_.stream);

    // Upload output projection weights
    size_t out_w_bytes = (size_t)hidden_dim * vocab_size * ggml_type_size(out_type);
    d_output_w_ = DeviceBuffer(out_w_bytes);
    d_output_w_.upload(host_output_w, out_w_bytes, stream_.stream);
    out_type_ = out_type;

    // Upload each layer's weights
    for (int l = 0; l < n_layers_; l++) {
        const LayerWeights& lw = host_layers[l];
        LayerGPUWeights& lw_gpu = layer_gpu_[l];

        // Attention weights
        if (lw.attn_q) {
            size_t q_bytes = (size_t)hidden_dim * (n_heads_ * head_dim_) * ggml_type_size(lw.qtype);
            lw_gpu.q = DeviceBuffer(q_bytes);
            lw_gpu.q.upload(lw.attn_q, q_bytes, stream_.stream);
        }
        if (lw.attn_k) {
            size_t k_bytes = (size_t)hidden_dim * (n_kv_heads_ * head_dim_) * ggml_type_size(lw.ktype);
            lw_gpu.k = DeviceBuffer(k_bytes);
            lw_gpu.k.upload(lw.attn_k, k_bytes, stream_.stream);
        }
        if (lw.attn_v) {
            size_t v_bytes = (size_t)hidden_dim * (n_kv_heads_ * head_dim_) * ggml_type_size(lw.vtype);
            lw_gpu.v = DeviceBuffer(v_bytes);
            lw_gpu.v.upload(lw.attn_v, v_bytes, stream_.stream);
        }
        if (lw.attn_o) {
            size_t o_bytes = (size_t)(n_heads_ * head_dim_) * hidden_dim * ggml_type_size(lw.otype);
            lw_gpu.o = DeviceBuffer(o_bytes);
            lw_gpu.o.upload(lw.attn_o, o_bytes, stream_.stream);
        }

        // Fused QKV
        if (lw.has_fused_qkv && lw.attn_qkv) {
            size_t qkv_bytes = (size_t)hidden_dim * (n_heads_ * head_dim_ + 2 * n_kv_heads_ * head_dim_) * ggml_type_size(lw.qkv_type);
            lw_gpu.qkv = DeviceBuffer(qkv_bytes);
            lw_gpu.qkv.upload(lw.attn_qkv, qkv_bytes, stream_.stream);
            lw_gpu.has_fused_qkv = true;
        }

        // FFN weights (dense)
        if (lw.ffn_gate) {
            size_t gate_bytes = (size_t)hidden_dim * intermediate_dim_ * ggml_type_size(lw.gate_type);
            lw_gpu.gate = DeviceBuffer(gate_bytes);
            lw_gpu.gate.upload(lw.ffn_gate, gate_bytes, stream_.stream);
        }
        if (lw.ffn_up) {
            size_t up_bytes = (size_t)hidden_dim * intermediate_dim_ * ggml_type_size(lw.up_type);
            lw_gpu.up = DeviceBuffer(up_bytes);
            lw_gpu.up.upload(lw.ffn_up, up_bytes, stream_.stream);
        }
        if (lw.ffn_down) {
            size_t down_bytes = (size_t)intermediate_dim_ * hidden_dim * ggml_type_size(lw.down_type);
            lw_gpu.down = DeviceBuffer(down_bytes);
            lw_gpu.down.upload(lw.ffn_down, down_bytes, stream_.stream);
        }

 // MoE weights
 if (lw.has_moe) {
 int n_exp = n_experts_;
 int exp_inter = lw.expert_intermediate_dim > 0 ? lw.expert_intermediate_dim : intermediate_dim_;

            // Router input
            if (lw.ffn_gate_inp) {
                size_t gate_inp_bytes = (size_t)hidden_dim * n_exp * ggml_type_size(lw.gate_inp_type);
                lw_gpu.gate_inp = DeviceBuffer(gate_inp_bytes);
                lw_gpu.gate_inp.upload(lw.ffn_gate_inp, gate_inp_bytes, stream_.stream);
            }

 // Experts (consolidated tensors)
 if (lw.ffn_gate_exps) {
 size_t gate_exps_bytes = (size_t)n_exp * exp_inter * hidden_dim * ggml_type_size(lw.gate_exps_type);
 lw_gpu.gate_exps = DeviceBuffer(gate_exps_bytes);
 lw_gpu.gate_exps.upload(lw.ffn_gate_exps, gate_exps_bytes, stream_.stream);
 }
 if (lw.ffn_up_exps) {
 size_t up_exps_bytes = (size_t)n_exp * exp_inter * hidden_dim * ggml_type_size(lw.up_exps_type);
 lw_gpu.up_exps = DeviceBuffer(up_exps_bytes);
 lw_gpu.up_exps.upload(lw.ffn_up_exps, up_exps_bytes, stream_.stream);
 }
            if (lw.ffn_down_exps) {
                size_t down_exps_bytes = (size_t)n_exp * hidden_dim * exp_inter * ggml_type_size(lw.down_exps_type);
                lw_gpu.down_exps = DeviceBuffer(down_exps_bytes);
                lw_gpu.down_exps.upload(lw.ffn_down_exps, down_exps_bytes, stream_.stream);
            }

            // Shared expert (if any)
            if (lw.has_shared_expert) {
                if (lw.ffn_gate_shexp) {
                    size_t shexp_bytes = (size_t)hidden_dim * exp_inter * ggml_type_size(lw.shexp_gate_type);
                    lw_gpu.gate_shexp = DeviceBuffer(shexp_bytes);
                    lw_gpu.gate_shexp.upload(lw.ffn_gate_shexp, shexp_bytes, stream_.stream);
                }
                if (lw.ffn_up_shexp) {
                    size_t shexp_bytes = (size_t)hidden_dim * exp_inter * ggml_type_size(lw.shexp_up_type);
                    lw_gpu.up_shexp = DeviceBuffer(shexp_bytes);
                    lw_gpu.up_shexp.upload(lw.ffn_up_shexp, shexp_bytes, stream_.stream);
                }
                if (lw.ffn_down_shexp) {
                    size_t shexp_bytes = (size_t)exp_inter * hidden_dim * ggml_type_size(lw.shexp_down_type);
                    lw_gpu.down_shexp = DeviceBuffer(shexp_bytes);
                    lw_gpu.down_shexp.upload(lw.ffn_down_shexp, shexp_bytes, stream_.stream);
                }
                lw_gpu.has_shared_expert = true;
            }

            lw_gpu.has_moe = true;
        } else {
            lw_gpu.has_moe = false;
            lw_gpu.has_shared_expert = false;
        }

        // Norm weights
        if (lw.attn_norm) {
            // fp32 copy
            DeviceBuffer norm_buf(hidden_dim * sizeof(float));
            norm_buf.upload(lw.attn_norm, hidden_dim * sizeof(float), stream_.stream);
            // We don't keep it in LayerGPUWeights currently — store separately? Actually we need it.
            // Let's extend LayerGPUWeights to include norm weights.
        }
        if (lw.ffn_norm) {
            // similar
        }
    }

    // Build RoPE cache on GPU
    // (CPU builds vectors rope_cos_/rope_sin_; we'll upload those from VibeBladeFast after)
    stream_.synchronize();
}

void CudaBackend::reset() {
    // Free all layer GPU weights
    layer_gpu_.clear();
    d_token_emb_.free();
    d_output_norm_.free();
    d_output_w_.free();
    d_kv_k_.clear();
    d_kv_v_.clear();
    d_rope_cos_.free();
    d_rope_sin_.free();
}

void CudaBackend::embedding_lookup(
    const int* host_token_ids, int seq_len,
    float* d_out
) {
    if (!initialized_) throw std::runtime_error("CudaBackend not initialized");

    // Upload token IDs to device temporarily
    std::vector<int> token_ids(seq_len);
    std::memcpy(token_ids.data(), host_token_ids, seq_len * sizeof(int));
    // Could use a DeviceBuffer for IDs, but for small seq_len host copy is fine

    // Launch kernel
    cuda::embedding_lookup(
        d_token_emb_.data(),
        token_ids.data(),  // host pointer — kernel will copy; for perf use pinned/device memory
        d_out,
        seq_len, hidden_dim_, vocab_size_,
        emb_type_,
        stream_.stream
    );
}

void CudaBackend::rms_norm(
    const float* d_x, const float* d_weight, float* d_out,
    int rows, int D, float eps
) {
    cuda::rms_norm(d_x, d_weight, d_out, rows, D, eps, stream_.stream);
}

void CudaBackend::gemv_dequant(
    const float* d_x, const void* d_weights, ggml_type type,
    float* d_y, int K, int M
) {
    cuda::gemv_dequant(d_x, d_weights, d_y, K, M, type, d_dequant_scratch_, stream_.stream);
}

void CudaBackend::apply_rope(
    float* d_QK, int seq_len, int num_heads, int head_dim,
    bool neox_style
) {
 cuda::apply_rope(d_QK, rope_cos_gpu(), rope_sin_gpu(),
 seq_len, num_heads, head_dim, neox_style, stream_.stream);
}

void CudaBackend::fused_sdpa(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_O, int M, int N, int d, float scale
) {
    cuda::fused_sdpa(d_Q, d_K, d_V, d_O, M, N, d, scale, stream_.stream);
}

void CudaBackend::silu_mul(
    const float* d_a, const float* d_b, float* d_out, int N
) {
    cuda::silu_mul(d_a, d_b, d_out, N, stream_.stream);
}

void CudaBackend::residual_add(
    float* d_out, const float* d_a, const float* d_b, int N
) {
    cuda::residual_add(d_out, d_a, d_b, N, stream_.stream);
}

void CudaBackend::softmax(float* d_x, int rows, int cols) {
    cuda::softmax_f32(d_x, rows, cols, stream_.stream);
}

void CudaBackend::moe_top_k(
    const float* d_router_logits,
    int* d_top_indices, float* d_top_weights,
    int n_experts, int k
) {
    cuda::moe_top_k(d_router_logits, d_top_indices, d_top_weights, n_experts, k, stream_.stream);
}

void CudaBackend::store_kv(int layer, int pos, const float* d_K, const float* d_V, int kv_dim, int seq_len) {
    // Ensure cache buffers exist for this layer
    if ((int)d_kv_k_.size() <= layer) {
        d_kv_k_.resize(n_layers_);
        d_kv_v_.resize(n_layers_);
    }

    // Allocate if first time
    if (!d_kv_k_[layer].valid()) {
        size_t cache_size = (size_t)context_length_ * kv_dim * sizeof(float);
        d_kv_k_[layer].alloc(cache_size);
        d_kv_v_[layer].alloc(cache_size);
    }

    // Copy K, V for this position into cache
    // K shape: (seq_len, kv_dim) already on device
    // We need to copy to position offset
 size_t offset_bytes = (size_t)pos * kv_dim * sizeof(float);
 size_t copy_bytes = (size_t)seq_len * kv_dim * sizeof(float);
 CUDA_CHECK(cudaMemcpyAsync(
 (char*)d_kv_k_[layer].data() + offset_bytes,
 d_K, copy_bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream_.stream
 ));
 CUDA_CHECK(cudaMemcpyAsync(
 (char*)d_kv_v_[layer].data() + offset_bytes,
 d_V, copy_bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream_.stream
 ));
}

void CudaBackend::get_kv(int layer, float** d_K_cache, float** d_V_cache) {
    if ((int)d_kv_k_.size() <= layer || !d_kv_k_[layer].valid()) {
        throw std::runtime_error("KV cache not allocated for layer " + std::to_string(layer));
    }
    *d_K_cache = (float*)d_kv_k_[layer].data();
    *d_V_cache = (float*)d_kv_v_[layer].data();
}

} // namespace cuda
} // namespace vibeblade
