#pragma once
// VibeBlade CUDA Kernels — GPU-accelerated inference primitives for NVIDIA GPUs.
// Targets sm_121 (Blackwell/GB10) with CUDA 13.0.
//
// All kernels operate on fp32 data (dequantized from GGML quantized formats).
// cuBLAS is used for GEMM/GEMV operations.

#ifdef VIBEBLADE_USE_CUDA

#include <cstdint>
#include <cstddef>
#include <string>

namespace vibeblade {
namespace cuda {

// ── Device Management ──
bool is_available();          // Check if CUDA GPU is accessible
int device_count();           // Number of CUDA devices
std::string device_name(int device = 0);  // GPU name
size_t device_memory(int device = 0);     // Total VRAM in bytes
int compute_capability(int device = 0);   // e.g. 121 for sm_121

// ── CUDA Stream ──
// Simple RAII wrapper around cudaStream_t
struct CudaStream {
    void* stream; // cudaStream_t
    CudaStream();
    ~CudaStream();
    void synchronize();
    operator bool() const { return stream != nullptr; }
};

// ── Device Memory Buffers ──
struct DeviceBuffer {
    void* ptr;
    size_t size;  // in bytes
    DeviceBuffer();
    DeviceBuffer(size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() { return ptr; }
    const void* data() const { return ptr; }
    size_t bytes() const { return size; }
    bool valid() const { return ptr != nullptr; }

    void alloc(size_t bytes);
    void free();
    void upload(const void* host_data, size_t bytes, void* stream = nullptr);
    void download(void* host_data, size_t bytes, void* stream = nullptr) const;
};

// ── GEMM via cuBLAS ──
// C = alpha * A @ B + beta * C
// A: (M, K), B: (K, N), C: (M, N) — all fp32, row-major
// cuBLAS uses column-major, so we compute C^T = B^T @ A^T
void gemm_f32(
    const float* d_A, const float* d_B, float* d_C,
    int M, int K, int N,
    float alpha = 1.0f, float beta = 0.0f,
    void* stream = nullptr
);

// GEMV: y = alpha * A @ x + beta * y
// A: (M, K), x: (K,), y: (M,) — all fp32
void gemv_f32(
    const float* d_A, const float* d_x, float* d_y,
    int M, int K,
    float alpha = 1.0f, float beta = 0.0f,
    void* stream = nullptr
);

// ── RMSNorm ──
// out = x * weight / sqrt(mean(x^2) + eps)
// x: (rows, D), weight: (D,), out: (rows, D) — fp32
void rms_norm(
    const float* d_x, const float* d_weight, float* d_out,
    int rows, int D, float eps = 1e-5f,
    void* stream = nullptr
);

// ── SiLU + multiply (SwiGLU gate) ──
// out = silu(a) * b — element-wise
// a: (N,), b: (N,), out: (N,) — fp32
void silu_mul(
    const float* d_a, const float* d_b, float* d_out,
    int N,
    void* stream = nullptr
);

// SiLU: out = x * sigmoid(x)
void silu(
    const float* d_x, float* d_out,
    int N,
    void* stream = nullptr
);

// ── Fused SDPA (Scaled Dot-Product Attention) ──
// O = softmax(Q @ K^T / scale) @ V
// Q: (M, d), K: (N, d), V: (N, d), O: (M, d) — all fp32
// Uses online softmax (flash-style) to avoid O(M*N*d) memory
void fused_sdpa(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_O, int M, int N, int d, float scale,
    void* stream = nullptr
);

// ── RoPE (Rotary Position Embeddings) ──
// Apply RoPE in-place on GPU
// x: (seq_len, num_heads, head_dim) — fp32
// cos: (seq_len, head_dim/2), sin: (seq_len, head_dim/2) — fp32
void apply_rope(
    float* d_x,
    const float* d_cos, const float* d_sin,
    int seq_len, int num_heads, int head_dim,
    bool neox_style = true,  // interleaved RoPE (Qwen/Llama style)
    void* stream = nullptr
);

// ── Softmax ──
// Row-wise softmax in-place
// x: (rows, cols) — fp32
void softmax_f32(
    float* d_x, int rows, int cols,
    void* stream = nullptr
);

// ── Dequantization Kernels ──
// Each dequantizes a row of quantized data to fp32 on GPU

// Dequantize Q4_0 block to fp32
void dequantize_row_q4_0(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize Q4_1 block to fp32
void dequantize_row_q4_1(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize Q5_0 block to fp32
void dequantize_row_q5_0(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize Q5_1 block to fp32
void dequantize_row_q5_1(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize Q8_0 block to fp32
void dequantize_row_q8_0(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize Q4_K block to fp32
void dequantize_row_q4_k(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize Q5_K block to fp32
void dequantize_row_q5_k(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize Q6_K block to fp32
void dequantize_row_q6_k(
    const void* d_blocks, float* d_out,
    int64_t n, void* stream = nullptr
);

// Dequantize FP16 to FP32
void dequantize_row_f16(
    const void* d_f16, float* d_out,
    int64_t n, void* stream = nullptr
);

// Generic dequantize dispatch (same interface as CPU gemv_dequant)
// Dequantizes a weight matrix row and computes gemv in one call
void gemv_dequant(
    const float* d_x,      // (K,) input vector on GPU
    const void* d_weights,  // quantized weight matrix on GPU
    float* d_y,             // (M,) output vector on GPU
    int K, int M,
    int ggml_type,          // GGML_TYPE enum value
    float* d_scratch,       // scratch buffer for dequant (at least M floats)
    void* stream = nullptr
);

// ── Weight Upload ──
// Upload quantized weights to GPU once at model load time.
// Returns a DeviceBuffer owning the GPU memory.
DeviceBuffer upload_weights(const void* host_data, size_t bytes);

// ── Residual add ──
// out = a + b (element-wise)
void residual_add(
    float* d_out, const float* d_a, const float* d_b,
    int N, void* stream = nullptr
);

// ── Token embedding lookup ──
// Given token IDs, look up embedding rows and dequantize to fp32
void embedding_lookup(
    const void* d_embeddings,  // (vocab_size, hidden_dim) quantized
    const int* d_token_ids,    // (seq_len,) token IDs
    float* d_out,              // (seq_len, hidden_dim) fp32 output
    int seq_len, int hidden_dim, int vocab_size,
    int ggml_type,
    void* stream = nullptr
);

// ── Top-k expert routing (MoE) ──
// Given router logits, compute softmax and select top-k experts
// router_logits: (n_experts,) fp32
// top_indices: (k,) int32 output
// top_weights: (k,) fp32 output
void moe_top_k(
    const float* d_router_logits,
    int* d_top_indices, float* d_top_weights,
    int n_experts, int k,
    void* stream = nullptr
);

} // namespace cuda
} // namespace vibeblade

#endif // VIBEBLADE_USE_CUDA
