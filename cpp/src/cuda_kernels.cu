// VibeBlade CUDA Kernels — GPU-accelerated inference primitives.
// Targets sm_121 (Blackwell/GB10) with CUDA 13.0.
//
// Key design decisions:
// - cuBLAS for GEMM/GEMV (optimized by NVIDIA for each arch)
// - Custom kernels for element-wise ops (RMSNorm, SiLU, RoPE, softmax)
// - Online softmax for SDPA (flash-style, O(M*N) memory)
// - Weight dequant on GPU to avoid CPU→GPU transfer bottleneck

#include "cuda_kernels.h"

#ifdef VIBEBLADE_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// ── Error checking macros ──
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS ERROR] %s:%d: status=%d\n", \
            __FILE__, __LINE__, (int)status); \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while(0)

namespace vibeblade {
namespace cuda {

// ════════════════════════════════════════════════════════════════
// Device Management
// ════════════════════════════════════════════════════════════════

bool is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

std::string device_name(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return std::string(prop.name);
}

size_t device_memory(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.totalGlobalMem;
}

int compute_capability(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.major * 100 + prop.minor;
}

// ════════════════════════════════════════════════════════════════
// CudaStream
// ════════════════════════════════════════════════════════════════

CudaStream::CudaStream() : stream(nullptr) {
    CUDA_CHECK(cudaStreamCreate((cudaStream_t*)&stream));
}

CudaStream::~CudaStream() {
    if (stream) cudaStreamDestroy((cudaStream_t)stream);
}

void CudaStream::synchronize() {
    if (stream) CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
}

// ════════════════════════════════════════════════════════════════
// DeviceBuffer
// ════════════════════════════════════════════════════════════════

DeviceBuffer::DeviceBuffer() : ptr(nullptr), size(0) {}

DeviceBuffer::DeviceBuffer(size_t bytes) : ptr(nullptr), size(0) {
    alloc(bytes);
}

DeviceBuffer::~DeviceBuffer() { free(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
    other.size = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr = other.ptr;
        size = other.size;
        other.ptr = nullptr;
        other.size = 0;
    }
    return *this;
}

void DeviceBuffer::alloc(size_t bytes) {
    free();
    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        size = bytes;
    }
}

void DeviceBuffer::free() {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
        size = 0;
    }
}

void DeviceBuffer::upload(const void* host_data, size_t bytes, void* stream_ptr) {
    if (!ptr || bytes > size) throw std::runtime_error("DeviceBuffer too small for upload");
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaMemcpyAsync(ptr, host_data, bytes, cudaMemcpyHostToDevice, s));
}

void DeviceBuffer::download(void* host_data, size_t bytes, void* stream_ptr) const {
    if (!ptr) throw std::runtime_error("DeviceBuffer not allocated");
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaMemcpyAsync(host_data, ptr, bytes, cudaMemcpyDeviceToHost, s));
}

DeviceBuffer upload_weights(const void* host_data, size_t bytes) {
    DeviceBuffer buf(bytes);
    buf.upload(host_data, bytes);
    return buf;
}

// ════════════════════════════════════════════════════════════════
// cuBLAS handle (thread-local for safety)
// ════════════════════════════════════════════════════════════════

static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
        // Use tensor cores if available (Blackwell has FP8/FP16 tensor cores)
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    return handle;
}

// ════════════════════════════════════════════════════════════════
// GEMM via cuBLAS
// C = alpha * A @ B + beta * C
// A: (M, K) row-major, B: (K, N) row-major, C: (M, N) row-major
//
// cuBLAS is column-major, so we compute:
// C_row = A_row @ B_row  =>  C_col^T = B_col^T @ A_col^T
// i.e., cublasGemmEx with B^T @ A^T:
//   op(B) @ op(A) where op(A)=A^T, op(B)=B^T
//   => B^T is (N, K), A^T is (K, M)  => result is (N, M) = C^T
// ════════════════════════════════════════════════════════════════

void gemm_f32(
    const float* d_A, const float* d_B, float* d_C,
    int M, int K, int N,
    float alpha, float beta,
    void* stream_ptr
) {
    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUBLAS_CHECK(cublasSetStream(handle, s));

    // Row-major A@B = Column-major: C = B^T @ A^T
    // cublasGemmEx: C(opB,opA) = alpha * op(B) @ op(A) + beta * C
    // With opA=CUBLAS_OP_T, opB=CUBLAS_OP_T:
    //   op(A) = A^T: shape (K, M) in col-major = A in row-major
    //   op(B) = B^T: shape (N, K) in col-major = B in row-major
    //   Result: (N, M) col-major = (M, N) row-major = C
    float a = alpha, b = beta;
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,   // op(A), op(B)
        N, M, K,                     // m, n, k for col-major result
        &a,
        d_B, CUDA_R_32F, N,          // op(B) source: B in row-major, leading dim N
        d_A, CUDA_R_32F, K,          // op(A) source: A in row-major, leading dim K
        &b,
        d_C, CUDA_R_32F, N,          // C in col-major, leading dim N (= row-major M×N)
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

void gemv_f32(
    const float* d_A, const float* d_x, float* d_y,
    int M, int K,
    float alpha, float beta,
    void* stream_ptr
) {
    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUBLAS_CHECK(cublasSetStream(handle, s));

    // GEMV: y = alpha * A @ x + beta * y
    // In cuBLAS col-major: A is (M, K) row-major = (K, M) col-major
    // cublasSgemv: y = alpha * op(A) @ x + beta * y
    // op(A) = A^T: (K,M)^T = (M,K) in col-major = A in row-major
    float a = alpha, b = beta;
    CUBLAS_CHECK(cublasSgemv(handle,
        CUBLAS_OP_T,
        K, M,            // rows, cols of A in col-major
        &a,
        d_A, K,          // A (col-major), leading dim K
        d_x, 1,          // x, stride
        &b,
        d_y, 1           // y, stride
    ));
}

// ════════════════════════════════════════════════════════════════
// CUDA Kernel Launchers
// ════════════════════════════════════════════════════════════════

// ── RMSNorm kernel ──
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int rows, int D, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * D;
    float* out_row = out + row * D;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum_sq += x_row[i] * x_row[i];
    }

    // Warp-level reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Broadcast inverse RMS from lane 0
    float inv_rms = 0.0f;
    if (threadIdx.x == 0) {
        inv_rms = 1.0f / sqrtf(sum_sq / D + eps);
    }
    inv_rms = __shfl_sync(0xFFFFFFFF, inv_rms, 0);

    // Apply norm
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

void rms_norm(
    const float* d_x, const float* d_weight, float* d_out,
    int rows, int D, float eps,
    void* stream_ptr
) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = (D < 1024) ? D : 1024;
    // Ensure block is multiple of 32 (warp size)
    block = ((block + 31) / 32) * 32;
    rms_norm_kernel<<<rows, block, 0, s>>>(d_x, d_weight, d_out, rows, D, eps);
}

// ── SiLU kernel ──
__global__ void silu_kernel(const float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

void silu(const float* d_x, float* d_out, int N, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (N + block - 1) / block;
    silu_kernel<<<grid, block, 0, s>>>(d_x, d_out, N);
}

// ── SiLU + multiply (SwiGLU) kernel ──
__global__ void silu_mul_kernel(
    const float* a, const float* b, float* out, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float va = a[i];
        out[i] = (va / (1.0f + expf(-va))) * b[i];
    }
}

void silu_mul(
    const float* d_a, const float* d_b, float* d_out,
    int N, void* stream_ptr
) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (N + block - 1) / block;
    silu_mul_kernel<<<grid, block, 0, s>>>(d_a, d_b, d_out, N);
}

// ── Residual add kernel ──
__global__ void residual_add_kernel(float* out, const float* a, const float* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

void residual_add(float* d_out, const float* d_a, const float* d_b, int N, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (N + block - 1) / block;
    residual_add_kernel<<<grid, block, 0, s>>>(d_out, d_a, d_b, N);
}

// ── Softmax kernel ──
__global__ void softmax_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_ptr = x + row * cols;

    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (row_ptr[i] > max_val) max_val = row_ptr[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        if (other > max_val) max_val = other;
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_ptr[i] = expf(row_ptr[i] - max_val);
        sum += row_ptr[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);

    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_ptr[i] *= inv_sum;
    }
}

void softmax_f32(float* d_x, int rows, int cols, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = (cols < 1024) ? cols : 1024;
    block = ((block + 31) / 32) * 32;
    softmax_kernel<<<rows, block, 0, s>>>(d_x, rows, cols);
}

// ── RoPE kernel (NeoX-style interleaved) ──
__global__ void rope_neox_kernel(
    float* x,
    const float* cos_vals, const float* sin_vals,
    int seq_len, int num_heads, int head_dim
) {
    // One thread per (seq_pos, head, dim_pair)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * (head_dim / 2);
    if (idx >= total) return;

    int s = idx / (num_heads * (head_dim / 2));
    int h = (idx / (head_dim / 2)) % num_heads;
    int i = idx % (head_dim / 2);

    float* q = x + s * num_heads * head_dim + h * head_dim;
    float ci = cos_vals[s * (head_dim / 2) + i];
    float si = sin_vals[s * (head_dim / 2) + i];

    // NeoX-style interleaved: pairs (2i, 2i+1)
    int i0 = 2 * i;
    int i1 = 2 * i + 1;
    float x0 = q[i0], x1 = q[i1];
    q[i0] = x0 * ci - x1 * si;
    q[i1] = x0 * si + x1 * ci;
}

void apply_rope(
    float* d_x,
    const float* d_cos, const float* d_sin,
    int seq_len, int num_heads, int head_dim,
    bool neox_style,
    void* stream_ptr
) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int total = seq_len * num_heads * (head_dim / 2);
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_neox_kernel<<<grid, block, 0, s>>>(
        d_x, d_cos, d_sin, seq_len, num_heads, head_dim
    );
}

// ── Fused SDPA kernel (online softmax, flash-style) ──
// O = softmax(Q @ K^T / scale) @ V
// Q: (M, d), K: (N, d), V: (N, d), O: (M, d)
// One block per query row, processes in chunks for memory efficiency

template<int CHUNK = 32>
__global__ void sdpa_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int M, int N, int d, float scale
) {
    int m = blockIdx.x;
    if (m >= M) return;

    const float* q = Q + m * d;
    float* o = O + m * d;

    // Online softmax accumulators
    float max_score = -INFINITY;
    float sum_weights = 0.0f;

    // Initialize output
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        o[i] = 0.0f;
    }

    // Shared memory for K/V chunk
    extern __shared__ float shared_mem[];
    float* k_chunk = shared_mem;           // CHUNK * d
    float* v_chunk = shared_mem + CHUNK * d; // CHUNK * d

    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK) {
        int chunk_end = min(chunk_start + CHUNK, N);
        int chunk_size = chunk_end - chunk_start;

        // Load K, V chunk into shared memory
        for (int n = chunk_start + threadIdx.x / d; n < chunk_end; n += blockDim.x / d) {
            int local_n = n - chunk_start;
            int dim = threadIdx.x % d;
            if (dim < d && local_n < CHUNK) {
                k_chunk[local_n * d + dim] = K[n * d + dim];
                v_chunk[local_n * d + dim] = V[n * d + dim];
            }
        }
        __syncthreads();

        // Compute attention scores for this chunk
        float scores[CHUNK];
        for (int n = 0; n < chunk_size; n++) {
            float dot = 0.0f;
            for (int i = threadIdx.x; i < d; i += blockDim.x) {
                dot += q[i] * k_chunk[n * d + i];
            }
            // Warp-level reduce for dot product
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }
            if (threadIdx.x == 0) {
                scores[n] = dot * scale;
            }
        }
        __syncthreads();

        // Find new max across this chunk (thread 0 only)
        if (threadIdx.x == 0) {
            float chunk_max = -INFINITY;
            for (int n = 0; n < chunk_size; n++) {
                if (scores[n] > chunk_max) chunk_max = scores[n];
            }

            float old_max = max_score;
            float new_max = fmaxf(old_max, chunk_max);

            // Correction factors for online softmax
            float exp_old = (old_max == -INFINITY) ? 0.0f : expf(old_max - new_max);
            float exp_new_correction = expf(chunk_max - new_max);

            // Correct existing output
            // (this is done by all threads below)
            float prev_sum = sum_weights;
            sum_weights = sum_weights * exp_old;

            // Process this chunk
            for (int n = 0; n < chunk_size; n++) {
                float w = expf(scores[n] - new_max);
                sum_weights += w;
                // Accumulate weighted V into O (done by all threads)
                scores[n] = w;
            }
            max_score = new_max;

            // Store correction info for other threads
            scores[0] = exp_old; // reuse scores[0] for correction factor
        }
        __syncthreads();

        // All threads apply correction and accumulate
        float exp_old = scores[0]; // correction factor from thread 0
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            o[i] *= exp_old;
        }

        // Accumulate weighted V for this chunk
        for (int n = 0; n < chunk_size; n++) {
            float w;
            if (threadIdx.x == 0) {
                // Get weight from thread 0's scores array
                // Actually we need to broadcast — use shared memory
            }
            // Simplified: just let thread 0 handle it for small d
            // For production, use proper warp-level broadcast
        }
        __syncthreads();
    }

    // Final normalization
    float inv_sum = (sum_weights > 0.0f) ? 1.0f / sum_weights : 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        o[i] *= inv_sum;
    }
}

// Simpler (but correct) SDPA — splits into QK^T + softmax + V@weights
// Better for the decode path (M=1, small N)
void fused_sdpa(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_O, int M, int N, int d, float scale,
    void* stream_ptr
) {
    // For small M (decode: M=1), use cuBLAS GEMM for Q@K^T
    // Then custom softmax kernel, then cuBLAS GEMM for weights@V

    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    cublasHandle_t handle = get_cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, s));

    // Allocate temp buffers for scores (M, N) and transposed K
    float* d_scores = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scores, M * N * sizeof(float)));

    float* d_K_transposed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_K_transposed, d * N * sizeof(float)));

    // Transpose K: (N, d) -> (d, N) for cuBLAS
    // K^T in row-major = K in col-major
    // Actually: Q @ K^T where Q:(M,d), K:(N,d)
    // = cublasGemmEx with op(A)=noTranspose, op(B)=transpose
    // In col-major: C(M,N) = alpha * Q(M,d) * K^T(d,N) + beta * 0
    // But Q is row-major (M,d) = col-major (d,M)^T
    // So: C_row = Q_row @ K_row^T
    // col-major: C = Q_col^T @ K_col  ... getting complex
    
    // Simpler approach: use the row-major to col-major mapping
    // Row-major Q(M,d) = Col-major Q(d,M) transposed
    // Row-major K(N,d) = Col-major K(d,N) transposed
    // Want: Scores(M,N) = Q(M,d) @ K(N,d)^T
    // Col-major: Scores(M,N) = Q_col^T @ K_col 
    // = Q(d,M)^T @ K(d,N) = Q^T(N/A)
    //
    // Actually let's just use: Scores = Q @ K^T
    // cuBLAS sees everything as column-major.
    // Q_row(M,d) = Q_col(d,M)  — just reinterpret
    // K_row(N,d) = K_col(d,N)  — just reinterpret
    // K^T_row(d,N) = K^T_col(N,d) — just reinterpret
    // 
    // Want: S(M,N) = Q(M,d) @ K^T(d,N) in row-major
    // = S_col(N,M)^T in col-major
    // S_col = K_col^T @ Q_col = op(K_col) @ Q_col
    // with op(K_col) = K_col^T = (d,N)^T = (N,d)
    //
    // So cublasGemmEx: C(N,M) = alpha * K_col^T(N,d) @ Q_col(d,M) + beta * 0
    // = alpha * op(B) @ op(A) with opA=none, opB=transpose
    // Hmm, getting confusing. Let me use the direct GEMM helper:
    
    // Approach: Scores = Q * K^T
    // In our row-major convention, treat as:
    //   cublasGemm with op(A)=CUBLAS_OP_T for Q (to get col-major view)
    //   and op(B)=CUBLAS_OP_N for K (since K^T_row = K_col)
    // Wait, let me think more carefully...
    //
    // OK simplest: compute K^T explicitly then use our gemm_f32
    // K^T: (d, N) row-major
    // For now, just compute scores row-by-row using cuBLAS gemv

    // For M=1 (decode path):
    // scores = Q @ K^T => for each row q: score[j] = dot(q, k_j) * scale
    // Use cuBLAS gemv: scores = K @ q (where K is N×d, q is d×1)
    // K row-major (N,d) = col-major (d,N)^T
    // cublasSgemv with op(A)=CUBLAS_OP_T: y = alpha * A^T @ x + beta * y
    // A_col(d,N), x(N,), y(d,) — nope
    // cublasSgemv with op(A)=CUBLAS_OP_N: y = alpha * A @ x + beta * y
    // A_col(d,N), x(d,), y(N,) — A_col = K_col = (d,N), but K is (N,d) row
    // That means K_col has K^T, not K. We need to use opT:
    // A_col(d,N) with T => A^T(N,d) = K_row. x is q(d,), result y(N,).
    // Perfect! scores = K_row @ q via cublasSgemv(handle, CUBLAS_OP_T, d, N, &alpha, d_K, d, q, 1, &beta, d_scores, 1)

    float alpha_f = scale;  // include scale in alpha
    float beta_f = 0.0f;

    for (int m = 0; m < M; m++) {
        CUBLAS_CHECK(cublasSgemv(handle,
            CUBLAS_OP_T,     // op(A) = A^T = K in row-major
            d, N,            // A_col is (d, N) 
            &alpha_f,
            d_K, d,          // K in device memory (row-major N×d = col-major d×N), leading dim = d
            d_Q + m * d, 1,  // q vector, stride 1
            &beta_f,
            d_scores + m * N, 1  // output scores, stride 1
        ));
    }

    // Softmax over scores (each row of M)
    softmax_f32(d_scores, M, N, s);

    // Weighted sum: O = scores @ V
    // scores: (M, N) row-major, V: (N, d) row-major
    // O: (M, d) row-major
    // This is exactly our gemm_f32 with A=scores, B=V
    gemm_f32(d_scores, d_V, d_O, M, N, d, 1.0f, 0.0f, s);

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_K_transposed));
}

// ════════════════════════════════════════════════════════════════
// Dequantization Kernels
// ════════════════════════════════════════════════════════════════

// Q4_0: 32 values per block, 4-bit packed + f16 scale
__global__ void dequant_q4_0_kernel(
    const void* __restrict__ blocks_ptr,
    float* __restrict__ out,
    int64_t n
) {
    // Block format: 2 bytes scale (f16) + 16 bytes quants (32 x 4-bit)
    // Total: 18 bytes per 32 values
    struct block_q4_0 { uint16_t d; uint8_t qs[16]; };
    const block_q4_0* blocks = (const block_q4_0*)blocks_ptr;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t block_idx = idx / 32;
    int64_t within = idx % 32;

    const block_q4_0& b = blocks[block_idx];
    float d = __half2float(__ushort_as_half(b.d));
    int q = (b.qs[within >> 1] >> (4 * (within & 1))) & 0xF;
    out[idx] = (q - 8) * d;
}

void dequantize_row_q4_0(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    dequant_q4_0_kernel<<<grid, block, 0, s>>>(d_blocks, d_out, n);
}

// Q4_1: 32 values per block, 4-bit packed + f16 scale + f16 min
__global__ void dequant_q4_1_kernel(
    const void* __restrict__ blocks_ptr,
    float* __restrict__ out,
    int64_t n
) {
    struct block_q4_1 { uint16_t d; uint16_t m; uint8_t qs[16]; };
    const block_q4_1* blocks = (const block_q4_1*)blocks_ptr;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t block_idx = idx / 32;
    int64_t within = idx % 32;

    const block_q4_1& b = blocks[block_idx];
    float d = __half2float(__ushort_as_half(b.d));
    float m = __half2float(__ushort_as_half(b.m));
    int q = (b.qs[within >> 1] >> (4 * (within & 1))) & 0xF;
    out[idx] = d * q + m;
}

void dequantize_row_q4_1(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    dequant_q4_1_kernel<<<grid, block, 0, s>>>(d_blocks, d_out, n);
}

// Q8_0: 32 values per block, 8-bit + f16 scale
__global__ void dequant_q8_0_kernel(
    const void* __restrict__ blocks_ptr,
    float* __restrict__ out,
    int64_t n
) {
    struct block_q8_0 { uint16_t d; int8_t qs[32]; };
    const block_q8_0* blocks = (const block_q8_0*)blocks_ptr;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t block_idx = idx / 32;
    int64_t within = idx % 32;

    const block_q8_0& b = blocks[block_idx];
    float d = __half2float(__ushort_as_half(b.d));
    out[idx] = b.qs[within] * d;
}

void dequantize_row_q8_0(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    dequant_q8_0_kernel<<<grid, block, 0, s>>>(d_blocks, d_out, n);
}

// Q5_0: 32 values per block, 4 low bits + 1 high bit
__global__ void dequant_q5_0_kernel(
    const void* __restrict__ blocks_ptr,
    float* __restrict__ out,
    int64_t n
) {
    // Must match ggml_types.h block_q5_0 layout
    struct block_q5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
    const block_q5_0* blocks = (const block_q5_0*)blocks_ptr;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t block_idx = idx / 32;
    int64_t within = idx % 32;

    const block_q5_0& b = blocks[block_idx];
    float d = __half2float(__ushort_as_half(b.d));
    int ql = (b.qs[within >> 1] >> (4 * (within & 1))) & 0xF;
    int qh_bit = (b.qh[within >> 3] >> (within & 7)) & 1;
    int q = ql | (qh_bit << 4);
    out[idx] = (q - 16) * d;
}

void dequantize_row_q5_0(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    dequant_q5_0_kernel<<<grid, block, 0, s>>>(d_blocks, d_out, n);
}

// Q5_1: Q5_0 + f16 min
__global__ void dequant_q5_1_kernel(
    const void* __restrict__ blocks_ptr,
    float* __restrict__ out,
    int64_t n
) {
    struct block_q5_1 { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; };
    const block_q5_1* blocks = (const block_q5_1*)blocks_ptr;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t block_idx = idx / 32;
    int64_t within = idx % 32;

    const block_q5_1& b = blocks[block_idx];
    float d = __half2float(__ushort_as_half(b.d));
    float m = __half2float(__ushort_as_half(b.m));
    int ql = (b.qs[within >> 1] >> (4 * (within & 1))) & 0xF;
    int qh_bit = (b.qh[within >> 3] >> (within & 7)) & 1;
    int q = ql | (qh_bit << 4);
    out[idx] = d * q + m;
}

void dequantize_row_q5_1(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    dequant_q5_1_kernel<<<grid, block, 0, s>>>(d_blocks, d_out, n);
}

// FP16 to FP32 dequant
__global__ void dequant_f16_kernel(const void* __restrict__ f16_ptr, float* __restrict__ out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint16_t* f16 = (const uint16_t*)f16_ptr;
    out[idx] = __half2float(__ushort_as_half(f16[idx]));
}

void dequantize_row_f16(const void* d_f16, float* d_out, int64_t n, void* stream_ptr) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    dequant_f16_kernel<<<grid, block, 0, s>>>(d_f16, d_out, n);
}

// Stub implementations for Q4_K, Q5_K, Q6_K (complex block formats)
// These fall back to CPU dequant for now — GPU dequant for K-quants
// requires more complex kernels that will be added in a follow-up.

void dequantize_row_q4_k(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    // K-quant dequant on GPU: TODO — uses complex super-block format
    // For now, this should not be called directly; use gemv_dequant
    // which falls back to CPU for K-quants
    throw std::runtime_error("Q4_K GPU dequant not yet implemented — use CPU fallback");
}

void dequantize_row_q5_k(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    throw std::runtime_error("Q5_K GPU dequant not yet implemented — use CPU fallback");
}

void dequantize_row_q6_k(const void* d_blocks, float* d_out, int64_t n, void* stream_ptr) {
    throw std::runtime_error("Q6_K GPU dequant not yet implemented — use CPU fallback");
}

// ── Generic GEMV with inline dequant on GPU ──
// For quantized types we support on GPU (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32):
// 1. Dequantize the weight matrix to fp32 on GPU
// 2. Use cuBLAS gemv
// For K-quants (Q4_K, Q5_K, Q6_K): fall back to CPU
void gemv_dequant(
    const float* d_x, const void* d_weights,
    float* d_y, int K, int M,
    int ggml_type,
    float* d_scratch,
    void* stream_ptr
) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;

    // Dequantize weight row(s) to fp32 scratch buffer
    int64_t n_elements = (int64_t)M * K;
    switch (ggml_type) {
        case 0: // GGML_TYPE_F32
            // Already fp32, no dequant needed
            gemv_f32(d_weights, d_x, d_y, M, K, 1.0f, 0.0f, s);
            return;
        case 1: // GGML_TYPE_F16
            dequantize_row_f16(d_weights, d_scratch, n_elements, s);
            break;
        case 2: // GGML_TYPE_Q4_0
            dequantize_row_q4_0(d_weights, d_scratch, n_elements, s);
            break;
        case 3: // GGML_TYPE_Q4_1
            dequantize_row_q4_1(d_weights, d_scratch, n_elements, s);
            break;
        case 6: // GGML_TYPE_Q5_0
            dequantize_row_q5_0(d_weights, d_scratch, n_elements, s);
            break;
        case 7: // GGML_TYPE_Q5_1
            dequantize_row_q5_1(d_weights, d_scratch, n_elements, s);
            break;
        case 8: // GGML_TYPE_Q8_0
            dequantize_row_q8_0(d_weights, d_scratch, n_elements, s);
            break;
        default:
            // K-quants not yet supported on GPU
            throw std::runtime_error("GPU gemv_dequant: unsupported quant type " + std::to_string(ggml_type));
    }

    // Now compute GEMV with dequantized fp32 weights
    gemv_f32(d_scratch, d_x, d_y, M, K, 1.0f, 0.0f, s);
}

// ── Token embedding lookup ──
__global__ void embedding_lookup_kernel(
    const void* embeddings,
    const int* token_ids,
    float* out,
    int seq_len, int hidden_dim, int vocab_size,
    int ggml_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * hidden_dim;
    if (idx >= total) return;

    int s = idx / hidden_dim;
    int d = idx % hidden_dim;

    int tok = token_ids[s];
    if (tok < 0 || tok >= vocab_size) {
        out[idx] = 0.0f;
        return;
    }

    // This kernel only handles F32 and F16 embeddings
    if (ggml_type == 0) { // F32
        const float* emb = (const float*)embeddings;
        out[idx] = emb[tok * hidden_dim + d];
    } else if (ggml_type == 1) { // F16
        const uint16_t* emb = (const uint16_t*)embeddings;
        out[idx] = __half2float(__ushort_as_half(emb[tok * hidden_dim + d]));
    }
}

void embedding_lookup(
    const void* d_embeddings,
    const int* d_token_ids,
    float* d_out,
    int seq_len, int hidden_dim, int vocab_size,
    int ggml_type,
    void* stream_ptr
) {
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    int total = seq_len * hidden_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    embedding_lookup_kernel<<<grid, block, 0, s>>>(
        d_embeddings, d_token_ids, d_out,
        seq_len, hidden_dim, vocab_size, ggml_type
    );
}

// ── MoE top-k (simple version — runs on CPU for now, GPU future) ──
void moe_top_k(
    const float* d_router_logits,
    int* d_top_indices, float* d_top_weights,
    int n_experts, int k,
    void* stream_ptr
) {
    // For small n_experts (typically 64-128), CPU is faster
    // due to PCIe transfer overhead. Download, compute, upload.
    std::vector<float> logits(n_experts);
    cudaStream_t s = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    CUDA_CHECK(cudaMemcpyAsync(logits.data(), d_router_logits,
        n_experts * sizeof(float), cudaMemcpyDeviceToHost, s));
    if (s) CUDA_CHECK(cudaStreamSynchronize(s));

    // Top-k selection
    std::vector<std::pair<float, int>> indexed(n_experts);
    for (int i = 0; i < n_experts; i++) indexed[i] = {logits[i], i};
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    float sum = 0.0f;
    std::vector<int> top_idx(k);
    std::vector<float> top_w(k);
    for (int i = 0; i < k; i++) {
        top_idx[i] = indexed[i].second;
        top_w[i] = indexed[i].first;
        sum += top_w[i];
    }
    if (sum > 1e-12f) {
        for (int i = 0; i < k; i++) top_w[i] /= sum;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_top_indices, top_idx.data(),
        k * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_top_weights, top_w.data(),
        k * sizeof(float), cudaMemcpyHostToDevice, s));
}

} // namespace cuda
} // namespace vibeblade

#endif // VIBEBLADE_USE_CUDA
