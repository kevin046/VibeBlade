// TurboStack Metal Bridge — Objective-C++ bridge between Python and Metal GPU
// Requires: macOS 13+, Xcode, Metal framework
// Build: pybind11 + Metal performance shaders

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>

namespace py = pybind11;

// ─────────────────────────────────────────────
// MetalBackend — manages Metal device, queues, pipelines
// ─────────────────────────────────────────────

class MetalBackend {
public:
    MetalBackend() {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            throw std::runtime_error("No Metal device found (requires Apple Silicon or Apple GPU)");
        }
        _commandQueue = [_device newCommandQueue];
        if (!_commandQueue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        // Compile default shader library
        NSError* error = nil;
        // Try to load pre-compiled metallib, fall back to runtime compilation
        _library = [_device newLibraryWithSource:[NSString stringWithUTF8String:METAL_SHADER_SOURCE]
                                         options:nil
                                           error:&error];
        if (!_library) {
            std::string errMsg = [[error localizedDescription] UTF8String];
            throw std::runtime_error("Metal shader compilation failed: " + errMsg);
        }

        // Cache pipeline states
        _pipelines = [[NSMutableDictionary alloc] init];
    }

    ~MetalBackend() {
        @autoreleasepool {
            [_pipelines release];
            [_library release];
            [_commandQueue release];
            [_device release];
        }
    }

    std::string device_name() const {
        return [_device.name UTF8String];
    }

    bool supports_family(uint32_t family) const {
        return [_device supportsFamily:(MTLGPUFamily)family];
    }

    // ── Buffer management ────────────────────

    py::array_t<float> zeros(uint64_t n) {
        auto buf = [_device newBufferWithLength:n * sizeof(float)
                                        options:MTLResourceStorageModeShared];
        auto result = py::array_t<float>({(ssize_t)n});
        std::memset(result.mutable_data(), 0, n * sizeof(float));
        return result;
    }

    // ── RotorQuant: 4-bit unpack + SO(4) rotation ──

    py::array_t<float> rotor_unpack(
        py::array_t<uint8_t> packed,
        py::array_t<float> rotor,
        uint64_t n)
    {
        auto buf_packed = [_device newBufferWithBytes:packed.data()
                                                length:packed.size()
                                               options:MTLResourceStorageModeShared];
        auto buf_output = [_device newBufferWithLength:n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        auto buf_rotor = [_device newBufferWithBytes:rotor.data()
                                               length:rotor.size()
                                              options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pipeline = get_pipeline("ts_rotor_unpack");
        id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_packed offset:0 atIndex:0];
        [encoder setBuffer:buf_output offset:0 atIndex:1];
        [encoder setBuffer:buf_rotor  offset:0 atIndex:2];

        uint n_val = (uint)n;
        [encoder setBytes:&n_val length:sizeof(uint) atIndex:3];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(
            std::min((uint)n, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);
        [encoder dispatchThreadgroups:gridSize
                 threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        auto result = py::array_t<float>({(ssize_t)n});
        std::memcpy(result.mutable_data(), buf_output.contents, n * sizeof(float));

        [buf_packed release];
        [buf_output release];
        [buf_rotor release];

        return result;
    }

    // ── dReLU activation ────────────────────

    py::array_t<float> drelu(py::array_t<float> input, uint64_t n) {
        auto buf_input = [_device newBufferWithBytes:input.data()
                                               length:input.size()
                                              options:MTLResourceStorageModeShared];
        auto buf_output = [_device newBufferWithLength:n * sizeof(float)
                                                options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pipeline = get_pipeline("ts_drelu");
        id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_input  offset:0 atIndex:0];
        [encoder setBuffer:buf_output offset:0 atIndex:1];

        uint n_val = (uint)n;
        [encoder setBytes:&n_val length:sizeof(uint) atIndex:2];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(
            std::min((uint)n, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);
        [encoder dispatchThreadgroups:gridSize
                 threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        auto result = py::array_t<float>({(ssize_t)n});
        std::memcpy(result.mutable_data(), buf_output.contents, n * sizeof(float));

        [buf_input release];
        [buf_output release];

        return result;
    }

    // ── SiLU activation ─────────────────────

    py::array_t<float> silu(py::array_t<float> input, uint64_t n) {
        auto buf_input = [_device newBufferWithBytes:input.data()
                                               length:input.size()
                                              options:MTLResourceStorageModeShared];
        auto buf_output = [_device newBufferWithLength:n * sizeof(float)
                                                options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pipeline = get_pipeline("ts_silu");
        id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_input  offset:0 atIndex:0];
        [encoder setBuffer:buf_output offset:0 atIndex:1];

        uint n_val = (uint)n;
        [encoder setBytes:&n_val length:sizeof(uint) atIndex:2];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(
            std::min((uint)n, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);
        [encoder dispatchThreadgroups:gridSize
                 threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        auto result = py::array_t<float>({(ssize_t)n});
        std::memcpy(result.mutable_data(), buf_output.contents, n * sizeof(float));

        [buf_input release];
        [buf_output release];

        return result;
    }

    // ── Matrix Multiply: C = A × B ─────────

    py::array_t<float> matmul(
        py::array_t<float> a,
        py::array_t<float> b,
        uint64_t M, uint64_t K, uint64_t N)
    {
        auto buf_a = [_device newBufferWithBytes:a.data()
                                            length:a.size()
                                           options:MTLResourceStorageModeShared];
        auto buf_b = [_device newBufferWithBytes:b.data()
                                            length:b.size()
                                           options:MTLResourceStorageModeShared];
        auto buf_c = [_device newBufferWithLength:M * N * sizeof(float)
                                           options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pipeline = get_pipeline("ts_matmul");
        id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_a offset:0 atIndex:0];
        [encoder setBuffer:buf_b offset:0 atIndex:1];
        [encoder setBuffer:buf_c offset:0 atIndex:2];

        uint m_val = (uint)M, k_val = (uint)K, n_val = (uint)N;
        [encoder setBytes:&m_val length:sizeof(uint) atIndex:3];
        [encoder setBytes:&k_val length:sizeof(uint) atIndex:4];
        [encoder setBytes:&n_val length:sizeof(uint) atIndex:5];

        uint tpg = std::min((uint)16, pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake((M + 15) / 16, (N + 15) / 16, 1);
        MTLSize threadGroupSize = MTLSizeMake(tpg, tpg, 1);
        [encoder dispatchThreadgroups:gridSize
                 threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        auto result = py::array_t<float>({(ssize_t)M, (ssize_t)N});
        std::memcpy(result.mutable_data(), buf_c.contents, M * N * sizeof(float));

        [buf_a release];
        [buf_b release];
        [buf_c release];

        return result;
    }

    // ── RMSNorm ─────────────────────────────

    py::array_t<float> rms_norm(
        py::array_t<float> input,
        py::array_t<float> weight,
        float eps, uint64_t dim)
    {
        uint64_t rows = input.shape(0) / dim;
        auto buf_input = [_device newBufferWithBytes:input.data()
                                                length:input.size()
                                               options:MTLResourceStorageModeShared];
        auto buf_output = [_device newBufferWithLength:input.size()
                                                  options:MTLResourceStorageModeShared];
        auto buf_weight = [_device newBufferWithBytes:weight.data()
                                                 length:weight.size()
                                                options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pipeline = get_pipeline("ts_rms_norm_vec");
        id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_input  offset:0 atIndex:0];
        [encoder setBuffer:buf_output offset:0 atIndex:1];
        [encoder setBuffer:buf_weight offset:0 atIndex:2];

        [encoder setBytes:&eps   length:sizeof(float) atIndex:3];
        uint dim_val = (uint)dim;
        [encoder setBytes:&dim_val length:sizeof(uint) atIndex:4];

        uint tpg = std::min((uint)rows, pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(tpg, 1, 1);
        [encoder dispatchThreadgroups:gridSize
                 threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        auto result = py::array_t<float>(input.shape());
        std::memcpy(result.mutable_data(), buf_output.contents, input.size());

        [buf_input release];
        [buf_output release];
        [buf_weight release];

        return result;
    }

private:
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLLibrary> _library;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* _pipelines;

    id<MTLComputePipelineState> get_pipeline(const char* name) {
        NSString* key = [NSString stringWithUTF8String:name];
        id<MTLComputePipelineState> cached = _pipelines[key];
        if (cached) return cached;

        NSError* error = nil;
        id<MTLFunction> func = [_library newFunctionWithName:key];
        if (!func) {
            throw std::runtime_error(std::string("Metal function not found: ") + name);
        }
        id<MTLComputePipelineState> pipeline =
            [_device newComputePipelineStateWithFunction:func error:&error];
        [func release];
        if (!pipeline) {
            std::string errMsg = [[error localizedDescription] UTF8String];
            throw std::runtime_error("Pipeline creation failed for " + std::string(name) + ": " + errMsg);
        }
        _pipelines[key] = pipeline;
        return pipeline;
    }

    // Embedded Metal shader source (loaded at compile time)
    static constexpr const char* METAL_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void ts_rotor_unpack(
    device const uint8_t* packed_weights [[buffer(0)]],
    device float* output           [[buffer(1)]],
    device const float* rotor      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint byte_idx = gid / 2;
    uint8_t byte_val = packed_weights[byte_idx];
    float val = (gid & 1u) ? (float)(byte_val & 0x0F)
                           : (float)((byte_val >> 4) & 0x0F);
    uint group_start = (gid / 4) * 4;
    output[gid] = val * rotor[group_start + gid % 4];
}

kernel void ts_drelu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = fmax(input[gid], 0.0f);
}

kernel void ts_silu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = x * (1.0f / (1.0f + exp(-x)));
}

kernel void ts_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M, constant uint& K, constant uint& N,
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.x, col = gid.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

kernel void ts_rms_norm_vec(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    constant uint& dim          [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    float4 sum_sq4 = float4(0.0f);
    uint row_off = row * dim;
    uint i = 0;
    for (; i + 4 <= dim; i += 4) {
        float4 v = float4(input[row_off+i], input[row_off+i+1],
                          input[row_off+i+2], input[row_off+i+3]);
        sum_sq4 += v * v;
    }
    float sum_sq = sum_sq4.x + sum_sq4.y + sum_sq4.z + sum_sq4.w;
    for (; i < dim; i++) sum_sq += input[row_off+i] * input[row_off+i];
    float rms = rsqrt(sum_sq / (float)dim + eps);
    for (i = 0; i + 4 <= dim; i += 4) {
        float4 v = float4(input[row_off+i], input[row_off+i+1],
                          input[row_off+i+2], input[row_off+i+3]);
        float4 w = float4(weight[i], weight[i+1], weight[i+2], weight[i+3]);
        float4 r = v * rms * w;
        output[row_off+i] = r.x; output[row_off+i+1] = r.y;
        output[row_off+i+2] = r.z; output[row_off+i+3] = r.w;
    }
    for (; i < dim; i++)
        output[row_off+i] = input[row_off+i] * rms * weight[i];
}
    )";
};

// ─────────────────────────────────────────────
// Python module registration
// ─────────────────────────────────────────────

PYBIND11_MODULE(_turbostack_metal, m) {
    m.doc() = "TurboStack Metal (Apple Silicon) GPU backend";

    py::class_<MetalBackend>(m, "MetalBackend")
        .def(py::init<>(), "Initialize Metal backend with default device")
        .def("device_name", &MetalBackend::device_name,
             "Return the Metal device name")
        .def("rotor_unpack", &MetalBackend::rotor_unpack,
             "4-bit weight unpack + SO(4) rotation",
             py::arg("packed"), py::arg("rotor"), py::arg("n"))
        .def("drelu", &MetalBackend::drelu,
             "dReLU activation sparsification",
             py::arg("input"), py::arg("n"))
        .def("silu", &MetalBackend::silu,
             "SiLU activation (x * sigmoid(x))",
             py::arg("input"), py::arg("n"))
        .def("matmul", &MetalBackend::matmul,
             "Matrix multiply C = A × B",
             py::arg("a"), py::arg("b"), py::arg("M"), py::arg("K"), py::arg("N"))
        .def("rms_norm", &MetalBackend::rms_norm,
             "RMS normalization",
             py::arg("input"), py::arg("weight"), py::arg("eps"), py::arg("dim"))
        .def("zeros", &MetalBackend::zeros,
             "Allocate zero-filled GPU buffer",
             py::arg("n"));
}
