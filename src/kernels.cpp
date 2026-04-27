#include <immintrin.h>
#include <stdint.h>
#include <cstddef>

// Refined RotorQuant 4-bit Unpacker with AVX-512
extern "C" void ts_rotor_unpack(
    const uint8_t* __restrict__ pinned_ram,
    float* __restrict__ output,
    const float* __restrict__ rotor_matrix,
    size_t n)
{
    // FIX #2: Process SIMD-width chunks, then scalar tail
    size_t vec_len = n - (n % 32);
    size_t i;
    for (i = 0; i < vec_len; i += 32) {
        // Load 128-bit chunk (contains 32 4-bit weights)
        __m128i raw = _mm_loadu_si128((__m128i*)&pinned_ram[i / 2]);

        // Expand and Unpack
        __m256i low = _mm256_and_si256(_mm256_cvtepu8_epi16(raw), _mm256_set1_epi8(0x0F));
        __m512 v_float = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(low)));

        // FIX #8: Advance rotor pointer per chunk
        __m512 rotor = _mm512_loadu_ps(rotor_matrix + i);
        __m512 result = _mm512_mul_ps(v_float, rotor);

        _mm512_storeu_ps(&output[i], result);
    }
    // Scalar tail for remaining elements
    for (; i < n; i++) {
        int byte_idx = i / 2;
        int nibble_shift = (i % 2) * 4;
        int val = (pinned_ram[byte_idx] >> nibble_shift) & 0x0F;
        output[i] = (float)val * rotor_matrix[i];
    }
}

// Refined dReLU activation for TurboSparse
extern "C" void ts_drelu_activation(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t n)
{
    // FIX #2: Process SIMD-width chunks, then scalar tail
    size_t vec_len = n - (n % 16);
    size_t i;
    for (i = 0; i < vec_len; i += 16) {
        __m512 in_vec = _mm512_loadu_ps(&input[i]);
        __m512 zero = _mm512_setzero_ps();
        __m512 result = _mm512_max_ps(in_vec, zero);  // ReLU: max(x, 0)
        _mm512_storeu_ps(&output[i], result);
    }
    // Scalar tail
    for (; i < n; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(_vibeblade_core, m) {
    m.doc() = "VibeBlade C++ AVX-512 kernels";

    m.def("ts_rotor_unpack", [](py::array_t<uint8_t> pinned_ram, py::array_t<float> output, py::array_t<float> rotor_matrix, size_t n) {
        auto buf_ram = pinned_ram.request();
        auto buf_out = output.request();
        auto buf_rotor = rotor_matrix.request();
        // Validate that n doesn't exceed buffer sizes
        if (n > 0 && (buf_ram.size < (py::ssize_t)((n + 1) / 2) ||
                      buf_out.size < (py::ssize_t)n ||
                      buf_rotor.size < (py::ssize_t)n)) {
            throw std::invalid_argument("Input buffer too small for n=" + std::to_string(n));
        }
        ts_rotor_unpack(
            static_cast<const uint8_t*>(buf_ram.ptr),
            static_cast<float*>(buf_out.ptr),
            static_cast<const float*>(buf_rotor.ptr),
            n
        );
    }, "4-bit weight unpacker using AVX-512");

    m.def("ts_drelu_activation", [](py::array_t<float> input, py::array_t<float> output, size_t n) {
        auto buf_in = input.request();
        auto buf_out = output.request();
        // Validate that n doesn't exceed buffer sizes
        if (n > 0 && (buf_in.size < (py::ssize_t)n ||
                      buf_out.size < (py::ssize_t)n)) {
            throw std::invalid_argument("Input buffer too small for n=" + std::to_string(n));
        }
        ts_drelu_activation(
            static_cast<const float*>(buf_in.ptr),
            static_cast<float*>(buf_out.ptr),
            n
        );
    }, "dReLU activation sparsification using AVX-512");
}
