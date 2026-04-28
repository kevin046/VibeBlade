cmake_minimum_required(VERSION 3.16)
project(vibeblade_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── OpenMP for multi-threaded gemv / attention ──
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP:          YES (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "OpenMP:          NO (single-threaded fallback)")
endif()

# ── Find dependencies ──
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# pybind11 via cmake config or pip
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    find_package(pybind11 QUIET)
endif()
if(NOT pybind11_FOUND)
    # Fallback: discover pybind11 cmake dir from Python package
    message(STATUS "pybind11 not found via find_package, trying Python discovery...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import pybind11, os; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))"
        RESULT_VARIABLE _PYBIND_CMD_RESULT
        OUTPUT_VARIABLE _PYBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_PYBIND_CMD_RESULT EQUAL 0 AND EXISTS "${_PYBIND_CMAKE_DIR}")
        message(STATUS "pybind11 cmake dir: ${_PYBIND_CMAKE_DIR}")
        set(pybind11_DIR "${_PYBIND_CMAKE_DIR}")
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR
            "pybind11 not found. Install with: pip install pybind11\n"
            "  Or set -Dpybind11_DIR=<path> to the cmake config directory."
        )
    endif()
endif()

# ── Detect SIMD ──
include(CheckCXXSourceCompiles)

# AVX-512 Foundation
check_cxx_source_compiles([=[
#include <immintrin.h>
int main() {
    __m512 a = _mm512_set1_ps(1.0f);
    (void)a;
    return 0;
}
]=] HAS_AVX512F)

# AVX-512 FP16 (Sapphire Rapids+)
check_cxx_source_compiles([=[
#include <immintrin.h>
int main() {
    __m512h a = _mm512_setzero_ph();
    (void)a;
    return 0;
}
]=] HAS_AVX512FP16)

# AVX-512 VNNI
check_cxx_source_compiles([=[
#include <immintrin.h>
int main() {
    __m512i a = _mm512_setzero_si512();
    __m512i b = _mm512_dpbusd_epi32(a, a, a);
    (void)b;
    return 0;
}
]=] HAS_AVX512VNNI)

# AVX2
check_cxx_source_compiles([=[
#include <immintrin.h>
int main() {
    __m256d a = _mm256_setzero_pd();
    (void)a;
    return 0;
}
]=] HAS_AVX2)

# NEON (ARM) — skip entirely on x86/x86_64
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86|i386|i686|AMD64")
    check_cxx_source_compiles([=[
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
#error "Not ARM"
#endif
#include <arm_neon.h>
int main() {
    float16x8_t a = vdupq_n_f16(0);
    (void)a;
    return 0;
}
]=] HAS_NEON_FP16)
endif()

message(STATUS "AVX-512F:        ${HAS_AVX512F}")
message(STATUS "AVX-512-FP16:    ${HAS_AVX512FP16}")
message(STATUS "AVX-512-VNNI:    ${HAS_AVX512VNNI}")
message(STATUS "AVX2:            ${HAS_AVX2}")
message(STATUS "NEON FP16:       ${HAS_NEON_FP16}")

# ── Build module ──
pybind11_add_module(_vibeblade_native
    src/bindings.cpp
    src/fp16_compat.cpp
    src/gemm.cpp
    src/norm.cpp
    src/activation.cpp
    src/quantize.cpp
    src/attention.cpp
    src/rope.cpp
    src/softmax.cpp
    src/gguf.cpp
    src/dequant.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/fast_model.cpp
)

target_include_directories(_vibeblade_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set optimization flags — Release uses -O3 + LTO + OpenMP
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_vibeblade_native PRIVATE -g -O0)
else()
    target_compile_options(_vibeblade_native PRIVATE -O3 -DNDEBUG)
    # LTO for cross-compilation-unit inlining
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED)
    if(IPO_SUPPORTED)
        set_property(TARGET _vibeblade_native PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO:             YES")
    endif()
endif()

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(_vibeblade_native PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_vibeblade_native PRIVATE TS_OPENMP)
endif()

# Add SIMD flags
set(SIMD_FLAGS "")
if(HAS_AVX512FP16)
    list(APPEND SIMD_FLAGS "-mavx512f" "-mavx512vl" "-mavx512fp16")
    target_compile_definitions(_vibeblade_native PRIVATE TS_AVX512FP16)
elseif(HAS_AVX512F)
    list(APPEND SIMD_FLAGS "-mavx512f" "-mavx512vl")
    target_compile_definitions(_vibeblade_native PRIVATE TS_AVX512F)
elseif(HAS_AVX2)
    list(APPEND SIMD_FLAGS "-mavx2" "-mfma")
    target_compile_definitions(_vibeblade_native PRIVATE TS_AVX2)
elseif(HAS_NEON_FP16)
    list(APPEND SIMD_FLAGS "-march=native")
    target_compile_definitions(_vibeblade_native PRIVATE TS_NEON_FP16)
endif()

if(SIMD_FLAGS)
    target_compile_options(_vibeblade_native PRIVATE ${SIMD_FLAGS})
    message(STATUS "SIMD flags:      ${SIMD_FLAGS}")
else()
    message(STATUS "SIMD:            scalar fallback (no SIMD detected)")
endif()

# On aarch64, -march=native is usually sufficient for NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(_vibeblade_native PRIVATE -march=native)
endif()

# Link math library on Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(_vibeblade_native PRIVATE m)
endif()
