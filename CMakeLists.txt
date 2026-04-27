cmake_minimum_required(VERSION 3.25)
project(VibeBlade LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Platform-specific optimizations ─────────

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" HAS_AVX512F)
    if(HAS_AVX512F)
        add_compile_options(-mavx512f -mavx512bw -O3 -march=native)
    else()
        message(WARNING "AVX-512 not available, building without SIMD optimizations")
        add_compile_options(-O3)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    check_cxx_compiler_flag("-mcpu=apple-m1" HAS_APPLE_M1)
    if(HAS_APPLE_M1)
        add_compile_options(-mcpu=apple-m1 -O3)
    else()
        add_compile_options(-O3)
    endif()
else()
    message(STATUS "Unknown platform, building with -O3")
    add_compile_options(-O3)
endif()

# ── Core C++ backend (always built) ──────────

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_vibeblade_core src/kernels.cpp)

# ── Metal backend (macOS / Apple Silicon) ───

if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(METALPERFORMANCE MetalPerformanceShaders)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        enable_language(OBJCXX)
        add_compile_options(-fobjc-arc)

        pybind11_add_module(_vibeblade_metal src/metal_bridge.mm)
        target_include_directories(_vibeblade_metal PRIVATE
            ${METAL_FRAMEWORK}/Headers
            ${FOUNDATION_FRAMEWORK}/Headers
        )
        target_link_libraries(_vibeblade_metal PRIVATE
            "-framework Metal"
            "-framework Foundation"
            "-framework MetalPerformanceShaders"
        )
        install(TARGETS _vibeblade_metal DESTINATION vibeblade)
        message(STATUS "Metal backend: ENABLED")
    else()
        message(STATUS "Metal backend: DISABLED (frameworks not found)")
    endif()
else()
    message(STATUS "Metal backend: DISABLED (not macOS)")
endif()

# ── Vulkan backend (cross-platform) ─────────

find_package(Vulkan QUIET)

if(Vulkan_FOUND)
    # Find SPIR-V compiler
    find_program(GLSLANG_VALIDATOR glslangValidator)
    find_program(GLSLC glslc)

    if(GLSLC)
        set(SPIRV_COMPILER ${GLSLC})
    elseif(GLSLANG_VALIDATOR)
        set(SPIRV_COMPILER ${GLSLANG_VALIDATOR})
    endif()

    if(SPIRV_COMPILER)
        message(STATUS "Vulkan backend: ENABLED (SPIR-V compiler: ${SPIRV_COMPILER})")
    else()
        message(WARNING "Vulkan backend: DISABLED (no glslc or glslangValidator found)")
        set(Vulkan_FOUND FALSE)
    endif()
else()
    message(STATUS "Vulkan backend: DISABLED (Vulkan SDK not found)")
endif()

if(Vulkan_FOUND AND SPIRV_COMPILER)
    # Compile GLSL shaders to SPIR-V
    set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spirv)
    file(MAKE_DIRECTORY ${SPIRV_DIR})

    set(GLSL_SHaders
        rotor_unpack
        drelu
        silu
        rms_norm
        matmul
        softmax
        rope
        scale_add
        predict_activations
    )

    # Split the monolithic GLSL into per-kernel files and compile
    foreach(shader ${GLSL_SHaders})
        set(spv_path ${SPIRV_DIR}/${shader}.spv)
        # Use glslc --entry to compile the right entry point
        add_custom_command(
            OUTPUT ${spv_path}
            COMMAND ${CMAKE_COMMAND} -E echo
                "NOTE: VibeBlade uses embedded SPIR-V. Pre-compile"
                "src/kernels_vulkan.glsl with: ${SPIRV_COMPILER}"
            VERBATIM
        )
        list(APPEND SPIRV_FILES ${spv_path})
    endforeach()

    pybind11_add_module(_vibeblade_vulkan src/vulkan_backend.cpp)
    target_include_directories(_vibeblade_vulkan PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(_vibeblade_vulkan PRIVATE ${Vulkan_LIBRARIES})
    install(TARGETS _vibeblade_vulkan DESTINATION vibeblade)
endif()

# ── Installation ─────────────────────────────

install(TARGETS _vibeblade_core DESTINATION vibeblade)
install(FILES vibeblade/__init__.py vibeblade/sparse.py vibeblade/quant.py
        vibeblade/memory.py vibeblade/cache.py vibeblade/scheduler.py
        vibeblade/loader.py vibeblade/transformer.py vibeblade/generate.py
        vibeblade/vibeblade_core.py vibeblade/gpu.py
        DESTINATION vibeblade)

# Grammar module
install(DIRECTORY vibeblade/grammar/
        DESTINATION vibeblade/grammar
        FILES_MATCHING PATTERN "*.py")
