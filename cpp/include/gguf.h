#pragma once
// VibeBlade GGUF reader — parses GGUF files for model loading.
//
// IMPORTANT (ARM64 platform bug workaround):
// All internal maps use unique_ptr and are initialized in the constructor
// body AFTER the large file buffer is allocated. On certain ARM64/libstdc++
// combinations, a large allocation (>128MB) corrupts previously-allocated
// map sentinel nodes. By creating maps after the buffer, their heap
// allocations land in non-corrupted address space.

#include "ggml_types.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

namespace vibeblade {

struct TensorInfo {
    std::string name;
    int n_dims;
    int64_t dims[4];
    ggml_type type;
    size_t offset;   // byte offset to tensor data from start of data section
    size_t size;     // total bytes of tensor data
};

struct GGUFFile {
    GGUFFile(const char* path);
    ~GGUFFile();

    // Non-copyable, non-movable (owns raw buffer + maps)
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    // Direct pointer to tensor data
    const void* tensor_data(const std::string& name) const;

    // Tensor info by name
    const TensorInfo* tensor_info(const std::string& name) const;

    // Scalar metadata accessors
    std::string meta_string(const std::string& key) const;
    int64_t   meta_int(const std::string& key, int64_t default_val = 0) const;
    float     meta_float(const std::string& key, float default_val = 0.0f) const;
    bool      meta_bool(const std::string& key, bool default_val = false) const;

    // Array metadata accessors
    std::vector<std::string> meta_string_array(const std::string& key) const;
    std::vector<int64_t>     meta_int_array(const std::string& key) const;
    std::vector<float>       meta_float_array(const std::string& key) const;

    // All tensor infos
    const std::vector<TensorInfo>& tensors() const { return tensor_infos_; }

    // File info
    uint32_t version() const { return version_; }
    size_t   file_size() const { return file_size_; }

private:
    void parse_header(const uint8_t* ptr);
    void parse_metadata(const uint8_t* ptr, size_t& offset);
    void parse_tensor_infos(const uint8_t* ptr, size_t& offset);
    void load_file(const char* path);
    void init_maps();

    int fd_ = -1;
    uint8_t* data_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;

    uint32_t version_ = 0;
    uint64_t n_tensors_ = 0;
    uint64_t n_kv_ = 0;

    // Maps initialized via unique_ptr AFTER file buffer is loaded
    std::unique_ptr<std::map<std::string, TensorInfo>> tensor_map_;
    std::vector<TensorInfo> tensor_infos_;
    std::unique_ptr<std::map<std::string, std::string>> meta_strings_;
    std::unique_ptr<std::map<std::string, int64_t>>     meta_ints_;
    std::unique_ptr<std::map<std::string, float>>       meta_floats_;
    std::unique_ptr<std::map<std::string, bool>>        meta_bools_;
    std::unique_ptr<std::map<std::string, std::vector<std::string>>> meta_string_arrays_;
    std::unique_ptr<std::map<std::string, std::vector<int64_t>>>     meta_int_arrays_;
    std::unique_ptr<std::map<std::string, std::vector<float>>>       meta_float_arrays_;
};

}  // namespace vibeblade
