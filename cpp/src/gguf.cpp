#include "gguf.h"
#include "win_compat.h"
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <memory>
#include <cstdlib>
#include <string>

namespace vibeblade {

static constexpr uint32_t GGUF_MAGIC   = 0x46554747;  // "GGUF" (ASCII LE)
static constexpr uint32_t GGUF_VERSION = 3;
static constexpr size_t   GGUF_DEFAULT_ALIGNMENT = 32;

static inline uint16_t read_u16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t read_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline int32_t  read_i32(const uint8_t* p) { int32_t v;  memcpy(&v, p, 4); return v; }
static inline uint64_t read_u64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline int64_t  read_i64(const uint8_t* p) { int64_t v;  memcpy(&v, p, 8); return v; }
static inline float    read_f32(const uint8_t* p) { float v;    memcpy(&v, p, 4); return v; }
static inline double   read_f64(const uint8_t* p) { double v;   memcpy(&v, p, 8); return v; }

static inline size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// ── GGUF string: { uint64 len, char[len] } ──
struct GGUFString {
    uint64_t len;
    const char* ptr;
};

static GGUFString read_string(const uint8_t* p, size_t& offset, size_t max_len) {
    GGUFString s;
    if (offset + 8 > max_len)
        throw std::runtime_error("GGUF: string length extends past end of file");
    s.len = read_u64(p + offset); offset += 8;
    if (s.len > max_len || offset + s.len > max_len)
        throw std::runtime_error("GGUF: string data extends past end of file");
    s.ptr = (const char*)(p + offset); offset += s.len;
    return s;
}

// Bounds-check helper
#define CHECK_OFFSET(n) do { \
    if (offset + (n) > file_size_) \
        throw std::runtime_error("GGUF: read past end of file at offset " + std::to_string(offset)); \
} while(0)

void GGUFFile::init_maps() {
    tensor_map_        = std::make_unique<std::map<std::string, TensorInfo>>();
    meta_strings_      = std::make_unique<std::map<std::string, std::string>>();
    meta_ints_         = std::make_unique<std::map<std::string, int64_t>>();
    meta_floats_       = std::make_unique<std::map<std::string, float>>();
    meta_bools_        = std::make_unique<std::map<std::string, bool>>();
    meta_string_arrays_= std::make_unique<std::map<std::string, std::vector<std::string>>>();
    meta_int_arrays_   = std::make_unique<std::map<std::string, std::vector<int64_t>>>();
    meta_float_arrays_ = std::make_unique<std::map<std::string, std::vector<float>>>();
}

void GGUFFile::parse_header(const uint8_t* ptr) {
    if (file_size_ < 24) throw std::runtime_error("GGUF file too small for header");
    uint32_t magic = read_u32(ptr);
    if (magic != GGUF_MAGIC) throw std::runtime_error("Not a GGUF file (bad magic)");

    version_ = read_u32(ptr + 4);
    if (version_ > GGUF_VERSION)
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version_));

    n_tensors_ = read_u64(ptr + 8);
    n_kv_      = read_u64(ptr + 16);

    if (n_tensors_ > 1000000 || n_kv_ > 1000000)
        throw std::runtime_error("GGUF: implausible tensor/kv count");
}

void GGUFFile::parse_metadata(const uint8_t* ptr, size_t& offset) {
    for (uint64_t i = 0; i < n_kv_; i++) {
        GGUFString key = read_string(ptr, offset, file_size_);
        std::string k(key.ptr, key.len);

        CHECK_OFFSET(4);
        uint32_t vtype = read_u32(ptr + offset); offset += 4;

        switch (vtype) {
            case 0: { CHECK_OFFSET(1); (*meta_ints_)[k] = ptr[offset]; offset += 1; break; }
            case 1: { CHECK_OFFSET(1); (*meta_ints_)[k] = (int8_t)ptr[offset]; offset += 1; break; }
            case 2: { CHECK_OFFSET(2); offset += 2; break; }
            case 3: { CHECK_OFFSET(2); offset += 2; break; }
            case 4: { CHECK_OFFSET(4); (*meta_ints_)[k] = (int32_t)read_u32(ptr + offset); offset += 4; break; }
            case 5: { CHECK_OFFSET(4); (*meta_ints_)[k] = read_i32(ptr + offset); offset += 4; break; }
            case 6: { CHECK_OFFSET(4); (*meta_floats_)[k] = read_f32(ptr + offset); offset += 4; break; }
            case 7: { CHECK_OFFSET(1); (*meta_bools_)[k] = ptr[offset] != 0; offset += 1; break; }
            case 8: {
                GGUFString s = read_string(ptr, offset, file_size_);
                (*meta_strings_)[k] = std::string(s.ptr, s.len);
                break;
            }
            case 9: { // ARRAY
                CHECK_OFFSET(12);
                uint32_t etype = read_u32(ptr + offset); offset += 4;
                uint64_t elen  = read_u64(ptr + offset); offset += 8;

                switch (etype) {
                    case 8: { // String array
                        std::vector<std::string> arr;
                        arr.reserve(elen);
                        for (uint64_t j = 0; j < elen; j++) {
                            GGUFString s = read_string(ptr, offset, file_size_);
                            arr.emplace_back(s.ptr, s.len);
                        }
                        (*meta_string_arrays_)[k] = std::move(arr);
                        break;
                    }
                    case 4: case 5: { // INT32/UINT32 -> int64
                        std::vector<int64_t> arr(elen);
                        for (uint64_t j = 0; j < elen; j++) {
                            arr[j] = read_i32(ptr + offset); offset += 4;
                        }
                        (*meta_int_arrays_)[k] = std::move(arr);
                        break;
                    }
                    case 10: case 11: { // UINT64/INT64
                        std::vector<int64_t> arr(elen);
                        for (uint64_t j = 0; j < elen; j++) {
                            arr[j] = read_i64(ptr + offset); offset += 8;
                        }
                        (*meta_int_arrays_)[k] = std::move(arr);
                        break;
                    }
                    case 6: { // FLOAT32
                        std::vector<float> arr(elen);
                        for (uint64_t j = 0; j < elen; j++) {
                            arr[j] = read_f32(ptr + offset); offset += 4;
                        }
                        (*meta_float_arrays_)[k] = std::move(arr);
                        break;
                    }
                    default: {
                        size_t esz = 0;
                        switch (etype) {
                            case 0: case 1: case 7: esz = 1; break;
                            case 2: case 3: esz = 2; break;
                            case 12: esz = 8; break;
                            default: esz = 4; break;
                        }
                        CHECK_OFFSET(elen * esz);
                        offset += elen * esz;
                        break;
                    }
                }
                break;
            }
            case 10: { CHECK_OFFSET(8); (*meta_ints_)[k] = (int64_t)read_u64(ptr + offset); offset += 8; break; }
            case 11: { CHECK_OFFSET(8); (*meta_ints_)[k] = read_i64(ptr + offset); offset += 8; break; }
            case 12: { CHECK_OFFSET(8); (*meta_floats_)[k] = (float)read_f64(ptr + offset); offset += 8; break; }
            default:
                throw std::runtime_error("Unknown GGUF metadata type: " + std::to_string(vtype));
        }
    }
}

void GGUFFile::parse_tensor_infos(const uint8_t* ptr, size_t& offset) {
    tensor_infos_.reserve(n_tensors_);

    for (uint64_t i = 0; i < n_tensors_; i++) {
        TensorInfo ti;
        GGUFString name = read_string(ptr, offset, file_size_);
        ti.name = std::string(name.ptr, name.len);

        CHECK_OFFSET(4);
        ti.n_dims = read_u32(ptr + offset); offset += 4;
        if (ti.n_dims > 4) throw std::runtime_error("GGUF: tensor has >4 dimensions");
        CHECK_OFFSET(ti.n_dims * 8);
        for (int d = 0; d < ti.n_dims && d < 4; d++) {
            ti.dims[d] = (int64_t)read_u64(ptr + offset); offset += 8;
        }
        for (int d = ti.n_dims; d < 4; d++) ti.dims[d] = 1;

        CHECK_OFFSET(12);
        ti.type = (ggml_type)read_u32(ptr + offset); offset += 4;
        ti.offset = read_u64(ptr + offset); offset += 8;

        // Compute total bytes
        int64_t n_values = ti.dims[0];
        for (int d = 1; d < ti.n_dims; d++) n_values *= ti.dims[d];
        ti.size = tensor_nbytes(ti.type, n_values);

        tensor_infos_.push_back(ti);
        (*tensor_map_)[ti.name] = ti;
    }

    data_offset_ = align_up(offset, GGUF_DEFAULT_ALIGNMENT);
    if (data_offset_ > file_size_)
        throw std::runtime_error("GGUF: data section starts past end of file");
}

void GGUFFile::load_file(const char* path) {
#ifdef _WIN32
    // Windows: use _open with binary flag
    fd_ = _open(path, _O_RDONLY | _O_BINARY);
    if (fd_ < 0) throw std::runtime_error(std::string("Cannot open GGUF: ") + path);

    struct _stat64 st;
    if (_fstat64(fd_, &st) < 0) { _close(fd_); throw std::runtime_error("Cannot stat GGUF"); }
    file_size_ = st.st_size;

    data_ = (uint8_t*)std::malloc(file_size_);
    if (!data_) { _close(fd_); throw std::runtime_error("GGUF: malloc failed for " + std::to_string(file_size_) + " bytes"); }

    size_t total = 0;
    while (total < file_size_) {
        int r = _read(fd_, data_ + total, (unsigned int)(file_size_ - total));
        if (r <= 0) {
            if (r == 0) { std::free(data_); data_ = nullptr; _close(fd_); throw std::runtime_error("GGUF: unexpected EOF"); }
            std::free(data_); data_ = nullptr;
            _close(fd_);
            throw std::runtime_error(std::string("GGUF: read error"));
        }
        total += r;
    }
    _close(fd_);
    fd_ = -1;
#else
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) throw std::runtime_error(std::string("Cannot open GGUF: ") + path);

    struct stat st;
    if (fstat(fd_, &st) < 0) { close(fd_); throw std::runtime_error("Cannot stat GGUF"); }
    file_size_ = st.st_size;

    data_ = (uint8_t*)std::malloc(file_size_);
    if (!data_) { close(fd_); throw std::runtime_error("GGUF: malloc failed for " + std::to_string(file_size_) + " bytes"); }

    size_t total = 0;
    while (total < file_size_) {
        ssize_t r = ::read(fd_, data_ + total, file_size_ - total);
        if (r <= 0) {
            if (r == 0) { std::free(data_); data_ = nullptr; close(fd_); throw std::runtime_error("GGUF: unexpected EOF"); }
            if (errno == EINTR) continue;
            std::free(data_); data_ = nullptr;
            close(fd_);
            throw std::runtime_error(std::string("GGUF: read error: ") + strerror(errno));
        }
        total += r;
    }
    close(fd_);
    fd_ = -1;
#endif
}

GGUFFile::GGUFFile(const char* path) {
    load_file(path);
    init_maps();
    const uint8_t* ptr = data_;
    size_t offset = 24;
    parse_header(ptr);
    parse_metadata(ptr, offset);
    parse_tensor_infos(ptr, offset);
}

GGUFFile::~GGUFFile() {
    if (fd_ >= 0) {
#ifdef _WIN32
        _close(fd_);
#else
        close(fd_);
#endif
    }
    if (data_) std::free(data_);
}

const void* GGUFFile::tensor_data(const std::string& name) const {
    auto it = tensor_map_->find(name);
    if (it == tensor_map_->end()) return nullptr;
    size_t end = data_offset_ + it->second.offset + it->second.size;
    if (end > file_size_)
        throw std::runtime_error("GGUF: tensor '" + name + "' data extends past end of file");
    return (const void*)(data_ + data_offset_ + it->second.offset);
}

const TensorInfo* GGUFFile::tensor_info(const std::string& name) const {
    auto it = tensor_map_->find(name);
    return (it != tensor_map_->end()) ? &it->second : nullptr;
}

std::string GGUFFile::meta_string(const std::string& key) const {
    auto it = meta_strings_->find(key);
    return (it != meta_strings_->end()) ? it->second : "";
}

int64_t GGUFFile::meta_int(const std::string& key, int64_t default_val) const {
    auto it = meta_ints_->find(key);
    return (it != meta_ints_->end()) ? it->second : default_val;
}

float GGUFFile::meta_float(const std::string& key, float default_val) const {
    auto it = meta_floats_->find(key);
    return (it != meta_floats_->end()) ? it->second : default_val;
}

bool GGUFFile::meta_bool(const std::string& key, bool default_val) const {
    auto it = meta_bools_->find(key);
    return (it != meta_bools_->end()) ? it->second : default_val;
}

std::vector<std::string> GGUFFile::meta_string_array(const std::string& key) const {
    auto it = meta_string_arrays_->find(key);
    return (it != meta_string_arrays_->end()) ? it->second : std::vector<std::string>();
}

std::vector<int64_t> GGUFFile::meta_int_array(const std::string& key) const {
    auto it = meta_int_arrays_->find(key);
    return (it != meta_int_arrays_->end()) ? it->second : std::vector<int64_t>();
}

std::vector<float> GGUFFile::meta_float_array(const std::string& key) const {
    auto it = meta_float_arrays_->find(key);
    return (it != meta_float_arrays_->end()) ? it->second : std::vector<float>();
}

}  // namespace vibeblade
