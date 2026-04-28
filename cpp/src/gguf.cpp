#include "gguf.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace vibeblade {

static constexpr uint32_t GGUF_MAGIC   = 0x46554755;  // "GGUF"
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

static GGUFString read_string(const uint8_t* p, size_t& offset) {
    GGUFString s;
    s.len = read_u64(p + offset); offset += 8;
    s.ptr = (const char*)(p + offset); offset += s.len;
    return s;
}

void GGUFFile::map_file(const char* path) {
    fd_ = open(path, O_RDONLY);
    if (fd_ < 0) throw std::runtime_error(std::string("Cannot open GGUF: ") + path);

    struct stat st;
    if (fstat(fd_, &st) < 0) { close(fd_); throw std::runtime_error("Cannot stat GGUF"); }
    file_size_ = st.st_size;

    data_ = (const uint8_t*)mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed for GGUF");
    }
}

void GGUFFile::parse_header(const uint8_t* ptr) {
    uint32_t magic = read_u32(ptr);
    if (magic != GGUF_MAGIC) throw std::runtime_error("Not a GGUF file (bad magic)");

    version_ = read_u32(ptr + 4);
    if (version_ > GGUF_VERSION)
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version_));

    n_tensors_ = read_u64(ptr + 8);
    n_kv_      = read_u64(ptr + 16);
}

void GGUFFile::parse_metadata(const uint8_t* ptr, size_t& offset) {
    for (uint64_t i = 0; i < n_kv_; i++) {
        GGUFString key = read_string(ptr, offset);
        std::string k(key.ptr, key.len);

        uint32_t vtype = read_u32(ptr + offset); offset += 4;

        switch (vtype) {
            case 0: { auto v = ptr[offset]; offset += 1; meta_ints_[k] = v; break; } // UINT8
            case 1: { auto v = (int8_t)ptr[offset]; offset += 1; meta_ints_[k] = v; break; } // INT8
            case 2: offset += 2; break; // UINT16 (skip)
            case 3: offset += 2; break; // INT16
            case 4: { auto v = read_u32(ptr + offset); offset += 4; meta_ints_[k] = v; break; }
            case 5: { auto v = read_i32(ptr + offset); offset += 4; meta_ints_[k] = v; break; }
            case 6: { auto v = read_f32(ptr + offset); offset += 4; meta_floats_[k] = v; break; }
            case 7: { auto v = ptr[offset] != 0; offset += 1; meta_bools_[k] = v; break; }
            case 8: {
                GGUFString s = read_string(ptr, offset);
                meta_strings_[k] = std::string(s.ptr, s.len);
                break;
            }
            case 9: { // ARRAY
                uint32_t etype = read_u32(ptr + offset); offset += 4;
                uint64_t elen  = read_u64(ptr + offset); offset += 8;
                // Compute element size and skip
                size_t esz = 0;
                switch (etype) {
                    case 0: case 1: case 7: esz = 1; break;
                    case 2: case 3: esz = 2; break;
                    case 4: case 5: case 6: esz = 4; break;
                    case 10: case 11: esz = 8; break;
                    case 12: esz = 8; break;
                    case 8: {
                        // Array of strings — skip each string
                        for (uint64_t j = 0; j < elen; j++) {
                            uint64_t slen = read_u64(ptr + offset); offset += 8;
                            offset += slen;
                        }
                        break;
                    }
                    default: esz = 4; break;
                }
                if (etype != 8) offset += elen * esz;
                break;
            }
            case 10: { auto v = read_u64(ptr + offset); offset += 8; meta_ints_[k] = (int64_t)v; break; }
            case 11: { auto v = read_i64(ptr + offset); offset += 8; meta_ints_[k] = v; break; }
            case 12: { auto v = read_f64(ptr + offset); offset += 8; meta_floats_[k] = (float)v; break; }
            default:
                throw std::runtime_error("Unknown GGUF metadata type: " + std::to_string(vtype));
        }
    }
}

void GGUFFile::parse_tensor_infos(const uint8_t* ptr, size_t& offset) {
    tensor_infos_.reserve(n_tensors_);

    for (uint64_t i = 0; i < n_tensors_; i++) {
        TensorInfo ti;
        GGUFString name = read_string(ptr, offset);
        ti.name = std::string(name.ptr, name.len);

        ti.n_dims = read_u32(ptr + offset); offset += 4;
        for (int d = 0; d < ti.n_dims && d < 4; d++) {
            ti.dims[d] = (int64_t)read_u64(ptr + offset); offset += 8;
        }
        // Zero remaining dims
        for (int d = ti.n_dims; d < 4; d++) ti.dims[d] = 1;

        ti.type = (ggml_type)read_u32(ptr + offset); offset += 4;

        // offset is relative to start of data section (set later)
        ti.offset = read_u64(ptr + offset); offset += 8;

        // Compute total bytes
        int64_t n_values = ti.dims[0];
        for (int d = 1; d < ti.n_dims; d++) n_values *= ti.dims[d];
        ti.size = tensor_nbytes(ti.type, n_values);

        tensor_infos_.push_back(ti);
        tensor_map_[ti.name] = ti;
    }

    // Data section starts here, aligned to GGUF_DEFAULT_ALIGNMENT
    data_offset_ = align_up(offset, GGUF_DEFAULT_ALIGNMENT);
}

GGUFFile::GGUFFile(const char* path) {
    map_file(path);
    size_t offset = 0;
    parse_header(data_);
    offset = 24; // header size
    parse_metadata(data_, offset);
    parse_tensor_infos(data_, offset);
}

GGUFFile::~GGUFFile() {
    if (data_ && data_ != MAP_FAILED) {
        munmap((void*)data_, file_size_);
    }
    if (fd_ >= 0) close(fd_);
}

const void* GGUFFile::tensor_data(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) return nullptr;
    return data_ + data_offset_ + it->second.offset;
}

const TensorInfo* GGUFFile::tensor_info(const std::string& name) const {
    auto it = tensor_map_.find(name);
    return (it != tensor_map_.end()) ? &it->second : nullptr;
}

std::string GGUFFile::meta_string(const std::string& key) const {
    auto it = meta_strings_.find(key);
    return (it != meta_strings_.end()) ? it->second : "";
}

int64_t GGUFFile::meta_int(const std::string& key, int64_t default_val) const {
    auto it = meta_ints_.find(key);
    return (it != meta_ints_.end()) ? it->second : default_val;
}

float GGUFFile::meta_float(const std::string& key, float default_val) const {
    auto it = meta_floats_.find(key);
    return (it != meta_floats_.end()) ? it->second : default_val;
}

bool GGUFFile::meta_bool(const std::string& key, bool default_val) const {
    auto it = meta_bools_.find(key);
    return (it != meta_bools_.end()) ? it->second : default_val;
}

}  // namespace vibeblade
