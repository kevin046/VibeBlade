#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include "kernels.h"
#include "fast_model.h"
#include "dequant.h"

namespace py = pybind11;
using namespace py::literals;
using namespace vibeblade;

// ── Helpers ──

static const uint16_t* f16_ptr(const py::array& arr) {
    if (arr.itemsize() != 2 || arr.dtype().kind() != 'f')
        throw std::runtime_error("Expected float16 array");
    return reinterpret_cast<const uint16_t*>(arr.data());
}

static uint16_t* f16_ptr_mut(py::array& arr) {
    if (arr.itemsize() != 2 || arr.dtype().kind() != 'f')
        throw std::runtime_error("Expected float16 array");
    return reinterpret_cast<uint16_t*>(arr.mutable_data());
}

static py::array make_f16(py::ssize_t d0, py::ssize_t d1 = 0) {
    if (d0 <= 0 || d1 < 0)
        throw std::invalid_argument("array dimensions must be positive");
    if (d1 > 0) {
        std::vector<py::ssize_t> shape = {d0, d1};
        return py::array(py::dtype("float16"), shape);
    }
    std::vector<py::ssize_t> shape = {d0};
    return py::array(py::dtype("float16"), shape);
}

// FIX #7: Validate axis parameter (0 or 1 only)
static void validate_axis(int axis) {
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 (per-token) or 1 (per-channel)");
}

// ════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_vibeblade_native, m) {
    m.doc() = "VibeBlade native C++ kernels — SIMD-optimized inference primitives";

    std::string simd = "scalar";
#ifdef TS_AVX512FP16
    simd = "AVX-512-FP16";
#elif defined(TS_AVX512F)
    simd = "AVX-512-F (fp32 path)";
#elif defined(TS_AVX2)
    simd = "AVX2+FMA";
#elif defined(TS_NEON_FP16)
    simd = "NEON-FP16";
#endif
    m.attr("SIMD_BACKEND") = simd;

    // ════════════════════ GEMM ════════════════════
    m.def("gemm", [](py::array a, py::array b, float alpha, float beta) -> py::array {
        auto ab = a.request(), bb = b.request();
        if (ab.ndim != 2 || bb.ndim != 2) throw std::runtime_error("gemm: 2-D required");
        int M = ab.shape[0], K = ab.shape[1];
        if (bb.shape[0] != K) throw std::runtime_error("gemm: shape mismatch");
        int N = bb.shape[1];
        py::array c = make_f16(M, N);
        gemm_f16(f16_ptr(a), f16_ptr(b), f16_ptr_mut(c), M, K, N, alpha, beta);
        return c;
    }, py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
       R"doc(C = alpha * A @ B + beta * C.  A:(M,K) B:(K,N) → C:(M,N) float16)doc");

    // ════════════════════ RMSNorm ════════════════════
    m.def("rms_norm", [](py::array x, py::array weight, float eps) -> py::array {
        auto xb = x.request(), wb = weight.request();
        if (xb.ndim != 2) throw std::runtime_error("rms_norm: x must be 2-D");
        int rows = xb.shape[0], D = xb.shape[1];
        py::array out = make_f16(rows, D);
        rms_norm(f16_ptr(x), f16_ptr(weight), f16_ptr_mut(out), rows, D, eps);
        return out;
    }, py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5f,
       R"doc(RMSNorm: out = x * weight / sqrt(mean(x^2) + eps))doc");

    // ════════════════════ Activations ════════════════════
    m.def("silu", [](py::array x) -> py::array {
        auto xb = x.request();
        py::array out = make_f16(xb.size);
        silu_f16(f16_ptr(x), f16_ptr_mut(out), xb.size);
        return out;
    }, py::arg("x"));

    m.def("silu_mul", [](py::array a, py::array b) -> py::array {
        auto ab = a.request();
        if (b.request().size != ab.size) throw std::runtime_error("shape mismatch");
        py::array out = make_f16(ab.size);
        silu_mul_f16(f16_ptr(a), f16_ptr(b), f16_ptr_mut(out), ab.size);
        return out;
    }, py::arg("a"), py::arg("b"),
       R"doc(Fused SwiGLU gate: out = silu(a) * b)doc");

    // ════════════════════ Quantization ════════════════════
    m.def("quantize_2bit", [](py::array x, int axis)
          -> std::tuple<py::array_t<uint8_t>, py::array_t<float>, py::array_t<float>> {
        validate_axis(axis);  // FIX #7
        auto xb = x.request();
        if (xb.ndim != 2) throw std::runtime_error("2-D required");
        int S = xb.shape[0], D = xb.shape[1];
        // FIX #3: Use size_t to prevent integer overflow in packed size calc
        py::ssize_t total = (py::ssize_t)S * (py::ssize_t)D;
        py::ssize_t out_bytes = (total + 3) / 4;
        py::array_t<uint8_t> packed(out_bytes);
        int scale_len = (axis == 1) ? D : S;
        py::array_t<float> scales(scale_len), mins(scale_len);
        quantize_2bit(f16_ptr(x), packed.mutable_data(), S, D, axis,
                      scales.mutable_data(), mins.mutable_data());
        return std::make_tuple(packed, scales, mins);
    }, py::arg("x"), py::arg("axis"));

    m.def("dequantize_2bit", [](py::array_t<uint8_t> packed,
                                 py::array_t<float> scales,
                                 py::array_t<float> mins,
                                 int S, int D, int axis) -> py::array {
        validate_axis(axis);  // FIX #7
        py::array out = make_f16(S, D);
        dequantize_2bit(packed.data(), f16_ptr_mut(out), S, D, axis,
                        scales.data(), mins.data());
        return out;
    }, py::arg("packed"), py::arg("scales"), py::arg("mins"),
       py::arg("S"), py::arg("D"), py::arg("axis"));

    m.def("quantize_4bit", [](py::array x, int axis)
          -> std::tuple<py::array_t<uint8_t>, py::array_t<float>, py::array_t<float>> {
        validate_axis(axis);  // FIX #7
        auto xb = x.request();
        if (xb.ndim != 2) throw std::runtime_error("2-D required");
        int S = xb.shape[0], D = xb.shape[1];
        // FIX #3: Use size_t to prevent integer overflow
        py::ssize_t total = (py::ssize_t)S * (py::ssize_t)D;
        py::array_t<uint8_t> packed((total + 1) / 2);
        int sl = (axis == 1) ? D : S;
        py::array_t<float> scales(sl), mins(sl);
        quantize_4bit(f16_ptr(x), packed.mutable_data(), S, D, axis,
                      scales.mutable_data(), mins.mutable_data());
        return std::make_tuple(packed, scales, mins);
    }, py::arg("x"), py::arg("axis"));

    m.def("dequantize_4bit", [](py::array_t<uint8_t> packed,
                                 py::array_t<float> scales,
                                 py::array_t<float> mins,
                                 int S, int D, int axis) -> py::array {
        validate_axis(axis);  // FIX #7
        py::array out = make_f16(S, D);
        dequantize_4bit(packed.data(), f16_ptr_mut(out), S, D, axis,
                        scales.data(), mins.data());
        return out;
    }, py::arg("packed"), py::arg("scales"), py::arg("mins"),
       py::arg("S"), py::arg("D"), py::arg("axis"));

    m.def("quantize_8bit_sym", [](py::array x, float max_abs)
          -> std::tuple<py::array_t<int8_t>, float> {
        auto xb = x.request();
        py::array_t<int8_t> out(xb.size);
        float scale = quantize_8bit_sym(f16_ptr(x), out.mutable_data(), xb.size, max_abs);
        return std::make_tuple(out, scale);
    }, py::arg("x"), py::arg("max_abs"));

    m.def("dequantize_8bit_sym", [](py::array_t<int8_t> x, float scale) -> py::array {
        auto xb = x.request();
        py::array out = make_f16(xb.size);
        dequantize_8bit_sym(x.data(), f16_ptr_mut(out), xb.size, scale);
        return out;
    }, py::arg("x"), py::arg("scale"));

    // ════════════════════ Attention ════════════════════
    m.def("fused_sdpa", [](py::array Q, py::array K, py::array V,
                            float scale) -> py::array {
        auto qb = Q.request(), kb = K.request(), vb = V.request();
        if (qb.ndim != 2 || kb.ndim != 2 || vb.ndim != 2)
            throw std::runtime_error("2-D required");
        int M = qb.shape[0], d = qb.shape[1];
        int N = kb.shape[0];
        if (scale < 0) scale = 1.0f / sqrtf((float)d);
        py::array O = make_f16(M, d);
        fused_sdpa(f16_ptr(Q), f16_ptr(K), f16_ptr(V), f16_ptr_mut(O), M, N, d, scale);
        return O;
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale") = -1.0f,
       R"doc(O = softmax(Q @ K^T / scale) @ V  (online softmax, flash-style))doc");

    // ════════════════════ RoPE ════════════════════
    m.def("apply_rope", [](py::array x, py::array freqs) -> py::array {
        auto xb = x.request();
        if (xb.ndim != 3) throw std::runtime_error("x must be 3-D (seq, heads, dim)");
        int seq = xb.shape[0], heads = xb.shape[1], dim = xb.shape[2];
        py::array out = py::array(py::dtype("float16"), {seq, heads, dim});
        std::memcpy(out.mutable_data(), x.data(), (size_t)xb.size * 2);
        apply_rope(f16_ptr_mut(out),
                   reinterpret_cast<const float*>(freqs.data()),
                   seq, heads, dim);
        return out;
    }, py::arg("x"), py::arg("freqs"),
       R"doc(x:(seq,heads,dim) float16, freqs:(seq, dim/2) float32 [cos,sin,...])doc");

    // ════════════════════ Softmax ════════════════════
    m.def("softmax", [](py::array x) -> py::array {
        auto xb = x.request();
        if (xb.ndim != 2) throw std::runtime_error("2-D required");
        int rows = xb.shape[0], cols = xb.shape[1];
        py::array out = py::array(py::dtype("float32"), {rows, cols});
        std::memcpy(out.mutable_data(), x.data(), (size_t)xb.size * sizeof(float));
        softmax_f32(reinterpret_cast<float*>(out.mutable_data()), rows, cols);
        return out;
    }, py::arg("x"));

    // ════════════════════ VibeBladeFast — GGUF inference engine ════════════════════
    py::class_<GenerateResult>(m, "GenerateResult",
        R"doc(Result from generate(): text, token_ids, tokens_per_second, stopped_eos.)doc")
        .def_readonly("text", &GenerateResult::text)
        .def_readonly("token_ids", &GenerateResult::token_ids)
        .def_readonly("tokens_per_second", &GenerateResult::tokens_per_second)
        .def_readonly("stopped_eos", &GenerateResult::stopped_eos);

    py::class_<VibeBladeFast>(m, "VibeBladeFast",
        R"doc(Llama.cpp-style C++ inference engine with mmap'd GGUF weights and inline dequant.
Full generate loop in C++: tokenize → prefill → decode → sample → detokenize.
Zero malloc in the decode loop. Supports Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32.)doc")
        .def(py::init<>())
        .def("load", &VibeBladeFast::load, py::arg("path"),
            R"doc(Load a GGUF model file (mmaps weights, parses config, loads tokenizer).)doc")

        // ── Full generate pipeline (one C++ call, zero Python in hot path) ──
        .def("generate", [](VibeBladeFast& self,
                            const std::string& prompt,
                            int max_tokens,
                            float temperature,
                            int top_k,
                            float top_p,
                            float repetition_penalty,
                            int seed,
                            py::object on_token_py
                        ) -> GenerateResult {
            std::function<void(int, const std::string&)> on_token;
            if (!on_token_py.is_none()) {
                // Capture the Python callback with GIL management
                py::function cb = on_token_py.cast<py::function>();
                on_token = [cb](int token_id, const std::string& piece) {
                    // Release GIL not needed here — pybind11 manages it
                    cb(token_id, piece);
                };
            }
            return self.generate(prompt, max_tokens, temperature, top_k, top_p,
                                 repetition_penalty, seed, on_token);
        },
            py::arg("prompt"),
            py::arg("max_tokens") = 128,
            py::arg("temperature") = 1.0f,
            py::arg("top_k") = 50,
            py::arg("top_p") = 0.9f,
            py::arg("repetition_penalty") = 1.0f,
            py::arg("seed") = -1,
            py::arg("on_token") = py::none(),
            R"doc(Full generate pipeline in C++: tokenize prompt → prefill → decode loop → sample → detokenize.
on_token: optional Python callback(token_id: int, piece: str) for streaming.)doc")

        // ── Tokenizer ──
        .def("tokenize", [](const VibeBladeFast& self, const std::string& text) -> std::vector<int> {
            return self.tokenize(text);
        }, py::arg("text"),
            R"doc(Encode text to token IDs using the GGUF BPE tokenizer.)doc")
        .def("detokenize", [](const VibeBladeFast& self, const std::vector<int>& ids) -> std::string {
            return self.detokenize(ids);
        }, py::arg("token_ids"),
            R"doc(Decode token IDs to text.)doc")
        .def("detokenize_token", [](const VibeBladeFast& self, int id) -> std::string {
            return self.detokenize_token(id);
        }, py::arg("token_id"),
            R"doc(Decode a single token ID to its text piece.)doc")

        // ── Individual steps (for advanced use) ──
        .def("prefill", [](VibeBladeFast& self, const std::vector<int>& tokens) -> py::array_t<float> {
            auto logits = self.prefill(tokens);
            // Copy into numpy — logits vector is temporary, must not hand out pointer
            py::array_t<float> out(logits.size());
            std::memcpy(out.mutable_data(), logits.data(), logits.size() * sizeof(float));
            return out;
        }, py::arg("token_ids"),
            R"doc(Prefill: process all prompt tokens, return logits for last position.)doc")
        .def("decode", [](VibeBladeFast& self, int token_id) -> py::array_t<float> {
            auto logits = self.decode(token_id);
            // Copy into numpy — logits vector is temporary, must not hand out pointer
            py::array_t<float> out(logits.size());
            std::memcpy(out.mutable_data(), logits.data(), logits.size() * sizeof(float));
            return out;
        }, py::arg("token_id"),
            R"doc(Decode: process one token, return full vocab logits.)doc")
        .def("decode_with_hidden", [](VibeBladeFast& self,
                                       int token_id,
                                       const std::vector<int>& layer_indices)
                                       -> std::tuple<py::array_t<float>, std::vector<py::array_t<float>>> {
            auto result = self.decode_with_hidden(token_id, layer_indices);
            py::array_t<float> logits_out(result[0].size());
            std::memcpy(logits_out.mutable_data(), result[0].data(), result[0].size() * sizeof(float));
            std::vector<py::array_t<float>> hidden_outs;
            hidden_outs.reserve(result.size() - 1);
            for (size_t i = 1; i < result.size(); i++) {
                py::array_t<float> h_out(result[i].size());
                std::memcpy(h_out.mutable_data(), result[i].data(), result[i].size() * sizeof(float));
                hidden_outs.push_back(std::move(h_out));
            }
            return std::make_tuple(logits_out, hidden_outs);
        }, py::arg("token_id"), py::arg("layer_indices"),
            "Decode + extract hidden states at specified layer indices.\n"
            "Returns (logits, [hidden_0, ...]). Each hidden vector has shape (hidden_dim,).")
.def("embedding", [](VibeBladeFast& self, int token_id) -> py::array_t<float> {
    std::vector<float> emb = self.embedding(token_id);
    py::array_t<float> result(emb.size());
    std::memcpy(result.mutable_data(), emb.data(), emb.size() * sizeof(float));
    return result;
}, py::arg("token_id"),
"Return the token embedding vector (float32) for a token ID.")

        .def("lm_head", [](VibeBladeFast& self, py::array_t<float> hidden) -> py::array_t<float> {
            py::buffer_info info = hidden.request();
            if (info.ndim != 1) {
                throw std::runtime_error("lm_head expects 1D hidden vector");
            }
            size_t dim = (size_t)info.shape[0];
            if (dim != self.config().hidden_dim) {
                throw std::runtime_error("lm_head: hidden dim mismatch, expected " +
                                         std::to_string(self.config().hidden_dim) +
                                         ", got " + std::to_string(dim));
            }
 const float* ptr = static_cast<const float*>(info.ptr);
 std::vector<float> hvec(ptr, ptr + dim);
 std::vector<float> logits = self.lm_head(hvec);
 py::array_t<float> result(logits.size());
 std::memcpy(result.mutable_data(), logits.data(), logits.size() * sizeof(float));
 return result;
        }, py::arg("hidden"),
            "Compute logits from a hidden-state vector (final norm + output layer).")



        .def("prefill_with_hidden", [](VibeBladeFast& self,
                                        const std::vector<int>& token_ids,
                                        const std::vector<int>& layer_indices)
                                        -> std::tuple<py::array_t<float>, std::vector<py::array_t<float>>> {
            auto result = self.prefill_with_hidden(token_ids, layer_indices);
            py::array_t<float> logits_out(result[0].size());
            std::memcpy(logits_out.mutable_data(), result[0].data(), result[0].size() * sizeof(float));
            std::vector<py::array_t<float>> hidden_outs;
            hidden_outs.reserve(result.size() - 1);
            for (size_t i = 1; i < result.size(); i++) {
                py::array_t<float> h_out(result[i].size());
                std::memcpy(h_out.mutable_data(), result[i].data(), result[i].size() * sizeof(float));
                hidden_outs.push_back(std::move(h_out));
            }
            return std::make_tuple(logits_out, hidden_outs);
        }, py::arg("token_ids"), py::arg("layer_indices"),
            "Prefill + extract hidden states at specified layer indices.\n"
            "Returns (logits, [hidden_layer_0, ...]). Each hidden is row-major (seq_len*hidden_dim).")

.def("reset", &VibeBladeFast::reset,
 R"doc(Reset KV cache and position to start a new conversation.)doc")

 // ── Speculative Decoding ──
 .def("speculative_decode", &VibeBladeFast::speculative_decode,
 "prompt"_a, "max_tokens"_a = 128, "temperature"_a = 1.0f,
 "top_k"_a = 50, "top_p"_a = 0.9f, "repetition_penalty"_a = 1.0f,
 "seed"_a = -1, "n_spec_tokens"_a = 4,
 R"doc(Speculative decoding: draft N tokens, verify all at once. Fast on large models.)doc")

 // ── Grammar Constraints ──
 .def("set_grammar", &VibeBladeFast::set_grammar, "gbnf"_a,
 R"doc(Set GBNF grammar for structured output (JSON, code, etc.).)doc")
 .def("clear_grammar", &VibeBladeFast::clear_grammar,
 R"doc(Clear grammar constraint.)doc")
 .def_property_readonly("has_grammar", &VibeBladeFast::has_grammar)

 // ── Properties ──
        .def_property_readonly("position", [](const VibeBladeFast& self) { return self.position(); })
        .def_property_readonly("eos_id", [](const VibeBladeFast& self) { return self.eos_id(); })
        .def_property_readonly("bos_id", [](const VibeBladeFast& self) { return self.bos_id(); })
        .def_property_readonly("config", [](const VibeBladeFast& self) -> py::dict {
            const auto& c = self.config();
            return py::dict(
                "n_layers"_a = c.n_layers,
                "n_heads"_a = c.n_heads,
                "n_kv_heads"_a = c.n_kv_heads,
                "head_dim"_a = c.head_dim,
                "hidden_dim"_a = c.hidden_dim,
                "intermediate_dim"_a = c.intermediate_dim,
                "vocab_size"_a = c.vocab_size,
                "context_length"_a = c.context_length,
                "norm_eps"_a = c.norm_eps,
                "arch"_a = c.arch,
                "n_experts"_a = c.n_experts,
                "n_experts_used"_a = c.n_experts_used
            );
        })
.def_property_readonly("kv_cache_bytes", [](const VibeBladeFast& self) {
    return self.kv_cache_bytes();
})

// ── Debug: dequantize a row of a tensor ──
.def("dequant_row", [](VibeBladeFast& self,
                        const std::string& tensor_name,
                        int row_idx,
                        int n_elements) -> py::array_t<float> {
    const auto& g = self.gguf();
    auto* info = g.tensor_info(tensor_name);
    if (!info) throw std::runtime_error("Tensor not found: " + tensor_name);
    const void* data = g.tensor_data(tensor_name);
    int64_t ne0 = info->dims[0];
    int64_t row_bytes_size = 0;
    switch (info->type) {
        case GGML_TYPE_F32: row_bytes_size = ne0 * 4; break;
        case GGML_TYPE_F16: row_bytes_size = ne0 * 2; break;
        case GGML_TYPE_Q4_0: row_bytes_size = (ne0 / 32) * 18; break;
        case GGML_TYPE_Q5_0: row_bytes_size = (ne0 / 32) * 22; break;
        case GGML_TYPE_Q8_0: row_bytes_size = (ne0 / 32) * 34; break;
        case GGML_TYPE_Q4_K: row_bytes_size = (ne0 / 256) * 144; break;
        case GGML_TYPE_Q5_K: row_bytes_size = (ne0 / 256) * 176; break;
        case GGML_TYPE_Q6_K: row_bytes_size = (ne0 / 256) * 210; break;
        default: throw std::runtime_error("Unsupported type for dequant_row");
    }
    int actual_elements = n_elements > 0 ? n_elements : (int)ne0;
    const uint8_t* row_ptr = (const uint8_t*)data + row_idx * row_bytes_size;
    std::vector<float> out(actual_elements);
    vibeblade::dequantize_row(row_ptr, out.data(), actual_elements, info->type);
    py::array_t<float> result(actual_elements);
    std::memcpy(result.mutable_data(), out.data(), actual_elements * sizeof(float));
    return result;
}, py::arg("tensor_name"), py::arg("row_idx") = 0, py::arg("n_elements") = 0,
"Dequantize and return a single row of a GGUF tensor for debugging.")

.def("debug_emb_type", [](const VibeBladeFast& self) -> py::dict {
    // Expose embedding internals for debugging
    // We need access to private members — use a trick
    // Actually, let's just re-derive from the GGUF file
    const auto& g = self.gguf();
    auto* info = g.tensor_info("token_embd.weight");
    if (!info) info = g.tensor_info("wte.weight");
    int type_id = info ? (int)info->type : -1;
    int64_t ne0 = info ? info->dims[0] : -1;
    int64_t ne1 = info ? info->dims[1] : -1;
    size_t offset = info ? info->offset : 0;
    size_t tsize = info ? info->size : 0;
    size_t rb = 0;
    if (info) {
        switch (info->type) {
            case GGML_TYPE_Q5_0: rb = (ne0 / 32) * 22; break;
            case GGML_TYPE_Q4_0: rb = (ne0 / 32) * 18; break;
            case GGML_TYPE_Q8_0: rb = (ne0 / 32) * 34; break;
            case GGML_TYPE_F32: rb = ne0 * 4; break;
            default: rb = 0;
        }
    }
    return py::dict(
        "type"_a = type_id,
        "ne0"_a = (int)ne0,
        "ne1"_a = (int)ne1,
        "offset"_a = (int)offset,
        "size"_a = (int)tsize,
        "row_bytes"_a = (int)rb
    );
})

// ── Debug: compute embedding via both paths and compare ──
.def("debug_embedding_compare", [](VibeBladeFast& self, int tok) -> py::dict {
    const auto& g = self.gguf();
    auto* info = g.tensor_info("token_embd.weight");
    if (!info) info = g.tensor_info("wte.weight");
    if (!info) throw std::runtime_error("No embedding tensor found");
    
    const void* data_via_gguf = g.tensor_data("token_embd.weight");
    if (!data_via_gguf) data_via_gguf = g.tensor_data("wte.weight");
    
    int hd = info->dims[0]; // 896 for Qwen
    int64_t ne0 = info->dims[0];
    size_t row_bytes = 0;
    switch (info->type) {
        case GGML_TYPE_Q5_0: row_bytes = (ne0 / 32) * 22; break;
        case GGML_TYPE_Q4_0: row_bytes = (ne0 / 32) * 18; break;
        case GGML_TYPE_Q8_0: row_bytes = (ne0 / 32) * 34; break;
        case GGML_TYPE_F32: row_bytes = ne0 * 4; break;
        default: row_bytes = 0;
    }
    
    // Path A: dequant via tensor_data (like dequant_row — WORKS)
    const uint8_t* ptr_a = (const uint8_t*)data_via_gguf + tok * row_bytes;
    std::vector<float> emb_a(hd);
    vibeblade::dequantize_row(ptr_a, emb_a.data(), hd, info->type);
    
    // Path B: use the embedding() method (BROKEN)
    std::vector<float> emb_b = self.embedding(tok);
    
    // Compare
    float max_diff = 0;
    for (int i = 0; i < hd; i++) {
        float d = fabsf(emb_a[i] - emb_b[i]);
        if (d > max_diff) max_diff = d;
    }
    
    // Also expose the raw pointer values for comparison
    auto ptr_val_a = (uintptr_t)ptr_a;
    auto ptr_val_b = (uintptr_t)data_via_gguf;
    
    py::array_t<float> arr_a(hd);
    py::array_t<float> arr_b(hd);
    std::memcpy(arr_a.mutable_data(), emb_a.data(), hd * sizeof(float));
    std::memcpy(arr_b.mutable_data(), emb_b.data(), hd * sizeof(float));
    
    return py::dict(
        "max_diff"_a = max_diff,
        "emb_a_first5"_a = py::cast(std::vector<float>(emb_a.begin(), emb_a.begin() + 5)),
        "emb_b_first5"_a = py::cast(std::vector<float>(emb_b.begin(), emb_b.begin() + 5)),
        "ptr_gguf_data"_a = (int)ptr_val_b,
        "ptr_row_a"_a = (int)ptr_val_a,
        "row_bytes"_a = (int)row_bytes,
        "hd"_a = hd,
        "type"_a = (int)info->type
    );
}, py::arg("tok"))
;
} // PYBIND11_MODULE
