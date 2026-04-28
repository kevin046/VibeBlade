// VibeBlade Tokenizer — supports GPT-2 BPE and SentencePiece from GGUF.
// Reads tokenizer.ggml.model to determine tokenizer type.
// Handles: llama, qwen2, mistral, phi, gemma, deepseek, falcon, etc.

#include "tokenizer.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <regex>
#include <sstream>

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  GPT-2 byte-level mapping tables
// ════════════════════════════════════════════════════════════════

static const unsigned char gpt2_bytes_to_unicode[] = {
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,
    65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,
    97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,
    161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,
    193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,
    225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255
};
static const int gpt2_n_bytes = 188;

// Reverse: unicode codepoint → byte value (512 entries to cover 256+ range)
static unsigned char gpt2_unicode_to_byte[512];

static void init_gpt2_tables() {
    static bool inited = false;
    if (inited) return;
    inited = true;
    memset(gpt2_unicode_to_byte, 0, sizeof(gpt2_unicode_to_byte));
    // bytes that map to themselves
    for (int i = 33; i <= 126; i++) gpt2_unicode_to_byte[i] = i;
    for (int i = 161; i <= 172; i++) gpt2_unicode_to_byte[i] = i;
    for (int i = 174; i <= 255; i++) gpt2_unicode_to_byte[i] = i;
    // non-printable bytes mapped to 256+
    int idx = 0;
    unsigned char mapped[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,128,129,130,
        131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,173};
    for (unsigned char b : mapped) {
        gpt2_unicode_to_byte[256 + idx] = b;
        idx++;
    }
}

// ════════════════════════════════════════════════════════════════
//  SentencePiece: byte-fallback + unigram (simplified)
//  SP models in GGUF store tokens as-is (already byte-encoded).
//  The pre-tokenization uses a regex pattern from GGUF metadata.
// ════════════════════════════════════════════════════════════════

// SentencePiece byte-fallback: each byte 0-255 maps to a unicode char
static std::string sp_byte_to_char(unsigned char b) {
    // SP maps bytes 0-255 to: chr(0x100 + b)
    // But in the GGUF vocab, bytes are stored as actual byte sequences
    // with the '▁' prefix for space.
    // We just return the byte as-is for lookup.
    return std::string(1, (char)b);
}

// ════════════════════════════════════════════════════════════════
//  Load tokenizer from GGUF
// ════════════════════════════════════════════════════════════════

void Tokenizer::load(const GGUFFile& gguf) {
    init_gpt2_tables();

    // Read vocab tokens
    auto vocab = gguf.meta_string_array("tokenizer.ggml.tokens");
    if (vocab.empty()) {
        throw std::runtime_error("GGUF missing tokenizer.ggml.tokens");
    }
    tokens_ = vocab;

    // Build reverse lookup
    token_to_id_.clear();
    for (int i = 0; i < (int)tokens_.size(); i++) {
        token_to_id_[tokens_[i]] = i;
    }

    // Determine tokenizer model type
    tokenizer_model_ = gguf.meta_string("tokenizer.ggml.model");
    if (tokenizer_model_.empty()) tokenizer_model_ = "gpt2";  // default

    bool is_sp = (tokenizer_model_ == "llama" || tokenizer_model_ == "sentencepiece" ||
                  tokenizer_model_ == "spm");

    // Read scores (token scores for BPE merge priority)
    auto scores = gguf.meta_float_array("tokenizer.ggml.scores");

    // Read BPE merges
    auto merge_strs = gguf.meta_string_array("tokenizer.ggml.merges");
    merges_.clear();
    merge_map_.clear();

    for (int i = 0; i < (int)merge_strs.size(); i++) {
        const std::string& m = merge_strs[i];
        size_t space = m.find(' ');
        if (space == std::string::npos) continue;

        std::string first = m.substr(0, space);
        std::string second = m.substr(space + 1);

        auto it1 = token_to_id_.find(first);
        auto it2 = token_to_id_.find(second);
        if (it1 == token_to_id_.end() || it2 == token_to_id_.end()) continue;

        Merge merge;
        merge.pair = {it1->second, it2->second};
        merge.result = it1->second;  // resolved later
        merge.score = (i < (int)scores.size()) ? scores[i] : (float)i;
        merge.rank = i;
        merges_.push_back(merge);
    }

    // Sort merges: by score (lower = higher priority), then by rank
    std::sort(merges_.begin(), merges_.end(), [](const Merge& a, const Merge& b) {
        if (a.score != b.score) return a.score < b.score;
        return a.rank < b.rank;
    });

    // Build merge lookup
    for (int i = 0; i < (int)merges_.size(); i++) {
        merge_map_[merges_[i].pair] = i;
    }

    // Special tokens
    bos_id_ = (int)gguf.meta_int("tokenizer.ggml.bos_token_id", 1);
    eos_id_ = (int)gguf.meta_int("tokenizer.ggml.eos_token_id", 2);
    pad_id_ = (int)gguf.meta_int("tokenizer.ggml.padding_token_id", -1);

    // Added tokens (special tokens like <|im_start|>, <|im_end|>, etc.)
    added_tokens_.clear();
    auto added = gguf.meta_string_array("tokenizer.ggml.added_tokens");
    for (int i = 0; i < (int)added.size(); i++) {
        auto it = token_to_id_.find(added[i]);
        if (it != token_to_id_.end()) {
            added_tokens_[added[i]] = it->second;
        }
    }

    // Pre-tokenization regex from GGUF metadata
    pre_token_regex_ = gguf.meta_string("tokenizer.ggml.precompiled_charsmap");
    if (pre_token_regex_.empty()) {
        pre_token_regex_ = gguf.meta_string("tokenizer.ggml.regex_pattern");
    }

    // BOS prefix mode: some models (qwen2, gemma) don't add BOS automatically
    add_bos_ = true;
    auto added_bos = gguf.meta_string("tokenizer.ggml.add_bos_token");
    if (added_bos == "false" || added_bos == "0") add_bos_ = false;

    // SentencePiece: detect if vocab uses ▁ (U+2581) for space encoding
    sp_byte_fallback_ = false;
    if (is_sp) {
        // Check if we have byte-fallback tokens (single byte tokens in vocab)
        int byte_token_count = 0;
        for (const auto& tok : tokens_) {
            if (tok.size() == 6 && tok.substr(0, 3) == "<0x") {
                byte_token_count++;
            }
        }
        sp_byte_fallback_ = (byte_token_count > 100);  // most have all 256
    }
}

// ════════════════════════════════════════════════════════════════
//  Pre-tokenize
// ════════════════════════════════════════════════════════════════

// Simplified GPT-2 regex: split on word boundaries while keeping whitespace
static std::vector<std::string> gpt2_pre_tokenize(const std::string& text) {
    std::vector<std::string> pieces;
    // GPT-2 pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // Simplified version that works for most cases:
    static const std::regex re(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+)",
        std::regex::ECMAScript
    );
    auto begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        pieces.push_back(it->str());
    }
    return pieces;
}

// SentencePiece pre-tokenize: split on spaces (SP replaces spaces with ▁)
// Real SP uses a complex regex; we approximate by splitting on whitespace
// and optionally applying a regex pattern from GGUF.
static std::vector<std::string> sp_pre_tokenize(const std::string& text) {
    std::vector<std::string> pieces;
    // SP splits text into chunks separated by whitespace
    // Each chunk gets a ▁ prefix (except the first if no leading space)
    std::istringstream stream(text);
    std::string word;
    bool first = true;
    while (stream >> word) {
        // Reconstruct with original spacing
        std::string piece = (first ? "" : " ") + word;
        pieces.push_back(piece);
        first = false;
    }
    if (pieces.empty() && !text.empty()) {
        pieces.push_back(text);
    }
    return pieces;
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;

    // First pass: extract special tokens
    size_t pos = 0;
    std::vector<std::pair<size_t, size_t>> special_ranges;
    while (pos < text.size()) {
        bool found = false;
        for (auto& [tok, id] : added_tokens_) {
            if (tok.size() > 2 && ((tok[0] == '<' && tok[tok.size()-1] == '>') ||
                tok == "<s>" || tok == "</s>")) {
                if (text.compare(pos, tok.size(), tok) == 0) {
                    special_ranges.push_back({pos, pos + tok.size()});
                    pos += tok.size();
                    found = true;
                    break;
                }
            }
        }
        if (!found) pos++;
    }

    // If we have special tokens, split text around them
    if (!special_ranges.empty()) {
        size_t last = 0;
        for (auto& [start, end] : special_ranges) {
            if (start > last) {
                std::string segment = text.substr(last, start - last);
                auto seg_pieces = (tokenizer_model_ == "gpt2" || tokenizer_model_ == "replit")
                    ? gpt2_pre_tokenize(segment)
                    : sp_pre_tokenize(segment);
                pieces.insert(pieces.end(), seg_pieces.begin(), seg_pieces.end());
            }
            pieces.push_back(text.substr(start, end - start));
            last = end;
        }
        if (last < text.size()) {
            std::string segment = text.substr(last);
            auto seg_pieces = (tokenizer_model_ == "gpt2" || tokenizer_model_ == "replit")
                ? gpt2_pre_tokenize(segment)
                : sp_pre_tokenize(segment);
            pieces.insert(pieces.end(), seg_pieces.begin(), seg_pieces.end());
        }
    } else {
        // No special tokens — apply appropriate pre-tokenizer
        pieces = (tokenizer_model_ == "gpt2" || tokenizer_model_ == "replit")
            ? gpt2_pre_tokenize(text)
            : sp_pre_tokenize(text);
    }

    return pieces;
}

// ════════════════════════════════════════════════════════════════
//  BPE: iteratively merge pairs by priority
// ════════════════════════════════════════════════════════════════

std::vector<int> Tokenizer::bpe(const std::vector<int>& word) const {
    if (word.size() <= 1) return word;

    // Simple BPE: repeatedly find the highest-priority merge pair
    std::vector<int> current = word;

    while (current.size() > 1) {
        // Find the best merge (lowest rank)
        int best_idx = -1;
        int best_rank = INT_MAX;

        for (int i = 0; i < (int)current.size() - 1; i++) {
            auto it = merge_map_.find({current[i], current[i + 1]});
            if (it != merge_map_.end()) {
                int rank = merges_[it->second].rank;
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                }
            }
        }

        if (best_idx == -1) break;

        // Apply merge: find the resulting token ID
        std::string merged_text = tokens_[current[best_idx]] + tokens_[current[best_idx + 1]];
        int merged_id = lookup(merged_text);

        if (merged_id < 0) {
            // Fallback: use the first token ID (merge wasn't in vocab)
            merged_id = current[best_idx];
        }

        // Replace pair with merged token
        current[best_idx] = merged_id;
        current.erase(current.begin() + best_idx + 1);
    }

    return current;
}

// ════════════════════════════════════════════════════════════════
//  SentencePiece encode: byte-level fallback
// ════════════════════════════════════════════════════════════════

std::vector<int> Tokenizer::sp_encode_word(const std::string& word) const {
    // Replace spaces with ▁ (U+2581, lower one eighth block)
    std::string normalized;
    for (size_t i = 0; i < word.size(); i++) {
        if (word[i] == ' ') {
            normalized += "\xe2\x96\x81";  // UTF-8 for ▁
        } else {
            normalized += word[i];
        }
    }

    // Try longest match first (greedy)
    std::vector<int> result;
    size_t i = 0;
    while (i < normalized.size()) {
        // Try to match longest token
        bool matched = false;
        // Try decreasing lengths from max to 1
        for (size_t len = std::min((size_t)64, normalized.size() - i); len >= 1; len--) {
            std::string sub = normalized.substr(i, len);
            int tok = lookup(sub);
            if (tok >= 0) {
                result.push_back(tok);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Byte fallback: encode each byte as <0xHH>
            unsigned char c = normalized[i];
            std::string byte_tok = "<0x" + std::string(1, "0123456789abcdef"[c >> 4])
                                          + std::string(1, "0123456789abcdef"[c & 0xf]) + ">";
            int tok = lookup(byte_tok);
            if (tok >= 0) {
                result.push_back(tok);
            }
            // Skip the UTF-8 bytes
            if ((c & 0x80) == 0) i++;
            else if ((c & 0xE0) == 0xC0) i += 2;
            else if ((c & 0xF0) == 0xE0) i += 3;
            else i += 4;
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════
//  Encode: text → token IDs
// ════════════════════════════════════════════════════════════════

int Tokenizer::lookup(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : -1;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    std::vector<int> result;

    // Add BOS if configured
    if (add_bos_ && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    auto pieces = pre_tokenize(text);

    bool is_sp = (tokenizer_model_ == "llama" || tokenizer_model_ == "sentencepiece" ||
                  tokenizer_model_ == "spm");

    for (const auto& piece : pieces) {
        // Check if it's a special/added token
        auto ait = added_tokens_.find(piece);
        if (ait != added_tokens_.end()) {
            result.push_back(ait->second);
            continue;
        }

        if (is_sp) {
            // SentencePiece encoding
            auto ids = sp_encode_word(piece);
            result.insert(result.end(), ids.begin(), ids.end());
        } else {
            // GPT-2 BPE encoding
            // Convert piece to byte-level tokens
            std::vector<int> byte_tokens;
            for (unsigned char c : piece) {
                // Map byte to GPT-2 unicode character
                std::string byte_str;
                if (c >= 33 && c <= 126) byte_str = std::string(1, (char)c);
                else if (c >= 161 && c <= 172) byte_str = std::string(1, (char)c);
                else if (c >= 174 && c <= 255) byte_str = std::string(1, (char)c);
                else {
                    // Map to 256+ range
                    static const unsigned char ctrl_bytes[] = {
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                        128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,
                        154,155,156,157,158,159,160,173
                    };
                    int idx = -1;
                    for (int j = 0; j < (int)sizeof(ctrl_bytes); j++) {
                        if (ctrl_bytes[j] == c) { idx = j; break; }
                    }
                    if (idx >= 0) {
                        byte_str = std::string(1, (char)(256 + idx));
                    }
                }
                if (!byte_str.empty()) {
                    int tok = lookup(byte_str);
                    if (tok >= 0) byte_tokens.push_back(tok);
                }
            }

            if (byte_tokens.empty()) continue;

            auto merged = bpe(byte_tokens);
            result.insert(result.end(), merged.begin(), merged.end());
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════
//  Decode: token IDs → text
// ════════════════════════════════════════════════════════════════

std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || id >= (int)tokens_.size()) return "";
    return tokens_[id];
}

// Decode raw byte from token piece
static std::string decode_bytes(const std::string& piece) {
    std::string result;
    for (size_t i = 0; i < piece.size(); i++) {
        unsigned char c = piece[i];

        // Handle UTF-8 multi-byte (pass through)
        if ((c & 0x80) != 0) {
            result += piece[i];
            continue;
        }

        if (c >= 256) {
            // Reverse GPT-2 byte mapping
            unsigned char orig = gpt2_unicode_to_byte[c & 0xFF];
            if (orig != 0) result += (char)orig;
        } else {
            result += (char)c;
        }
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    bool is_sp = (tokenizer_model_ == "llama" || tokenizer_model_ == "sentencepiece" ||
                  tokenizer_model_ == "spm");

    std::string result;
    for (int id : ids) {
        if (id == bos_id_ || id == eos_id_ || id == pad_id_) continue;

        std::string piece = decode_token(id);
        if (piece.empty()) continue;

        // Check for special/added token
        auto ait = added_tokens_.find(piece);
        if (ait != added_tokens_.end()) continue;

        if (is_sp) {
            // SentencePiece: ▁ → space
            // Replace the UTF-8 encoding of ▁ (U+2581 = 0xE2 0x96 0x81) with space
            for (size_t i = 0; i < piece.size(); i++) {
                if (i + 2 < piece.size() &&
                    (unsigned char)piece[i] == 0xE2 &&
                    (unsigned char)piece[i+1] == 0x96 &&
                    (unsigned char)piece[i+2] == 0x81) {
                    result += ' ';
                    i += 2;
                } else if (piece.size() == 6 && piece.substr(0, 3) == "<0x" && piece[5] == '>') {
                    // Byte fallback: <0xHH>
                    int hi = piece[3] >= 'a' ? piece[3] - 'a' + 10 : piece[3] - '0';
                    int lo = piece[4] >= 'a' ? piece[4] - 'a' + 10 : piece[4] - '0';
                    result += (char)((hi << 4) | lo);
                    break;
                } else {
                    result += piece[i];
                }
            }
        } else {
            // GPT-2: reverse byte-level mapping
            result += decode_bytes(piece);
        }
    }
    return result;
}

}  // namespace vibeblade
