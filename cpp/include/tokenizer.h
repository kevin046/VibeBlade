#pragma once
// VibeBlade BPE Tokenizer — reads tokenizer metadata from GGUF.
// Supports GPT-2 style BPE (llama, qwen2, mistral, etc.).
// Zero-copy: vocab tokens point into GGUF mmap'd memory.

#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace vibeblade {

// Hash for std::pair<int,int> keys (must be before Tokenizer class)
struct pair_hash {
    size_t operator()(const std::pair<int, int>& p) const {
        return (size_t(p.first) << 32) | size_t(p.second);
    }
};

class Tokenizer {
public:
    Tokenizer() = default;

    // Load tokenizer from GGUF metadata (tokens, merges, scores)
    void load(const GGUFFile& gguf);

    // Encode text → token IDs (BPE)
    std::vector<int> encode(const std::string& text) const;

    // Decode token IDs → text
    std::string decode(const std::vector<int>& ids) const;

    // Decode single token ID → text piece
    std::string decode_token(int id) const;

    // Special token IDs
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }

    // Vocab size
    int vocab_size() const { return (int)tokens_.size(); }

private:
    // BPE merge: pair of token IDs → merged token ID
    struct Merge {
        std::pair<int, int> pair;
        int result;
        float score;  // priority (lower = merge first in older models)
        int rank;     // insertion order (used as fallback)
    };

    // Pre-tokenize text into word pieces (simplified GPT-2 pre-tokenizer)
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Apply BPE merges to a word (list of token IDs) → final token IDs
    std::vector<int> bpe(const std::vector<int>& word) const;

    // Lookup token text in vocab
    int lookup(const std::string& token) const;

    // Vocab: index → text, text → index
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> token_to_id_;

    // BPE merges sorted by priority
    std::vector<Merge> merges_;

    // Hash for pair<int,int> keys
    std::unordered_map<std::pair<int, int>, int, pair_hash> merge_map_;

    // Special token IDs
    int bos_id_ = 1;
    int eos_id_ = 2;
    int pad_id_ = -1;

    // Added tokens
    std::unordered_map<std::string, int> added_tokens_;
};

}  // namespace vibeblade
