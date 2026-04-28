// VibeBlade BPE Tokenizer — GPT-2 style BPE from GGUF metadata.
// Supports llama, qwen2, mistral, phi, gemma architectures.

#include "tokenizer.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <regex>
#include <sstream>

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  Load tokenizer from GGUF
// ════════════════════════════════════════════════════════════════

void Tokenizer::load(const GGUFFile& gguf) {
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

    // Read scores (token scores for BPE merge priority)
    auto scores = gguf.meta_float_array("tokenizer.ggml.scores");

    // Read BPE merges
    auto merge_strs = gguf.meta_string_array("tokenizer.ggml.merges");
    merges_.clear();
    merge_map_.clear();

    for (int i = 0; i < (int)merge_strs.size(); i++) {
        const std::string& m = merge_strs[i];
        // Format: "first second" — two token strings separated by space
        size_t space = m.find(' ');
        if (space == std::string::npos) continue;

        std::string first = m.substr(0, space);
        std::string second = m.substr(space + 1);

        auto it1 = token_to_id_.find(first);
        auto it2 = token_to_id_.find(second);
        if (it1 == token_to_id_.end() || it2 == token_to_id_.end()) continue;

        int id = (int)tokens_.size() + i;  // merges come after base vocab
        // Actually, merges produce token IDs in the vocab itself
        // The merge result token IS tokens_[id]
        // But the ID is just the position in the merged vocab
        Merge merge;
        merge.pair = {it1->second, it2->second};
        merge.result = it1->second;  // placeholder, will be set by the actual merge target
        merge.score = (i < (int)scores.size()) ? scores[i] : 0.0f;
        merge.rank = i;
        merges_.push_back(merge);
    }

    // Sort merges: by score (lower = higher priority), then by rank
    std::sort(merges_.begin(), merges_.end(), [](const Merge& a, const Merge& b) {
        if (a.score != b.score) return a.score < b.score;
        return a.rank < b.rank;
    });

    // Build merge lookup: (token_a, token_b) → index in merges_ (for fast BPE)
    for (int i = 0; i < (int)merges_.size(); i++) {
        merge_map_[merges_[i].pair] = i;
    }

    // Special tokens
    bos_id_ = (int)gguf.meta_int("tokenizer.ggml.bos_token_id", 1);
    eos_id_ = (int)gguf.meta_int("tokenizer.ggml.eos_token_id", 2);
    pad_id_ = (int)gguf.meta_int("tokenizer.ggml.padding_token_id", -1);

    // Added tokens (special tokens, regex tokens, etc.)
    added_tokens_.clear();
    auto added = gguf.meta_string_array("tokenizer.ggml.added_tokens");
    for (int i = 0; i < (int)added.size(); i++) {
        // added tokens are in the vocab already, just note their text
        auto it = token_to_id_.find(added[i]);
        if (it != token_to_id_.end()) {
            added_tokens_[added[i]] = it->second;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Pre-tokenize: GPT-2 style regex splitting
// ════════════════════════════════════════════════════════════════

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    // Check for BOS/EOS special tokens
    // Simple pre-tokenizer: split on whitespace boundaries while keeping spaces
    // attached to the following word (GPT-2 style)

    std::vector<std::string> pieces;

    // First, check for special tokens in added_tokens_
    // (simplified: handle <|...|> patterns)
    size_t pos = 0;
    while (pos < text.size()) {
        // Check for special token match at current position
        bool found_special = false;
        for (auto& [tok, id] : added_tokens_) {
            if (tok.size() > 2 && tok[0] == '<' && tok[tok.size()-1] == '>') {
                if (text.compare(pos, tok.size(), tok) == 0) {
                    pieces.push_back(tok);
                    pos += tok.size();
                    found_special = true;
                    break;
                }
            }
        }
        if (found_special) continue;

        // Find next space or end
        size_t next_space = text.find(' ', pos);

        if (next_space == std::string::npos) {
            // Rest of string
            if (pos < text.size()) {
                pieces.push_back(text.substr(pos));
            }
            break;
        }

        if (next_space == pos) {
            // Leading space — attach to next word
            pos++;
            continue;
        }

        // Word from pos to next_space
        pieces.push_back(text.substr(pos, next_space - pos + 1));  // include trailing space
        pos = next_space + 1;
    }

    return pieces;
}

// ════════════════════════════════════════════════════════════════
//  BPE: iteratively merge the most frequent pair
// ════════════════════════════════════════════════════════════════

std::vector<int> Tokenizer::bpe(const std::vector<int>& word) const {
    if (word.size() <= 1) return word;

    // Work with mutable copy of pairs
    std::vector<std::pair<int, int>> parts;
    for (int i = 0; i < (int)word.size() - 1; i++) {
        parts.push_back({word[i], word[i+1]});
    }

    // Track which parts are still active
    std::vector<bool> active(parts.size(), true);
    int active_count = (int)parts.size();

    while (active_count > 1) {
        // Find the best (lowest score) merge among active pairs
        int best_idx = -1;
        int best_rank = INT_MAX;
        float best_score = 1e30f;

        for (int i = 0; i < (int)parts.size(); i++) {
            if (!active[i]) continue;
            auto it = merge_map_.find(parts[i]);
            if (it != merge_map_.end()) {
                const Merge& m = merges_[it->second];
                if (m.rank < best_rank || (m.rank == best_rank && m.score < best_score)) {
                    best_rank = m.rank;
                    best_score = m.score;
                    best_idx = i;
                }
            }
        }

        if (best_idx == -1) break;  // No more merges possible

        // Apply the merge: mark best_idx's left neighbor as inactive,
        // update best_idx's pair to merge the result with the next token
        // Merge: (a, b) → result token
        const Merge& merge = merges_[merge_map_.at(parts[best_idx])];

        // The merged token ID is the one that corresponds to this merge result
        // In GGUF, merge result token text is tokens[base_vocab_size + merge_rank]
        // But actually, the merge result token IS in the vocab with its own ID
        // The merge pair (a, b) produces a single token. We need to find its ID.
        // The merge result is: the text of a concatenated with text of b
        std::string merged_text = tokens_[parts[best_idx].first] +
                                  tokens_[parts[best_idx].second];
        int merged_id = lookup(merged_text);

        // Deactivate left neighbor
        if (best_idx > 0 && active[best_idx - 1]) {
            active[best_idx - 1] = false;
            active_count--;
        }

        // Update current pair: first part becomes merged, second stays
        parts[best_idx].first = merged_id >= 0 ? merged_id : parts[best_idx].first;
        // (keep second part same for now — will be resolved in next iteration)
    }

    // Convert back to token IDs
    std::vector<int> result;
    for (int i = 0; i < (int)parts.size(); i++) {
        if (active[i]) {
            result.push_back(parts[i].first);
        }
    }
    // Always add the last token of the word
    if (!word.empty()) {
        result.push_back(word.back());
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
    if (text.empty()) return {bos_id_};

    std::vector<int> result = {bos_id_};

    auto pieces = pre_tokenize(text);

    for (const auto& piece : pieces) {
        // Check if it's a special/added token
        auto ait = added_tokens_.find(piece);
        if (ait != added_tokens_.end()) {
            result.push_back(ait->second);
            continue;
        }

        // Convert to byte-level tokens for BPE
        // GPT-2 uses byte-level BPE: each byte becomes a unicode character
        std::vector<int> byte_tokens;
        for (unsigned char c : piece) {
            // Map byte to GPT-2 byte-level unicode
            // Printable ASCII stays as-is, others get mapped to 256+ range
            int mapped;
            if (c >= 33 && c <= 126) mapped = c;           // printable ASCII
            else if (c >= 161 && c <= 172) mapped = c;
            else if (c >= 174 && c <= 255) mapped = c;
            else {
                // Map control chars to GPT-2 byte range (256-511)
                static const int gpt2_byte_map[256] = {
                    256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,
                    272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,
                    288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,
                    304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,
                    320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,
                    336,337,338,339,340,341,342,343,160,161,162,163,164,165,166,167,
                    168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,
                    184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,
                    200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,
                    216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,
                    232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,
                    248,249,250,251,252,253,254,255
                };
                mapped = gpt2_byte_map[c];
            }
            // Look up the byte character as a token
            std::string byte_str(1, (char)mapped);
            int tok = lookup(byte_str);
            if (tok >= 0) {
                byte_tokens.push_back(tok);
            }
        }

        if (byte_tokens.empty()) continue;

        // Apply BPE merges
        auto merged = bpe(byte_tokens);
        result.insert(result.end(), merged.begin(), merged.end());
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

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == bos_id_ || id == eos_id_ || id == pad_id_) continue;

        std::string piece = decode_token(id);
        if (piece.empty()) continue;

        // Reverse byte-level mapping
        for (unsigned char c : piece) {
            if (c >= 161 && c <= 255) {
                result += (char)c;
            } else if (c >= 256) {
                // Reverse GPT-2 byte mapping
                static const unsigned char gpt2_byte_reverse[] = {
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
                    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
                    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
                    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
                    80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
                    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,
                    112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
                    128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,
                    144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
                    160,173,155,156,157,158,159,160,161,162,163,164,165,166,167,168,
                    169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,
                    185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,
                    200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,
                    216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,
                    232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,
                    248,249,250,251,252,253,254,255
                };
                int idx = c - 256;
                if (idx >= 0 && idx < 256) {
                    result += (char)gpt2_byte_reverse[idx];
                }
            } else {
                result += c;
            }
        }
    }
    return result;
}

}  // namespace vibeblade
