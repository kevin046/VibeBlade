// VibeBlade Sampler — all sampling strategies in C++.
// No numpy, no Python. Called from the generate loop.
// Supports: greedy, temperature, top-k, top-p, repetition penalty, frequency/presence penalty.

#pragma once

#include <vector>
#include <random>
#include <cmath>

namespace vibeblade {

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int mirostat = 0;        // 0=off, 1=v1, 2=v2
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;
};

class Sampler {
public:
    Sampler() : rng_(std::random_device{}()) {}
    explicit Sampler(uint64_t seed) : rng_(seed) {}

    // Sample a single token from logits.
    // logits: (vocab_size,) raw logits from the model.
    // token_output: all tokens generated so far (for repetition penalty).
    // Returns sampled token ID.
    int sample(
        const float* logits,
        int vocab_size,
        SamplerConfig cfg,
        const std::vector<int>& token_output
    );

    // Set random seed for reproducibility
    void set_seed(uint64_t seed) { rng_.seed(seed); }

private:
    // Apply repetition penalty in-place
    void apply_repetition_penalty(
        float* logits, int vocab_size,
        const std::vector<int>& token_output,
        float penalty
    );

    // Apply frequency and presence penalty
    void apply_frequency_presence_penalty(
        float* logits, int vocab_size,
        const std::vector<int>& token_output,
        float freq_penalty, float pres_penalty
    );

    // Top-k filtering: keep only top-k logits, set rest to -inf
    void top_k_filter(float* logits, int vocab_size, int k);

    // Top-p (nucleus) filtering: keep smallest set with cumprob >= top_p
    void top_p_filter(float* logits, int vocab_size, float p);

    // Softmax in-place, returns max prob token
    float softmax(float* probs, int vocab_size);

    // Mirostat sampling (v1 and v2)
    int mirostat_sample(
        float* logits, int vocab_size,
        int mirostat_version, float tau, float eta
    );

    std::mt19937 rng_;

    // Scratch buffer to avoid allocation in hot path
    std::vector<float> scratch_;
};

}  // namespace vibeblade
