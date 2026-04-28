// VibeBlade Sampler — temperature, top-k, top-p, repetition penalty.
// No Python, no numpy. Entire sampling pipeline in C++.

#include "sampler.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <cmath>

namespace vibeblade {

// ════════════════════════════════════════════════════════════════
//  Main sample entry point
// ════════════════════════════════════════════════════════════════

int Sampler::sample(
    const float* logits,
    int vocab_size,
    SamplerConfig cfg,
    const std::vector<int>& token_output
) {
    if (vocab_size <= 0) return 0;

    // Copy logits to scratch buffer (we modify in-place)
    scratch_.resize(vocab_size);
    std::memcpy(scratch_.data(), logits, vocab_size * sizeof(float));

    float* l = scratch_.data();

    // 1. Repetition penalty
    if (cfg.repetition_penalty != 1.0f && !token_output.empty()) {
        apply_repetition_penalty(l, vocab_size, token_output, cfg.repetition_penalty);
    }

    // 2. Frequency & presence penalty
    if ((cfg.frequency_penalty != 0.0f || cfg.presence_penalty != 0.0f) && !token_output.empty()) {
        apply_frequency_presence_penalty(l, vocab_size, token_output,
                                         cfg.frequency_penalty, cfg.presence_penalty);
    }

    // 3. Temperature
    if (cfg.temperature > 0.0f && cfg.temperature != 1.0f) {
        float inv_temp = 1.0f / cfg.temperature;
        for (int i = 0; i < vocab_size; i++) {
            l[i] *= inv_temp;
        }
    }

    // 4. Mirostat (bypasses top-k/top-p)
    if (cfg.mirostat > 0) {
        return mirostat_sample(l, vocab_size, cfg.mirostat, cfg.mirostat_tau, cfg.mirostat_eta);
    }

    // 5. Greedy (temperature == 0)
    if (cfg.temperature == 0.0f) {
        return int(std::max_element(l, l + vocab_size) - l);
    }

    // 6. Top-K filtering
    if (cfg.top_k > 0 && cfg.top_k < vocab_size) {
        top_k_filter(l, vocab_size, cfg.top_k);
    }

    // 7. Softmax (converts to probabilities)
    softmax(l, vocab_size);

    // 8. Top-P (nucleus) filtering
    if (cfg.top_p < 1.0f) {
        top_p_filter(l, vocab_size, cfg.top_p);
    }

    // 9. Renormalize after top-p
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += l[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < vocab_size; i++) l[i] *= inv;
    }

    // 10. Sample from distribution
    std::discrete_distribution<int> dist(l, l + vocab_size);
    return dist(rng_);
}

// ════════════════════════════════════════════════════════════════
//  Repetition penalty
// ════════════════════════════════════════════════════════════════

void Sampler::apply_repetition_penalty(
    float* logits, int vocab_size,
    const std::vector<int>& token_output,
    float penalty
) {
    // Count token frequencies
    std::vector<int> counts(vocab_size, 0);
    for (int tok : token_output) {
        if (tok >= 0 && tok < vocab_size) counts[tok]++;
    }

    // Apply penalty: if logit > 0, divide; if < 0, multiply
    for (int i = 0; i < vocab_size; i++) {
        if (counts[i] == 0) continue;
        if (logits[i] > 0.0f) {
            logits[i] /= penalty;
        } else {
            logits[i] *= penalty;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Frequency & presence penalty
// ════════════════════════════════════════════════════════════════

void Sampler::apply_frequency_presence_penalty(
    float* logits, int vocab_size,
    const std::vector<int>& token_output,
    float freq_penalty, float pres_penalty
) {
    std::vector<int> counts(vocab_size, 0);
    for (int tok : token_output) {
        if (tok >= 0 && tok < vocab_size) counts[tok]++;
    }

    for (int i = 0; i < vocab_size; i++) {
        if (counts[i] == 0) continue;
        logits[i] -= counts[i] * freq_penalty;
        logits[i] -= (counts[i] > 0 ? 1.0f : 0.0f) * pres_penalty;
    }
}

// ════════════════════════════════════════════════════════════════
//  Top-K filter
// ════════════════════════════════════════════════════════════════

void Sampler::top_k_filter(float* logits, int vocab_size, int k) {
    // Partial sort to find the k-th largest
    // We use a simple approach: find threshold, zero out below it
    std::vector<float> copy(logits, logits + vocab_size);
    std::nth_element(copy.begin(), copy.begin() + k - 1, copy.end(), std::greater<float>());
    float threshold = copy[k - 1];

    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) {
            logits[i] = -1e30f;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  Top-P (nucleus) filter
// ════════════════════════════════════════════════════════════════

void Sampler::top_p_filter(float* logits, int vocab_size, float p) {
    // Sort indices by probability (descending)
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });

    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[indices[i]];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out tokens beyond the nucleus
    for (int i = cutoff; i < vocab_size; i++) {
        logits[indices[i]] = 0.0f;
    }
}

// ════════════════════════════════════════════════════════════════
//  Softmax
// ════════════════════════════════════════════════════════════════

float Sampler::softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;

    return 1.0f / n;  // uniform baseline for mirostat
}

// ════════════════════════════════════════════════════════════════
//  Mirostat sampling
// ════════════════════════════════════════════════════════════════

int Sampler::mirostat_sample(
    float* logits, int vocab_size,
    int mirostat_version, float tau, float eta
) {
    // Apply softmax
    softmax(logits, vocab_size);

    // Estimate surprise (negative log prob)
    // For simplicity, use the Mirostat v2 approach:
    // Dynamic truncation based on target surprise

    if (mirostat_version == 2) {
        // Mirostat v2: sample from top tokens until surprise reaches tau
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return logits[a] > logits[b];
        });

        float surprise_sum = 0.0f;
        int cutoff = 1;
        for (int i = 0; i < vocab_size; i++) {
            float p = logits[indices[i]];
            if (p <= 0.0f) break;
            float surprise = -log2f(p);
            surprise_sum += surprise;
            cutoff = i + 1;
            if (surprise_sum >= tau) break;
        }

        // Renormalize and sample from truncated distribution
        std::vector<float> probs(cutoff);
        float sum = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            probs[i] = logits[indices[i]];
            sum += probs[i];
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < cutoff; i++) probs[i] *= inv;

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return indices[dist(rng_)];

    } else {
        // Mirostat v1: simpler approach — just use top-1 (greedy with adaptation)
        // This is a simplified version; full v1 tracks mu dynamically
        return int(std::max_element(logits, logits + vocab_size) - logits);
    }
}

}  // namespace vibeblade
