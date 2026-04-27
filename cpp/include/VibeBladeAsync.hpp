#ifndef VIBEBlade_ASYNC_HPP
#define VIBEBlade_ASYNC_HPP

/// @file VibeBladeAsync.hpp
/// @brief Asynchronous dual-stream MoE executor for VibeBlade v1.1.
///
/// Architecture:
///   Stream 1 (GPU/main): Attention + Hot Expert FFN
///   Stream 2 (CPU pool):  Prefetch + Cold Expert FFN
///   Barrier:              Zero-copy merge at end of FFN block
///
/// While GPU processes Layer L attention, CPU simultaneously:
///   1. Prefetches predicted experts for Layer L+1 (via ExpertOracle)
///   2. Computes cold expert FFN for Layer L

#include <vector>
#include <future>
#include <memory>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <atomic>

namespace vibeblade {

// ── Data Structures ────────────────────────────────────────────────────────

/// A single MoE expert's weight set residing in System RAM or SSD.
struct ExpertWeights {
    int id;
    void* data_ptr;       ///< Pointer to weight data (gate/up/down projections)
    size_t size;          ///< Total bytes
    bool is_in_vram;      ///< True if currently pinned in GPU VRAM
};

/// Result of a CPU-side cold expert computation.
struct ExpertOutput {
    int id;
    std::vector<float> activation_delta;  ///< Output activation (hidden_dim,)
    float weight;                         ///< Routing weight from softmax
    double compute_time_ms;               ///< Wall-clock compute time
};

/// Per-layer execution statistics for async dual-stream pipeline.
struct AsyncLayerStats {
    double hot_compute_ms;      ///< Time spent on GPU hot-expert path
    double cold_compute_ms;     ///< Time spent on CPU cold-expert path
    double cold_wait_ms;        ///< Time spent waiting at barrier
    int prefetch_hit_count;     ///< Prefetched experts actually used
    int cold_expert_count;      ///< Number of cold experts dispatched
    double gpu_overlap_pct;     ///< % of cold compute overlapping with GPU

    /// Serialize stats to a flat vector (for Python binding).
    std::vector<double> to_vector() const {
        return {
            hot_compute_ms,
            cold_compute_ms,
            cold_wait_ms,
            static_cast<double>(prefetch_hit_count),
            static_cast<double>(cold_expert_count),
            gpu_overlap_pct,
        };
    }
};

/// Markov-chain expert predictor (first-order).
/// Tracks which expert E_j follows expert E_i at the next layer.
class ExpertOracle {
public:
    explicit ExpertOracle(size_t num_experts)
        : num_experts_(num_experts),
          transitions_(num_experts, std::vector<int>(num_experts, 0)),
          total_predictions_(0),
          correct_predictions_(0) {}

    /// Record observed expert selections for a layer.
    void observe(const std::vector<int>& layer_selected) {
        if (last_selected_.empty()) {
            last_selected_ = layer_selected;
            return;
        }

        // Update transition counts: last_selected[i] → layer_selected[j]
        for (int src : last_selected_) {
            for (int dst : layer_selected) {
                transitions_[src][dst]++;
            }
        }

        // Check prediction accuracy
        if (!last_prediction_.empty()) {
            total_predictions_++;
            int hits = 0;
            for (int predicted : last_prediction_) {
                for (int actual : layer_selected) {
                    if (predicted == actual) { hits++; break; }
                }
            }
            if (hits > 0) correct_predictions_++;
        }

        last_selected_ = layer_selected;
    }

    /// Predict experts for next layer given current selections.
    std::vector<std::pair<int, float>> predict(
        const std::vector<int>& current_experts,
        size_t top_k = 4
    ) {
        std::vector<std::pair<int, float>> scores;
        scores.reserve(num_experts_);

        for (size_t dst = 0; dst < num_experts_; ++dst) {
            float score = 0.0f;
            for (int src : current_experts) {
                score += static_cast<float>(transitions_[src][dst]);
            }
            score /= static_cast<float>(current_experts.size());
            scores.emplace_back(static_cast<int>(dst), score);
        }

        // Sort descending by score
        std::partial_sort(
            scores.begin(),
            scores.begin() + std::min(top_k, scores.size()),
            scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; }
        );

        scores.resize(std::min(top_k, scores.size()));

        // Store for accuracy tracking
        last_prediction_.clear();
        last_prediction_.reserve(scores.size());
        for (const auto& [id, _] : scores) {
            last_prediction_.push_back(id);
        }

        return scores;
    }

    float accuracy() const {
        return total_predictions_ > 0
            ? static_cast<float>(correct_predictions_) / static_cast<float>(total_predictions_)
            : 0.0f;
    }

    void reset() {
        for (auto& row : transitions_) std::fill(row.begin(), row.end(), 0);
        total_predictions_ = 0;
        correct_predictions_ = 0;
        last_selected_.clear();
        last_prediction_.clear();
    }

private:
    size_t num_experts_;
    std::vector<std::vector<int>> transitions_;
    std::vector<int> last_selected_;
    std::vector<int> last_prediction_;
    size_t total_predictions_;
    size_t correct_predictions_;
};

// ── Async MoE Manager ──────────────────────────────────────────────────────

/// Dual-stream MoE executor: GPU (hot experts) || CPU (cold experts).
///
/// Usage pattern:
///   1. predict_and_prefetch(L, experts) — called at START of Layer L
///   2. dispatch_cold_experts(state, cold_ids) — called AFTER router selects
///   3. ... GPU processes hot experts ...
///   4. merge_activations(gpu_state, cpu_future) — BARRIER at end of FFN
class AsyncMoEManager {
public:
    AsyncMoEManager(size_t num_experts, size_t top_k, size_t cpu_threads = 8)
        : num_experts_(num_experts),
          top_k_(top_k),
          oracle_(num_experts),
          pool_(cpu_threads),
          active_futures_(0) {}

    ~AsyncMoEManager() {
        shutdown();
    }

    // Non-copyable, non-movable
    AsyncMoEManager(const AsyncMoEManager&) = delete;
    AsyncMoEManager& operator=(const AsyncMoEManager&) = delete;

    /// Predict and prefetch experts for next layer.
    /// Call this at START of current layer — runs in background.
    ///
    /// @param current_layer_id  Current layer index
    /// @param last_selected     Expert IDs selected at current layer
    /// @return  Future resolving to predicted expert IDs
    std::future<std::vector<int>> predict_and_prefetch(
        int current_layer_id,
        const std::vector<int>& last_selected
    ) {
        return pool_.submit([this, current_layer_id, last_selected]() {
            auto predicted = oracle_.predict(last_selected, top_k_);

            std::vector<int> to_load;
            for (const auto& [id, _] : predicted) {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                if (ram_cache_.find(id) == ram_cache_.end()) {
                    to_load.push_back(id);
                }
            }

            // Pre-load from SSD → RAM
            for (int id : to_load) {
                load_to_system_cache(id);
            }

            return to_load;
        });
    }

    /// Dispatch cold experts to CPU threadpool.
    /// Call this AFTER the router selects experts for the current layer.
    ///
    /// @param input_hidden  Input hidden state (shared, read-only)
    /// @param cold_expert_ids  Expert IDs to compute on CPU
    /// @return  Future resolving to cold expert outputs
    std::future<std::vector<ExpertOutput>> dispatch_cold_experts(
        std::shared_ptr<const std::vector<float>> input_hidden,
        const std::vector<int>& cold_expert_ids,
        const std::vector<float>& routing_weights
    ) {
        active_futures_.fetch_add(1, std::memory_order_relaxed);
        return pool_.submit([this, input_hidden, cold_expert_ids, routing_weights]() {
            std::vector<ExpertOutput> results;
            results.reserve(cold_expert_ids.size());

            for (size_t i = 0; i < cold_expert_ids.size(); ++i) {
                int id = cold_expert_ids[i];
                auto t0 = std::chrono::high_resolution_clock::now();

                ExpertOutput out;
                out.id = id;
                out.weight = (i < routing_weights.size()) ? routing_weights[i] : 0.0f;

                // Load weights (should be pre-fetched into RAM cache)
                const float* weights = get_cached_weights(id);
                if (weights != nullptr) {
                    // Simplified FFN: activation @ gate @ up @ down
                    out.activation_delta.resize(input_hidden->size(), 0.0f);
                    // Real implementation would do actual matmul here
                    // using AVX-512/AMX kernels
                    compute_expert_ffn(
                        input_hidden->data(),
                        weights,
                        out.activation_delta.data(),
                        input_hidden->size()
                    );
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                out.compute_time_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();

                results.push_back(std::move(out));
            }

            active_futures_.fetch_sub(1, std::memory_order_relaxed);
            return results;
        });
    }

    /// Barrier: merge GPU and CPU results via zero-copy buffer.
    ///
    /// @param gpu_hidden  GPU-side hidden state (modified in-place)
    /// @param cpu_future  Future from dispatch_cold_experts()
    /// @return  Layer execution stats
    AsyncLayerStats merge_activations(
        std::vector<float>& gpu_hidden,
        std::future<std::vector<ExpertOutput>>& cpu_future,
        double hot_compute_ms
    ) {
        auto barrier_start = std::chrono::high_resolution_clock::now();

        // Wait for CPU results (blocks only if CPU not finished)
        auto cold_results = cpu_future.get();

        auto barrier_end = std::chrono::high_resolution_clock::now();
        double cold_wait_ms =
            std::chrono::duration<double, std::milli>(barrier_end - barrier_start).count();

        // Zero-copy merge: accumulate weighted cold expert outputs
        // Using pre-allocated merge buffer to avoid allocations
        merge_buffer_.resize(gpu_hidden.size(), 0.0f);
        for (const auto& res : cold_results) {
            for (size_t i = 0; i < gpu_hidden.size() && i < res.activation_delta.size(); ++i) {
                merge_buffer_[i] += res.weight * res.activation_delta[i];
            }
        }

        // Single merge into GPU state
        for (size_t i = 0; i < gpu_hidden.size(); ++i) {
            gpu_hidden[i] += merge_buffer_[i];
        }

        // Compute stats
        double total_cold_ms = 0.0;
        for (const auto& res : cold_results) {
            total_cold_ms += res.compute_time_ms;
        }
        if (!cold_results.empty()) {
            total_cold_ms /= cold_results.size();
        }

        double overlap = (cold_wait_ms < total_cold_ms)
            ? ((total_cold_ms - cold_wait_ms) / total_cold_ms) * 100.0
            : 0.0;

        return {
            hot_compute_ms,
            total_cold_ms,
            cold_wait_ms,
            static_cast<int>(cold_results.size()),
            static_cast<int>(cold_results.size()),
            overlap,
        };
    }

    /// Observe expert selections (updates oracle).
    void observe(const std::vector<int>& selected_experts) {
        std::lock_guard<std::mutex> lock(oracle_mutex_);
        oracle_.observe(selected_experts);
    }

    float oracle_accuracy() const {
        std::lock_guard<std::mutex> lock(oracle_mutex_);
        return oracle_.accuracy();
    }

    void shutdown() {
        pool_.shutdown();
    }

private:
    size_t num_experts_;
    size_t top_k_;
    ExpertOracle oracle_;
    std::mutex oracle_mutex_;

    // Thread pool (simplified — real impl would use a proper pool)
    struct ThreadPool {
        size_t max_threads;
        std::mutex mtx;
        std::atomic<size_t> active{0};
        bool stopped{false};

        explicit ThreadPool(size_t n) : max_threads(n) {}

        template<typename F>
        std::future<typename std::invoke_result<F>::type> submit(F&& f) {
            auto task = std::make_shared<std::packaged_task<typename std::invoke_result<F>::type()>>(std::forward<F>(f));
            auto future = task->get_future();
            // In production, this would queue to a real thread pool
            std::thread([task]() { (*task)(); }).detach();
            return future;
        }

        void shutdown() { stopped = true; }
    };

    ThreadPool pool_;
    std::atomic<size_t> active_futures_;

    // Weight cache: expert_id → weight data in RAM
    std::unordered_map<int, std::vector<float>> ram_cache_;
    std::mutex cache_mutex_;

    // Pre-allocated merge buffer (zero-copy)
    std::vector<float> merge_buffer_;

    void load_to_system_cache(int expert_id) {
        // Placeholder: In production, mmap from SSD file
        // expert_{expert_id}.bin → ram_cache_[expert_id]
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (ram_cache_.find(expert_id) == ram_cache_.end()) {
            ram_cache_[expert_id] = std::vector<float>(4096, 0.0f);  // placeholder
        }
    }

    const float* get_cached_weights(int expert_id) const {
        auto it = ram_cache_.find(expert_id);
        return (it != ram_cache_.end()) ? it->second.data() : nullptr;
    }

    /// Placeholder for AVX-512/AMX expert FFN kernel.
    /// Real implementation would use _mm512_* intrinsics or AMX instructions.
    static void compute_expert_ffn(
        const float* input,
        const float* weights,
        float* output,
        size_t hidden_dim
    ) {
        // Placeholder: identity pass-through
        // Real implementation: x @ gate_w → silu → @ up_w → @ down_w
        std::memcpy(output, input, hidden_dim * sizeof(float));
    }
};

} // namespace vibeblade

#endif // VIBEBlade_ASYNC_HPP
