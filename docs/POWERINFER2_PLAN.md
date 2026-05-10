# PowerInfer 2 Integration Plan

## Target Platform
- Oracle A1 ARM64 (4 OCPU, 24GB RAM)
- No NPU, no flash storage — pure CPU
- llama.cpp fork with existing EMA-based "PowerInfer" heuristic

## What PowerInfer 2 Actually Does (from paper arxiv 2406.06282)
Neuron-cluster-based sparse inference. NOT speculative decoding.
Key: decompose FFN ops into neuron clusters, route by activation pattern,
pipeline computation with I/O, tiered caching.

## Components to Build

### 1. Neuron Cluster Abstraction (`neuron_cluster.h/c`)
- Group FFN neurons by activation frequency into clusters
- Hot clusters: top N% by activation rate (dense, always computed)
- Cold clusters: remainder (sparse, predictor-gated)
- Cluster size adapts to L2 cache line size (ARM NEON: 512KB L2 per core)
- Offline profiling: run model over calibration data, record per-neuron activation rates
- Classification: sort neurons by activation freq, threshold at hot_budget percentile

### 2. Activation Predictor (`pi2_predictor.h/c`)
- Replace EMA heuristic with trained per-layer predictor
- Small 2-layer MLP per FFN layer (input: hidden state → output: binary activation mask)
- Training: run calibration prompts, collect (hidden_state, activation_pattern) pairs
- Inference: given hidden state, predict which cold neurons will fire
- Paper reports >90% accuracy on ReLU-based models, ~80% on SiLU

### 3. Segmented Neuron Cache (`neuron_cache.h/c`)
- Two regions: hot (cluster-level LRU) + cold (neuron-level LRU)
- Hot region: contiguous memory, pre-loaded, rarely evicted
- Cold region: individual neurons, demand-loaded on prediction
- On CPU-only: cache is DRAM, "I/O" is cache-miss prefetch from main memory

### 4. Neuron-Cluster Pipeline (`cluster_pipeline.h/c`)
- 5 stages per cluster: Predict → GateIO → GateCompute → UpDownIO → UpDownCompute
- Overlap: while computing Gate cluster K, prefetch UpDown cluster K
- Break matrix barrier: pipeline across Gate/Up/Down matrices within a layer
- Multi-threaded: N-1 compute threads + 1 prefetch thread

### 5. Adaptive Engine (CPU-only variant) (`adaptive_engine.h/c`)
- Original: NPU does hot (dense), CPU does cold (sparse)
- Our variant: CPU threads split between hot-cluster dense compute and cold-cluster sparse compute
- Thread allocation: hot_clusters get ceil(N*hot_ratio) threads, cold get the rest
- Dynamic: adjust hot_ratio based on actual activation sparsity at runtime

### 6. Flexible Bundle Loading (`bundle_loader.h/c`)
- Bundle Gate[i] + Up[i] + Down[i] weights contiguously in memory
- 80% co-activation → save 2 of 3 I/O operations on average
- Two-phase: load Gate first, verify non-zero, then load Up+Down only if needed
- On CPU-only: this improves cache locality (bundled weights in same cache line)

### 7. Offline Profiler (`pi2_profiler.py`)
- Run model on calibration data (Wikipedia subset, ~1M tokens)
- Record per-neuron activation frequencies per layer
- Generate: hot/cold classification, cluster assignments, predictor training data

## File Layout
```
llama.cpp/src/
  pi2/
    neuron_cluster.h/c    — cluster abstraction
    pi2_predictor.h/c     — trained activation predictor
    neuron_cache.h/c      — segmented LRU cache
    cluster_pipeline.h/c  — 5-stage pipeline
    adaptive_engine.h/c   — CPU-only adaptive engine
    bundle_loader.h/c     — Gate+Up+Down bundling
    pi2_common.h          — shared types, constants
```

## Integration Points
- `llama.cpp/src/llama-graph.cpp`: replace EMA sparse FFN with cluster-based FFN
- `llama.cpp/ggml/src/ggml-cpu/ops.cpp`: add cluster-aware matmul kernels
- `vibeblade/powerinfer2.py`: Python bindings (enable/disable/configure)
- `vibeblade/auto_tune.py`: add PI2 profile to auto-tuner

## Key Differences from Paper (CPU-only Adaptation)
1. No NPU → all compute on CPU threads, split by cluster type
2. No flash storage → "I/O" becomes L2 cache prefetch, "loading" = memcpy
3. Pipeline still helps: prefetch cluster N+1 while computing cluster N
4. Bundle loading still helps: cache locality for Gate+Up+Down per neuron
5. Predictor still helps: skip cold neurons that won't activate

## Expected Gains (CPU-only)
- Paper: 2.24-2.48× over llama.cpp (with NPU+CPU hybrid on Bamboo-7B, all in memory)
- Our baseline EMA heuristic: already 1.0-2.5× over baseline
- PI2 should beat EMA by 1.3-2.0× additional on models with high sparsity
- Dense models (Gemma family): minimal gain, activation sparsity too low
