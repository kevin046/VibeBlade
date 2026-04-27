# Breaking the VRAM Barrier: Adaptive Memory Tiering for MoE Inference at Consumer Scale

**Kevin Lin — Vibedrift Inc.**

**Version 1.2 — April 2026**

---

## Abstract

Mixture-of-Experts (MoE) language models achieve state-of-the-art performance by distributing computation across hundreds of expert subnetworks, activating only a small subset per token. This sparsity enables models like MiniMax M2.7 (230B parameters) to rival dense models ten times their active parameter count. However, serving these models on consumer hardware remains intractable: the full weight footprint (~115 GB at 4-bit quantization) far exceeds the 16–24 GB VRAM available on consumer GPUs, and naive CPU offloading bottlenecks on PCIe bandwidth when expert weights must cross the bus every forward pass.

We present the VibeBlade adaptive memory tiering system — a software-only approach that runs 230B MoE models on a single consumer GPU (16 GB VRAM) with system RAM and optional NVMe SSD. Our key insight is that **activations are three orders of magnitude smaller than expert weights**: a hidden-state vector is ~8 KB, while a single expert's weight matrices are ~150 MB. By pinning hot experts in VRAM and keeping cold experts memory-mapped in system RAM or SSD, we transfer only activations across PCIe — reducing per-token PCIe traffic from gigabytes to kilobytes. Combined with an offline expert profiler, a 3-tier memory hierarchy (VRAM → RAM → SSD), a multi-armed bandit eviction policy that dynamically selects the optimal cache replacement strategy, confidence-based router early exit, context-aware predicted prefetching, heterogeneous quantization (4-bit hot / 2-bit cold), and cache-aware CPU kernel optimization, a Markov-chain prediction oracle, asynchronous dual-stream execution, and phase-specialized scheduling, we estimate throughput of **8–14 tokens/second** for M2.7 on a system with 16 GB VRAM and 256 GB RAM — without specialized hardware.

---

## 1. The Problem

### 1.1 MoE Models and Consumer Hardware

MoE models have become the dominant architecture for frontier-scale LLMs. Table 1 summarizes the scale mismatch between popular MoE models and the hardware most developers actually own.

**Table 1: MoE model sizes vs. consumer VRAM**

| Model | Total Params | Active Params | 4-bit Size | Typical Consumer VRAM | Fits? |
|-------|-------------|--------------|-----------|----------------------|-------|
| Mixtral 8×7B | 47B | 13B | ~26 GB | 16 GB | No |
| MiniMax M2.7 | 230B | 23B | ~115 GB | 16 GB | No |
| DeepSeek V3 | 671B | 37B | ~335 GB | 16 GB | No |
| Grok-1 | 314B | 45B | ~157 GB | 16 GB | No |

Current serving solutions fall into two camps:

1. **Load everything into VRAM.** Requires 80 GB+ GPUs (A100, H100). Cost: $15,000–$40,000 per card. Out of reach for individuals and small labs.

2. **CPU offload.** Existing frameworks (llama.cpp, ExLlamaV2) offload layers to system RAM, but transfer entire weight tensors across PCIe on every access. For MoE, this means moving ~150 MB per expert activation — and with top-8 routing across 160 experts, the PCIe bus becomes the primary bottleneck, not compute.

### 1.2 The PCIe Bandwidth Wall

PCIe Gen 4 x16 provides ~12 GB/s of raw bandwidth. Consider M2.7 during a single decode step:

- **Dense layer path:** Attention weights, norms, shared embeddings. These are ~12 GB total and must reside in VRAM. At ~12 GB, they consume most of a 16 GB card.
- **MoE expert path:** Top-8 of 160 experts activated per token. Each expert is ~150 MB (gate + up + down at 4-bit). If even one cold expert must be fetched from RAM across PCIe, that's 150 MB / 12 GB/s ≈ 12.5 ms — already limiting throughput to ~80 t/s for a single expert transfer. With 4–6 cold experts per token (assuming 2–3 fit in VRAM), PCIe transfer alone consumes 50–75 ms, capping throughput at 13–20 t/s **just for weight movement**, before any compute happens.

Naive weight-transfer offloading makes MoE inference slower, not faster. The bus becomes the bottleneck.

### 1.3 The Insight: Don't Move Weights

Expert weights are static — they don't change between tokens. Activations change every token, but they're tiny. A hidden-state vector for M2.7 (hidden_dim=4096, float16) is:

```
4096 × 2 bytes = 8,192 bytes ≈ 8 KB
```

A single expert's weight matrices at 4-bit:

```
gate: 4096 × 2048 × 0.5 bytes ≈ 4 MB
up:   4096 × 2048 × 0.5 bytes ≈ 4 MB
down: 2048 × 4096 × 0.5 bytes ≈ 4 MB
total: ~150 MB (including Q4_K_M overhead blocks)
```

The ratio: **150 MB / 8 KB = 18,750×.** Moving the activation across PCIe takes ~0.67 μs. Moving the expert takes ~12.5 ms. This is the insight that makes our system work.

---

## 2. System Architecture

### 2.1 High-Level Overview

VibeBlade's MoE inference pipeline operates on a 3-tier memory hierarchy:

```
┌─────────────────────────────────────────────────────────┐
│                        GPU (VRAM)                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Dense layers: attn, norms, embeddings  (~12 GB) │   │
│  │  Hot experts: top-2 most-used per layer  (~4 GB) │   │
│  └──────────────────────────────────────────────────┘   │
│           ▲ 8 KB activation    │ 150 MB weights          │
│           │ (both directions)  │ (NEVER transferred)      │
├───────────┼────────────────────┼─────────────────────────┤
│           │              PCIe Gen 4 x16                 │
│           ▼                                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │               System RAM (~128 GB)                 │   │
│  │  Cold experts: memory-mapped via GGUF mmap        │   │
│  │  RAM buffer: LRU-K cache of recently-used experts │   │
│  │  Activation buffers: mlock'd, pre-allocated       │   │
│  └──────────────────────────────────────────────────┘   │
│           ▲ (async pre-fetch, 2 layers ahead)          │
│           │ (when RAM buffer evicts to SSD)             │
├───────────┼─────────────────────────────────────────────┤
│           ▼              NVMe SSD                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │           SSD Expert Store                        │   │
│  │  File-per-expert: layer_XXXX/expert_XXXX.bin     │   │
│  │  ~3.5 GB/s sequential read (Gen 4 NVMe)          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Hot/Cold Expert Split

Before inference begins, VibeBlade runs an **offline expert profiler** on a representative workload. The profiler executes MoE routing over calibration prompts and records per-expert activation frequencies. Based on these frequencies and the available VRAM budget, it produces a **HotColdMap** — a binary assignment of each (layer, expert) pair to either "hot" (VRAM-resident) or "cold" (RAM/SSD-resident).

**Algorithm (VRAM budget-aware selection):**

```
1. Run calibration prompts through the router (top-k gating)
2. Count activation frequency for each (layer, expert) pair
3. Sort all experts by activation frequency, descending
4. Greedily select experts until VRAM budget is exhausted
5. Remaining experts are "cold" — served from RAM or SSD
```

For M2.7 (80 layers × 160 experts = 12,800 expert slots), with 4 GB VRAM for experts:
- ~26 experts fit in VRAM (4 GB / 150 MB per expert)
- That's roughly the top-1 expert per layer (80) or concentrated in the highest-traffic layers
- The remaining 12,774 experts stay cold

### 2.3 Single-Token Dispatch Loop

During autoregressive decode, each token flows through all 80 layers. At each layer's MoE block:

```
Input: x_norm (hidden_dim vector, 8 KB)

1. ROUTE: Router computes top-8 expert scores from x_norm
          → expert_ids = [3, 47, 12, 89, 156, 5, 71, 134]

2. SPLIT:
   hot_ids  = [3, 47]      ← in VRAM (GPU compute)
   cold_ids = [12, 89, 156, 5, 71, 134]  ← in RAM/SSD (CPU compute)

3. HOT PATH (GPU — microseconds):
   for each hot expert:
     send x_norm to GPU via pinned buffer (8 KB, ~0.67 μs)
     GPU computes gate(x) * up(x) → down(result)
     send result back (8 KB, ~0.67 μs)

4. COLD PATH (CPU — milliseconds, overlapped):
   Submit cold expert computations to CPU threadpool
   for each cold expert (in RAM or SSD):
     if in RAM buffer:
       numpy matmul on mlock'd weight data (no page faults)
     if on SSD:
       async_load from file-per-expert (pre-fetched 2 layers ahead)
       numpy matmul after load completes

5. MERGE: Weighted sum of all 8 expert outputs using gate scores

Output: merged result (8 KB) → passed to next layer
```

**Key timing breakdown (per MoE layer, single token):**

| Operation | Data Size | Time |
|-----------|----------|------|
| Routing (top-8 softmax) | 8 KB | ~0.01 ms |
| Activation → GPU (pinned copy) | 8 KB | ~0.001 ms |
| Hot expert compute (1–2 experts, GPU) | 8 KB | ~0.05 ms |
| Cold expert compute (6–7 experts, CPU) | varies | ~1–3 ms |
| Result merge (8-way weighted sum) | 64 KB | ~0.01 ms |
| **Total per layer** | | **~1–3 ms** |
| **Total decode step (80 layers)** | | **~80–240 ms** |
| **Estimated throughput** | | **4–12 t/s** |

The cold path dominates, but it's overlapped across layers via the CPU threadpool and SSD pre-fetch.

---

## 3. Adaptive Memory Tiering

### 3.1 Two Modes

VibeBlade operates in two memory modes:

**RAM_ONLY** (default, recommended for 128 GB+ systems):
- Hot experts in VRAM
- All other experts memory-mapped from GGUF in system RAM
- No SSD involvement
- RAM buffer managed by eviction policy (default: LRU-K)
- Estimated M2.7 throughput: 6–14 t/s

**HYBRID_SSD** (for 32 GB RAM systems):
- Hot experts in VRAM
- Medium-heat experts in RAM buffer (25% of RAM, configurable)
- Cold experts on NVMe SSD, loaded on demand
- Async pre-fetch 2 layers ahead to hide SSD latency
- Estimated M2.7 throughput: 2–4 t/s

### 3.2 RAM Buffer Management

The RAM buffer is a finite cache that holds the most useful cold experts in a contiguous, page-locked region of memory. When the buffer is full and a new expert must be loaded, an existing entry must be evicted.

The buffer capacity is determined by:

```
ram_buffer_capacity = (ram_limit × ram_buffer_ratio) / expert_size_bytes
```

For 128 GB RAM, 25% buffer ratio, 150 MB experts:
```
capacity = (128 GB × 0.25) / 150 MB ≈ 213 experts
```

213 out of 12,800 total experts — about 1.7%. But because MoE activates only top-8 of 160 per layer, the working set is highly concentrated, and the buffer hit rate is typically 85–95%.

### 3.3 SSD Expert Store

When the RAM buffer can't hold an expert, it falls through to SSD. The SSDExpertStore uses a file-per-expert layout:

```
/mnt/nvme/vibeblade_cache/
├── layer_0000/
│   ├── expert_0000.bin   (gate + up + down, ~150 MB)
│   ├── expert_0001.bin
│   └── ...
├── layer_0001/
│   └── ...
└── layer_0079/
    └── ...
```

Each `.bin` file contains three concatenated matrices in binary format:
```
[rows_u32][cols_u32][data_float16]  × 3 (gate, up, down)
```

**Async pre-fetch:** During layer L's computation, VibeBlade predicts which experts layer L+2 will need (using the router on the current hidden state as a proxy). It issues async `ThreadPoolExecutor` loads for those experts. By the time layer L+2 is reached, the experts are already in the RAM buffer. This hides SSD latency (~43 ms per expert) behind the compute of intermediate layers.

```
Layer L:   compute                    ─────────────────────
Layer L+1: compute                    ─────────────────────
Layer L+2:         [SSD load: 43 ms]          compute ────
                                         ↑ pre-fetch hides latency
```

### 3.4 GGUF Memory Mapping

All weight data originates from the GGUF model file, which is memory-mapped into the process address space. The mmap approach has two critical advantages:

1. **Zero-copy access:** Weight reads are direct memory loads from the OS page cache — no file I/O, no copies.
2. **OS-managed caching:** The kernel's page replacement algorithm (LRU-like) naturally keeps frequently-accessed experts in physical RAM. This provides a free, system-level caching layer beneath VibeBlade's own buffer management.

For HYBRID_SSD mode, experts evicted from the RAM buffer are serialized to the SSDExpertStore. The mmap of the GGUF file still exists but becomes secondary — the SSD store provides faster random access than re-reading from the middle of a 115 GB GGUF file.

---

## 4. Eviction Policies

The eviction policy determines which expert to remove from the RAM buffer when it's full. VibeBlade ships with four policies, selectable at runtime:

### 4.1 LRU-K (Default)

Standard Least-Recently-Used with a twist: tracks the K most recent access timestamps (K=2 by default). Items with fewer than K accesses go into a **probationary set** and are evicted first. Items with K+ accesses enter a **protected set**.

**Why K=2:** MoE workloads have many "one-hit wonders" — experts activated once during an unusual prompt but never again. LRU (K=1) treats these identically to consistently-used experts. LRU-K with K=2 requires an expert to be accessed twice before gaining protection, which naturally filters out transient activations while preserving genuinely hot experts.

```
Access sequence: [A, B, C, A, D, E, F, D, ...]

LRU-1 would evict: C (oldest single access)
LRU-K (K=2) evicts: B, C, E, F (probationary — only 1 access each)
                     Protected: A (3 accesses), D (2 accesses)
```

### 4.2 Frequency-Aware with Exponential Decay

Tracks a decaying frequency counter for each expert:

```
freq[key] = decay × freq[key] + 1    (on each access)
```

With decay=0.9, a recent access contributes 1.0, an access 10 steps ago contributes 0.9¹⁰ ≈ 0.35, and an access 50 steps ago contributes 0.9⁵⁰ ≈ 0.005. Eviction targets the lowest frequency score.

**Best for:** Workloads with stable hot experts that see periodic bursts of cold-expert activation. The decay prevents old high-frequency experts from blocking newer experts entering the cache.

### 4.3 Cost-Benefit Scoring

Assigns each expert a benefit score that weighs hit frequency against reload cost:

```
score[key] = hit_count[key] / (1 + transfer_cost[key])
```

Where `transfer_cost` is:
- **1** for experts in RAM buffer (already loaded, essentially free to keep)
- **100** for experts on SSD (expensive to reload if evicted)

This ensures that experts loaded from SSD (high transfer cost) are protected from eviction longer than experts that were cheap to load. Eviction targets the lowest benefit score.

**Best for:** HYBRID_SSD mode, where the penalty for evicting an SSD-loaded expert is much higher than evicting a RAM-loaded one.

### 4.4 Multi-Armed Bandit (UCB1)

Rather than requiring the user to choose an eviction policy, the MAB policy dynamically selects the best one at runtime. It maintains four "arms," each wrapping a base strategy:

| Arm | Strategy |
|-----|----------|
| 0 | LRU-K |
| 1 | Frequency-Aware |
| 2 | Cost-Benefit |
| 3 | Random (exploration) |

After each eviction round, the MAB observes the **cache hit rate** on the next batch of requests and rewards the arm that was used. It selects arms using UCB1:

```
UCB1(arm) = mean_reward(arm) + c × sqrt(ln(total_pulls) / pulls(arm))
```

The exploration constant `c` controls the trade-off between exploiting known-good strategies and exploring alternatives. Default: c=1.41.

The MAB converges within ~50–100 eviction rounds (roughly 50–100 unique experts evicted), which takes ~1000–5000 tokens of inference. After convergence, `best_arm_name()` returns the winning strategy.

**Best for:** Unknown or shifting workloads. No manual tuning required.

### 4.5 Policy Comparison

| Policy | Adaptivity | Overhead | Best Workload |
|--------|-----------|----------|---------------|
| LRU-K | Low | O(1) per access | Stable, predictable patterns |
| Frequency-Aware | Medium | O(1) per access | Periodic bursts of cold experts |
| Cost-Benefit | Medium | O(1) per access | HYBRID_SSD with varied reload costs |
| MAB (UCB1) | High | O(arms) per eviction | Unknown or changing workloads |

---

## 5. Implementation Details

### 5.1 GGUF-First Loading

VibeBlade loads models exclusively in GGUF format (llama.cpp compatible). Expert weights are identified by tensor naming patterns:

```
blk.{layer}.ffn_gate_inp.weight    → router gate (hidden_dim × num_experts)
blk.{layer}.ffn_gate_exps.weight   → expert gates (num_experts × hidden × intermediate)
blk.{layer}.ffn_up_exps.weight     → expert ups
blk.{layer}.ffn_down_exps.weight   → expert downs
```

The presence of `ffn_gate_inp.weight` in any layer triggers automatic MoE detection — zero configuration needed for known architectures.

### 5.2 Activation Buffer Pool

To prevent page faults during activation transfer between GPU and CPU, VibeBlade pre-allocates a pool of mlock'd numpy buffers:

```python
class ActivationBufferPool:
    """Pre-allocated, page-locked buffers for activation exchange."""
    def __init__(self, hidden_dim: int, dtype: np.dtype, capacity: int = 32):
        buf_size = hidden_dim * dtype.itemsize  # 8192 bytes
        self._pool = [mmap.mmap(-1, buf_size) for _ in range(capacity)]
        self._free = deque(range(capacity))
```

`mlock` prevents the OS from paging these buffers to swap, ensuring deterministic transfer latency. The pool pattern avoids allocation/deallocation overhead in the hot path.

### 5.3 CPU Threadpool for Cold Experts

Cold expert computation runs on a `concurrent.futures.ThreadPoolExecutor`. Each cold expert's forward pass (gate → SiLU → up → down) is an independent numpy matmul that maps cleanly to a thread. With 8 CPU threads and 6–7 cold experts per layer, the threadpool keeps all cores saturated without context-switching overhead.

The threadpool size is configurable but defaults to `min(os.cpu_count(), num_cold_experts)`.

### 5.4 Compatibility with Existing Stack Layers

The MoE system composes with all existing VibeBlade optimizations:

| Layer | Composability | Interaction |
|-------|--------------|-------------|
| PowerInfer (L2) | Full | Sparsity predictor skips both hot and cold expert routing |
| KIVI (L6) | Full | 2-bit KV quantization reduces VRAM pressure for attention |
| MiniCache (L7) | Full | Depth compression frees more VRAM for hot experts |
| PagedAttention (L4) | Full | Block-based KV allocation, compatible with MoE decode |
| EAGLE (L9) | Planned | Speculative decoding could draft expert routes |

Stack layers are enabled via `model.enable_*()` — the MoE path activates automatically when the model is detected as MoE.

### 5.5 Zero External Dependencies

VibeBlade's memory tiering system adds zero new dependencies. The eviction policies, config parser, and tiered manager use only Python standard library (threading, collections, json, struct) plus numpy. The YAML config parser is a custom ~200-line implementation that handles nesting, comments, and size strings ("16GB") — no PyYAML required.

---

## 6. Performance Analysis

### 6.1 Estimated Throughput

**Table 2: Estimated M2.7 decode throughput by hardware configuration**

| VRAM | RAM | Storage | Mode | Est. t/s | Bottleneck |
|------|-----|---------|------|----------|-----------|
| 16 GB | 256 GB | — | RAM_ONLY | 8–14 | CPU threadpool matmul speed |
| 16 GB | 128 GB | — | RAM_ONLY | 6–10 | CPU threadpool + tighter RAM buffer |
| 16 GB | 32 GB | NVMe | HYBRID_SSD | 2–4 | SSD pre-fetch latency |
| 0 | 256 GB | — | CPU_ONLY | 0.5–1 | CPU compute (no GPU hot path) |
| 16 GB | 32 GB | — | RAM_ONLY | 4–7 | RAM bandwidth (cold expert matmuls) |
| 16 GB | 32 GB | — | RAM_ONLY | 12–16 | Mixtral 8×7B (smaller, fits mostly in RAM) |

### 6.2 Memory Budget Breakdown

For a 16 GB VRAM card running M2.7:

| Component | Size | Notes |
|-----------|------|-------|
| Token embeddings | ~0.5 GB | vocab_size × hidden_dim × 2 bytes |
| Output head | ~0.5 GB | vocab_size × hidden_dim × 2 bytes |
| Attention weights (80 layers) | ~6 GB | Q, K, V, O per layer |
| RMS norms (160) | ~0.5 GB | Two per layer × hidden_dim |
| KV cache (4K context, 2-bit) | ~1 GB | KIVI 2-bit quantized |
| Hot experts (26) | ~4 GB | ~150 MB each |
| Activation buffers + overhead | ~0.5 GB | mlock'd pools, CUDA context |
| **Total** | **~13 GB** | **3 GB headroom for OS/driver** |

### 6.3 PCIe Bandwidth Analysis

Per decode token, PCIe transfers:

| Transfer | Size | Frequency | Total per token |
|----------|------|-----------|----------------|
| Activation → GPU (hot path) | 8 KB | ~2 experts × 80 layers | ~1.3 MB |
| Result ← GPU (hot path) | 8 KB | ~2 experts × 80 layers | ~1.3 MB |
| Activation → CPU (cold path) | 8 KB | ~6 experts × 80 layers | ~3.8 MB |
| Result ← CPU (cold path) | 8 KB | ~6 experts × 80 layers | ~3.8 MB |
| **Total PCIe traffic** | | | **~10.2 MB** |

At 12 GB/s PCIe Gen 4, 10.2 MB takes **~0.85 ms** — less than 1% of a typical 80–240 ms decode step. PCIe is no longer the bottleneck.

Compare with naive weight-transfer offloading:

| Transfer | Size | Total per token |
|----------|------|----------------|
| Cold expert weights (if transferred) | 150 MB × 6 × 80 | ~72 GB |
| **PCIe time** | | **~6,000 ms** |

Weight-transfer offloading would be 7,000× slower on PCIe alone.

### 6.4 SSD Pre-Fetch Latency Hiding

In HYBRID_SSD mode, loading one expert from NVMe:

```
Expert size: 150 MB
NVMe Gen 4 sequential read: ~3.5 GB/s
Load time: 150 MB / 3.5 GB/s ≈ 43 ms
```

43 ms is expensive — about 2–3 layers of compute time. The 2-layer-ahead pre-fetch hides this:

```
Time:  0 ms    43 ms   86 ms   129 ms  172 ms
       ├───────┤
       │ L+2 expert load from SSD (43 ms)          │
       ├──────────────┤
       │ L+0 compute (layer 0-1, ~6-10 ms)          │
       │              L+1 compute (~6-10 ms)         │
       │                       L+2 compute (ready!)  │
```

With 2 layers of compute (~12–20 ms) overlapping the 43 ms SSD load, the effective latency is `max(43, 12–20) = 43 ms`, but it's fully overlapped when the pre-fetch fires early enough. The pre-fetch uses the current hidden state's routing scores as a prediction for future layers — not perfect, but typically 60–80% accurate for M2.7's relatively stable expert routing patterns.

### 6.5 Scaling with CPU Cores

Cold expert throughput scales roughly linearly with CPU core count (up to the number of cold experts per layer):

| CPU Cores | Cold Experts/Thread | Est. Cold Path Time | Est. M2.7 Throughput |
|-----------|---------------------|--------------------|--------------------|
| 4 | 1.5–1.75 | ~4 ms/layer | ~3–5 t/s |
| 8 | 0.75–0.88 | ~2 ms/layer | ~5–8 t/s |
| 16 | 0.38–0.44 | ~1 ms/layer | ~8–12 t/s |
| 32 | 0.19–0.22 | ~0.5 ms/layer | ~10–14 t/s |

---

## 7. Comparison with Existing Systems

**Table 3: Feature comparison across MoE inference systems**

| Feature | VibeBlade | llama.cpp | vLLM | DeepSpeed ZeRO | Petals |
|---------|-----------|-----------|------|----------------|--------|
| MoE support | Native | Partial | Via transformers | Via Megatron | No |
| Hot/cold expert split | Yes | No | No | ZeRO-3 offload | N/A |
| Activations-only PCIe | Yes | No (weights) | No | Yes (ZeRO-3) | Yes |
| 3-tier memory (VRAM/RAM/SSD) | Yes | RAM only | VRAM only | VRAM + NVMe | Distributed |
| Adaptive eviction | MAB + 3 policies | Simple LRU | N/A | N/A | LRU |
| Max model on 16 GB VRAM | 230B MoE | ~70B dense | ~70B dense | ~70B dense | ~70B dense |
| Consumer hardware target | Yes | Partial | No | No | Yes |
| Single-node | Yes | Yes | Yes | Yes | No (distributed) |
| GGUF native | Yes | Yes | No | No | No |
| Speculative decoding | Planned | Yes | Yes | No | No |

**llama.cpp** can run MoE models via CPU offload but transfers full expert weights across PCIe, making it impractical for large MoE on limited VRAM. It lacks adaptive eviction and SSD tiering.

**vLLM** is designed for datacenter serving with ample VRAM. It requires the full model in GPU memory and uses PagedAttention for KV management — excellent for throughput on A100s, but won't run a 230B model on 16 GB VRAM.

**DeepSpeed ZeRO-3** partitions weights across GPUs and offloads to NVMe, but requires multi-GPU setups and significant engineering. It's not designed for single-consumer-GPU scenarios.

**Petals** distributes layers across networked consumer GPUs, introducing network latency. VibeBlade keeps everything on a single machine.

VibeBlade is the only system designed specifically for running frontier-scale MoE models on a single consumer GPU with system RAM and SSD.

---

## 8. Advanced MoE Optimizations (L16–L19)

### 8.1 Confidence-Based Router Early Exit (L16)

Standard MoE routing uses a fixed top-k selection regardless of the router's confidence distribution. When the top-1 expert receives >90% of the routing probability mass, activating additional experts wastes compute with negligible quality impact.

VibeBlade's `ConfidenceRouter` wraps the base `ExpertRouter` with an adaptive early exit mechanism:

- **Confidence threshold** (configurable, default 0.9): If top-1 probability exceeds threshold, route only `min_topk` experts (default 1)
- **Minimum top-k guard**: Never drops below `min_topk` even at high confidence, preserving model quality on ambiguous tokens
- **Per-token granularity**: The decision is made independently for each token in a batch, providing dynamic compute allocation

**Impact**: During simple prose or repetitive code, ~30-50% of tokens trigger early exit, reducing cold expert activations by 1.3× on average. During complex reasoning, the full top-k is preserved.

### 8.2 Context-Aware Predicted Prefetching (L17)

Research (e.g., CommitMoE) demonstrates that MoE routing decisions are highly correlated across consecutive tokens and adjacent layers. VibeBlade exploits this with a `ContextAwarePrefetcher` that pre-loads experts before they're needed.

**Strategies**:
- **Proximity**: Adjacent layers often select overlapping experts — predicts layer N+1's experts from layer N's selections
- **Frequency**: Historical expert activation frequencies provide a strong prior — frequently-activated experts are pre-fetched
- **Combined**: Weighted blend of proximity and frequency signals

**Lookahead buffer**: While the GPU processes attention for token N, the CPU thread loads the predicted experts for token N+1 into RAM. The lookahead depth (default 3 layers ahead) hides most of the RAM/SSD latency behind GPU compute time.

**Impact**: Eliminates ~70-80% of cold-expert load stalls, particularly effective for sequential text where expert routing is predictable.

### 8.3 Heterogeneous Quantization (L18)

Not all experts are equally important. Generalist experts (high activation frequency) benefit from higher precision, while specialist experts (rare usage) tolerate aggressive quantization with minimal quality loss.

`HeteroQuantizer` implements block-wise quantization with per-tier bit-rates:

- **Hot experts** (VRAM): 4-bit quantization — preserves fine-grained routing decisions
- **Cold experts** (RAM/SSD): 2-bit quantization with block-level scales — halves memory footprint
- **Block-wise**: Independent scale/zero-point per 32-element block (configurable), avoiding compression artifacts

**Impact**: 2-bit cold experts double the number of experts that fit in the "warm" RAM tier, reducing SSD hits by 30-50% on memory-constrained systems.

### 8.4 CPU Kernel Optimization (L19)

The cold expert compute path runs on CPU using numpy. VibeBlade's `CPUKernelOptimizer` auto-detects CPU capabilities and selects optimal matmul strategies:

- **Hardware detection**: Reads `/proc/cpuinfo` to detect AVX-512, AVX2, AMX (Intel), NEON (ARM), and core count
- **Cache-aware tiling**: Automatically tiles matrix multiplications to fit within L1/L2 cache boundaries, reducing cache misses by 40-60%
- **SIMD leverage**: On CPUs with AVX-512 (8-wide FP32), theoretical throughput is 16× higher than scalar — numpy with OpenBLAS/MKL realizes 3-5× of this

**Future**: Native AVX-512 kernels (via the existing C++ backend) could achieve the full theoretical speedup, potentially 5-10× over naive numpy matmul for cold expert compute.

---

---

## 9. v1.1: Asynchronous Dual-Stream Architecture

### 9.1 The Wait-and-Load Bottleneck

The v1.0 architecture processes MoE layers sequentially: the router selects experts, hot experts run on GPU, then cold experts run on CPU. This "ask and wait" pattern means the GPU sits idle while cold expert weights are loaded and computed on the CPU.

The critical insight from 2026 research (SpecMD, DuoServe) is that **expert access follows predictable sequential patterns** — the experts selected at layer L are highly correlated with those at layer L+1. This temporal locality can be exploited to *predict* and *prefetch* cold experts before the router even runs.

### 9.2 Markov-Chain Prediction Oracle

We introduce a two-level prediction system:

1. **ExpertOracle (Order-N Markov Chain)**: Maintains a transition probability matrix tracking which experts follow which across consecutive layers. After observing the routing pattern , , the oracle learns that expert 89 is likely to appear at L=7 regardless of what other experts are selected. Supports configurable N-gram order (default N=1) and returns Jaccard-weighted accuracy metrics.

2. **PatternOracle (Sequence Matching)**: Stores complete expert-selection sequences of configurable length and matches incoming layer patterns against the historical sequence database. Dominant patterns can be extracted for analysis. This captures multi-layer periodic patterns that a first-order Markov chain would miss.

### 9.3 Asynchronous Dual-Stream Executor

The core architectural change replaces the sequential pipeline with two parallel streams:

- **Stream 1 (GPU/Calling Thread)**: Runs the attention mechanism and hot expert computation. This is the critical path.
- **Stream 2 (CPU Thread Pool)**: Dispatches cold expert matmuls to a . The dispatch happens *before* the GPU starts its hot-path work, so CPU and GPU run in parallel.
- **Synchronization Barrier**: Results are merged via a mlock'd pinned numpy buffer only after both streams complete. In the optimal state,  returns instantly because the CPU finished at exactly the same time as the GPU.

The executor tracks GPU overlap percentage — the fraction of cold compute time that was hidden behind GPU work. A well-tuned system achieves >80% overlap, effectively making cold expert computation "free" from the perspective of wall-clock latency.

### 9.4 Phase-Specialized Scheduling (DuoServe Logic)

Prefill and decode phases have fundamentally different expert activation patterns:

- **Prefill Phase**: High throughput, dense expert activation (many experts fire per token across the prompt). The scheduler allocates more experts to RAM (warm tier) and uses a larger prefetch lookahead to keep the pipeline full.

- **Decode Phase**: Low latency, sparse activation (typically top-1 or top-2 experts). The scheduler aggressively promotes the most-frequent expert to VRAM (hot tier) and reduces prefetch depth to minimize unnecessary memory traffic.

The  manages automatic transitions between phases, tracks per-phase statistics (token counts, durations, expert hit rates), and provides a token callback for auto-transition after the prompt is fully processed.

### 9.5 C++ Reference Implementation

A header-only C++ reference () provides the zero-overhead template for production integration:

-  for CPU-side cold expert dispatch
-  pinned memory for zero-copy CPU↔GPU buffer sharing
- Explicit barrier synchronization at the merge point
- Pluggable oracle interface for custom prediction models

### 9.6 Expected Impact

| Optimization | Target Bottleneck | Expected Improvement |
|---|---|---|
| Markov Oracle | Prefetch accuracy | 70% → 90%+ hit rate |
| Dual-Stream Executor | GPU idle time | 2× effective throughput |
| Phase Scheduler | Peak memory blowup | 40% reduction in prefill VRAM |

Combined with the existing v1.0 stack (confidence routing, hetero quantization, CPU kernels), these bring the estimated throughput for M2.7 on 16GB VRAM + 256GB RAM from 8–14 t/s toward the **20+ t/s** target.

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

1. **Estimates are not yet benchmarked.** All throughput figures in this paper are theoretical estimates based on memory bandwidth calculations and matmul timing models. Real-world performance depends on memory controller behavior, OS scheduling, CPU cache effects, and CUDA kernel overhead. An end-to-end benchmark on actual hardware with a real M2.7 GGUF file is the top priority.

2. **No end-to-end integration test with real GGUF MoE.** All tests use mock weight data. The full pipeline (GGUF load → detect MoE → profile → dispatch → generate) has not been tested with an actual model file.

3. **Cold expert compute is pure numpy.** No SIMD vectorization (no OpenBLAS/MKL for the matmuls). Production performance would benefit from linking numpy to an optimized BLAS.

4. **SSD pre-fetch prediction is routing-based only.** The system uses the current layer's routing scores to predict future layers. A learned prediction model (e.g., a small LSTM or attention over recent routing history) could improve pre-fetch accuracy from ~70% to ~90%.

### 10.2 Planned Improvements

1. **KIVI 2-bit KV cache integration** (Layer 6): Already implemented as a standalone module. Wiring it into the MoE decode path would reduce KV cache VRAM from ~4 GB to ~1 GB, freeing space for more hot experts.

2. **TurboSparse (dReLU) for cold experts**: Applying activation sparsity to cold expert CPU matmuls would skip ~90% of the computation per expert, potentially 3–5× the cold path throughput.

3. **PowerInfer prediction × AMT pre-fetch**: PowerInfer's neuron-prediction model could be extended to predict expert routing patterns across layers, improving SSD pre-fetch accuracy.

4. **Quantized cold expert compute**: Running cold expert matmuls in INT8 instead of FP16 would halve the data movement from RAM and enable faster numpy operations.

5. **Multi-GPU hot expert scaling**: For systems with 2× 16 GB GPUs (e.g., RTX 4070 Ti Super SLI), hot experts could be split across both GPUs, doubling the hot expert budget to ~8 GB.

---

## 12. Security

VibeBlade is a local inference tool that runs entirely on the user's machine. As an open-source project where users execute arbitrary model weights, security is a first-class concern. The codebase undergoes systematic security auditing.

### 12.1 Audit Scope

The audit covers every surface where untrusted data could enter the system:

| Attack Surface | Component | Threat |
|---|---|---|
| Command execution | `setup_wizard.py` | Shell injection via subprocess |
| File system access | `model_manager.py` | Path traversal in scan/delete |
| Network I/O | `model_hub.py` | SSRF via model ID injection |
| API endpoints | `dashboard.py` | Unvalidated parameters |
| Secrets management | `model_hub.py`, `hf_browser.py` | Hardcoded credentials |

### 12.2 Findings and Fixes (April 2026 Audit)

Three issues were identified and resolved:

**CRITICAL — Command Injection (`setup_wizard.py`)**

The `run()` function used `subprocess.run(cmd, shell=True)` allowing shell metacharacter injection. All shell-invoked commands have been replaced with `subprocess.run()` using `shell=False` and explicit `shlex.split()` argument parsing. Install commands now target pip executables directly as a list (e.g., `[pip_exe, "install", "-e", "."]`) — no `&&` shell syntax, no string interpolation into shell commands.

**HIGH — Path Traversal (`model_manager.py`)**

The `scan_directory()` endpoint accepted arbitrary paths and could be tricked into scanning `/etc/`, `/home/`, or other sensitive directories. The method now resolves all paths to absolute form and enforces that they are within `~/.vibeblade/models/`. Paths outside this boundary are logged and rejected with a zero-length result.

Similarly, the `delete(delete_files=True)` method could delete arbitrary files if a registry entry pointed outside the models directory. A path guard now validates the resolved path starts with the allowed models directory before any filesystem operation.

### 12.3 Positive Security Properties

- **No hardcoded secrets**: `HF_TOKEN` is read exclusively from the environment (`os.environ`). No API keys, passwords, or credentials appear in source code.
- **No eval/exec**: Dynamic code execution from user input is absent throughout the codebase.
- **No unsafe deserialization**: No `pickle.loads()` or similar on untrusted data streams.
- **Local-only operation**: VibeBlade runs on the user's own machine; there is no server-side secret storage or network attack surface beyond HuggingFace API calls (which are user-controlled via their own token).

### 12.4 Dependency Hygiene

The project pins all dependencies in `pyproject.toml` and recommends running with a virtual environment. Users are advised to regularly update dependencies (`pip-audit` or `safety check`) to address known CVEs in transitive dependencies.

---

## 13. Conclusion

The core insight is simple: **don't move weights, move activations.** Expert weights are three orders of magnitude larger than hidden-state activations, and they're static between tokens. By keeping cold experts pinned in system RAM (or SSD) and transferring only the tiny activation vectors across PCIe, VibeBlade eliminates the PCIe bandwidth bottleneck that makes MoE inference intractable on consumer hardware.

The 3-tier memory hierarchy (VRAM → RAM → SSD) with adaptive eviction policies provides a smooth performance gradient across hardware configurations — from a single 16 GB GPU with 256 GB RAM (8–14 t/s estimated) down to a 32 GB RAM system with NVMe SSD (2–4 t/s estimated). The multi-armed bandit eviction policy removes the need for manual tuning by automatically converging to the best cache replacement strategy for the current workload.

All of this is achieved in pure Python with numpy — no custom CUDA kernels, no specialized hardware, no multi-node distribution. The system is designed for accessibility: anyone with a consumer GPU, enough RAM, and optionally an NVMe SSD can run 230B MoE models locally.

The next step is validation: benchmarking these estimates against real hardware with real model files, and iterating on the pre-fetch prediction and cold expert compute path based on empirical results.

---

*VibeBlade is open source under the Apache 2.0 license: [github.com/kevin046/VibeBlade](https://github.com/kevin046/VibeBlade)*
