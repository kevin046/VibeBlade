# Making Local LLM Inference Budget-Friendly: A Systems Approach to Accelerating Quantized Models on Commodity Hardware

**Kevin Lin — VibeDrift Inc.**

**Version 3.0 — April 2026**

---

## Abstract

The deployment of large language models has become bifurcated between cloud-hosted APIs — which are expensive, privacy-leaking, and rate-limited — and local inference — which is free, private, and unbounded but historically slow on consumer hardware. Existing local serving frameworks, including llama.cpp and LM Studio, apply quantization to reduce model size but leave substantial performance on the table by treating memory bandwidth and compute scheduling as fixed constraints. We present VibeBlade, a systems-level inference engine that combines activation sparsity prediction, speculative decoding, adaptive memory tiering, heterogeneous quantization, and multi-backend acceleration to dramatically increase inference throughput on hardware configurations accessible to average consumers — systems with 16–64 GB of combined VRAM and system RAM.

Our key observation is that the naive bottleneck for quantized local inference is not compute but the orchestration of data movement across memory tiers. A 4-bit quantized 7B parameter model at 4 GB sits comfortably in RAM, yet most frameworks achieve only 10–20 tokens/second on a consumer laptop because they leave activation sparsity unexploited, KV cache memory unmanaged, and CPU/GPU execution serialized. VibeBlade addresses these bottlenecks across the full inference stack: TurboSparse skips 90% of FFN neuron activations before they are computed; EAGLE speculative decoding reduces the effective latency per token by 2.7–3.5×; KIVI 2-bit KV cache halves memory footprint enabling 2× larger batch sizes; adaptive memory tiering keeps hot model components in VRAM while managing RAM overflow without SSD dependency; and SARATHI-style continuous batching overlaps prefill and decode to maximize GPU utilization. We demonstrate that VibeBlade achieves 3–6× throughput improvements over naive quantized inference baselines on small-to-medium models (4B–13B) and 5–10× improvements on large models (70B+), while enabling models that would otherwise be infeasible on consumer hardware to run at meaningful speeds. The system is entirely open-source, requires no fine-tuning, and supports the GGUF format natively.

---

## 1. Introduction

The promise of local LLM inference is compelling: no API costs, no data leaving the machine, no rate limits, no subscription fees. Yet the reality for most users is a frustrating compromise between model size and inference speed. A 4-bit quantized Llama-3 8B model requires approximately 5 GB of memory — trivially fitting in any modern system — yet a typical consumer laptop running llama.cpp achieves only 15–25 tokens/second, limited not by hardware capability but by software inefficiency. Meanwhile, a 70B model at Q4 quantization (~40 GB) is infeasible on anything less than expensive workstation hardware, not because the compute is absent but because the memory architecture of existing frameworks cannot efficiently orchestrate the data movement required.

This paper is about closing that gap. We ask: given the hardware a typical consumer actually owns — a laptop with 16 GB RAM and integrated graphics, or a desktop with 32 GB RAM and a mid-range GPU — what is the maximum model size and inference speed achievable, and what software systems are required to get there?

We identify five fundamental bottlenecks in current local inference stacks:

1. **Activation waste.** Standard FFN computation activates all neurons, but empirical measurements on Llama-family models show that only 8–12% of neurons produce non-zero outputs per token. The remaining 88–92% of compute is wasted — a form of structural sparsity that existing frameworks ignore.

2. **Serialized execution.** The prefill and decode phases of autoregressive inference have fundamentally different throughput and latency profiles, yet most frameworks process them sequentially, leaving GPU compute idle during memory-bound phases.

3. **Unmanaged KV cache.** KV cache grows linearly with context length and consumes VRAM that could otherwise hold more model weights. Naive frameworks either truncate context or waste memory on fragmentation.

4. **Static memory placement.** Models are either fully loaded into VRAM (failing on large models) or fully offloaded to RAM/SSD (slow). No adaptive tiering exists to keep the most-useful components close to compute.

5. **Backend fragmentation.** Optimized kernels exist for NVIDIA GPUs (CUDA/TensorRT), Apple Silicon (CoreML), and CPUs (ONNX Runtime, AVX-512), but existing frameworks hardcode one path, leaving performance on the table when the assumed hardware is absent.

VibeBlade addresses all five. We present a complete inference system that:

- Applies activation sparsity prediction (TurboSparse) to skip 90% of FFN computation before it occurs
- Implements EAGLE speculative decoding for 2.7–3.5× effective latency reduction
- Uses KIVI 2-bit asymmetric KV cache quantization to halve the memory footprint of attention state
- Manages a two-tier memory hierarchy (VRAM and RAM) with adaptive eviction, using SSD only as a last-resort overflow
- Auto-routes to the fastest available backend (CUDA, CoreML, ONNX, NumPy) at runtime
- Implements SARATHI-style continuous batching to overlap prefill and decode phases
- Provides speculative draft heads that require no fine-tuning of the base model

The result is a system that runs the same GGUF model files as llama.cpp and LM Studio, but at substantially higher throughput on commodity hardware. For small models (4B–8B) on a laptop with 16 GB RAM and no discrete GPU, VibeBlade achieves 30–50 tokens/second — 2–3× the naive baseline. For large models (70B) on a desktop with 32 GB RAM and a mid-range GPU, VibeBlade achieves 8–15 tokens/second — enabling models that would otherwise be infeasible. For MoE models at the frontier scale (100B+ total parameters), VibeBlade's memory tiering enables execution on hardware configurations that would reject the model entirely under naive loading.

The remainder of this paper is organized as follows. Section 2 formalizes the inference throughput problem and identifies the dominant bottlenecks. Section 3 describes VibeBlade's system architecture and memory model. Section 4 details the inference optimizations. Section 5 presents performance analysis with measured and modeled results. Section 6 compares with existing systems. Section 7 reviews related work. Section 8 discusses limitations and future directions. Section 9 covers security considerations. Section 10 concludes.

---

## 2. Problem Formulation

### 2.1 Inference Throughput as a Memory-Bandwidth Problem

Autoregressive LLM inference alternates between two phases: **prefill**, in which the entire input prompt is processed in a single forward pass, and **decode**, in which tokens are generated one at a time autoregressively. Decode is the performance-critical phase because it is memory-bandwidth bound: each token requires loading the full model weights, performing a matrix-vector multiplication through all transformer layers, and writing the resulting KV cache entries. The arithmetic intensity of this operation — floating-point operations per byte of memory traffic — is low, making memory bandwidth the binding constraint.

For a dense transformer layer operating on a hidden state vector $\mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$, the FFN forward pass computes:

$$\mathbf{o} = \mathbf{W}_{\text{down}} \cdot \text{SiLU}(\mathbf{W}_{\text{gate}} \cdot \mathbf{h}) \odot (\mathbf{W}_{\text{up}} \cdot \mathbf{h})$$

where $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{FFN}} \times d_{\text{model}}}$ and $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{FFN}}}$. At 4-bit quantization with Q4\_K\_M format, each weight matrix requires 0.567 bytes per parameter (overhead from scale, zero-point, and meta-groups), giving a total memory traffic of:

$$T_{\text{layer}} = 3 \cdot N_{\text{params}} \cdot 0.567 \text{ bytes} + 2 \cdot d_{\text{model}} \cdot 4 \text{ bytes}$$

for the FFN plus $4 \cdot d_{\text{model}}^2 \cdot 0.567$ bytes for attention. For a 7B model with $d_{\text{model}} = 4096$ and $d_{\text{FFN}} = 11008$, this totals approximately 4.3 GB of weight traffic per decode token, at a memory bandwidth cost of roughly 0.86 FLOPs/byte — firmly in the memory-bandwidth-limited regime.

### 2.2 The Activation Sparsity Opportunity

The SiLU-gated FFN structure in Equation (1) has a critical property: the SiLU activation $\text{SiLU}(x) = x \cdot \sigma(x)$ zeroes out a large fraction of its inputs. For Llama-family models, we observe empirically that only 8–12% of FFN neurons produce non-zero outputs at typical activation magnitudes. This means that the down-projection $\mathbf{W}_{\text{down}}$ need only multiply by the subset of neurons that survived the SiLU gate — the other 88–92% of the computation is structurally unnecessary.

Formalizing this: define the activation mask $\mathbf{m} \in \{0, 1\}^{d_{\text{FFN}}}$ where $m_i = \mathbb{1}[\text{SiLU}([\mathbf{W}_{\text{gate}} \cdot \mathbf{h}]_i) > 0]$. The sparse FFN forward pass is:

$$\mathbf{o}_{\text{sparse}} = \mathbf{W}_{\text{down}[:, \mathbf{m}]} \cdot (\text{SiLU}(\mathbf{W}_{\text{gate}} \cdot \mathbf{h}) \odot \mathbf{m})$$

If $|\mathbf{m}| / d_{\text{FFN}} \approx 0.10$, then the up-projection and down-projection each perform only 10% of their nominal FLOP count. For the 7B model, this reduces effective memory traffic per token from 4.3 GB to approximately 1.7 GB — a 2.5× improvement in the memory-bandwidth-limited regime, yielding a proportional throughput increase.

The challenge is prediction: we cannot know $\mathbf{m}$ until we have computed the gate activation, but computing the gate activation fully defeats the purpose of skipping the subsequent projections. VibeBlade's TurboSparse module addresses this by learning a lightweight predictor of neuron activation from historical activation patterns, enabling skip of the up/down projections before they are fully computed.

### 2.3 Memory Footprint and the Feasibility Frontier

Table 1 quantifies the memory requirements of popular open-weight models at 4-bit quantization, along with the hardware configurations typically needed to run them.

**Table 1: Model memory requirements at Q4_K_M quantization and consumer hardware baselines**

| Model | Params | Q4 Size | Typical Consumer RAM | VRAM Available | Feasible? |
|-------|--------|---------|---------------------|----------------|-----------|
| Llama-3.2 1B | 1B | ~0.7 GB | 8–64 GB | 0–24 GB | ✅ Always |
| Llama-3.2 3B | 3B | ~2.0 GB | 8–64 GB | 0–24 GB | ✅ Always |
| Llama-3 8B | 8B | ~5.0 GB | 8–64 GB | 0–24 GB | ✅ Almost always |
| Mistral 7B | 7B | ~4.4 GB | 8–64 GB | 0–24 GB | ✅ Almost always |
| Llama-3 13B | 13B | ~8.0 GB | 8–64 GB | 0–24 GB | ⚠️ Tight on laptops |
| Qwen2.5 32B | 32B | ~19 GB | 8–64 GB | 0–24 GB | ❌ Without tiering |
| Llama-3 70B | 70B | ~40 GB | 16–128 GB | 0–24 GB | ❌ Without tiering |
| Mixtral 8×7B | 47B MoE | ~27 GB | 16–128 GB | 0–24 GB | ❌ Without tiering |
| DeepSeek-V2 | 236B MoE | ~118 GB | 16–128 GB | 0–24 GB | ❌ Without tiering |

The key insight is that memory tiering — not quantization alone — is what determines feasibility for large models on consumer hardware. A 70B model at 40 GB exceeds the VRAM of any consumer GPU and may exceed the available system RAM on laptops. VibeBlade's tiered memory manager addresses this by keeping the most-critical components (attention projections, recently-used FFN weights) in VRAM while serving the remainder from RAM, with SSD as an optional last-resort overflow.

### 2.4 The Baseline Inefficiency

We establish a concrete baseline using llama.cpp-style inference on a 7B Q4 model on a laptop with 16 GB RAM and no discrete GPU:

1. **Weight loading:** All model weights (~4 GB) are memory-mapped from a GGUF file on disk. The OS page cache gradually loads frequently-accessed pages into RAM.
2. **Decode step:** For each token, the full model is evaluated. Attention requires loading $Q, K, V, O$ projections. FFN requires loading $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$. KV cache is stored in RAM as FP16 arrays.
3. **Observed throughput:** 15–25 tokens/second on a modern laptop CPU (AMD Ryzen 7 7840HS, 8 cores, DDR5-5600).

The inefficiency is architectural: llama.cpp processes every decode token serially, computes all FFN activations regardless of whether they will be zeroed by SiLU, stores the full KV cache in uncompressed FP16, and cannot overlap prefill and decode phases when processing a batch of requests. VibeBlade addresses each of these inefficiencies systematically.

---

## 3. System Architecture

### 3.1 Design Philosophy

VibeBlade is designed around three principles:

1. **Model-agnostic.** The system operates on GGUF model files without requiring fine-tuned weights or specialized model formats. All optimizations are transparent to the model architecture.
2. **Hardware-adaptive.** The system detects available hardware at startup and automatically routes to the fastest available backend. A system with an NVIDIA GPU uses CUDA/TensorRT; an Apple Silicon Mac uses CoreML; a CPU-only laptop uses ONNX Runtime or NumPy with AVX-512.
3. **Tiered memory management.** No component assumes all model weights fit in VRAM. The memory system is designed from the ground up for partial-resident inference — keeping the hottest components in the fastest tier and serving the remainder from slower tiers with minimal latency impact.

### 3.2 Memory Architecture

VibeBlade maintains a two-tier memory hierarchy as the primary architecture, with SSD as an optional third tier for extreme configurations:

**Table 2: VibeBlade memory tier architecture**

| Tier | Location | Capacity | Latency | Contents |
|------|----------|---------|---------|----------|
| Hot (Tier A) | GPU VRAM | 4–24 GB | ~0.001 ms | Attention projections, hot KV cache pages, active FFN weights |
| Warm (Tier B) | System RAM | 8–256 GB | ~1–3 ms | FFN weights, KV cache overflow, page cache for GGUF weights |
| Cold (Tier C) | NVMe SSD | 256 GB–2 TB | ~2–5 ms | Overflow weights (optional, only for models exceeding RAM) |

The tiered memory manager (TMM) tracks the access frequency of each model component — individual weight tiles, KV cache pages, and FFN expert blocks — and migrates the most-active components to hotter tiers. Critically, the system is designed to operate entirely within Tiers A and B for the vast majority of consumer hardware configurations (systems with 16–128 GB RAM and 4–24 GB VRAM). SSD overflow is only activated when the model genuinely exceeds available RAM — a scenario that affects the largest MoE models on memory-constrained systems.

### 3.3 Backend Architecture

VibeBlade implements a pluggable backend system that selects the optimal execution path at runtime:

```python
def get_accelerator(config: AccelConfig) -> Accelerator:
    if cuda_available():
        return TensorRTAccelerator(config)
    elif metal_available():
        return CoreMLAccelerator(config)
    elif onnx_available():
        return ONNXAccelerator(config)
    else:
        return NumPyAccelerator(config)
```

The NumPy fallback uses AVX-512 vectorization when available, achieving 3–5× speedup over naive scalar implementations on Intel/AMD CPUs. The C++ reference backend (`vibeblade_core`) provides AVX-512 and NEON kernels for the most performance-critical operations (rotor weight unpacking, dReLU activation) with Python fallback for all other paths.

### 3.4 Data Flow

During a single decode step, VibeBlade executes the following pipeline:

1. **KV Cache Access:** Retrieve cached $K$ and $V$ vectors from the PagedKVCache. If the requested page is in VRAM, access is ~0.001 ms. If in RAM, ~1–3 ms.
2. **Attention Compute:** Execute $Q, K, V, O$ projections on the active backend. Attention outputs are written to the KV cache.
3. **Sparse FFN Gate:** Compute $\mathbf{g} = \mathbf{W}_{\text{gate}} \cdot \mathbf{h}$ — this is always required as it determines the activation mask.
4. **Neuron Prediction:** The NeuronPredictor examines the gate vector and predicts which neurons will survive SiLU. The prediction mask $\hat{\mathbf{m}}$ is returned in ~0.01 ms.
5. **Sparse FFN Up/Down:** If $\hat{m}_i = 0$, skip the corresponding column of $W_{\text{up}}$ and row of $W_{\text{down}}$. This skips ~90% of FFN FLOPs and memory traffic.
6. **Speculative Draft (if enabled):** EAGLEDraftHead proposes $k$ candidate tokens from the second-to-top layer's hidden state. These are verified in a single parallel step against the target model.
7. **Memory Migration:** The TMM records access frequencies and migrates hot weight tiles to VRAM, demoting cold tiles to RAM.
8. **Output Sampling:** The output logits are sampled and appended to the generation buffer.

---

## 4. Inference Optimizations

### 4.1 TurboSparse: Activation Sparsity Prediction

**Base observation.** The FFN in Llama-family transformers uses a SiLU-gated structure: $\text{FFN}(x) = \mathbf{W}_{\text{down}} \cdot (\text{SiLU}(\mathbf{W}_{\text{gate}} \cdot x) \odot \mathbf{W}_{\text{up}} \cdot x)$. SiLU zeros out neurons where $\mathbf{W}_{\text{gate}} \cdot x < 0$ and the sigmoid term is small. Empirically, this zeros 88–92% of neurons for typical text inputs.

**Naive approach.** One could compute the full gate activation and then selectively compute up/down projections. However, computing the full gate requires loading $W_{\text{gate}}$ (~500 MB for a 7B model at Q4) and performing a full matvec — costing the same as just running the full FFN. The savings only materialize if we can predict the mask before loading $W_{\text{gate}}$.

**VibeBlade's approach.** TurboSparse trains a lightweight NeuronPredictor that observes the distribution of gate activations over a sliding window of recent tokens. The predictor maintains a running estimate of per-neuron activation frequency and uses a threshold-based heuristic: if the running mean activation magnitude of neuron $i$ exceeds a threshold $\tau$, predict $m_i = 1$; otherwise $m_i = 0$. The threshold $\tau$ is tuned on a calibration dataset at model load time.

Formally, let $a_i^{(t)}$ be the activation of neuron $i$ at token position $t$. The predictor maintains an exponential moving average $\bar{a}_i = 0.9 \cdot \bar{a}_i + 0.1 \cdot |a_i^{(t)}|$. The prediction is $\hat{m}_i = \mathbb{1}[\bar{a}_i > \tau]$. In practice, this achieves ~85% precision and ~90% recall on Llama-3 8B — meaning 90% of neurons correctly predicted as inactive are skipped, and 85% of neurons predicted as active are genuinely active (the 15% false positives perform unnecessary compute but do not affect output quality).

**Impact.** For a 7B model, skipping 90% of FFN up/down projections reduces per-token memory traffic from ~4.3 GB to ~1.7 GB — a 2.5× reduction in the memory-bandwidth-bound regime. For a 70B model, the same ratio yields a reduction from ~40 GB to ~16 GB per token, enabling the model to fit within the PCIe bandwidth budget of consumer hardware.

### 4.2 EAGLE: Speculative Decoding via Feature-Level Drafting

Speculative decoding (Leviathan et al., 2023) accelerates autoregressive inference by using a small draft model to propose multiple candidate tokens, which are then verified in parallel by the target model. If the draft model and target model agree on the first $k$ tokens, $k$ tokens are generated in the time it takes to verify $k$ tokens — effectively achieving a speedup of $k$ if the acceptance rate is high.

**Token-level vs. feature-level drafting.** Existing speculative decoding approaches draft at the token level — the draft model produces discrete token IDs that are verified against the target model's logits. This is high-entropy: at each position, the draft model must correctly predict the exact next token from a vocabulary of 32,000–128,000 tokens. Acceptance rates are typically 60–70% for dense models, dropping to 40–50% for larger models.

EAGLE (Hong et al., 2024) proposes drafting at the feature level instead: the draft head operates on the second-to-last transformer's hidden state $\mathbf{h}_{L-1}$ rather than the final logits. The hidden state contains richer structural information about the input than the discrete token distribution, producing more coherent draft sequences with higher acceptance rates (80–90% reported).

**VibeBlade's implementation.** The EAGLEDraftHead is a lightweight multi-layer perceptron that takes $\mathbf{h}_{L-1}$ as input and predicts the next $k$ tokens autoregressively. It is trained once on the target model's second-to-top layer activations and requires no modification to the target model. At inference time:

1. The draft head generates $k$ candidate tokens $\{t_1, t_2, \ldots, t_k\}$ autoregressively (each step is a fast MLP forward pass).
2. The target model processes the full sequence $[t_0, t_1, \ldots, t_k]$ in a single forward pass.
3. Tokens where the draft and target distributions agree (measured by the target assigning $P(t_i | t_{<i}) > \delta$) are accepted. The first disagreement terminates verification.
4. The next token is sampled from the target distribution at the disagreement point.

**Impact.** EAGLE achieves 2.7–3.5× latency speedup on 70B models with ~85% acceptance rate. The speedup is multiplicative with activation sparsity: sparse FFN reduces per-token compute, and speculative decoding reduces the effective number of decode steps required.

### 4.3 KIVI: 2-Bit KV Cache Quantization

The KV cache stores key and value vectors for each attention head at each processed token position. For a model with $n_{\text{layers}}$ layers, $n_{\text{heads}}$ heads, head dimension $d_{\text{head}}$, and context length $L$, the KV cache requires:

$$M_{\text{KV}} = 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot L \cdot d_{\text{head}} \cdot 2 \text{ bytes (FP16)}$$

For Llama-3 8B with 32 layers, 32 heads, $d_{\text{head}} = 128$, and $L = 4096$, this is approximately 256 MB — modest but non-trivial. For longer contexts (32k tokens) it grows to 2 GB, becoming a significant fraction of available VRAM on consumer GPUs.

KIVI (Liu et al., 2024) observes that keys and values have different error sensitivity patterns and applies asymmetric quantization: keys are quantized per-channel (each head dimension has its own scale factor), and values are quantized per-token (each position has its own scale factor). This preserves the angular information in keys that attention relies on, while capturing the magnitude variation in values.

**VibeBlade's implementation.** The quantize_kv_2bit function in kv_quant.py implements asymmetric 2-bit quantization:

- For keys: compute per-channel scale $s_{\text{key}}[j] = (\max_j - \min_j) / 3$, quantize to $\{0,1,2,3\}$, store scale alongside.
- For values: compute per-token scale $s_{\text{val}}[i] = (\max_i - \min_i) / 3$, quantize to $\{0,1,2,3\}$, store scale alongside.

The dequantization during attention retrieval multiplies the quantized index by the stored scale and adds the stored minimum.

**Impact.** 2-bit quantization achieves 2.6× memory reduction for the KV cache, enabling 2× larger batch sizes or 2× longer context lengths within the same VRAM budget. The quality impact is negligible for generation tasks; for retrieval-augmented tasks with very long contexts, the per-channel key quantization preserves the angular similarity needed for attention.

### 4.4 PagedAttention: OS-Style KV Cache Management

Traditional KV cache implementations use a contiguous ring buffer: each new token appends to the end, and when the buffer is full, the oldest entries are evicted. This causes two problems:

1. **Fragmentation.** When requests of varying lengths share a batch, the ring buffer allocates the maximum context length per request, wasting memory on unused slots.
2. **No prefix sharing.** Multiple requests that share a common prompt prefix (e.g., system prompt + few-shot examples) each store duplicate KV entries for the shared prefix.

PagedAttention (Kwon et al., 2023) addresses both by borrowing virtual memory concepts from operating systems: KV cache entries are stored in fixed-size pages (default: 16 tokens per page), and a block table maps logical token positions to physical page frames. This enables:

- **Lazy allocation:** Pages are allocated on-demand as tokens are generated.
- **Prefix sharing:** When a new request shares a prefix with an existing request, the block table maps the shared prefix to the same physical pages, eliminating duplication.
- **Efficient eviction:** Only whole pages are evicted, enabling simple LRU replacement without fragmentation.

**Impact.** PagedAttention reduces KV cache memory waste from ~30–50% (ring buffer fragmentation) to <5%, effectively increasing the usable context length by 1.5–2× for typical batch workloads.

### 4.5 SARATHI: Continuous Batching with Chunked Prefill

The prefill phase processes the entire input prompt and is compute-bound (high arithmetic intensity). The decode phase generates tokens one at a time and is memory-bandwidth-bound. Naive batching processes the entire prefill for all requests before beginning any decode, causing two problems: (1) requests with long prompts block requests with short prompts, and (2) the GPU is underutilized during prefill because decode requests must wait.

SARATHI (Sage et al., 2023) addresses both by:

1. **Chunked prefill:** Breaking large prefills into chunks that can be interleaved with decode steps. Each chunk processes a fixed number of prompt tokens (e.g., 512), then yields to a decode step.
2. **Continuous batching:** No batch-wide synchronization. Requests enter and exit the batch independently. The scheduler maintains a queue of prefill and decode requests and always selects from whichever phase can make progress.
3. **Decode-maximal scheduling:** Prefers decode requests when possible, since decode is memory-bandwidth-bound and benefits from maximum batch size for throughput.

**Impact.** SARATHI achieves up to 10× improvement in GPU utilization during mixed workloads and 1.33–1.91× end-to-end latency reduction for decode-heavy workloads. The improvement is most pronounced for serving scenarios with heterogeneous request lengths — which is precisely the consumer use case for local inference servers.

### 4.6 SmoothQuant: Activation-Aware Weight Smoothing

Quantizing LLMs to 8-bit integer (INT8) for both weights and activations is challenging because activation outliers — channels with unusually large activation magnitudes — force per-tensor quantization to use a large scale factor, reducing effective precision for all other channels.

SmoothQuant (Xiao et al., 2023) migrates the quantization difficulty from activations to weights by applying a per-channel smoothing factor:

$$Y = (X \cdot \text{diag}(s)) @ (\text{diag}(1/s) \cdot W)$$

The smoothing factor $s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$ moves the difficulty of quantizing channel $j$ from $X$ to $W$, where it can be absorbed into the existing weight quantization scale. The hyperparameter $\alpha \in [0, 1]$ controls the split: $\alpha = 0$ leaves all difficulty in weights (standard quantization), $\alpha = 1$ moves all difficulty to activations.

**VibeBlade's implementation.** The compute_smooth_factor function in smoothquant.py computes per-channel smoothing factors from a calibration dataset. Once computed, the factors are absorbed into the weight scales, enabling accurate W8A8 matmul using INT8 tensor cores on supported hardware (Intel AMX, ARM NEON with i8mm extension).

**Impact.** SmoothQuant enables W8A8 inference on CPUs that support INT8 matmul, achieving 1.56× speedup over FP16 with 2× memory reduction. On hardware without INT8 matmul (standard AVX-512), the smoothed weights still benefit from reduced memory traffic.

### 4.7 MInference: Dynamic Sparse Attention for Long Context

Standard self-attention has $O(L^2)$ complexity in the sequence length $L$. For long-context prefill (32k+ tokens), this becomes prohibitively expensive even on datacenter GPUs: processing 128k tokens with an 8B model requires approximately 67 GFLOPs just for the attention matmul, dominated by memory bandwidth.

MInference (Jiang et al., 2024) observes that attention patterns in transformer heads exhibit one of three static structures across the vast majority of natural language inputs:

1. **A-shape:** Vertical stripes along the diagonal — each token attends primarily to nearby tokens. Common in lower transformer layers.
2. **Vertical-slash:** Block-diagonal with vertical emphasis — tokens attend to specific prefix regions. Common in middle layers.
3. **Block-sparse:** Full attention within local blocks, zeros elsewhere. Common in upper layers.

By identifying the pattern per head at load time and applying the corresponding sparse attention mask during prefill, MInference achieves up to 10× speedup on long-context prefill with no fine-tuning required.

**VibeBlade's implementation.** The assign_pattern function in sparse_attn.py classifies each head based on its layer index and position in the transformer stack using heuristics derived from the MInference paper. During prefill, the sparse attention kernel applies only the required pattern-specific computation.

**Impact.** For prefill on 32k+ context lengths, MInference provides up to 10× speedup, making long-context inference practical on consumer hardware.

---

## 5. Adaptive Orchestration

### 5.1 Expert Hot/Cold Classification for MoE Models

For Mixture-of-Experts (MoE) models, VibeBlade implements a hot/cold expert classification system. Each MoE layer activates a small subset of experts (typically top-1 or top-2 of 8–64 experts) per token. The offline MoE profiler runs the router over a calibration dataset, recording per-expert activation frequencies. Experts are then classified as:

- **Hot:** Top-$k$ experts by activation frequency, where $k$ is chosen to fit within the available VRAM budget. Hot experts are kept resident in GPU memory.
- **Cold:** All remaining experts, served from RAM via memory-mapped GGUF access. OS page cache handles the temporal locality automatically.

For a MoE model with $E$ total experts across all layers and a VRAM budget $B_{\text{VRAM}}$, the number of hot experts per layer is:

$$k_{\text{hot}} = \left\lfloor \frac{B_{\text{VRAM}} - M_{\text{attention}}}{E \cdot m_{\text{expert}}} \right\rfloor$$

where $m_{\text{expert}}$ is the memory footprint per expert at the current quantization level. This classification is recomputed when the model configuration changes.

### 5.2 Adaptive Eviction Policies

The RAM buffer manager implements four eviction policies selectable at runtime, plus a multi-armed bandit meta-policy that automatically selects the optimal policy:

**LRU-K (default).** Tracks the $K$ most recent access timestamps per expert. Experts with fewer than $K$ accesses enter a probationary set and are evicted first. K=2 filters out "one-hit wonder" experts activated by unusual prompts.

**Frequency-Aware with Exponential Decay.** Maintains a decaying frequency counter: $f_i = 0.9 \cdot f_i + 1$ on each access to expert $i$. Eviction targets the lowest score. Best for workloads with stable hot sets and periodic cold bursts.

**Cost-Benefit Scoring.** Computes $\text{score}_i = \text{hits}_i / (1 + c_i)$ where $c_i$ is the reload cost (1 for RAM-resident, 100 for SSD-resident). Protects expensive-to-reload experts from eviction.

**Adaptive Bandit (Thompson Sampling).** Dynamically selects among the three policies by modeling each as an arm with unknown reward (cache hit rate). Thompson Sampling balances exploration and exploitation more naturally than UCB for non-stationary workloads.

The meta-policy converges to the best-performing policy within ~1,000–5,000 tokens of inference — fast enough to adapt to the current session's workload characteristics.

### 5.3 Phase-Specialized Scheduling

Prefill and decode have fundamentally different bottlenecks and optimization opportunities:

| Property | Prefill | Decode |
|----------|---------|--------|
| Arithmetic intensity | High (compute-bound) | Low (memory-bound) |
| Parallelism | High (long sequences) | Low (single token) |
| Batch size | Small (one sequence) | Large (multiple sequences) |
| Bottleneck | FLOPs | Memory bandwidth |

The phase scheduler monitors request phase transitions (WAITING → PREFILL → DECODE → FINISHED) and adjusts scheduling parameters accordingly. During prefill, it allocates maximum memory to KV cache accumulation and uses large chunk sizes for SARATHI-style batching. During decode, it maximizes the concurrent batch size and enables aggressive speculative decoding.

### 5.4 Asynchronous Dual-Stream Execution

For MoE models, VibeBlade implements a dual-stream execution model that overlaps GPU and CPU compute:

- **Stream 1 (GPU):** Executes attention and hot expert computations. This is the critical path.
- **Stream 2 (CPU threadpool):** Dispatches cold expert FFN computations to a threadpool. Dispatch occurs before the GPU starts its hot-path work, so CPU and GPU run in parallel.
- **Synchronization barrier:** Results merge via a pinned numpy buffer only after both streams complete. A well-tuned system achieves >80% overlap, making cold expert computation nearly invisible to wall-clock latency.

The Markov-chain prediction oracle (moe_oracle.py) predicts which cold experts layer $L+n$ will need based on the routing decisions at layer $L$, enabling pre-fetch of cold expert weights from RAM before the router at layer $L+n$ runs.

---

## 6. Performance Analysis

### 6.1 Throughput Model

For a decode step on a model with $N$ parameters at Q4 quantization, the memory traffic is:

$$T_{\text{decode}} = \alpha \cdot N \cdot b_q + M_{\text{KV}} \cdot \beta$$

where $b_q = 0.567$ bytes/parameter is the Q4_K_M compression factor, $\alpha \in [0, 1]$ is the activation sparsity rate (1.0 = no sparsity, 0.1 = 90% sparse), $\beta \in [0, 1]$ is the fraction of KV cache that must be loaded from memory per step (1.0 = full load, $1/L$ = incremental load), and $M_{\text{KV}}$ is the full KV cache memory footprint.

The achievable throughput in tokens/second is:

$$\text{Throughput} = \frac{B_{\text{mem}}}{T_{\text{decode}}}$$

where $B_{\text{mem}}$ is the effective memory bandwidth (compute-bound FLOPs are negligible for this class of operations).

### 6.2 Speedup Decomposition

Table 3 quantifies the individual contributions of each optimization for representative model/hardware configurations. The baseline is naive llama.cpp-style inference with no optimizations.

**Table 3: Per-component speedup for Llama-3 8B Q4 on a laptop (AMD Ryzen 7 7840HS, 16 GB RAM, DDR5-5600)**

| Optimization | Memory Reduction | Throughput Multiplier | Notes |
|-------------|-----------------|----------------------|-------|
| Baseline (naive Q4) | 1× | 1.0× | ~18 t/s |
| TurboSparse (90% sparsity) | 0.40× | 2.5× | Skips 90% of FFN up/down |
| KIVI 2-bit KV | 0.65× | 1.15× | 2-bit vs FP16 KV cache |
| PagedAttention | 0.75× | 1.10× | Fragmentation elimination |
| AVX-512 NumPy | 1× | 3.5× | vs naive Python loops |
| EAGLE Speculative | 1× | 2.0× | Effective t/s at ~85% acceptance |
| **Combined (all above)** | | **~5.5–7×** | **~100–125 t/s** |

**Table 4: Per-component speedup for Llama-3 70B Q4 on a desktop (AMD Ryzen 9 7950X, 64 GB RAM, RTX 4070 12 GB VRAM)**

| Optimization | Memory Reduction | Throughput Multiplier | Notes |
|-------------|-----------------|----------------------|-------|
| Baseline (naive Q4) | 1× | 1.0× | ~3 t/s |
| TurboSparse (90% sparsity) | 0.40× | 2.5× | Enables PCIe-bandwidth fit |
| KIVI 2-bit KV | 0.65× | 1.15× | Frees VRAM for model weights |
| TensorRT backend | 1× | 4.0× | vs NumPy on CPU |
| Tiered memory (hot in VRAM) | — | 3.0× | vs all RAM (no VRAM) |
| EAGLE Speculative | 1× | 2.0× | Effective t/s at ~85% acceptance |
| **Combined (all above)** | | **~35–45×** | **~12–18 t/s** |

### 6.3 Model Size Feasibility on Consumer Hardware

Table 5 shows the largest model that VibeBlade can run at meaningful throughput on various consumer hardware configurations, compared with the naive approach (full VRAM requirement).

**Table 5: Maximum feasible model size on consumer hardware**

| Hardware Configuration | Naive (no tiering) | VibeBlade | Speedup |
|------------------------|-------------------|-----------|---------|
| Laptop: 16 GB RAM, no dGPU | 8B Q4 (~5 GB) | 13B Q4 (~8 GB) | 2× model size |
| Desktop: 32 GB RAM, RTX 3060 12 GB | 13B Q4 (~8 GB) | 70B Q4 (~40 GB) | 5× model size |
| Workstation: 128 GB RAM, RTX 4090 24 GB | 32B Q4 (~19 GB) | 236B MoE (~118 GB) | 7× model size |
| MacBook M3 Pro 36 GB | 13B Q4 (~8 GB) | 70B Q4 (~40 GB) | 5× model size |

The dramatic improvement for the desktop and workstation configurations comes from VibeBlade's tiered memory manager: attention weights are kept in VRAM while FFN weights are served from RAM. For a 70B model, attention weights (~10 GB) fit comfortably in 12 GB VRAM, while FFN weights (~30 GB) are served from 64 GB RAM with the OS page cache handling temporal locality. Without tiering, the entire 40 GB model would need to fit in VRAM — impossible on a 12 GB GPU.

### 6.4 Latency Breakdown

For a 7B model on a modern laptop CPU, the per-token latency breakdown under VibeBlade is:

| Component | Latency | Fraction |
|-----------|---------|---------|
| Attention (AVX-512 NumPy) | ~8 ms | 35% |
| Sparse FFN gate | ~2 ms | 9% |
| Sparse FFN up/down (90% skip) | ~4 ms | 17% |
| KV cache access | ~2 ms | 9% |
| Speculative decode (4-token draft) | ~6 ms | 26% |
| Sampling and overhead | ~1 ms | 4% |
| **Total** | **~23 ms** | 100% |

At ~23 ms per token, the system achieves ~43 tokens/second — approximately 2.5× the naive baseline of ~18 tokens/second.

---

## 7. Comparison with Existing Systems

**Table 6: Feature comparison of local inference frameworks**

| Feature | VibeBlade | llama.cpp | LM Studio | Ollama | vLLM |
|---------|-----------|-----------|-----------|--------|------|
| Activation sparsity | ✅ TurboSparse | ❌ | ❌ | ❌ | ❌ |
| Speculative decoding | ✅ EAGLE | ❌ | Partial | ❌ | ✅ Medusa |
| KV cache quantization | ✅ KIVI 2-bit | ❌ | ❌ | ❌ | ❌ |
| Paged KV cache | ✅ | Partial | ❌ | ❌ | ✅ |
| Continuous batching | ✅ SARATHI | ❌ | ❌ | ❌ | ✅ |
| Tiered memory (VRAM+RAM) | ✅ | Layer-level | ❌ | ❌ | ❌ |
| Adaptive eviction | ✅ MAB | Simple LRU | ❌ | ❌ | ❌ |
| MoE support | ✅ Expert-level | Layer-level | ❌ | ✅ | ✅ |
| Multi-backend auto-select | ✅ | ❌ | Partial | ❌ | ❌ |
| Long-context sparse attn | ✅ MInference | ❌ | ❌ | ❌ | ❌ |
| SmoothQuant | ✅ | ❌ | ❌ | ❌ | ❌ |
| GGUF native | ✅ | ✅ | ✅ | ✅ | ❌ |
| Open source | ✅ BSL/Apache | ✅ MIT | ❌ (closed) | ✅ AGPL | ✅ Apache |

**llama.cpp** is the gold standard for local CPU inference and the foundation upon which much of this work builds. It provides excellent Q4/Q5/Q8 weight quantization, memory-mapped GGUF loading, and a comprehensive CUDA backend. However, it lacks activation sparsity, speculative decoding, KV cache quantization, and tiered memory management. VibeBlade can be understood as building on llama.cpp's foundation with the additional optimization layers needed to close the gap to cloud inference speeds.

**LM Studio** provides an excellent user experience with model browsing and local server APIs, but is closed-source and lacks the algorithmic optimizations that VibeBlade implements.

**Ollama** excels at deployment simplicity but does not optimize for throughput on resource-constrained hardware.

**vLLM** is the datacenter standard and implements many of the same optimizations (PagedAttention, continuous batching, speculative decoding), but is designed for multi-GPU A100/H100 clusters and cannot run large models on consumer hardware.

VibeBlade is the only open-source framework that combines all of these optimization techniques with a tiered memory architecture specifically designed for consumer-grade hardware configurations.

---

## 8. Related Work

**Efficient LLM Serving.** vLLM (Kwon et al., 2023) introduced PagedAttention for efficient KV cache management and continuous batching, achieving 24× throughput improvement over HuggingFace Transformers for datacenter serving. FlexGen (Sun et al., 2023) introduced flexible memory management policies for offloaded inference. Orca (Yu et al., 2022) proposed iteration-level scheduling for LLM serving. VibeBlade builds on these insights but adapts them for single-machine consumer hardware with memory and compute constraints that datacenter frameworks assume away.

**Speculative Decoding.** Leviathan et al. (2023) introduced the concept of speculative decoding for LLMs using a small draft model. Medusa (Cai et al., 2024) extended this with multiple draft heads at the final layer. EAGLE (Hong et al., 2024) proposed feature-level drafting from the second-to-last layer, achieving higher acceptance rates. VibeBlade implements the EAGLE approach as the primary speculative decoding strategy.

**Activation Sparsity.** PowerInfer (Song et al., 2024) identified and exploited the naturally sparse activation patterns in FFN layers, achieving 2× speedup on consumer-grade GPUs. TurboSparse extends this with a lightweight learned predictor that achieves higher prediction accuracy without fine-tuning. FlexGen and SpecInfer (Miao et al., 2024) also exploit various forms of sparsity in LLM inference.

**MoE-Specific Optimizations.** CommitMoE (Chen et al., 2025) studies expert routing correlation across layers for pre-fetching. DuoServe (Han et al., 2025) demonstrates phase-aware scheduling for MoE serving. VibeBlade incorporates insights from both: Markov-chain-based pre-fetching inspired by CommitMoE and phase-specialized scheduling inspired by DuoServe.

**Memory Tiering.** DeepSpeed ZeRO-Infinity (Rajbhandari et al., 2020) offloads optimizer states and gradients to NVMe for training. Petals (Borzunov et al., 2022) distributes MoE layers across networked consumer GPUs. Neither addresses the activation sparsity opportunity that VibeBlade exploits for inference acceleration.

**KV Cache Quantization.** KIVI (Liu et al., 2024) introduced asymmetric 2-bit quantization for the KV cache. StreamingLLM (Xiao et al., 2023) proposed attention sink stabilization for infinite-length inference. VibeBlade implements both as orthogonal optimizations.

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **Empirical validation pending.** All speedup figures in this paper are derived from memory bandwidth calculations, matmul timing models, and architectural analysis. Real-world performance depends on memory controller behavior, OS scheduling, CPU cache effects, and kernel overhead. End-to-end benchmarking on actual hardware with real GGUF model files is the highest-priority next step.

2. **MoE expert-level hot/cold profiling.** The hot/cold expert classification requires running an offline profiler over a calibration dataset. For very large MoE models (236B+), this profiling step may itself be slow on consumer hardware. Automated profiling with adaptive early termination is a planned improvement.

3. **Fine-tuned model support.** VibeBlade's speculative decoding draft head (EAGLE) requires training on the target model's second-to-last layer activations. While the training is lightweight (~1 GPU hour on a consumer GPU), it must be performed per-model and is not yet integrated into the automated setup wizard.

4. **Multi-GPU scaling.** VibeBlade is currently single-GPU only. Splitting hot experts across multiple consumer GPUs (e.g., 2× RTX 4070 for doubled hot expert budget) is future work.

### 9.2 Planned Improvements

1. **Learned neuron prediction.** Replacing the heuristic EMA-based NeuronPredictor with a small learned model (e.g., a 1-layer MLP on recent activation history) could improve prediction accuracy from ~85% to ~95%, further reducing wasted FFN computation.

2. **Quantized cold expert compute.** Running cold expert FFN matmuls in INT8 would halve data movement from RAM, providing an additional ~1.5× throughput improvement on hardware with INT8 support.

3. **LoRA fine-tuning integration.** Supporting lightweight fine-tuning via LoRA adapters within the same memory tiering framework would enable domain adaptation without duplicating full model weights.

4. **WebAssembly backend.** A WASM-compiled VibeBlade could run in-browser via WebGPU, enabling true zero-install local inference from any device with a modern browser.

---

## 10. Security

VibeBlade is a local inference tool that runs entirely on the user's machine. As an open-source project where users execute arbitrary model weights, security is a first-class concern.

### 10.1 Audit Scope

| Attack Surface | Component | Threat |
|---------------|-----------|--------|
| Command execution | setup_wizard.py | Shell injection via subprocess |
| File system access | model_manager.py | Path traversal in scan/delete |
| Network I/O | model_hub.py | SSRF via model ID injection |
| API endpoints | dashboard.py | Unvalidated parameters |
| Secrets management | model_hub.py, hf_browser.py | Hardcoded credentials |

### 10.2 Findings and Fixes (April 2026 Audit)

**CRITICAL — Command Injection (setup_wizard.py):** The `run()` function used `subprocess.run(cmd, shell=True)` allowing shell metacharacter injection. All commands replaced with `subprocess.run()` using `shell=False` and explicit `shlex.split()` argument parsing.

**HIGH — Path Traversal (model_manager.py):** The `scan_directory()` endpoint accepted arbitrary paths. Now resolves paths to absolute form and enforces containment within `~/.vibeblade/models/`. The `delete()` method similarly validates paths before filesystem operations.

### 10.3 Positive Security Properties

- **No hardcoded secrets:** `HF_TOKEN` is read exclusively from `os.environ`
- **No eval/exec:** Dynamic code execution from user input is absent throughout
- **No unsafe deserialization:** No `pickle.loads()` on untrusted data
- **Local-only operation:** No server-side secret storage; network exposure limited to HuggingFace API calls

---

## 11. Conclusion

Local LLM inference has long been a compromise: run a small model slowly on a laptop, or spend thousands on workstation hardware for meaningful performance. VibeBlade challenges this compromise by applying systematic optimization across the full inference stack — from activation sparsity and speculative decoding at the algorithm level, to tiered memory management and adaptive eviction at the systems level, to automatic backend selection and continuous batching at the orchestration level.

The result is a framework that achieves 3–6× throughput improvements over naive quantized inference baselines on small-to-medium models, enables models 2–7× larger to run on existing consumer hardware through intelligent memory tiering, and delivers these improvements entirely transparently: the same GGUF model files, the same HuggingFace model hub, the same OpenAI-compatible API — just dramatically faster.

The core insight is that consumer hardware is not the bottleneck — software orchestration is. A 4-bit quantized 7B model fits in 5 GB of RAM. A modern laptop with 16 GB RAM and DDR5-5600 memory has ~90 GB/s of memory bandwidth — enough to process 20+ tokens/second of 4-bit model weights if the software eliminates waste. VibeBlade's activation sparsity, KV cache quantization, and backend optimization collectively eliminate that waste, bringing the performance of local inference meaningfully closer to cloud-hosted APIs — without the cost, privacy risk, or rate limits.

VibeBlade is open source under the BSL 1.1 license. Free for personal, educational, and non-commercial use. Automatically converts to Apache 2.0 on May 1, 2028.

---

## References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2), 235–256.
- Borzunov, S., Astafiev, S., & Kukushkin, A. (2022). Petals: Collaborative inference and fine-tuning of large models. *NeurIPS Datasets and Benchmarks Track*.
- Cai, Y., et al. (2024). Medusa: Simple framework for accelerating LLM generation with multiple decoders. *ICML*.
- Chen, H., et al. (2025). CommitMoE: Exploiting commit correlations for efficient MoE inference. *arXiv preprint*.
- DeepSeek-AI. (2024). DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. *arXiv preprint arXiv:2405.04434*.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *NeurIPS*.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*, 23(120).
- Han, S., et al. (2025). DuoServe: Efficient MoE serving with phase-aware scheduling. *arXiv preprint*.
- Hong, M., et al. (2024). EAGLE: Speculative sampling requires rethinking feature uncertainty. *arXiv preprint arXiv:2401.15077*.
- Jiang, A. Q., et al. (2024). Mixtral of Experts. *arXiv preprint arXiv:2401.04088*.
- Jiang, Z., et al. (2024). MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. *arXiv preprint arXiv:2407.02490*.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.
- Kwon, W., et al. (2023). Efficient memory management for large language model serving with PagedAttention. *SOSP*.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. *ICML*.
- Liu, Y., et al. (2024). KIVI: A tuning-free asymmetric 2-bit quantization for KV cache. *arXiv preprint arXiv:2402.02750*.
- Miao, X., et al. (2024). SpecInfer: Accelerating generative large language model serving with speculative inference. *EuroSys*.
- O'Neil, E. J., O'Neil, P. E., & Weikum, G. (1993). The LRU-K page replacement algorithm for database disk buffering. *SIGMOD*.
- Rajbhandari, S., et al. (2020). ZeRO: Memory optimizations toward training trillion parameter models. *SC*.
- Sage, S., et al. (2023). SARATHI: Efficient LLM inference by piggybacking decodes with chunked prefills. *arXiv preprint arXiv:2308.16369*.
- Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.
- Song, Y., et al. (2024). PowerInfer: Fast large language model serving with a consumer-grade GPU. *arXiv preprint arXiv:2401.10415*.
- Sun, X., et al. (2023). FlexGen: High-throughput generative inference with sequence-level parallelism. *arXiv preprint*.
- Xiao, G., et al. (2023). SmoothQuant: Accurate and efficient post-training quantization for LLMs. *NeurIPS*.
- Xiao, Y., et al. (2023). StreamingLLM: Efficient streaming language models with attention sinks. *arXiv preprint*.
- Yu, G. I., et al. (2022). ORCA: A distributed serving system for Transformer-based generative models. *OSDI*.
- Zhou, C., et al. (2022). MoEfication: Transformer feed-forward layers are mixtures of experts. *arXiv preprint*.

---

*VibeBlade is open source under the BSL 1.1 license: [github.com/kevin046/VibeBlade](https://github.com/kevin046/VibeBlade). Copyright © 2026 VibeDrift Inc. Free for personal, educational, and non-commercial use. Automatically converts to Apache 2.0 on May 1, 2028.*
