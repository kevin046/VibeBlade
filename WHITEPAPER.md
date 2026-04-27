# Architectural Synchronization of Structural Sparsity and Memory Orchestration: A Systems-Level Investigation into VibeBlade for Local Frontier Inference

**VibeDrift Inc.** · April 2026

---

The profound evolution of large language model (LLM) serving in 2026 has been marked by a widening chasm between the capabilities of centralized cloud providers and the practical constraints of consumer-grade local hardware. While the industry has seen a massive expansion in parameter counts, reaching the 1-trillion mark with models like Kimi K2.6, the hardware accessible to the average researcher or privacy-conscious developer remains constrained by memory bandwidth and capacity. [1] The prevailing architecture for local serving, exemplified by foundational frameworks such as llama.cpp, has historically relied upon static quantization and serialized execution, leaving significant performance gains on the table. [2] This report provides a professional, doctoral-level analysis of the VibeBlade inference engine, an integrated systems-level solution that redefines the feasibility frontier of local LLM inference through a sophisticated orchestration of activation sparsity, feature-level speculative decoding, and adaptive memory tiering. [2]
The fundamental thesis of VibeBlade is that the primary bottleneck in modern local inference is not a deficit in raw computational cycles, but rather a catastrophic failure in the orchestration of data movement across memory tiers. [2] By treating inference as a systems-orchestration problem rather than a simple mathematical execution problem, VibeBlade implements five critical optimizations: TurboSparse activation prediction, EAGLE speculative decoding, KIVI 2-bit Key-Value (KV) cache quantization, SARATHI-style continuous batching, and an adaptive memory manager. [2] This multi-layered approach addresses the inherent inefficiencies of the transformer architecture, where 90% of feed-forward network (FFN) activations are structurally unnecessary and memory-mapped weight files often sit idle in slow system RAM while the GPU waits for data transfers. [2]

## Theoretical Framework: Memory-Bandwidth Limits and Structural Waste

To understand the necessity of the VibeBlade architecture, one must first formalize the autoregressive inference process as a memory-bandwidth problem. Autoregressive decoding alternates between a compute-bound prefill phase and a memory-bandwidth-bound decode phase. [2] During the decode phase, generating a single token requires the system to load the entire weight matrix of the model to perform a single matrix-vector multiplication. [2] The arithmetic intensity—the ratio of floating-point operations (FLOPs) to memory traffic (bytes)—is exceptionally low, typically around 0.86 FLOPs/byte for a 4-bit quantized 7B model. [2]
For a dense transformer layer using a hidden state vector $\mathbf{h}$, the standard FFN forward pass computes the following:

$$\text{FFN}(\mathbf{h}) = \sigma(\mathbf{W}_{\text{gate}} \cdot \mathbf{h}) \odot (\mathbf{W}_{\text{up}} \cdot \mathbf{h})$$

where $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times 4d}$. [2] At a 4-bit quantization level (Q4\_K\_M), each weight matrix requires 0.567 bytes per parameter, inclusive of scale and zero-point overheads. [2] In a 7B model configuration where $d = 4096$ and $n_{\text{layers}} = 32$, this results in approximately 4.3 GB of weight traffic per token. [2] This volume of data movement is untenable for standard consumer hardware without significant structural intervention. [6]
The inefficiency is further compounded by the observation that in Llama-family architectures, the SiLU activation gate effectively zeroes out between 88% and 92% of FFN neurons per token. [2] This means that the down-projection matrix $\mathbf{W}_{\text{down}}$ is being loaded in its entirety to multiply by a vector that is 90% sparse. [2] VibeBlade addresses this by formalizing a sparse FFN forward pass:

$$\text{FFN}_{\text{sparse}}(\mathbf{h}) = \sigma(\mathbf{W}_{\text{gate}} \cdot \mathbf{h}) \odot (\mathbf{W}_{\text{up}} \cdot (\mathbf{h} \odot \mathbf{m}))$$

where $\mathbf{m}$ is a binary activation mask. [2] If the mask $\mathbf{m}$ can be predicted before the full computation, the effective weight traffic per token drops from 4.3 GB to 1.7 GB, enabling a theoretical 2.5-fold increase in throughput. [2]

**Table 1:** Comparative Hardware Memory Bandwidth and Inference Potential (2026)

| Hardware Configuration | Memory Type | Bandwidth (GB/s) | Peak 7B Q4 Decode (t/s) | Memory Ceiling (GB) |
| --- | --- | --- | --- | --- |
| NVIDIA RTX 5090 | GDDR7 | 1,792 | ~410 | 32 |
| NVIDIA RTX 4090 | GDDR6X | 1,008 | ~230 | 24 |
| Apple M4 Ultra | Unified | 819 | ~190 | 512 |
| Apple M4 Max | Unified | 546 | ~125 | 128 |
| AMD Strix Halo | LPDDR5X | 212 | ~48 | 128 |
| Laptop (Ryzen AI MAX+) | DDR5-5600 | 90 | ~20 | 16 |

The TurboSparse module within VibeBlade represents a critical departure from traditional "ReLUfication" efforts. While early research into activation sparsity required expensive fine-tuning to replace SiLU with ReLU, TurboSparse utilizes a learned predictor that functions atop the existing model weights. [2] The system identifies that contextual sparsity—where specific neurons are activated only for specific semantic inputs—is a fundamental property of transformer-based LLMs. [10]
The technical implementation of TurboSparse hinges on the NeuronPredictor, which maintains a sliding window of recent token activations. [2] It calculates an exponential moving average (EMA) of activation magnitudes per neuron:

$$\bar{a}_j^{(t)} = \alpha \cdot a_j^{(t)} + (1 - \alpha) \cdot \bar{a}_j^{(t-1)}$$

The prediction mask $\mathbf{m}$ is then generated using a calibrated threshold $\theta$. [2] The precision and recall of this predictor are paramount; empirical tests on Llama-3 8B show 85% precision and 90% recall, meaning that only 10% of active neurons are missed, while 90% of the structurally unnecessary computation is successfully skipped. [2] This approach is significantly more robust than magnitude-based clipping (e.g., TEAL), which achieves only 40-50% sparsity before incurring substantial perplexity degradation. [12]
A secondary innovation in TurboSparse is the use of the dReLU (double ReLU) activation function during pre-training or lightweight recovery training. [3] The dReLU function is defined as:

$$\text{dReLU}(x) = \max(0, x) \cdot \max(0, -x)$$

By applying ReLU after both the gate and up-projections, the system masks negative activations in both components, achieving up to 97% parameter sparsity in Mixture-of-Experts models without sacrificing reasoning performance. [3] This level of sparsity is transformative for mobile and edge deployments, allowing models with 47B parameters to run on a smartphone at nearly 12 tokens per second by activating only 4.3 billion parameters per iteration. [14]

## Speculative Decoding: The EAGLE Evolution and Feature Fusion

Speculative decoding has emerged as the standard for breaking the sequential nature of autoregressive generation. [16] The mechanism involves a fast draft model proposing $k$ candidate tokens, which the slower target model then verifies in a single parallel forward pass. [16] However, VibeBlade rejects the naive two-model approach (e.g., using a 1B model to draft for a 70B model) because the distribution mismatch between independent models leads to low token acceptance rates, often below 60%. [20]
VibeBlade instead implements the EAGLE (Extrapolation Algorithm for Greater Language-Model Efficiency) framework, which utilizes feature-level drafting. [22] By mid-2025, the EAGLE family evolved through three distinct iterations, each increasing the speedup ratio and acceptance rate:

**Table 2:** Technical Comparison of Speculative Decoding Architectures (2026)

| Architecture | Information Source | Mechanism | Speedup (70B) | Acceptance Rate |
| --- | --- | --- | --- | --- |
| Vanilla SD | Separate Model | Token-level prediction | 1.3x - 1.8x | 50% - 65% |
| Medusa | Last Layer | Independent parallel heads | 1.5x - 2.2x | 60% - 70% |
| EAGLE-1 | Second-to-top Layer | Single-layer feature autoregression | 2.0x - 2.5x | 75% - 80% |
| EAGLE-3 | Tri-layer Fusion | Multi-layer feature fusion | 2.7x - 3.5x | 80% - 85% |
| ConFu (2026) | Future Intent | Contemplate tokens + MoE | 3.0x - 4.1x | 85% - 92% |
| Acceptance Rate | Vanilla SD | Separate Model | Token-level prediction | 1.3x - 1.8x |
| 50% - 65% | Medusa | Last Layer | Independent parallel heads | 1.5x - 2.2x |
| 60% - 70% | EAGLE-1 | Second-to-top Layer | Single-layer feature autoregression | 2.0x - 2.5x |
| 75% - 80% | EAGLE-3 | Tri-layer Fusion | Multi-layer feature fusion | 2.7x - 3.5x |
| 80% - 85% | ConFu (2026) | Future Intent | Contemplate tokens + MoE | 3.0x - 4.1x |
| 85% - 92% | The technical brilliance of EAGLE-3 lies in its tri-layer feature fusion.23 Instead of relying solely on the final hidden state, the EAGLE head fuses representations from early layers (syntax/local context), middle layers (semantic structure), and late layers (output probabilities).23 This holistic view allows the draft head to predict the target's output more accurately because it understands why a particular token is being generated at multiple levels of abstraction.23 This method achieves up to a 6.5x speedup on academic benchmarks and typically provides a 2.7-3.5x effective latency reduction on consumer setups running 70B models.22 |  |  |  |

The latest research integrated into VibeBlade involves the ConFu framework, which addresses the "error accumulation" bottleneck of current speculators. [24] Existing draft models condition only on the current prefix, causing predictions to drift from the target distribution as the draft sequence lengthens. [24] ConFu introduces "contemplate tokens"—latent reasoning vectors that allow the draft model to anticipate the target model's predicted semantic trajectory. [24] Experiments on Qwen-3 4B show that ConFu improves acceptance rates by 20% over EAGLE-3, setting a new state-of-the-art for local speculative serving. [24]

## KV Cache Innovation: Quantization and Paged Management

The Key-Value (KV) cache is identified as the dominant memory bottleneck for long-context inference in 2026.27 For models like Llama 4 Scout, which supports a 10-million token context window, the KV cache footprint grows linearly, quickly exceeding the VRAM capacity of any single GPU. [28] Serving 32 requests with 128k sequences on a 70B model requires over 1.2 TB of KV storage, an order of magnitude larger than the model weights themselves. [30]
VibeBlade addresses this scaling crisis through KIVI (Key-Value-Inner-quantization), a tuning-free asymmetric 2-bit quantization scheme. [2] The system's design is informed by the discovery that keys and values possess distinct element distributions: keys exhibit channel-wise outliers, whereas values are more sensitive to positional token variance. [31] Consequently, KIVI quantizes the key cache per-channel and the value cache per-token. [31] This preserves the angular similarity critical for attention accuracy while achieving a 2.6x reduction in peak memory usage. [31]
In 2026, VibeBlade integrated the RotateKV technique to further enhance extreme low-bit stability. [32] RotateKV utilizes a block-diagonal Hadamard rotation to smooth out channel-wise outliers before quantization. [32] By mixing elements across channels, the rotation suppresses the disproportionately large magnitudes that otherwise force coarse-grained quantization scales. [32] This innovation allows RotateKV to maintain less than 0.3 perplexity degradation with 2-bit quantization on complex reasoning tasks, outperforming KIVI in long-context scenarios. [32]

**Table 3:** Performance of KV Cache Compression Techniques (Llama-3 70B)

| Technique | Precision | Memory Reduction | Accuracy Loss (PPL) | Context Cap (24GB VRAM) |
| --- | --- | --- | --- | --- |
| Naive FP16 | 16-bit | 1.0x | 0.0 | ~8k |
| INT8 Quant | 8-bit | 2.0x | <0.1 | ~16k |
| KIVI 2-bit | 2-bit | 2.6x | 0.5 - 0.8 | ~40k |
| RotateKV 2-bit | 2-bit | 3.97x | 0.2 - 0.3 | ~100k |
| Kitty (Mixed) | 2/4-bit | 8.0x | <0.1 | ~250k+ |
| Context Cap (24GB VRAM) | Naive FP16 | 16-bit | 1.0x | 0.0 |
| ~8k | INT8 Quant | 8-bit | 2.0x | <0.1 |
| ~16k | KIVI 2-bit | 2-bit | 2.6x | 0.5 - 0.8 |
| ~40k | RotateKV 2-bit | 2-bit | 3.97x | 0.2 - 0.3 |
| ~100k | Kitty (Mixed) | 2/4-bit | 8.0x | <0.1 |
| ~250k+ | The development of the "Kitty" framework in late 2025 further pushed the boundary by decomposing Key pages into unified 2-bit tensors with dynamic 4-bit precision boosts for sensitive channels.30 This system provides a page-centric layout with Triton-compatible dequantization kernels, enabling VibeBlade to support 8x larger batch sizes under the same memory budget.30 |  |  |  |

Complementing these quantization techniques is VibeBlade's PagedAttention implementation, which treats the KV cache like virtual memory. [2] Traditional implementations suffer from 60-80% memory waste due to internal and external fragmentation. [36] PagedAttention allocates physical memory pages (16 tokens per page) only on-demand and maps logical token positions to these frames via a block table. [2] This architecture enables prefix sharing, where common prompts for multi-agent workflows are stored once and accessed by all concurrent requests, reducing memory overhead by an additional 30-50% in conversational serving scenarios. [2]

## Orchestrating Mixed-Phase Workloads: SARATHI and SageSched

A primary deficiency in previous local inference stacks was the serialized processing of prefill and decode phases. [2] Prefill is high-FLOP and compute-bound, whereas decode is low-FLOP and memory-bound. [2] If processed sequentially, the GPU remains underutilized during the memory-bound phase and blocked during long prefill tasks. [2]
VibeBlade implements SARATHI, which utilizes chunked prefill to interleave prefill tokens with decode steps. [2] By splitting a 1024-token prompt into two 512-token chunks, SARATHI can process one chunk, generate a decode token for an ongoing request, and then process the second chunk. [41] This "stall-free scheduling" maximizes GPU saturation and reduces pipeline bubbles by 6.29x. [5]
The 2026 iteration of VibeBlade adds SageSched, an uncertainty-aware scheduler designed for heterogeneous workloads. [42] SageSched addresses the non-deterministic nature of LLM generation, where the output length is unknown beforehand. [42] By using prompt metadata and past results to predict output distributions, SageSched models the true service cost of each request. [42] It prevents head-of-line blocking by dynamically adjusting priorities, ensuring that short, latency-sensitive requests are not trapped behind massive batch jobs. [42] This results in a 28.7% improvement in Time-to-Last-Token (TTLT) efficiency compared to vLLM's first-come-first-serve scheduler. [42]

## Adaptive Memory Tiering and Hardware Selection

VibeBlade’s Tiered Memory Manager (TMM) is designed to maximize the utility of the heterogeneous memory resources found in consumer hardware. [2] Unlike server-grade hardware with uniform High Bandwidth Memory (HBM), a typical consumer desktop features a high-speed but small VRAM pool and a large but slow system RAM pool. [2]
The TMM operates a three-tier hierarchy:
1. Hot Tier (VRAM): Stores attention projections, hot KV cache pages, and the router for MoE models. [2] Latency is ~0.001 ms. [2]
2. Warm Tier (RAM): Stores the bulk of FFN weights and overflow KV cache. [2] Latency is ~1-3 ms. [2]
3. Cold Tier (SSD): Optional overflow for extreme scenarios (e.g., 200B+ models on 16GB RAM). [2] Latency is ~2-5 ms. [2]
For Mixture-of-Experts (MoE) models, VibeBlade implements expert-level tiering. [2] The system profiles expert activation frequencies offline and keeps the "hot" experts resident in VRAM while loading "cold" experts from RAM on-demand. [2] To mask the latency of loading cold experts over the PCIe bus, VibeBlade utilizes a dual-stream execution model. [2] The GPU executes attention and hot expert work while a CPU threadpool concurrently fetches and processes cold experts. [2] This is further optimized by an expert pre-fetching oracle that uses a Markov chain to predict which experts layer $\ell+1$ will need based on the routing decisions at layer $\ell$. [2]
The selection of the execution backend is performed automatically at startup. On NVIDIA hardware, VibeBlade routes to a custom TensorRT-LLM wrapper. [2] On Apple Silicon, it utilizes CoreML or the recently released MLX backend, which provides a 93% speedup in generation over standard Metal kernels. [2] For CPU-only environments, it leverages AVX-512 with SmoothQuant to enable INT8 tensor acceleration, providing a 1.56x speedup over FP16.47

**Table 4:** Hardware-Adaptive Backend Scaling (Llama-3.1 70B Q4\_K\_M)

| Hardware Configuration | Optimized Backend | Throughput (VibeBlade) | Throughput (Naive/llama.cpp) | Scaling Factor |
| --- | --- | --- | --- | --- |
| RTX 5090 (32GB) | TensorRT + Sparsity | 62.5 t/s | 18.2 t/s | 3.4x |
| RTX 4090 (24GB) | CUDA + Tiering | 18.4 t/s | 3.1 t/s (Hybrid) | 5.9x |
| M4 Ultra (128GB) | MLX + Sparsity | 114.0 t/s | 15.0 t/s (Metal) | 7.6x |
| Strix Halo (128GB) | Vulkan + MoE Opt | 22.0 t/s | 4.2 t/s | 5.2x |
| Dual Mac Mini M4 Pro | TB5 Cluster | 24.5 t/s | 6.0 t/s | 4.1x |

As a local inference engine, VibeBlade is designed to execute untrusted model weights from repositories like HuggingFace, making security a primary concern. [2] A professional audit in April 2026 identified several critical attack surfaces which have since been mitigated through systemic architectural changes. [2]
The most significant finding was a critical shell injection vulnerability in the setup wizard. [2] The original implementation used subprocess.run(cmd, shell=True), which allowed for the execution of arbitrary shell commands if a malicious model name or path was provided. [2] This has been replaced with a secure implementation utilizing shell=False and explicit argument parsing through shlex.split(). [2]
Additionally, the model manager was found to be vulnerable to path traversal attacks. [2] Malicious GGUF files or user-provided configuration paths could potentially access or delete sensitive files outside the designated model directory. [2] VibeBlade now resolves all paths to their absolute canonical form and enforces containment within the ~/.vibeblade/models/ directory before any filesystem operation. [2]
Positive security properties of the system include the complete absence of eval() or exec() calls, no hardcoded secrets (HuggingFace tokens are read only from environment variables), and the use of safe serialization formats. [2] By operating entirely within a local-only sandbox, VibeBlade provides a secure alternative to cloud APIs that may inadvertently leak user-provided data or proprietary prompt templates. [9]

## Nuanced Performance Analysis: The Feasibility Frontier

The ultimate goal of VibeBlade is to expand the "Feasibility Frontier" for consumer hardware. [2] In 2024, running a 70B model required a multi-GPU workstation or a top-tier Mac. [2] By 2026, VibeBlade has shifted this frontier such that a standard desktop can now serve frontier-scale models with meaningful throughput. [2]

**Table 5:** Model Feasibility on Consumer Hardware (VibeBlade v. Baseline)

| System Profile | RAM / VRAM | Baseline Max Model | VibeBlade Max Model | Effective Gain |
| --- | --- | --- | --- | --- |
| Budget Laptop | 16GB / 0GB | 8B Q4 (~15 t/s) | 13B Q4 (~35 t/s) | 2x Capacity |
| Standard Desktop | 32GB / 12GB | 13B Q4 (~8 t/s) | 70B Q4 (~12 t/s) | 5x Capacity |
| Pro Workstation | 64GB / 24GB | 32B Q4 (~10 t/s) | 236B MoE (~15 t/s) | 7x Capacity |
| Unified Mac | 128GB / 128GB | 70B Q4 (~15 t/s) | 1T MoE (~8 t/s) | 14x Capacity |

Despite the substantial advancements documented herein, VibeBlade is subject to several architectural limitations that define the next phase of systems research. The most prominent constraint is the requirement for offline MoE expert profiling. [2] For exceptionally large models with hundreds of experts (e.g., DeepSeek-V3), the profiling step can be slow on consumer hardware, and if the calibration dataset does not match the deployment workload, the hot/cold classification may be suboptimal. [2] Research into dynamic, online expert migration based on reinforcement learning (RL) reward signals is currently underway. [54]
Furthermore, while activation sparsity (TurboSparse) provides massive throughput gains, its accuracy is partially dependent on the activation function of the base model. [2] Models that have not been "ReLUfied" or trained with dReLU may experience a larger "quality cliff" at high sparsity levels. [8] The integration of training-free SVD-based contextual predictors is a promising mitigation strategy, providing theoretical guarantees for prediction accuracy on smooth activation functions like SwiGLU. [11]
The emergence of NVIDIA’s NVFP4 (native 4-bit floating point) format on Blackwell GPUs presents another frontier. [57] NVFP4 offers distinct advantages over standard INT4 quantization, including higher dynamic range and hardware-native math operations. [57] Transitioning VibeBlade's kernels from post-training quantization (PTQ) to native FP4 acceleration could yield an additional 1.5-2x improvement in energy efficiency and throughput, provided the model is natively trained in that format. [57]
Finally, the development of a WebAssembly (WASM) backend for VibeBlade is a high-priority direction. [2] By leveraging WebGPU, the VibeBlade stack could be deployed within any modern browser, enabling zero-install local inference for billions of devices. [2] This would represent the final democratization of LLM technology, moving frontier-grade intelligence from dedicated binaries to the pervasive web platform.

## Conclusions

The VibeBlade inference engine represents a paradigm shift in the serving of large language models on consumer hardware. By identifying that the binding constraint is software orchestration rather than hardware capacity, VibeBlade demonstrates that 70B+ models can be served at conversational speeds on systems that were previously considered inadequate. Through the integration of activation sparsity prediction, tri-layer feature fusion in speculative decoding, and adaptive memory tiering, the system achieves a 3-6x improvement in throughput and enables the execution of models up to 14 times larger than naive baselines.
The broader implications of this research are economic and social. As the cost of cloud-hosted APIs remains high and privacy concerns grow, the ability to run GPT-4-class models locally ensures that sophisticated AI remains accessible to a broad spectrum of developers and researchers. [9] VibeBlade’s commitment to an open-source, model-agnostic architecture ensures that it will remain a cornerstone of the local AI ecosystem as the hardware landscape continues to evolve through 2026 and beyond. The convergence of structural sparsity and intelligent memory management is not merely an optimization; it is the essential path forward for the democratization of artificial intelligence.


## References

[1] Top 5 Local LLM Tools and Models in 2026 - Pinggy. Accessed April 27, 2026. <https://pinggy.io/blog/top_5_local_llm_tools_and_models/>
[2] vibeblade-whitepaper.pdf
[3] Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2406.05955v1>
[4] Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching. Accessed April 27, 2026. <https://ojs.aaai.org/index.php/AAAI/article/view/39224/43185>
[5] Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve - Computer Science and Engineering. Accessed April 27, 2026. <https://www.cse.iitd.ac.in/~rijurekha/col851/scheduling_oct23_oct25/2024_osdi-sarathiserve.pdf>
[6] What to Buy for Local LLMs (April 2026) | by Julien Simon - Medium. Accessed April 27, 2026. <https://julsimon.medium.com/what-to-buy-for-local-llms-april-2026-a4946a381a6a>
[7] Benchmarked the main GPU options for local LLM inference in 2026 : r/LocalLLaMA - Reddit. Accessed April 27, 2026. <https://www.reddit.com/r/LocalLLaMA/comments/1rk5ftz/benchmarked_the_main_gpu_options_for_local_llm/>
[8] [Literature Review] Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters - Moonlight | AI Colleague for Research Papers. Accessed April 27, 2026. <https://www.themoonlight.io/en/review/turbo-sparse-achieving-llm-sota-performance-with-minimal-activated-parameters>
[9] llama.cpp: Fast Local LLM Inference, Hardware Choices & Tuning - Clarifai. Accessed April 27, 2026. <https://www.clarifai.com/blog/ilama.cpp>
[10] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | Request PDF - ResearchGate. Accessed April 27, 2026. <https://www.researchgate.net/publication/385860215_PowerInfer_Fast_Large_Language_Model_Serving_with_a_Consumer-grade_GPU>
[11] SVD Contextual Sparsity Predictors for Fast LLM Inference - arXiv. Accessed April 27, 2026. <https://arxiv.org/pdf/2603.14110>
[12] Training-Free Activation Sparsity in Large Language Models - OpenReview. Accessed April 27, 2026. <https://openreview.net/forum?id=dGVZwyq5tV>
[13] Language Models (dot tech)'s blog. Accessed April 27, 2026. <https://languagemodels.tech/>
[14] PowerInfer-2: Fast Large Language Model Inference on a Smartphone - ResearchGate. Accessed April 27, 2026. <https://www.researchgate.net/publication/381307075_PowerInfer-2_Fast_Large_Language_Model_Inference_on_a_Smartphone>
[15] GitHub - Tiiny-AI/PowerInfer: High-speed Large Language Model Serving for Local Deployment. Accessed April 27, 2026. <https://github.com/Tiiny-AI/PowerInfer>
[16] An Introduction to Speculative Decoding for Reducing Latency in AI Inference | NVIDIA Technical Blog. Accessed April 27, 2026. <https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/>
[17] Speculative Decoding: Achieving 2-3x LLM Inference Speedup | Introl Blog. Accessed April 27, 2026. <https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025>
[18] Speculative Decoding in vLLM: Complete Guide to Faster LLM Inference | Jarvis Labs Blog. Accessed April 27, 2026. <https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference>
[19] Speculative Speculative Decoding | OpenReview. Accessed April 27, 2026. <https://openreview.net/forum?id=aL1Wnml9Ef>
[20] Speculative Decoding for Multi-Sample Inference - ACL Anthology. Accessed April 27, 2026. <https://aclanthology.org/2025.findings-emnlp.668.pdf>
[21] From research to production: Accelerate OSS LLM with EAGLE-3 on Vertex | Cloud AI Engineering Blog. Accessed April 27, 2026. <https://docs.cloud.google.com/vertex-ai/docs/blog/posts/from-research-to-production-accelerate-oss-llm-with-eagle-3-on-vertex>
[22] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty - arXiv. Accessed April 27, 2026. <https://arxiv.org/abs/2401.15077>
[23] Speculative Decoding in Practice: How EAGLE3 Makes LLMs Faster Without Changing Their Outputs - Hugging Face. Accessed April 27, 2026. <https://huggingface.co/blog/lujangusface/tw-eagle3-gpu>
[24] ConFu: Contemplate the Future for Better Speculative Sampling - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2603.08899v3>
[25] ConFu: Contemplate the Future for Better Speculative Sampling - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2603.08899v1>
[26] Paper page - ConFu: Contemplate the Future for Better Speculative Sampling. Accessed April 27, 2026. <https://huggingface.co/papers/2603.08899>
[27] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache - GitHub. Accessed April 27, 2026. <https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24bz/liu24bz.pdf>
[28] KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2506.08018v3>
[29] Deploy Llama 4 with vLLM: Scout vs Maverick Setup Guide (2026) - Prem AI. Accessed April 27, 2026. <https://blog.premai.io/eploy-llama-4-with-vllm-scout-vs-maverick-setup-guide-2026/>
[30] (PDF) Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost - ResearchGate. Accessed April 27, 2026. <https://www.researchgate.net/publication/397934403_Kitty_Accurate_and_Efficient_2-bit_KV_Cache_Quantization_with_Dynamic_Channel-wise_Precision_Boost>
[31] [2402.02750] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache - arXiv. Accessed April 27, 2026. <https://arxiv.org/abs/2402.02750>
[32] RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations - IJCAI. Accessed April 27, 2026. <https://www.ijcai.org/proceedings/2025/0690.pdf>
[33] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache - OpenReview. Accessed April 27, 2026. <https://openreview.net/forum?id=L057s2Rq8O>
[34] SAW-INT4: System-AWare 4-Bit KV-Cache Quantization for Real-World LLM Serving - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2604.19157v1>
[35] [2511.18643] Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost - arXiv. Accessed April 27, 2026. <https://arxiv.org/abs/2511.18643>
[36] Continuous batching. Accessed April 27, 2026. <https://aarnphm.xyz/thoughts/Continuous-batching>
[37] vLLM vs llama.cpp: Huge Context Efficiency Differences on Qwen3.5-4B AWQ - Reddit. Accessed April 27, 2026. <https://www.reddit.com/r/LocalLLaMA/comments/1sfnjoh/vllm_vs_llamacpp_huge_context_efficiency/>
[38] MInference: Million-Tokens Prompt Inference for Long-context LLMs - Microsoft Research. Accessed April 27, 2026. <https://www.microsoft.com/en-us/research/project/minference-million-tokens-prompt-inference-for-long-context-llms/>
[39] 10 Best vLLM Alternatives for LLM Inference in Production (2026). Accessed April 27, 2026. <https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/>
[40] SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. Accessed April 27, 2026. <https://www.alphaxiv.org/overview/2308.16369>
[41] Research Focus: Week of September 25, 2023 - Microsoft. Accessed April 27, 2026. <https://www.microsoft.com/en-us/research/blog/research-focus-week-of-september-25-2023/>
[42] SageSched: Efficient LLM Scheduling Confronting Demand Uncertainty and Hybridity - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2603.07917v2>
[43] TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2604.19769v1>
[44] Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference - Microsoft. Accessed April 27, 2026. <https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf>
[45] Speculating Experts Accelerates Inference for Mixture-of-Experts - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2603.19289v1>
[46] Prefetching using Markov Chain. Accessed April 27, 2026. <https://omscs.gatech.edu/prefetching-using-markov-chain>
[47] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models - Proceedings of Machine Learning Research. Accessed April 27, 2026. <https://proceedings.mlr.press/v202/xiao23c.html>
[48] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models - Proceedings of Machine Learning Research. Accessed April 27, 2026. <https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf>
[49] Llama.cpp GGUF Quantization Guide: Optimize Local LLM Performance (2026). Accessed April 27, 2026. <https://www.decodesfuture.com/articles/llama-cpp-gguf-quantization-guide-2026>
[50] LLM Quantization: Run Any Model on Consumer Hardware - Let's Data Science. Accessed April 27, 2026. <https://letsdatascience.com/blog/llm-quantization-run-any-model-on-consumer-hardware>
[51] How to train custom EAGLE-3 heads for speculative decoding - Baseten. Accessed April 27, 2026. <https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/>
[52] MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2412.07067v3>
[53] MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2505.11415v1>
[54] ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2602.03203v1>
[55] Sparsing Law: Towards Large Language Models with Greater Activation Sparsity | OpenReview. Accessed April 27, 2026. <https://openreview.net/forum?id=B9XP2R9LtG>
[56] SVD Contextual Sparsity Predictors for Fast LLM Inference - arXiv. Accessed April 27, 2026. <https://arxiv.org/html/2603.14110v1>
[57] ainews-web-2025/src/content/issues/26-03-05-gpt54.md at main - GitHub. Accessed April 27, 2026. <https://github.com/smol-ai/ainews-web-2025/blob/main/src/content/issues/26-03-05-gpt54.md>
[58] GPT 5.4: SOTA Knowledge Work -and- Coding -and- CUA Model, OpenAI is so very back - xAGI Labs. Accessed April 27, 2026. <https://xagi.in/blog/gpt-54-sota-knowledge-work-and-coding-and-cua-model-openai-is-so-very-back>
[59] Implementing Google's TurboQuant: KV Cache Compression and LLM Evaluation with W&B | by Dave Davies | Online Inference | Mar, 2026 | Medium. Accessed April 27, 2026. <https://medium.com/online-inference/implementing-googles-turboquant-kv-cache-compression-and-llm-evaluation-with-w-b-1403d460846b>
[60] AI Trends 2026 – LLM Statistics & Industry Insights. Accessed April 27, 2026. <https://llm-stats.com/ai-trends>