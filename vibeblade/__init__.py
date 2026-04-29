"""VibeBlade — Unified CPU/RAM Sparse Inference Protocol

Bypass the VRAM Wall. Run large language models on system RAM with
activation sparsity, 4-bit quantization, and neuron-prediction scheduling.
"""

import numpy as np

from .memory import UnifiedMemoryPool, ActivationBufferPool
from .sparse import (
    drelu_activation,
    predict_activations,
    compute_sparsity,
    sparse_matmul,
    topk_activation_mask,
    batch_drelu,
)
from .quant import (
    pack_nibbles,
    unpack_nibbles,
    build_so4_rotor,
    quantize_4bit,
    dequantize_4bit,
    quantization_error,
)
from .cache import KVCache
from .scheduler import NeuronPredictor, PowerInferScheduler
from .loader import GGUFLoader, load_model, _LazyWeights
from .transformer import (
    rms_norm, silu, rope, build_rope_cache,
    attention, ffn_silu,
    forward_token, forward_prefill, forward_decode_single,
)
from .generate import TextGenerator
from .vibeblade_core import rotor_unpack, drelu, _CPP_BACKEND
from .gpu import GPUBackend, available_backends
from .moe import (
    MoEConfig, ExpertRouter, MoEExpertSet,
    detect_moe_config, load_moe_weights_from_layer,
)
from .moe_profiler import HotColdMap, ExpertProfiler
from .moe_executor import HotColdExecutor, ExecutorStats
from .config import (
    OffloadMode, OffloadConfig, VibeBladeConfig,
    load_config, default_config, parse_size, ConfigError,
)
from .tiered_memory import (
    MemoryTier, TieredMemoryManager, LRUKPolicy, SSDExpertStore,
)
from .eviction import (
    EvictionPolicy, FrequencyAwarePolicy, CostBenefitScorer,
    AdaptiveBanditPolicy,
)
from .moe_advanced import (
    ConfidenceRouter, ContextAwarePrefetcher, HeteroQuantizer,
    CPUKernelOptimizer,
)

from .moe_oracle import ExpertOracle, PatternOracle
from .async_executor import AsyncMoEExecutor, AsyncStats, ColdExpertResult
from .phase_scheduler import PhaseScheduler, PhaseConfig, InferencePhase

# v1.4: Whitepaper algorithm implementations
from .rotatekv import (
    hadamard_rotation_matrix,
    rotate_kv,
    inverse_rotate_kv,
    RotateKVCache,
)
from .confu import (
    ContemplateTokenLayer,
    ConFuDraftModel,
    ConFuSpeculator,
    ConFuStats,
)
from .sarathi import (
    SarathiConfig,
    SarathiRequest,
    SarathiScheduler,
)
from .sagesched import (
    SageConfig,
    SageRequest,
    SageSched,
    entropy_from_logits,
    entropy_from_probs,
)
from .sparse import (
    drelu_gate,
    EMANeuronPredictor,
)
# ONNX Runtime / TensorRT accelerated inference (lazy imports — requires onnxruntime)
# Loaded via __getattr__ at bottom of file to avoid import errors when packages aren't installed.

# Grammar-constrained decoding (pure Python, no optional deps)
from .grammar import (
    GrammarConstraint,
    RegexGrammar,
    JsonSchemaGrammar,
    EbnfGrammar,
)

__version__ = "1.4.0-alpha"
__all__ = [
    # High-level API
    "VibeBladeModel",
    # Memory
    "UnifiedMemoryPool",
    # Sparse
    "drelu_activation", "predict_activations", "compute_sparsity",
    "sparse_matmul", "topk_activation_mask", "batch_drelu",
    # Quantization
    "pack_nibbles", "unpack_nibbles", "build_so4_rotor",
    "quantize_4bit", "dequantize_4bit", "quantization_error",
    # Cache
    "KVCache",
    # Scheduler
    "NeuronPredictor", "PowerInferScheduler",
    # Loader
    "GGUFLoader", "load_model",
    # Transformer
    "rms_norm", "silu", "rope", "build_rope_cache",
    "attention", "ffn_silu",
    "forward_token", "forward_prefill", "forward_decode_single",
    # Generation
    "TextGenerator",
    # C++ bridge
    "rotor_unpack", "drelu", "_CPP_BACKEND",
    # GPU backends
    "GPUBackend", "available_backends",
    # ONNX Runtime / TensorRT acceleration
    "ORTOps", "detect_providers", "platform_info",
    "TensorRTEngine", "is_available",
    "AcceleratedBackend", "get_accelerator",
    # Grammar-constrained decoding
    "GrammarConstraint", "RegexGrammar", "JsonSchemaGrammar", "EbnfGrammar",
    # MoE
    "MoEConfig", "ExpertRouter", "MoEExpertSet", "HotColdMap", "ExpertProfiler",
    "HotColdExecutor", "ExecutorStats",
    # Adaptive Memory Tiering
    "OffloadMode", "OffloadConfig", "VibeBladeConfig",
    "load_config", "default_config", "parse_size", "ConfigError",
    "MemoryTier", "TieredMemoryManager", "LRUKPolicy", "SSDExpertStore",
    "ActivationBufferPool",
    "EvictionPolicy", "FrequencyAwarePolicy", "CostBenefitScorer",
    "AdaptiveBanditPolicy",
    # Advanced MoE optimizations
    "ConfidenceRouter", "ContextAwarePrefetcher", "HeteroQuantizer",
    "CPUKernelOptimizer",
    # v1.1: Async dual-stream, predictive oracle, phase scheduling
    "ExpertOracle", "PatternOracle",
    "AsyncMoEExecutor", "AsyncStats", "ColdExpertResult",
    "PhaseScheduler", "PhaseConfig", "InferencePhase",
    # v1.4: Whitepaper algorithm implementations
    # RotateKV — outlier-aware 2-bit KV quantization (§3)
    "hadamard_rotation_matrix", "rotate_kv", "inverse_rotate_kv", "RotateKVCache",
    # ConFu — contemplate-token speculative decoding (§2)
    "ContemplateTokenLayer", "ConFuDraftModel", "ConFuSpeculator", "ConFuStats",
    # SARATHI — chunked prefill scheduler (§4)
    "SarathiConfig", "SarathiRequest", "SarathiScheduler",
    # SageSched — uncertainty-aware scheduler (§4)
    "SageConfig", "SageRequest", "SageSched", "entropy_from_logits", "entropy_from_probs",
    # TurboSparse — EMA neuron prediction + dReLU gating (§1)
    "drelu_gate", "EMANeuronPredictor",
]


class VibeBladeModel:
    """High-level model wrapper for VibeBlade inference.
    
    Supports loading GGUF format models and running inference with:
    - TurboSparse activation sparsity (dReLU)
    - RotorQuant 4-bit weight compression
    - PowerInfer neuron-prediction scheduling
    - KV cache for fast autoregressive generation
    """
    
    def __init__(self, model_path: str, hot_budget: float = 0.1, use_sparse: bool = True,
                 progress_cb=None):
        """Load a GGUF model.
        
        Args:
            model_path: path to .gguf model file
            hot_budget: fraction of neurons for PowerInfer fast path (0.0-1.0)
            use_sparse: enable TurboSparse activation sparsity
            progress_cb: optional callback(name, done, total, loading=bool)
        """
        from pathlib import Path
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.path = str(path)
        self.use_sparse = use_sparse
        self.hot_budget = hot_budget
        self._progress_cb = progress_cb
        
        # Load model
        if path.suffix == ".gguf":
            data = load_model(model_path, progress_cb=self._progress_tick)
            self.metadata = data["metadata"]
            self.config = data["config"]
            self.weights = data["tensors"]
            self._extract_config()
            # Store EOS token ID from metadata
            arch = self.metadata.get("general.architecture", "llama")
            self.eos_token_id = int(self.metadata.get(
                f"{arch}.eos_token_id",
                self.metadata.get("tokenizer.ggml.eos_token_id", 2),
            ))
        else:
            # Raw numpy weights directory or file
            self.pool = UnifiedMemoryPool(model_path)
            self.metadata = {}
            self.config = {
                "hidden_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "intermediate_dim": 1024,
                "vocab_size": 1000,
            }
        
        self._setup_components()
        print(f"🚀 VibeBlade v{__version__} initialized")
        print(f"   Backend: {'C++ AVX-512' if _CPP_BACKEND else 'NumPy (pure Python)'}")
        print(f"   Sparse: {'ON' if use_sparse else 'OFF'}")
        print(f"   Hot budget: {hot_budget:.1%}")
        if hasattr(self.weights, 'memory_info'):
            mi = self.weights.memory_info()
            mode = "Lazy" if isinstance(self.weights, _LazyWeights) else "Eager"
            gpu = " + GPU" if mi.get("gpu_offload") else ""
            print(f"   Weights: {mode} ({mi['total_tensors']} tensors, "
                  f"cache {mi['cached_mb']:.0f}/{mi['max_mb']:.0f} MB{gpu})")
            # Debug: show blk.0 tensor names so we can verify naming convention
            if isinstance(self.weights, _LazyWeights):
                blk0 = sorted(k for k in self.weights._name_map if k.startswith("blk.0."))
                print(f"   blk.0 tensors: {blk0}")
    
    def _extract_config(self):
        """Extract model config from GGUF metadata."""
        arch = self.metadata.get("general.architecture", "llama")
        def get(key, default):
            return self.metadata.get(f"{arch}.{key}", self.metadata.get(key, default))
        
        self.config.update({
            "hidden_dim": get("embedding_length", 512),
            "num_heads": get("attention.head_count", 8),
            "num_kv_heads": get("attention.head_count_kv", get("attention.head_count", 8)),
            "num_layers": get("block_count", 6),
            "intermediate_dim": get("feed_forward_length", 1024),
            "vocab_size": get("vocab_size", 1000),
            "context_length": get("context_length", 2048),
        })

    def _progress_tick(self, name: str, done: int, total: int, **_kw) -> None:
        """Forward progress to the external callback (if any)."""
        if self._progress_cb is not None:
            self._progress_cb(name, done, total, loading=True)
    
    def _setup_components(self):
        """Initialize scheduler, cache, generator, and MoE components."""
        self._progress_tick("probe", 0, 1)

        # If lazy weights with fused QKV, probe the actual tensor to get
        # correct attention dimensions (metadata may be wrong for hybrid models)
        if isinstance(self.weights, _LazyWeights) and self.weights._has_fused_qkv:
            self._probe_qkv_dimensions()
            # Propagate corrected dims to _LazyWeights so _load_split_qkv
            # uses the SAME dimensions (no independent re-inference)
            if "head_dim" in self.config:
                self.weights._qkv_head_dim = self.config["head_dim"]
                self.weights._qkv_n_heads = self.config["num_heads"]
                self.weights._qkv_n_kv_heads = self.config.get(
                    "num_kv_heads", self.config["num_heads"]
                )

        self._progress_tick("scheduler", 0, 1)
        self.scheduler = PowerInferScheduler(
            hidden_size=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            hot_budget=self.hot_budget,
        )
        self.cache = KVCache(
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            head_dim=self.config.get("head_dim", self.config["hidden_dim"] // self.config["num_heads"]),
            max_seq_len=self.config.get("context_length", 2048),
            quantize=True,
        )
        self.generator = TextGenerator(temperature=1.0, top_k=50, top_p=0.9)

        # Initialize BPE tokenizer from GGUF metadata (if available)
        self.tokenizer = None
        if self.metadata and "tokenizer.ggml.tokens" in self.metadata:
            from .tokenizer import GGUFTokenizer
            try:
                self.tokenizer = GGUFTokenizer.from_gguf_metadata(self.metadata)
                print(f"   Tokenizer: BPE (vocab {len(self.tokenizer.tokens)}, "
                      f"{len(self.tokenizer.merges)} merges, EOS={self.tokenizer.eos_token_id})")
            except Exception as e:
                print(f"   Tokenizer: fallback (BPE init failed: {e})")
        else:
            print(f"   Tokenizer: byte-level fallback")

        # Build RoPE cache (cos/sin tables for rotary positional embeddings)
        head_dim = self.config.get("head_dim", self.config["hidden_dim"] // self.config["num_heads"])
        max_ctx = self.config.get("context_length", 2048)
        self._progress_tick("rope cache", 0, 1)
        self.cos_cache, self.sin_cache = build_rope_cache(head_dim, max_ctx)

        # Auto-detect MoE architecture
        self._progress_tick("moe detect", 0, 1)
        self.moe_config = detect_moe_config(self.weights)
        self._moe_executor = None
        if self.moe_config is not None:
            self.config["num_experts"] = self.moe_config.num_experts
            self.config["num_active_experts"] = self.moe_config.num_active
            self.config["expert_dim"] = self.moe_config.expert_dim
            print(f"   MoE: {self.moe_config.num_experts} experts "
                  f"(top-{self.moe_config.num_active} per token)")

    def _probe_qkv_dimensions(self):
        """Probe the fused QKV tensor to infer actual attention dimensions.

        For hybrid attention+SSM models (e.g. Qwen3.6), GGUF metadata may
        report wrong head counts.  Reading just the tensor *shape* from
        the GGUF header (no dequant) lets us correct the config.
        """
        arch = self.metadata.get("general.architecture", "")
        meta = self.metadata

        # Find the QKV and attn_gate tensor shapes (no dequant needed)
        qkv_shape = None
        gate_shape = None
        for ti in self.weights._loader.tensor_infos:
            name = ti["name"]
            if "attn_qkv" in name and len(ti["shape"]) == 2 and ".0." in name:
                qkv_shape = ti["shape"]
            if "attn_gate" in name and len(ti["shape"]) == 2 and ".0." in name:
                gate_shape = ti["shape"]

        if qkv_shape is None:
            return

        qkv_rows, hidden_dim = qkv_shape

        n_heads_meta = int(meta.get(f"{arch}.attention.head_count", 0) or 0)
        n_kv_meta = int(meta.get(f"{arch}.attention.head_count_kv", 0) or 0)
        head_dim_meta = hidden_dim // max(n_heads_meta, 1)

        # Get all valid candidates from _infer_qkv_split
        candidates = self._collect_qkv_candidates(
            qkv_rows, hidden_dim, n_heads_meta, n_kv_meta, head_dim_meta, meta, arch
        )

        # Cross-validate with attn_gate (output projection) shape if available
        # attn_gate.weight layout depends on the model:
        #   Standard: (hidden_dim, n_heads * head_dim)  — gate_shape[1] = attn_out_dim
        #   Qwen3.6+: (n_heads * head_dim, hidden_dim)  — gate_shape[0] = attn_out_dim
        # So we validate against whichever dimension != hidden_dim
        if gate_shape is not None and len(gate_shape) == 2:
            # The attention output dim is the one that's NOT hidden_dim
            gate_dims = set(gate_shape)
            attn_out_candidates = gate_dims - {hidden_dim}
            # If both dims differ from hidden_dim, try both
            if not attn_out_candidates:
                attn_out_candidates = gate_dims
            validated = []
            for n_h, n_kv, hd in candidates:
                attn_out_dim = n_h * hd
                if attn_out_dim in gate_dims:
                    validated.append((n_h, n_kv, hd))
            if validated:
                candidates = validated

        if not candidates:
            return

        n_heads, n_kv_heads, head_dim = candidates[0]

        old_heads = self.config["num_heads"]
        old_kv = self.config.get("num_kv_heads", old_heads)
        old_hd = self.config["hidden_dim"] // max(old_heads, 1)

        if n_heads != old_heads or n_kv_heads != old_kv:
            print(f"   Attention: QKV probe corrected heads "
                  f"{old_heads}→{n_heads}, "
                  f"kv_heads {old_kv}→{n_kv_heads}, "
                  f"head_dim {old_hd}→{head_dim}")
            self.config["num_heads"] = n_heads
            self.config["num_kv_heads"] = n_kv_heads
            self.config["head_dim"] = head_dim

    def _collect_qkv_candidates(self, qkv_rows, hidden_dim,
                                 n_heads_meta, n_kv_meta, head_dim_meta,
                                 meta, arch):
        """Collect all valid (n_heads, n_kv_heads, head_dim) candidates."""
        candidates = []

        # 1. Metadata values directly
        hd = head_dim_meta
        if (n_heads_meta + 2 * n_kv_meta) * hd == qkv_rows and n_heads_meta > 0 and n_kv_meta > 0:
            candidates.append((n_heads_meta, n_kv_meta, hd))

        # 2. With explicit key_length as head_dim
        explicit_hd = meta.get(f"{arch}.attention.key_length")
        if explicit_hd:
            explicit_hd = int(explicit_hd)
            if (n_heads_meta + 2 * n_kv_meta) * explicit_hd == qkv_rows:
                candidates.append((n_heads_meta, n_kv_meta, explicit_hd))

        # 3. Brute-force: enumerate plausible splits
        def divisors(n):
            if n <= 0:
                return []
            divs = set()
            for i in range(1, min(int(n**0.5) + 1, 257)):
                if n % i == 0:
                    divs.add(i)
                    divs.add(n // i)
            return sorted(divs)

        head_dim_candidates = [hd for hd in set(divisors(hidden_dim)) | set(divisors(qkv_rows))
                               if 16 <= hd <= hidden_dim and qkv_rows % hd == 0]

        for hd in sorted(head_dim_candidates):
            total_heads = qkv_rows // hd
            for n_kv in range(1, min(total_heads // 3 + 1, 129)):
                n_q = total_heads - 2 * n_kv
                if n_q <= 0:
                    continue
                # Prefer configs where n_q matches metadata or n_kv divides n_q (GQA)
                if n_q % n_kv == 0:
                    if n_q == n_heads_meta or n_heads_meta == 0:
                        candidates.append((n_q, n_kv, hd))
                        break

        # 4. Without GQA constraint
        if not candidates or (len(candidates) == 1 and candidates[0] == (n_heads_meta, n_kv_meta, head_dim_meta)):
            for hd in sorted(head_dim_candidates):
                total_heads = qkv_rows // hd
                if total_heads % 4 == 0:
                    n_kv = total_heads // 4
                    n_q = total_heads - 2 * n_kv
                    if n_q > 0:
                        candidates.append((n_q, n_kv, hd))
                if total_heads % 3 == 0:
                    n_q = total_heads // 3
                    candidates.append((n_q, n_q, hd))

        return candidates
    
    def generate(
        self,
        prompt: str = None,
        token_ids=None,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        stream: bool = True,
        on_token=None,
    ) -> tuple[str, float]:
        """Generate text from a prompt.
        
        Args:
            prompt: text prompt (requires tokenizer in model metadata)
            token_ids: pre-tokenized input (alternative to prompt)
            max_tokens: maximum new tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering
            top_p: nucleus (top-p) filtering
            stream: print tokens as they generate
            on_token: optional callback(token_id, pos) for streaming
        
        Returns:
            (generated_text, tokens_per_second) tuple
        """
        self.generator.temperature = temperature
        self.generator.top_k = top_k
        self.generator.top_p = top_p
        
        import time
        n_layers = self.config["num_layers"]
        n_heads = self.config["num_heads"]
        n_kv_heads = self.config.get("num_kv_heads", n_heads)
        head_dim = self.config.get(
            "head_dim", self.config["hidden_dim"] // max(n_heads, 1)
        )
        
        if token_ids is None:
            if prompt is None:
                token_ids = np.array([self.tokenizer.bos_token_id]
                                     if self.tokenizer and self.tokenizer.bos_token_id is not None
                                     else [1])  # BOS token
            elif self.tokenizer is not None:
                # Use proper BPE tokenizer
                token_ids = np.array(self.tokenizer.encode(prompt))
            else:
                # Fallback: raw byte tokenization
                token_ids = np.array(list(prompt.encode("utf-8")) + [1])
        
        # Get shared weights
        token_emb = self.weights["token_embd.weight"]
        output_norm_w = self.weights["output_norm.weight"]
        output_w = self.weights["output.weight"]
        
        # Step 1: Prefill — process all prompt tokens at once
        logits, kv_caches_k, kv_caches_v = forward_prefill(
            token_ids=token_ids,
            token_emb=token_emb,
            output_norm_w=output_norm_w,
            output_w=output_w,
            weights=self.weights,
            n_layers=n_layers,
            cos_cache=self.cos_cache,
            sin_cache=self.sin_cache,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        
        prompt_len = len(token_ids)
        position = prompt_len
        
        # Step 2: Decode — generate one token at a time using KV cache
        output_tokens = []
        gen_start = time.time()
        
        # Normalize logits to 1D for sampling
        # Prefill returns (seq_len, vocab) — take last position
        # Decode returns (vocab,) — use directly
        if logits.ndim > 1:
            logits = logits[-1]
        
        for i in range(max_tokens):
            # Sample from logits (always 1D at this point)
            next_token = self.generator.sample(logits)
            output_tokens.append(next_token)
            
            # Stream the decoded token text
            if on_token is not None:
                on_token(next_token, i)
            elif stream:
                if self.tokenizer is not None:
                    chunk = self.tokenizer.decode([next_token], stop_at_eos=False)
                    print(chunk, end="", flush=True)
                else:
                    try:
                        print(chr(next_token), end="", flush=True)
                    except (ValueError, OverflowError):
                        pass
            
            eos_id = self.tokenizer.eos_token_id if self.tokenizer else 2
            if next_token == eos_id:
                break
            
            # Forward single token through all layers with KV cache
            logits, kv_caches_k, kv_caches_v, _ = forward_decode_single(
                token_id=next_token,
                position=position,
                token_emb=token_emb,
                output_norm_w=output_norm_w,
                output_w=output_w,
                weights=self.weights,
                n_layers=n_layers,
                kv_caches_k=kv_caches_k,
                kv_caches_v=kv_caches_v,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
            )
            position += 1
        
        gen_elapsed = time.time() - gen_start
        tps = len(output_tokens) / max(gen_elapsed, 1e-6)
        
        if stream and on_token is None:
            print()
        
        # Decode output tokens to text
        if self.tokenizer is not None:
            text = self.tokenizer.decode(output_tokens)
        else:
            try:
                text = "".join(chr(t) for t in output_tokens if 32 <= t < 127)
            except (ValueError, OverflowError):
                text = f"[{len(output_tokens)} tokens generated at {tps:.1f} t/s]"
        
        return text, tps

    @property
    def is_moe(self) -> bool:
        """Whether this model uses Mixture-of-Experts."""
        return self.moe_config is not None

    def enable_moe_hot_cold(
        self,
        hot_cold_map: "HotColdMap",
        gpu_backend: "GPUBackend" = None,
        cpu_threads: int = 8,
    ) -> None:
        """Enable hot/cold MoE split execution.

        Args:
            hot_cold_map: Pre-computed hot/cold expert assignment (from profiler)
            gpu_backend: Optional GPUBackend for hot expert compute (None = pure CPU)
            cpu_threads: Number of CPU threads for cold expert compute
        """
        if not self.is_moe:
            raise RuntimeError("Model is not MoE — cannot enable hot/cold split")

        from .moe import ExpertRouter

        routers: dict = {}
        experts: dict = {}
        shared: dict = {}

        for layer_idx in range(self.config["num_layers"]):
            router_w, expert_set, extras = load_moe_weights_from_layer(
                self.weights, layer_idx
            )
            if router_w is None:
                continue
            topk = min(8, self.moe_config.num_active)
            routers[layer_idx] = ExpertRouter(router_w, topk=topk)
            experts[layer_idx] = expert_set
            if "shared_gate" in extras:
                shared[layer_idx] = (
                    extras["shared_gate"],
                    extras["shared_up"],
                    extras["shared_down"],
                )

        self._moe_executor = HotColdExecutor(
            hot_cold_map=hot_cold_map,
            routers=routers,
            experts=experts,
            shared_experts=shared,
            gpu_backend=gpu_backend,
            cpu_threads=cpu_threads,
        )
        mem = self._moe_executor.memory_estimate(
            expert_dim=self.moe_config.expert_dim,
            shared_dim=self.config["hidden_dim"],
        )
        print(f"   Hot/Cold MoE: {mem['total_hot_experts']} hot ({mem['hot_vram_gb']:.1f} GB), "
              f"{mem['total_cold_experts']} cold ({mem['cold_ram_gb']:.1f} GB)")

    def profile_experts(
        self,
        prompt_tokens: np.ndarray,
        max_tokens: int = 128,
        hot_ratio: float = 0.1,
    ) -> "HotColdMap":
        """Run a calibration pass to build hot/cold expert map.

        Args:
            prompt_tokens: Token IDs for calibration prompts
            max_tokens: How many tokens to generate during calibration
            hot_ratio: Fraction of experts to mark as hot per layer

        Returns:
            HotColdMap with per-layer expert assignments
        """
        if not self.is_moe:
            raise RuntimeError("Model is not MoE — cannot profile experts")

        from .moe import ExpertRouter
        from .moe_profiler import ExpertProfiler

        profiler = ExpertProfiler(
            num_layers=self.config["num_layers"],
            num_experts=self.moe_config.num_experts,
        )

        # Build per-layer routers
        routers: dict = {}
        for li in range(self.config["num_layers"]):
            router_w, _, _ = load_moe_weights_from_layer(self.weights, li)
            if router_w is not None:
                topk = min(8, self.moe_config.num_active)
                routers[li] = ExpertRouter(router_w, topk=topk)

        # Run forward passes and record expert activations
        token_ids = prompt_tokens.copy()
        cos_cache, sin_cache = build_rope_cache(
            self.config["hidden_dim"] // self.config["num_heads"],
            self.config.get("context_length", 2048),
        )
        kv_k = [np.zeros((self.config["num_heads"], 0, self.config["hidden_dim"] // self.config["num_heads"]))
                for _ in range(self.config["num_layers"])]
        kv_v = list(kv_k)

        # Prefill (record expert selections for each token in prompt)
        for pos in range(len(token_ids)):
            x = self.weights["token_embd.weight"][token_ids[pos:pos+1]]
            for li in range(self.config["num_layers"]):
                h = rms_norm(
                    x,
                    self.weights[f"blk.{li}.ffn_norm.weight"],
                    self.config.get("attention.layer_norm_rms_epsilon", 1e-5),
                )
                if li in routers:
                    logits = routers[li].predict(h)
                    indices = np.argsort(logits)[-self.moe_config.num_active:]
                    profiler.record(li, indices)

        # Decode pass
        import time
        t0 = time.time()
        for _ in range(max_tokens):
            logits_all, new_kv_k, new_kv_v, stats = forward_decode_single(
                token_ids[-1], len(token_ids) - 1,
                self.weights["token_embd.weight"],
                self.weights["output_norm.weight"],
                self.weights["output.weight"],
                self.weights, self.config["num_layers"],
                kv_k, kv_v, cos_cache, sin_cache,
                self.config["num_heads"], self.config.get("num_heads", self.config["num_heads"]),
            )
            kv_k, kv_v = new_kv_k, new_kv_v
            next_token = int(np.argmax(logits_all))
            token_ids = np.append(token_ids, next_token)
            if next_token == self.eos_token_id:
                break

            # Record MoE stats from the decode pass
            if "moe" in stats:
                for layer_stat in stats["moe"]["layers"]:
                    li = layer_stat["layer"]
                    if "expert_indices" in layer_stat and li in routers:
                        profiler.record(li, layer_stat["expert_indices"])

        elapsed = time.time() - t0
        hot_map = profiler.compute_hot_cold_map(hot_ratio=hot_ratio)
        print(f"   Profiling: {profiler.total_tokens} tokens in {elapsed:.1f}s")
        print(f"   Hot/Cold: {hot_map.overall_hit_rate():.0%} avg hit rate")
        return hot_map

    def __repr__(self):
        return (f"VibeBladeModel(path={self.path!r}, "
                f"backend={'C++ AVX-512' if _CPP_BACKEND else 'NumPy'}, "
                f"layers={self.config.get('num_layers', '?')}, "
                f"hidden={self.config.get('hidden_dim', '?')})")


# ── Lazy imports for ONNX Runtime / TensorRT (requires onnxruntime package) ──

def __getattr__(name):
    """Lazy-load ONNX/TensorRT backends to avoid import errors when packages aren't installed."""
    _lazy = {
        "ORTOps": "vibeblade.onnx_backend",
        "detect_providers": "vibeblade.onnx_backend",
        "platform_info": "vibeblade.onnx_backend",
        "TensorRTEngine": "vibeblade.tensorrt_backend",
        "is_available": "vibeblade.tensorrt_backend",
        "AcceleratedBackend": "vibeblade.accelerated",
        "get_accelerator": "vibeblade.accelerated",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'vibeblade' has no attribute {name!r}")
