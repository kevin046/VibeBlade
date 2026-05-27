"""
DFlash (Block Diffusion for Flash Speculative Decoding) integration for VibeBlade.

Reference: arxiv.org/abs/2602.06036 — Chen et al. 2026
Code base: github.com/z-lab/dflash — Z Lab

DFlash replaces autoregressive drafting with a lightweight block diffusion model
that generates a block of B tokens in a SINGLE forward pass (vs sequential
AR drafting which does B forward passes for B tokens).

Key insight from the paper: "The target knows best" — a large target LLM's
hidden features implicitly contain information about multiple future tokens.
Conditioning the draft on target hidden features → high acceptance rates.

Architecture:
  - DFlash draft model: small transformer with custom Qwen3DFlashAttention
  - Takes target_hidden (from target model) + noise_embedding (noisy block tokens)
  - Generates denoised block tokens in parallel via single forward pass
  - Draft tokens verified in batch by llama.cpp target model

Integration with VibeBlade:
  - Target model: loaded via llama.cpp (LlamaCppBackend / SpeculativeBackend)
  - DFlash draft: loaded via HuggingFace transformers (CPU PyTorch)
  - DFlash generates block_size tokens per forward pass
  - llama.cpp batch-verifies all draft tokens in a single decode call
  - Net effect: (block_size + 1) tokens per target decode step instead of 1

Usage:
    from vibeblade.dflash import DFlashDraftHead, dflash_generate
    from vibeblade.speculative import SpeculativeBackend

    spec = SpeculativeBackend()
    spec.load("models/qwen3-8b-instruct-q4km.gguf", n_ctx=2048, n_threads=4)

    # Enable DFlash drafting
    dflash = DFlashDraftHead(
        draft_model_name="z-lab/Qwen3-8B-DFlash-b16",
        target_model_name="Qwen/Qwen3-8B",
    )
    spec.set_draft_model(dflash)

    # Generate with DFlash speculative decoding
    result = spec.generate(prompt, max_tokens=256, speculative=True)
    print(f"Throughput: {result.tokens_per_second:.2f} t/s")
    print(f"Spec stats: {spec.spec_stats}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy imports (torch + transformers are optional dependencies) ─────
_torch = None


def _torch_lazy():
    global _torch
    if _torch is None:
        try:
            import torch as _torch
        except ImportError:
            raise ImportError(
                "DFlash requires PyTorch: pip install torch — see "
                "https://pytorch.org/get-started/locally/"
            )
    return _torch


_HF = None


def _get_transformers():
    global _HF
    if _HF is None:
        try:
            import transformers as _HF
        except ImportError:
            raise ImportError(
                "DFlash requires HuggingFace transformers: pip install transformers"
            )
    return _HF


@dataclass
class DFlashStats:
    """Track DFlash speculative decoding efficiency."""
    n_draft_blocks: int = 0
    n_tokens_generated: int = 0
    n_tokens_accepted: int = 0
    n_target_decodes: int = 0
    time_draft_ms: float = 0.0
    time_verify_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.n_tokens_generated == 0:
            return 0.0
        return self.n_tokens_accepted / self.n_tokens_generated

    @property
    def effective_speedup(self) -> float:
        if self.n_target_decodes == 0:
            return 1.0
        return self.n_tokens_generated / self.n_target_decodes

    def __str__(self) -> str:
        return (
            f"blocks={self.n_draft_blocks} "
            f"accept={self.acceptance_rate:.2%} "
            f"({self.n_tokens_accepted}/{self.n_tokens_generated}) "
            f"draft={self.time_draft_ms:.1f}ms "
            f"verify={self.time_verify_ms:.1f}ms"
        )


def _record_stats(head: "DFlashDraftHead", t0: float, filtered: list[int]) -> None:
    """Update DFlashStats after a draft call."""
    elapsed_ms = (time.time() - t0) * 1000
    head.stats.time_draft_ms += elapsed_ms
    head.stats.n_draft_blocks += 1
    head.stats.n_tokens_generated += len(filtered)


def sample(logits: _torch_lazy().Tensor, temperature: float = 0.0) -> _torch_lazy().Tensor:
    """Sample from logits (greedy if temperature ≈ 0)."""
    if temperature < 1e-5:
        return _torch_lazy().argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size) / temperature
    probs = _torch_lazy().softmax(logits_flat, dim=-1)
    return _torch_lazy().multinomial(probs, num_samples=1).view(bsz, seq_len)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Map draft layer index → target layer index for hidden feature extraction.

    From DFlash paper §4.2 / modeling_dflash.py:
    - Single draft layer: use middle target layer (num_target_layers // 2)
    - Multiple draft layers: evenly space between layers 1 and (num_target_layers - 3)
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / max(1, num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[_torch_lazy().Tensor],
    layer_ids: Optional[list[int]],
) -> _torch_lazy().Tensor:
    """Concatenate hidden states from selected target layers.

    From DFlash paper: "extract context feature from target hidden states
    by concatenating the hidden states at the selected layers."
    """
    if layer_ids is None:
        return hidden_states[0]
    # Offset +1: hidden_states[0] = input embeddings, first real layer = index 1
    offset = 1
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return _torch_lazy().cat(selected, dim=-1)


# -----------------------------------------------------------------------
# Minimal DFlash forward (self-conditioned, CPU-safe)
#
# We reimplement the core DFlash forward here to avoid requiring the full
# HF pipeline and to support CPU-only inference. The key operations are:
#
#   1. Project target_hidden via self.fc to match draft hidden dim
#   2. Run draft transformer layers (each: attention + MLP)
#      - Attention: q = hidden_states, k/v = target_hidden + hidden_states
#      - MLPs preserve hidden_states for next layer
#   3. Final RMSNorm → project to vocab via lm_head
#
# For CPU inference, we use eager attention (no FlashAttention on CPU).
# -----------------------------------------------------------------------


def _dflash_forward_single_pass(
    dflash_model,
    target_hidden: _torch_lazy().Tensor,
    noise_embedding: _torch_lazy().Tensor,
    position_ids: _torch_lazy().LongTensor,
    past_key_values: Optional,
    use_cache: bool = False,
    attention_mode: str = "eager",
) -> tuple[_torch_lazy().Tensor, Optional]:
    """Run one DFlash forward pass.

    Args:
        dflash_model: The loaded DFlash model (DFlashDraftModel from HF).
        target_hidden: [1, seq_ctx, hidden_dim] — context features from target.
        noise_embedding: [1, block_size, hidden_dim] — embeddings of noisy block.
        position_ids: [1, block_size] — position indices for RoPE.
        past_key_values: Optional KV cache from previous draft passes.
        use_cache: Whether to return updated KV cache.

    Returns:
        (logits [1, block_size, vocab_size], updated_kv or None)
    """
    hidden = noise_embedding  # [1, block_size, hidden_dim]
    rotary_emb = dflash_model.rotary_emb

    # Project target_hidden to match draft hidden dimension
    # self.fc: [len(target_layer_ids) * target_hidden_dim] → [draft_hidden_dim]
    # Cast to fc weight dtype to avoid BFloat16 != Half crash when target model
    # outputs bf16 but draft weights are fp16 (or vice versa).
    target_proj = dflash_model.fc(
        target_hidden.to(dflash_model.fc.weight.dtype)
    )  # [1, seq_ctx, draft_hidden]
    target_proj = dflash_model.hidden_norm(target_proj)

    bsz, block_size, _ = hidden.shape


    past_kv = past_key_values

    for layer_idx, layer in enumerate(dflash_model.layers):
        # Rotary embeddings for this layer's attention
        pos_emb = rotary_emb(hidden, position_ids)

        # Build attention: q = hidden_states, k/v = target_hidden + hidden_states
        # Qwen3DFlashAttention forward signature:
        #   hidden_states: q source (noisy tokens)
        #   target_hidden: k/v source (conditioning from target)
        #   position_embeddings: (cos, sin) for RoPE
        #   attention_mask: None (causal mask handled internally)
        #   past_key_values: KV cache
        #   use_cache: return updated KV
        attn_out = layer.self_attn(
            hidden_states=hidden,
            target_hidden=target_proj,
            position_embeddings=pos_emb,
            attention_mask=None,
            past_key_values=past_kv,
            use_cache=use_cache,
        )
        if use_cache and past_kv is not None:
            attn_out_val, present = attn_out[0], attn_out[1]
            past_kv = present
        else:
            attn_out_val = attn_out[0] if isinstance(attn_out, tuple) else attn_out

        # Residual connection + MLP
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        hidden = attn_out_val + residual
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden) + residual

    # Final norm → vocab projection
    hidden = dflash_model.norm(hidden)
    logits = dflash_model.lm_head(hidden)
    return logits, past_kv if use_cache else logits


# -----------------------------------------------------------------------
# Block diffusion generation loop
# -----------------------------------------------------------------------


@dataclass
class BlockDiffusionResult:
    """Result of one DFlash block generation step."""
    draft_tokens: list[int]          # sampled draft token IDs
    draft_logits: list[_torch_lazy().Tensor]  # raw logits per position (for debugging)
    target_hidden: _torch_lazy().Tensor       # context features (retained for next block)
    past_kv: Optional                 # draft KV cache for next pass


def _block_diffusion_step(
    dflash_model,
    target_hidden: _torch_lazy().Tensor,
    prev_token_ids: list[int],
    tokenizer,
    block_size: int,
    temperature: float = 0.0,
    max_position: int = 0,
) -> BlockDiffusionResult:
    """One block diffusion step: generate block_size draft tokens in parallel.

    From DFlash paper §4.1:
    1. Sample first token greedily from target model (done in SpeculativeBackend)
    2. Noise the remaining block_size-1 positions
    3. DFlash denoises all positions in ONE forward pass
    4. Sample from denoised logits

    We implement a simplified version: DFlash forward on the current context
    embedding, sampling block_size tokens from the output distribution.
    """
    device = target_hidden.device
    bsz = 1

    # Build input: last token embedding + masked block tokens
    # mask_token_id is used during training; at inference we use last token + zeros
    mask_token_id = getattr(dflash_model, "mask_token_id", tokenizer.pad_token_id or 0)

    # Get last token embedding as seed
    if prev_token_ids:
        last_id = _torch_lazy().tensor([[prev_token_ids[-1]]], dtype=_torch_lazy().long, device=device)
    else:
        last_id = _torch_lazy().tensor([[tokenizer.bos_token_id or 1]], dtype=_torch_lazy().long, device=device)

    # Embed last token to get hidden_dim
    try:
        dflash_model.model.embed_tokens(last_id)
    except AttributeError:
        # Fallback: embed via embedding layer if accessible
        dflash_model.embed_tokens(last_id)

    # Create block: first token = last_token, rest = zeros (or masked)
    # This gives DFlash something to denoise/extend from
    block_seq = _torch_lazy().zeros((bsz, block_size), dtype=_torch_lazy().long, device=device)
    block_seq[:, 0] = last_id.item()
    block_seq[:, 1:] = mask_token_id

    # Embed block tokens
    try:
        noise_emb = dflash_model.model.embed_tokens(block_seq)
    except AttributeError:
        noise_emb = dflash_model.embed_tokens(block_seq)

    # Position IDs for RoPE
    start_pos = max_position
    pos_ids = _torch_lazy().arange(start_pos, start_pos + block_size, device=device).unsqueeze(0)

    # Run DFlash forward (no KV cache for first pass)
    logits, _ = _dflash_forward_single_pass(
        dflash_model,
        target_hidden=target_hidden,
        noise_embedding=noise_emb,
        position_ids=pos_ids,
        past_key_values=None,
        use_cache=False,
    )

    # Sample block_size tokens from logits
    sampled = sample(logits, temperature=temperature)  # [1, block_size]

    # Filter special tokens
    draft_tokens = []
    draft_logits = []
    eos_id = tokenizer.eos_token_id or 0
    bos_id = tokenizer.bos_token_id or 1
    pad_id = tokenizer.pad_token_id or 0

    for i in range(block_size):
        tok_id = sampled[0, i].item()
        tok_logits = logits[0, i].clone()
        # Skip special tokens (except BOS/EOS)
        if tok_id in (eos_id, pad_id) and i > 0:
            continue
        if tok_id == bos_id and i > 0:
            continue
        draft_tokens.append(tok_id)
        draft_logits.append(tok_logits)

    return BlockDiffusionResult(
        draft_tokens=draft_tokens,
        draft_logits=draft_logits,
        target_hidden=target_hidden,  # pass through for next iteration
        past_kv=None,
    )


# -----------------------------------------------------------------------
# DFlash Draft Head — main integration class
# -----------------------------------------------------------------------


class DFlashDraftHead:
    """
    DFlash block diffusion draft head for speculative decoding.

    Loads a pretrained DFlash model from HuggingFace and uses it as a
    parallel draft generator. In each draft step:
      1. Extracts target context features (from target model or self-conditioned)
      2. Runs DFlash forward → block_size token logits in ONE pass
      3. Returns sampled draft token IDs
      4. Draft tokens are batch-verified by llama.cpp target model

    This replaces the autoregressive n-gram or neural draft head, which
    generates tokens sequentially (block_size forward passes for block_size
    tokens). DFlash generates all block_size tokens in a single forward pass.

    Key parameters from paper:
      - block_size: number of tokens drafted per step (default 16)
      - num_draft_layers: layers in the draft model (from config)
      - target_layer_ids: which target layers to extract features from

    Attributes:
        block_size: Tokens drafted per step.
        model: The loaded DFlash model (DFlashDraftModel).
        tokenizer: Shared tokenizer with target model.
        stats: DFlashStats tracking efficiency.
    """

    def __init__(
        self,
        draft_model_name: str,
        target_model_name: Optional[str] = None,
        block_size: int = 16,
        temperature: float = 0.0,
        device: str = "cpu",
        torch_dtype: Optional[str] = "float32",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        target_vocab_size: Optional[int] = None,
    ):
        """
        Initialize DFlash draft head.

        Args:
            draft_model_name: HuggingFace model ID or local path to DFlash draft model.
                              e.g. "z-lab/Qwen3-8B-DFlash-b16"
            target_model_name: HuggingFace model ID for target model (used for tokenizer
                               and target-hidden feature extraction). If None, uses same
                               tokenizer as draft model.
            block_size: Number of tokens to draft per block diffusion step.
                       DFlash models are trained for specific block sizes (-b16 = 16).
            temperature: Sampling temperature for draft token generation.
            device: PyTorch device ("cpu" for VibeBlade CPU inference).
            torch_dtype: PyTorch dtype ("float32" or "float16").
            trust_remote_code: Allow custom model code from HF (required for DFlash).
            token: HF token for gated models.
            target_vocab_size: Vocab size of the target model. Draft tokens outside
                this range are clamped to prevent decode failures from vocab mismatch.
        """
        self.draft_model_name = draft_model_name
        self.target_model_name = target_model_name or draft_model_name
        self.block_size = block_size
        self.temperature = temperature
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.token = token
        self.target_vocab_size = target_vocab_size  # set by set_draft_model_dflash()

        # Stats tracking
        self.stats = DFlashStats()

        # Model / tokenizer loaded lazily on first use
        self._model: Optional = None
        self._target: Optional = None
        self._tokenizer: Optional = None
        self._target_layer_ids: Optional[list[int]] = None
        self._past_kv = None  # KV cache across draft steps

    # ── Lazy loading ────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load models and tokenizer on first draft call."""
        if self._model is not None:
            return

        tf = _get_transformers()

        dtype_map = {"float32": _torch_lazy().float32, "float16": _torch_lazy().float16, "bfloat16": _torch_lazy().bfloat16}
        dtype = dtype_map.get(self.torch_dtype, _torch_lazy().float32)

        # Load draft model — AutoModel resolves DFlashDraftModel via auto_map config
        self._model = tf.AutoModel.from_pretrained(
            self.draft_model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
        )
        self._model.eval()

        # Extract DFlash-specific config
        if hasattr(self._model, 'block_size'):
            self.block_size = self._model.block_size
        if hasattr(self._model, 'target_layer_ids'):
            self._target_layer_ids = self._model.target_layer_ids

        # Load target model for tokenizer (and hidden feature extraction if available)
        # Note: we load target lazily; for self-conditioned DFlash we use the draft
        # model's own hidden features as a proxy for target hidden features.
        try:
            self._tokenizer = tf.AutoTokenizer.from_pretrained(
                self.target_model_name,
                trust_remote_code=self.trust_remote_code,
                token=self.token,
            )
        except Exception:
            # Fallback to draft model tokenizer
            self._tokenizer = tf.AutoTokenizer.from_pretrained(
                self.draft_model_name,
                trust_remote_code=self.trust_remote_code,
                token=self.token,
            )

        # Ensure pad token is set
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = 0

        # Extract target_layer_ids from draft model config
        self._target_layer_ids = getattr(self._model, "target_layer_ids", None)
        if self._target_layer_ids is None:
            # Infer from config
            num_draft_layers = getattr(self._model.config, "num_hidden_layers", 1)
            num_target_layers = getattr(self._model.config, "num_target_layers", 32)
            self._target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)

        # Block size from config (override init arg)
        cfg_block_size = getattr(self._model.config, "block_size", None)
        if cfg_block_size is not None:
            self.block_size = cfg_block_size

        # KV cache for draft model
        self._past_kv = None

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def target_layer_ids(self) -> list[int]:
        self._ensure_loaded()
        return self._target_layer_ids

    # ── Draft generation ────────────────────────────────────────────────

    def draft(
        self,
        history: list[int],
        draft_max_override: int = 0,
        target_hidden=None,
    ) -> list[int]:
        """
        Generate a block of draft tokens.

        When *target_hidden* is provided (hidden states from the target model at
        ``target_layer_ids``), uses **DFlash block diffusion** — a single forward
        pass that produces *block_size* tokens in parallel, conditioned on the
        target's intermediate representations.

        Without *target_hidden*, falls back to standard autoregressive generation
        via ``model.generate()`` (one forward pass per token).

        Args:
            history: Full token history (prompt + generated tokens so far).
            draft_max_override: Override draft length.
            target_hidden: ``[1, seq, len(target_layer_ids) * target_hidden_dim]``
                hidden states from the target model. When provided, enables the
                DFlash parallel block diffusion path.

        Returns:
            List of draft token IDs (length ≤ block_size).
        """
        self._ensure_loaded()
        t0 = time.time()
        torch = _torch_lazy()

        n_draft = (
            min(self.block_size, draft_max_override)
            if draft_max_override > 0
            else self.block_size
        )
        n_draft = max(n_draft, 1)

        # ── DFlash block diffusion (parallel single-pass drafting) ──────
        if target_hidden is not None:
            mask_token_id = getattr(
                self._model, "mask_token_id",
                self._tokenizer.pad_token_id or 0,
            )

            if history:
                last_id = torch.tensor(
                    [[history[-1]]], dtype=torch.long, device=self.device,
                )
            else:
                last_id = torch.tensor(
                    [[self._tokenizer.bos_token_id or 1]],
                    dtype=torch.long, device=self.device,
                )

            # Build block: last token seed + masked denoise positions
            block_seq = torch.zeros(
                (1, n_draft), dtype=torch.long, device=self.device,
            )
            block_seq[:, 0] = last_id.item()
            block_seq[:, 1:] = mask_token_id

            # Embed block tokens
            try:
                noise_emb = self._model.model.embed_tokens(block_seq)
            except AttributeError:
                noise_emb = self._model.embed_tokens(block_seq)

            # Position IDs for RoPE
            max_position = len(history)
            pos_ids = torch.arange(
                max_position, max_position + n_draft,
                device=self.device,
            ).unsqueeze(0)

            with torch.inference_mode():
                logits, _ = _dflash_forward_single_pass(
                    self._model,
                    target_hidden=target_hidden,
                    noise_embedding=noise_emb,
                    position_ids=pos_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                sampled = sample(logits, self.temperature)  # [1, n_draft]

            filtered = self._filter_tokens(sampled, n_draft)
            _record_stats(self, t0, filtered)
            return filtered

        # ── No target_hidden provided ──────────────────────────────────────
        # DFlash is a diffusion head conditioned on target model features.
        # Without target_hidden, it cannot produce meaningful draft tokens.
        # Two options:
        #   1. Use spec_generate() if a local target model is loaded
        #   2. Return empty draft (caller should use n-gram or EAGLE instead)
        if self._target is not None:
            # Local target available — use spec_generate for DFlash block diffusion
            with torch.inference_mode():
                input_ids = torch.tensor(
                    [history[-512:]], dtype=torch.long, device=self.device,
                )
                try:
                    output_ids = self._model.spec_generate(
                        self._target,
                        input_ids,
                        block_size=n_draft,
                        temperature=self.temperature,
                    )
                    drafted = output_ids[0, input_ids.shape[1]:].tolist()
                except Exception as e:
                    logger.debug(f"spec_generate failed: {e}")
                    drafted = []
        else:
            # No local target — DFlash can't draft without target hidden states.
            # Log once per session and return empty.
            if not getattr(self, '_warned_no_target', False):
                logger.info(
                    "DFlash: no target_hidden and no local target model. "
                    "Draft returns empty — use n-gram/EAGLE for HTTP backends "
                    "or load a local target model for DFlash block diffusion."
                )
                self._warned_no_target = True
            drafted = []

        filtered = self._filter_token_list(drafted)
        _record_stats(self, t0, filtered)
        return filtered

    def _filter_tokens(self, sampled, n_draft):
        """Filter special tokens from a sampled tensor [1, n_draft]."""
        _torch_lazy()
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id or 0
        bos_id = self._tokenizer.bos_token_id or 0
        vocab_max = self.target_vocab_size if self.target_vocab_size else 2**31
        special_ids = {eos_id, pad_id, bos_id}

        filtered = []
        for i in range(n_draft):
            tok_id = sampled[0, i].item()
            if tok_id in special_ids:
                continue
            filtered.append(max(0, min(tok_id, vocab_max - 1)))
        return filtered

    def _filter_token_list(self, drafted):
        """Filter special tokens from a token ID list."""
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id or 0
        bos_id = self._tokenizer.bos_token_id or 0
        vocab_max = self.target_vocab_size if self.target_vocab_size else 2**31
        special_ids = {eos_id, pad_id, bos_id, -1}
        special_ids.discard(-1)

        filtered = []
        for t in drafted:
            if t in special_ids:
                continue
            filtered.append(max(0, min(t, vocab_max - 1)))
        return filtered

    def _extract_draft_hidden_features(self, hidden_states: list[_torch_lazy().Tensor]) -> _torch_lazy().Tensor:
        """Extract concatenated features from draft model hidden states.

        Uses target_layer_ids to select specific layers, then concatenates.
        Projects to draft hidden dim via self.fc.
        """
        layer_ids = self._target_layer_ids
        # hidden_states[0] = embeddings, [1..N] = layers
        offset = 1
        selected = [hidden_states[lid + offset] for lid in layer_ids]
        concat = _torch_lazy().cat(selected, dim=-1)  # [1, seq, num_layers * hidden_dim]
        # fc projects concatenated multi-layer features to single hidden dim
        return self._model.fc(concat)

    def free(self) -> None:
        """Release model memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._target is not None:
            del self._target
            self._target = None
        self._past_kv = None
        _torch_lazy().cuda.empty_cache() if _torch_lazy().cuda.is_available() else None


# -----------------------------------------------------------------------
# Standalone DFlash generation (for non-speculative use)
# -----------------------------------------------------------------------


@dataclass
class DFlashGenerateResult:
    """Result from dflash_generate()."""
    text: str
    tokens: list[int]
    tokens_per_second: float
    stats: DFlashStats
    time_total: float


def dflash_generate(
    target_model_path: str,
    draft_model_name: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    block_size: int = 16,
    n_ctx: int = 2048,
    n_threads: int = 4,
    verbose: bool = False,
    add_bos: bool = False,
) -> DFlashGenerateResult:
    """
    End-to-end DFlash speculative generation with VibeBlade + HuggingFace DFlash.

    This function:
      1. Loads the target model via llama.cpp (VibeBlade backend)
      2. Loads the DFlash draft model via HuggingFace transformers
      3. Runs DFlash speculative decoding: draft → batch verify → accept

    Args:
        target_model_path: Path to target model GGUF file.
        draft_model_name: HuggingFace model ID for DFlash draft (e.g. "z-lab/Qwen3-8B-DFlash-b16").
        prompt: Input prompt string.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        block_size: DFlash block size (draft tokens per step).
        n_ctx: llama.cpp context size.
        n_threads: llama.cpp thread count.
        verbose: Print per-step debug info.
        add_bos: Prepend BOS token to prompt.

    Returns:
        DFlashGenerateResult with generated text, tokens, throughput, and stats.
    """
    from vibeblade.speculative import SpeculativeBackend

    t0 = time.time()

    # Load target model via VibeBlade llama.cpp backend
    spec = SpeculativeBackend(draft_n=1, draft_max=block_size)
    spec.load(target_model_path, n_ctx=n_ctx, n_threads=n_threads)

    # Attach DFlash draft head
    dflash = DFlashDraftHead(
        draft_model_name=draft_model_name,
        block_size=block_size,
        temperature=temperature,
    )
    spec.set_draft_model_dflash(dflash)

    # Verify tokenizer compatibility
    draft_vocab = len(dflash.tokenizer)
    target_vocab = spec._vocab_n_tokens if hasattr(spec, "_vocab_n_tokens") else 0
    if target_vocab > 0 and draft_vocab != target_vocab:
        import warnings
        warnings.warn(
            f"Tokenizer mismatch: DFlash vocab={draft_vocab}, Target vocab={target_vocab}. "
            f"Acceptance rate will be near zero. Use matching tokenizers."
        )

    # Generate
    result = spec.generate(prompt, max_tokens=max_tokens, temperature=temperature,
                           add_bos=add_bos, speculative=True)

    t_total = time.time() - t0

    return DFlashGenerateResult(
        text=result.text,
        tokens=result.tokens,
        tokens_per_second=result.tokens_per_second,
        stats=dflash.stats,
        time_total=t_total,
    )
