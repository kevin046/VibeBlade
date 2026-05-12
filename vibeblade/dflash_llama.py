"""
DFlash Integration for VibeBlade — Qwen3-4B + Qwen3-4B-DFlash-b16.

Architecture:
  - Target:    Qwen3-4B-Q4_K_M.gguf via VibeBladeFast (mmap'd, Q4_K_M, 32 layers)
  - Draft:     z-lab/Qwen3-4B-DFlash-b16 via HF Transformers (5 layers, block_size=16)

DFlash needs target hidden states at specific layers [1, 9, 17, 25, 33] from a 32-layer target.
These are extracted during target forward passes (prefill for prompt, decode for each generated token).

Usage:
    from vibeblade.dflash_llama import DFlashIntegration
    integ = DFlashIntegration(
        gguf_path="~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
        dflash_model_path="/tmp/qwen-dflash",
    )
    result = integ.generate("Hello world", max_tokens=128)
    print(result.text)
"""

from __future__ import annotations

import os
import sys
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# Add location of DFlash Qwen3 model to path
sys.path.insert(0, "/tmp")   # for the `qwen-dflash` package
sys.path.insert(0, "/home/ubuntu/VibeBlade")  # for vibeblade module
from vibeblade._vibeblade_native import VibeBladeFast

# ── DFlash model ───────────────────────────────────────────────────────────────
import torch
from dflash.model import DFlashDraftModel
from transformers import AutoTokenizer

# ── Qwen3-4B DFlash config ────────────────────────────────────────────────────
QWEN3_LAYER_INDICES = [1, 9, 17, 25, 33]   # 5 layers from 32-layer target
QWEN3_VOCAB_SIZE    = 151_936


@dataclass
class DFlashGenerateResult:
    text: str
    token_ids: list[int]
    tokens_per_second: float
    accepted_tokens: int
    total_drafted: int
    acceptance_rate: float
    blocks: int
    stopped_eos: bool
    latency_s: float


class DFlashIntegration:
    """
    DFlash speculative decoding: target (VibeBladeFast GGUF) + draft (HF DFlash).

    The DFlash draft model is trained on Qwen3 hidden states. It takes:
      - target_hidden: hidden states at layers [1,9,17,25,33] from the target, for context tokens
      - noise_embedding: embeddings of masked draft tokens
    And denoises them to predict the next `block_size` tokens.

    Verification loop:
      1. Condition DFlash on target_hidden (from most recent context tokens)
      2. Generate draft block tokens
      3. For each draft token:
           - Compare draft token to argmax of target's current posterior
           - If mismatch → accept correction token (from target), stop block
           - If match → accept draft token and feed it to target to get next posterior and hidden
      4. After full block acceptance, also accept one extra token from target's posterior
      5. Roll context: next block conditions on the hidden states of the newly added tokens
    """

    def __init__(
        self,
        gguf_path: str,
        dflash_model_path: str,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        n_ctx: int = 2048,
        n_threads: int = 4,
    ):
        self.gguf_path = gguf_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.n_ctx = n_ctx

        # ── Load target (VibeBladeFast GGUF) ─────────────────────────────────
        print(f"[DFlash] Loading target: {gguf_path}", flush=True)
        t0 = time.perf_counter()
        self._target = VibeBladeFast()
        self._target.load(gguf_path)
        cfg = self._target.config
        print(f"[DFlash] Target loaded in {time.perf_counter()-t0:.1f}s")
        print(f"[DFlash]   n_layers={cfg['n_layers']}, hidden_dim={cfg['hidden_dim']}, "
              f"vocab={cfg['vocab_size']}, arch={cfg['arch']}", flush=True)

        self._target_vocab_size = cfg["vocab_size"]
        self._target_hidden_dim  = cfg["hidden_dim"]
        self._target_n_layers    = cfg["n_layers"]

        # Validate layer indices
        bad = [i for i in QWEN3_LAYER_INDICES if i >= self._target_n_layers]
        if bad:
            raise ValueError(
                f"Layer indices {bad} exceed target layers ({self._target_n_layers}). "
                f"Adjust QWEN3_LAYER_INDICES for this model."
            )

        # ── Load draft (HF DFlash) ───────────────────────────────────────────
        print(f"[DFlash] Loading draft: {dflash_model_path}", flush=True)
        t0 = time.perf_counter()
        self._draft = DFlashDraftModel.from_pretrained(
            dflash_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self._draft = self._draft.eval().to(device)
        print(f"[DFlash] Draft loaded in {time.perf_counter()-t0:.1f}s")
        print(f"[DFlash]   params={sum(p.numel() for p in self._draft.parameters())/1e6:.0f}M, "
              f"block_size={self._draft.block_size}", flush=True)

        # ── Load tokenizer (Qwen3 public) ─────────────────────────────────────
        self._tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B",
            trust_remote_code=True,
            use_fast=False,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._eos = self._tokenizer.eos_token_id
        # Use pad token as the mask token — any token works as a placeholder
        self._mask_token = self._tokenizer.pad_token_id

        # Draft model should use same vocab size as target
        draft_vocab = self._draft.config.vocab_size
        if draft_vocab != self._target_vocab_size:
            print(f"[DFlash] WARNING: draft vocab ({draft_vocab}) != target vocab ({self._target_vocab_size})")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sample_greedy(self, logits: np.ndarray) -> int:
        """Greedy argmax from float32 numpy logits."""
        return int(np.argmax(logits))

    def _build_target_hidden_tensor(self, context_hiddens_np: np.ndarray) -> torch.Tensor:
        """
        Convert a context hidden array (seq_len, concat_dim) into a torch tensor
        with shape (1, seq_len, concat_dim) on the correct device/dtype.
        """
        t = torch.from_numpy(context_hiddens_np).to(self.device, dtype=self.torch_dtype)
        return t.unsqueeze(0)  # [1, seq_len, concat]

    def _draft_tokens_to_torch(self, tokens: List[int]) -> torch.Tensor:
        """Embed draft token IDs using the target model's token embedding table
        (via VibeBladeFast.embedding). Returns shape [1, seq_len, hidden]."""
        emb_list = []
        for t in tokens:
            emb_np = self._target.embedding(t)           # numpy array [hidden_dim]
            t_emb   = torch.from_numpy(emb_np).to(self.device, dtype=self.torch_dtype)
            emb_list.append(t_emb)
        return torch.stack(emb_list, dim=0).unsqueeze(0)   # [1, seq_len, hidden]

    def _sample_draft(self, draft_logits: torch.Tensor) -> List[int]:
        """Greedy sample block_size tokens from draft logits (shape [1, block, vocab])."""
        return torch.argmax(draft_logits, dim=-1)[0].tolist()

    # ── Main generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        block_size: Optional[int] = None,
        verbose: bool = False,
    ) -> DFlashGenerateResult:
        if temperature > 1e-5:
            raise NotImplementedError("Temperature sampling not yet implemented; use temperature=0")

        block_size = block_size or self._draft.block_size

        # ── Tokenize prompt ────────────────────────────────────────────────────
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        input_len = input_ids.shape[1]
        tokens = input_ids[0].tolist()

        t_start = time.perf_counter()

        # STEP 1: Prefill prompt + extract hidden states at target layers
        # Returns: (logits_np, [hidden_np_layer0, hidden_np_layer1, ...])
        logits_np, hidden_layers = self._target.prefill_with_hidden(
            tokens, QWEN3_LAYER_INDICES
        )
        prompt_len = len(tokens)
        hd = self._target_hidden_dim

        # Reshape each hidden layer to (prompt_len, hd) and concatenate across layers
        hidden_reshaped = [h.reshape(prompt_len, hd) for h in hidden_layers]
        context_hiddens_np = np.concatenate(hidden_reshaped, axis=-1)  # (prompt_len, 5*hd)

        # prev_logits holds target posterior for the *next* token (position after prompt)
        prev_logits = logits_np

        # Stats
        stats = {"accepted": 0, "drafted": 0, "blocks": 0, "n_rejected": 0}

        # ── Main speculative loop ──────────────────────────────────────────────
        while len(tokens) - input_len < max_new_tokens:
            if tokens and tokens[-1] == self._eos:
                break

            stats["blocks"] += 1

            # Current target position equals number of tokens already consumed (prompt + generated)
            start_pos = len(tokens)   # matches target position after prefill + decodes

            # ── Build conditioning tensor from recent context hidden states ──────
            # For first iteration this is the full prompt hidden.
            # After each block we replace context with hiddens of tokens we just added.
            target_hidden_torch = self._build_target_hidden_tensor(context_hiddens_np)

            # ── Draft: generate block_size masked tokens ────────────────────────
            noise_ids  = [self._mask_token] * block_size
            noise_emb  = self._draft_tokens_to_torch(noise_ids)   # [1, block, hidden]

            # Compute full-sequence position embeddings (context + block) so that
            # the draft's rotary embeddings have the right seq_len for its attention
            # which concatenates target_hidden (context) and noise_embedding (query).
            context_len = context_hiddens_np.shape[0]
            total_len   = context_len + block_size
            pos_ids_full = torch.arange(total_len, device=self.device, dtype=torch.long).unsqueeze(0)
            # Use the draft's own rotary_emb to compute cos/sin for all positions
            dummy_hidden = torch.zeros(
                1, total_len, self._draft.config.hidden_size,
                device=self.device, dtype=self.torch_dtype
            )
            position_embeddings = self._draft.rotary_emb(dummy_hidden, pos_ids_full)  # (cos, sin)
            # Slice for the block's absolute positions
            pos_ids_block = pos_ids_full[:, context_len:]   # shape (1, block_size)

            with torch.inference_mode():
                denoised = self._draft(
                    noise_embedding=noise_emb,
                    target_hidden=target_hidden_torch,
                    position_ids=pos_ids_block,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    is_causal=False,
                )   # shape [1, block, hidden]

            # Project through target's LM head to get vocab logits (per-token)
            # DFlash outputs already pass through its final norm
            hidden_np = denoised.squeeze(0).float().cpu().numpy()  # [block, hidden]
            logits_list = []
            for i in range(hidden_np.shape[0]):
                logits_i = self._target.lm_head(hidden_np[i])      # returns list/array of [vocab]
                logits_list.append(np.array(logits_i, dtype=np.float32))
            draft_logits_np = np.stack(logits_list, axis=0)        # [block, vocab]
            draft_logits = torch.from_numpy(draft_logits_np).unsqueeze(0).to(self.device)  # [1, block, vocab]
            draft_tokens = self._sample_draft(draft_logits)

            if verbose:
                print(f"[Block {stats['blocks']}] Draft: {draft_tokens[:8]}...", flush=True)

            # ── Verification: accept/reject draft tokens against target posterior ─
            accepted_this_block = []      # tokens appended this block
            new_hiddens          = []      # per-token concatenated hidden (each (concat,))
            accepted_draft_count = 0
            rejected             = False

            for i, dt in enumerate(draft_tokens):
                greedy_next = int(np.argmax(prev_logits))
                if greedy_next != dt:
                    # Mismatch → accept the correction token (greedy_next), reject rest
                    correction = greedy_next
                    accepted_this_block.append(correction)
                    # Feed correction token to target to update KV and get hidden + next logits
                    logits_next, hidden_layers_corr = self._target.decode_with_hidden(
                        correction, QWEN3_LAYER_INDICES
                    )
                    # Concatenate layers for this token
                    hidden_concat = np.concatenate([h for h in hidden_layers_corr], axis=0)
                    new_hiddens.append(hidden_concat)
                    prev_logits = logits_next
                    rejected = True
                    stats["n_rejected"] += 1
                    if verbose:
                        print(f"  [Reject at pos {i}] draft={dt} accepted={correction}", flush=True)
                    break
                else:
                    # Draft token accepted
                    accepted_this_block.append(dt)
                    accepted_draft_count += 1
                    # Feed accepted draft token to target to get hidden and next logits
                    logits_next, hidden_layers_dt = self._target.decode_with_hidden(
                        dt, QWEN3_LAYER_INDICES
                    )
                    hidden_concat = np.concatenate([h for h in hidden_layers_dt], axis=0)
                    new_hiddens.append(hidden_concat)
                    prev_logits = logits_next
                    # continue checking next draft token

            else:
                # All draft tokens accepted → need one extra token from target posterior
                next_token = int(np.argmax(prev_logits))
                accepted_this_block.append(next_token)
                logits_next, hidden_layers_next = self._target.decode_with_hidden(
                    next_token, QWEN3_LAYER_INDICES
                )
                hidden_concat = np.concatenate([h for h in hidden_layers_next], axis=0)
                new_hiddens.append(hidden_concat)
                prev_logits = logits_next

            # ── Update tokens and stats ────────────────────────────────────────
            tokens.extend(accepted_this_block)
            stats["accepted"] += len(accepted_this_block)
            stats["drafted"] += len(draft_tokens)

            # New context for next block = hidden states of tokens we just added
            context_hiddens_np = np.stack(new_hiddens, axis=0)   # shape (new_len, concat)

            # Stop if EOS reached
            if tokens[-1] == self._eos:
                break

        # ── Detokenize ────────────────────────────────────────────────────────
        gen_tokens = tokens[input_len:]
        gen_text   = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)

        t_total = time.perf_counter() - t_start
        n_out   = len(gen_tokens)
        tps     = n_out / t_total if t_total > 0 else 0.0
        accept  = stats["accepted"] / max(1, stats["drafted"])

        return DFlashGenerateResult(
            text=gen_text,
            token_ids=gen_tokens,
            tokens_per_second=tps,
            accepted_tokens=stats["accepted"],
            total_drafted=stats["drafted"],
            acceptance_rate=accept,
            blocks=stats["blocks"],
            stopped_eos=(tokens[-1] == self._eos),
            latency_s=t_total,
        )


# ── Convenience CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFlash generation CLI")
    parser.add_argument("--gguf", default="/home/ubuntu/.vibeblade/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--dflash", default="/tmp/qwen-dflash")
    parser.add_argument("--prompt", default="Write a short story about a robot.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    integ = DFlashIntegration(
        gguf_path=args.gguf,
        dflash_model_path=args.dflash,
    )
    result = integ.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        block_size=args.block_size,
        verbose=args.verbose,
    )

    print(f"\n=== DFlash Result ===")
    print(f"Text: {result.text}")
    print(f"Tokens: {len(result.token_ids)}")
    print(f"Speed: {result.tokens_per_second:.1f} tok/s")
    print(f"Blocks: {result.blocks}")
    print(f"Acceptance: {result.acceptance_rate:.1%} ({result.accepted_tokens}/{result.total_drafted})")
