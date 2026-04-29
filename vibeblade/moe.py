"""VibeBlade MoE (Mixture of Experts) Module

Provides CPU/RAM-only Mixture of Experts support for LLaMA-style models,
following the llama.cpp GGUF tensor naming conventions.

Supported tensor layouts:
  - llama.cpp consolidated: blk.{i}.ffn_gate_exps.weight, blk.{i}.ffn_up_exps.weight, etc.
  - Per-expert alternate:    blk.{i}.expert.{e}.ffn_gate.weight, blk.{i}.expert.{e}.ffn_up.weight, etc.
  - Shared (always-on) expert: blk.{i}.ffn_gate_shunt.weight (DeepSeek style)

All operations use numpy only — no torch, no GPU dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MoEConfig",
    "ExpertRouter",
    "MoEExpertSet",
    "detect_moe_config",
    "moe_ffn_silu",
    "load_moe_weights_from_layer",
    "_silu",
    "_dense_ffn",
]


# ── helpers (self-contained so the module has no cross-imports) ─────────────


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x).  Matches transformer.silu()."""
    x32 = x.astype(np.float32)
    return x32 * (1.0 / (1.0 + np.exp(-x32)))


def _dense_ffn(
    x: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
) -> np.ndarray:
    """SwiGLU FFN: down(silu(gate(x)) * up(x)).

    Uses **MoE weight convention** (no transpose):
        gate_w: (shared_dim, expert_dim)
        up_w:   (shared_dim, expert_dim)
        down_w: (expert_dim, shared_dim)

    Handles 1D (hidden_dim,) and 2D (batch, hidden_dim) inputs.
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]  # (1, hidden_dim)
        squeeze = True
    else:
        squeeze = False
    gate = x @ gate_w      # (batch, expert_dim)
    up = x @ up_w          # (batch, expert_dim)
    hidden = _silu(gate) * up
    out = hidden @ down_w  # (batch, hidden_dim)
    if squeeze:
        out = out[0]
    return out


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along *axis*."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MoEConfig
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MoEConfig:
    """Architecture parameters for a single MoE FFN layer."""

    num_experts: int
    num_active: int          # top-k experts selected per token
    expert_dim: int          # intermediate dimension *per expert*
    shared_dim: int          # model hidden dimension
    router_topk: int | None = None  # defaults to num_active
    has_shared_expert: bool = False

    def __post_init__(self):
        if self.router_topk is None:
            self.router_topk = self.num_active

    # ── convenience -----------------------------------------------------------

    @classmethod
    def from_tensors(
        cls,
        router_w: np.ndarray,
        gate_exps: np.ndarray,
        up_exps: np.ndarray,
        down_exps: np.ndarray,
        num_active: int = 2,
    ) -> "MoEConfig":
        """Derive config from raw weight tensors.

        router_w:  (shared_dim, num_experts)
        gate_exps: (num_experts, shared_dim, expert_dim)
        up_exps:   (num_experts, shared_dim, expert_dim)
        down_exps: (num_experts, expert_dim, shared_dim)
        """
        return cls(
            num_experts=int(router_w.shape[1]),
            num_active=num_active,
            expert_dim=int(gate_exps.shape[2]),
            shared_dim=int(gate_exps.shape[1]),
        )

    def __repr__(self) -> str:
        parts = [f"num_experts={self.num_experts}",
                 f"num_active={self.num_active}",
                 f"expert_dim={self.expert_dim}",
                 f"shared_dim={self.shared_dim}"]
        if self.has_shared_expert:
            parts.append("shared_expert=True")
        return f"MoEConfig({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertRouter
# ═══════════════════════════════════════════════════════════════════════════════


class ExpertRouter:
    """Routes tokens to top-k experts via a learned gating network.

    The router computes ``logits = x @ W_router``, applies softmax, then
    selects the top-k experts (by weight) for each token.
    """

    def __init__(self, router_weight: np.ndarray, topk: int = 2):
        """Initialise the router.

        Args:
            router_weight: (shared_dim, num_experts) — gating projection.
            topk: number of experts to select per token.
        """
        self.weight = np.asarray(router_weight, dtype=np.float32)
        self.topk = int(topk)
        self.num_experts = int(self.weight.shape[1])
        self.shared_dim = int(self.weight.shape[0])

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Select top-k experts for input *x*.

        Args:
            x: (batch, shared_dim) or (shared_dim,) — hidden states.

        Returns:
            expert_indices: (batch, topk) — selected expert IDs (int).
            expert_weights: (batch, topk) — softmax weights for selected experts.
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        logits = x @ self.weight  # (batch, num_experts)
        probs = _softmax(logits, axis=-1)  # (batch, num_experts)

        # Top-k selection (descending)
        topk_idx = np.argpartition(probs, -self.topk, axis=-1)[..., -self.topk:]
        # Gather corresponding weights
        topk_weights = np.take_along_axis(probs, topk_idx, axis=-1)

        # Re-normalise the selected weights (standard practice in MoE)
        weight_sum = topk_weights.sum(axis=-1, keepdims=True)
        # Guard against division by zero (shouldn't happen with softmax)
        weight_sum = np.where(weight_sum < 1e-12, 1.0, weight_sum)
        topk_weights = topk_weights / weight_sum

        if squeeze:
            topk_idx = topk_idx[0]
            topk_weights = topk_weights[0]

        return topk_idx, topk_weights.astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return full routing logits (before top-k).  Useful for profiling."""
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]
        logits = x @ self.weight  # (batch, num_experts)
        if squeeze:
            logits = logits[0]
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# MoEExpertSet
# ═══════════════════════════════════════════════════════════════════════════════


class MoEExpertSet:
    """Holds all expert weight tensors for a single MoE layer.

    GGUF/llama.cpp consolidated layout:
        gate_weights: (num_experts, expert_dim, shared_dim)
        up_weights:   (num_experts, expert_dim, shared_dim)
        down_weights: (num_experts, shared_dim, expert_dim)
    """

    def __init__(
        self,
        gate_weights: np.ndarray,
        up_weights: np.ndarray,
        down_weights: np.ndarray,
    ):
        self.gate = np.asarray(gate_weights, dtype=np.float32)
        self.up = np.asarray(up_weights, dtype=np.float32)
        self.down = np.asarray(down_weights, dtype=np.float32)

        assert self.gate.shape == self.up.shape, \
            f"gate {self.gate.shape} != up {self.up.shape}"
        # gate/up: (E, expert_dim, shared_dim), down: (E, shared_dim, expert_dim)
        assert self.gate.shape[0] == self.down.shape[0] and \
               self.gate.shape[2] == self.down.shape[1], \
            f"gate {self.gate.shape} incompatible with down {self.down.shape}"

    @property
    def num_experts(self) -> int:
        return int(self.gate.shape[0])

    @property
    def expert_dim(self) -> int:
        return int(self.gate.shape[2])

    @property
    def shared_dim(self) -> int:
        return int(self.gate.shape[1])

    def get_expert(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (gate_w, up_w, down_w) for expert *idx*.

        gate_w: (shared_dim, expert_dim)
        up_w:   (shared_dim, expert_dim)
        down_w: (expert_dim, shared_dim)
        """
        # self.gate/up: (expert_dim, shared_dim) [GGUF], transpose to (shared_dim, expert_dim)
        return self.gate[idx].T, self.up[idx].T, self.down[idx]

    def get_experts_batch(
        self, indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return batched expert weights.

        Args:
            indices: 1-D array of expert IDs, shape (batch,).

        Returns:
            gate_w: (batch, shared_dim, expert_dim)
            up_w:   (batch, shared_dim, expert_dim)
            down_w: (batch, expert_dim, shared_dim)
        """
        # self.gate/up are (E, expert_dim, shared_dim) [GGUF format]
        # Transpose to (E, shared_dim, expert_dim) for einsum compatibility
        gate_batch = np.transpose(self.gate[indices], (0, 2, 1))
        up_batch   = np.transpose(self.up[indices],   (0, 2, 1))
        # self.down is already (E, shared_dim, expert_dim) — index directly
        return gate_batch, up_batch, self.down[indices]


# ═══════════════════════════════════════════════════════════════════════════════
# detect_moe_config
# ═══════════════════════════════════════════════════════════════════════════════

# Regex patterns for the two supported MoE naming conventions
_RE_GGUF_ROUTER = re.compile(r"^blk\.\d+\.ffn_gate_inp\.weight$")
_RE_GGUF_GATE   = re.compile(r"^blk\.\d+\.ffn_gate_exps\.weight$")
_RE_GGUF_UP     = re.compile(r"^blk\.\d+\.ffn_up_exps\.weight$")
_RE_GGUF_DOWN   = re.compile(r"^blk\.\d+\.ffn_down_exps\.weight$")
_RE_GGUF_SHARED = re.compile(r"^blk\.\d+\.ffn_gate_shunt\.weight$")

_RE_ALT_ROUTER  = re.compile(r"^blk\.\d+\.ffn_router\.weight$")
_RE_ALT_EXPERT  = re.compile(r"^blk\.(\d+)\.expert\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.weight$")


def detect_moe_config(weights: dict[str, np.ndarray]) -> MoEConfig | None:
    """Auto-detect MoE architecture from weight tensor names.

    Checks for ``ffn_gate_inp.weight`` (router) and ``ffn_gate_exps.weight``
    (consolidated experts), or the alternate per-expert naming scheme.

    Returns ``None`` if the model is dense (no MoE tensors found).
    """
    keys = set(weights.keys())

    # ── llama.cpp consolidated layout ──
    gguf_router_keys = [k for k in keys if _RE_GGUF_ROUTER.match(k)]
    gguf_gate_keys   = [k for k in keys if _RE_GGUF_GATE.match(k)]

    if gguf_router_keys and gguf_gate_keys:
        router_w = weights[gguf_router_keys[0]]
        gate_w   = weights[gguf_gate_keys[0]]
        # Derive dimensions from tensors
        num_experts = int(router_w.shape[1])
        shared_dim  = int(router_w.shape[0])
        expert_dim  = int(gate_w.shape[2])

        # Check for shared expert
        has_shared = any(_RE_GGUF_SHARED.match(k) for k in keys)

        # Default num_active to 2 (common for Mixtral/Qwen MoE)
        # Could be overridden by caller
        return MoEConfig(
            num_experts=num_experts,
            num_active=2,
            expert_dim=expert_dim,
            shared_dim=shared_dim,
            has_shared_expert=has_shared,
        )

    # ── alternate per-expert layout ──
    alt_router_keys = [k for k in keys if _RE_ALT_ROUTER.match(k)]
    alt_expert_keys = [k for k in keys if _RE_ALT_EXPERT.match(k)]

    if alt_router_keys and alt_expert_keys:
        router_w = weights[alt_router_keys[0]]
        num_experts = int(router_w.shape[1])
        shared_dim  = int(router_w.shape[0])

        # Get expert dim from an expert GATE weight specifically
        gate_keys = [k for k in alt_expert_keys if ".ffn_gate.weight" in k]
        sample = gate_keys[0] if gate_keys else alt_expert_keys[0]
        expert_gate = weights[sample]

        # Default: assume (expert_dim, shared_dim) transformer convention
        expert_dim = int(expert_gate.shape[0])

        # If second dim == shared_dim, could be either convention.
        # gate: (expert_dim, shared_dim) → shape[1]=shared_dim → expert_dim=shape[0] ✓
        # gate: (shared_dim, expert_dim) → shape[1]=expert_dim → need shape[0] as expert_dim... no
        # Since both conventions have shape[1]==shared_dim when (expert_dim, shared_dim),
        # and (shared_dim, expert_dim) has shape[0]==shared_dim, we can disambiguate:
        if expert_gate.ndim == 2 and expert_gate.shape[0] == shared_dim:
            # MoE convention: (shared_dim, expert_dim)
            expert_dim = int(expert_gate.shape[1])
        # else: transformer convention (expert_dim, shared_dim), expert_dim = shape[0] already set

        return MoEConfig(
            num_experts=num_experts,
            num_active=2,
            expert_dim=expert_dim,
            shared_dim=shared_dim,
        )

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# moe_ffn_silu  — the main MoE dispatch function
# ═══════════════════════════════════════════════════════════════════════════════


def moe_ffn_silu(
    x: np.ndarray,
    router: ExpertRouter,
    experts: MoEExpertSet,
    shared_expert_gate: np.ndarray | None = None,
    shared_expert_up: np.ndarray | None = None,
    shared_expert_down: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """MoE SwiGLU FFN — routes to top-k experts, weighted sum.

    Algorithm:
      1. (optional) Compute shared expert output.
      2. Route: ``indices, weights = router.route(x)``.
      3. For each selected expert *e*:
             ``expert_out = down(silu(gate(x)) * up(x))``
      4. Weighted sum: ``output = Σ w_i * expert_out_i + shared_out``.

    Args:
        x: (batch, shared_dim) or (shared_dim,) — hidden states after norm.
        router: ExpertRouter instance.
        experts: MoEExpertSet instance.
        shared_expert_gate / up / down: optional shared (always-on) expert
            weights with the same shapes as a single expert.

    Returns:
        output: (batch, shared_dim) or (shared_dim,)
        stats: dict with ``expert_indices``, ``expert_weights``,
               ``num_experts``, ``has_shared_expert``.
    """
    orig_ndim = x.ndim
    squeeze = orig_ndim == 1
    if squeeze:
        x = x[np.newaxis, :]  # (1, shared_dim)

    batch_size = x.shape[0]
    topk = router.topk

    # ── Step 1: shared expert (always-on dense FFN) ──
    has_shared = shared_expert_gate is not None
    if has_shared:
        shared_out = _dense_ffn(
            x, shared_expert_gate, shared_expert_up, shared_expert_down,
        )  # (batch, shared_dim) — _dense_ffn handles squeeze internally
        # Ensure 2D
        if shared_out.ndim == 1:
            shared_out = shared_out[np.newaxis, :]
    else:
        shared_out = np.zeros_like(x)

    # ── Step 2: route ──
    indices, weights = router.route(x)  # each (batch, topk)

    # ── Step 3–4: expert computation + weighted sum ──
    moe_out = np.zeros_like(x)  # (batch, shared_dim)

    for k in range(topk):
        # Expert IDs for this k-slot across the batch
        expert_ids = indices[:, k]  # (batch,)
        slot_weights = weights[:, k]  # (batch,)

        # Gather expert weights: (batch, shared_dim, expert_dim) etc.
        gate_w, up_w, down_w = experts.get_experts_batch(expert_ids)

        # Compute each expert's output in one batched matmul
        gate_proj = np.einsum("bs,bse->be", x, gate_w)   # (batch, expert_dim)
        up_proj   = np.einsum("bs,bse->be", x, up_w)     # (batch, expert_dim)
        hidden = _silu(gate_proj) * up_proj               # (batch, expert_dim)
        expert_out = np.einsum("be,bed->bd", hidden, down_w)  # (batch, shared_dim)

        # Accumulate weighted contribution
        moe_out += slot_weights[:, np.newaxis] * expert_out

    output = moe_out + shared_out

    # ── Build stats ──
    stats = {
        "expert_indices": indices,
        "expert_weights": weights,
        "num_experts": experts.num_experts,
        "topk": topk,
        "has_shared_expert": has_shared,
        "batch_size": batch_size,
    }

    if squeeze:
        output = output[0]

    return output, stats


# ═══════════════════════════════════════════════════════════════════════════════
# load_moe_weights_from_layer  — weight extraction helper
# ═══════════════════════════════════════════════════════════════════════════════


def load_moe_weights_from_layer(
    weights: dict[str, np.ndarray],
    layer_idx: int,
) -> tuple[np.ndarray | None, MoEExpertSet | None, dict]:
    """Extract MoE router + expert weights for a single layer.

    Supports both naming conventions:
      * llama.cpp consolidated:
            ``blk.{i}.ffn_gate_inp.weight``  (router)
            ``blk.{i}.ffn_gate_exps.weight`` (expert gates)
            ``blk.{i}.ffn_up_exps.weight``   (expert ups)
            ``blk.{i}.ffn_down_exps.weight`` (expert downs)
      * Per-expert alternate:
            ``blk.{i}.ffn_router.weight``
            ``blk.{i}.expert.{e}.ffn_gate.weight``
            ``blk.{i}.expert.{e}.ffn_up.weight``
            ``blk.{i}.expert.{e}.ffn_down.weight``

    Also collects any shared-expert or auxiliary weights into *extra*.

    Returns:
        router_weight (np.ndarray | None) — (shared_dim, num_experts) or None
        MoEExpertSet | None               — consolidated expert weights
        extra (dict)                      — e.g. ``shared_gate``, ``shared_up``,
                                          ``shared_down``
    """
    pfx = f"blk.{layer_idx}"
    keys = set(weights.keys())

    router_w = None
    gate_w = None
    up_w = None
    down_w = None
    extra: dict[str, np.ndarray] = {}

    # ── Try llama.cpp consolidated layout first ──
    router_key = f"{pfx}.ffn_gate_inp.weight"
    gate_key   = f"{pfx}.ffn_gate_exps.weight"
    up_key     = f"{pfx}.ffn_up_exps.weight"
    down_key   = f"{pfx}.ffn_down_exps.weight"

    if router_key in keys and gate_key in keys:
        router_w = weights[router_key]
        gate_w   = weights[gate_key]
        up_w     = weights.get(up_key)
        down_w   = weights.get(down_key)

        # Shared expert (DeepSeek-style "shunt" or Qwen-style "shexp")
        for shared_suffix, alias in [("shexp", "shexp"), ("shunt", "shunt")]:
            shared_gate_key = f"{pfx}.ffn_gate_{shared_suffix}.weight"
            shared_up_key   = f"{pfx}.ffn_up_{shared_suffix}.weight"
            shared_down_key = f"{pfx}.ffn_down_{shared_suffix}.weight"
            if shared_gate_key in keys:
                extra["shared_gate"] = weights[shared_gate_key]
                if shared_up_key in keys:
                    extra["shared_up"] = weights[shared_up_key]
                if shared_down_key in keys:
                    extra["shared_down"] = weights[shared_down_key]
                break

    else:
        # ── Try alternate per-expert layout ──
        alt_router_key = f"{pfx}.ffn_router.weight"
        if alt_router_key in keys:
            router_w = weights[alt_router_key]
            num_experts = int(router_w.shape[1])

            # Collect per-expert weights
            g_list, u_list, d_list = [], [], []
            for e in range(num_experts):
                gk = f"{pfx}.expert.{e}.ffn_gate.weight"
                uk = f"{pfx}.expert.{e}.ffn_up.weight"
                dk = f"{pfx}.expert.{e}.ffn_down.weight"
                if gk in keys:
                    g_list.append(weights[gk])
                if uk in keys:
                    u_list.append(weights[uk])
                if dk in keys:
                    d_list.append(weights[dk])

            if g_list and u_list and d_list:
                gate_w = np.stack(g_list, axis=0)
                up_w   = np.stack(u_list, axis=0)
                down_w = np.stack(d_list, axis=0)

    # ── Build results ──
    if router_w is None or gate_w is None or up_w is None or down_w is None:
        return None, None, extra

    expert_set = MoEExpertSet(gate_w, up_w, down_w)
    return router_w, expert_set, extra


# Keep the private name as an alias for the public one (backward compat)
_load_moe_weights_from_layer = load_moe_weights_from_layer
