"""Generalized DFlash2 layer-hook math for arbitrary target depth.

DFlash (z-layer-blocked speculative decoding) hooks the TARGET's hidden
states at a fixed set of layer indices. The original VibeBlade integration
hardcoded Qwen3-4B's 32-layer layout: [1, 9, 17, 25, 33] (5 hooks, one
being the final norm output). This module generalizes that to ANY depth so
Qwen3.8-27B (and future models) get their correct hook set.

Rules (derived from the DFlash paper + z-lab checkpoints):
  - 5 hooks for any depth: 4 interior + final
  - interior hooks evenly spaced, odd-indexed, biased toward the first 2/3
    of the network (early layers carry most of the drafting signal)
  - final hook always = last layer output (post-final-norm equivalent)

For 32 layers → [1, 9, 17, 25, 33] exactly matches the shipped config,
so the generalization is strictly backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DFlash2Config:
    n_layers: int
    layer_indices: tuple
    block_size: int = 16
    draft_layers: int = 5
    # DFlash2 additions over DFlash1:
    adaptive_blocks: bool = True   # vary block size with acceptance rate
    min_block: int = 8
    max_block: int = 64

    def __post_init__(self):
        if self.layer_indices and len(self.layer_indices) != self.draft_layers:
            raise ValueError(
                f"DFlash2 needs {self.draft_layers} layer hooks, got {len(self.layer_indices)}"
            )


def dflash2_layer_indices(n_layers: int, draft_layers: int = 5) -> tuple:
    """Compute layer hook indices for a target with n_layers transformer
    blocks (output includes the final read at index n_layers as the last hook).

    n_layers=32 → (1, 9, 17, 25, 33)
    """
    if n_layers < 4:
        raise ValueError("need >= 4 layers")
    interior = draft_layers - 1
    # reference: qwen3-4b 32-layer → interior hooks 1,9,17,25 = 1 + k*8, k=0..3
    # step 8 = n_layers / 4 exactly; generalize with n//interior (32→8 ✓)
    step = max(1, n_layers // interior)
    idxs = [1 + k * step for k in range(interior)]
    # keep them strictly increasing and below the final layer
    idxs = sorted(set(idxs))[:interior]
    while len(idxs) < interior:  # degenerate tiny nets
        nxt = idxs[-1] + 1 if idxs else 1
        if nxt >= n_layers:
            break
        idxs.append(nxt)
    idxs.append(n_layers + 1)  # final hook: post-final-norm read (matches 32→33)
    return tuple(idxs)


def make_dflash2_config(n_layers: int, block_size: int = 16) -> DFlash2Config:
    idxs = dflash2_layer_indices(n_layers)
    return DFlash2Config(
        n_layers=n_layers,
        layer_indices=idxs,
        block_size=block_size,
    )


# ── adaptive block sizing (DFlash2 core idea) ────────────────────────────────

def next_block_size(acceptance_rate: float, current: int,
                    cfg: DFlash2Config | None = None) -> int:
    """Adapt draft block length from recent acceptance rate.

    High acceptance → longer blocks (more speculative depth); low → shrink.
    Hysteresis-free, clamped to [min_block, max_block].
    """
    cfg = cfg or DFlash2Config(n_layers=32, layer_indices=(1, 9, 17, 25, 33))
    if acceptance_rate >= 0.85:
        target = current * 2
    elif acceptance_rate >= 0.70:
        target = current + 4
    elif acceptance_rate >= 0.50:
        target = current
    else:
        target = max(cfg.min_block, current // 2)
    return int(max(cfg.min_block, min(cfg.max_block, target)))
