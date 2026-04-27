"""VibeBlade PagedAttention — OS-style paged KV cache with prefix sharing.

Replaces the contiguous ring buffer with paged allocation, reducing memory
waste from fragmentation and enabling cross-request prefix sharing (vLLM-style).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PagedKVCache:
    """Paged KV cache with lazy allocation and prefix sharing.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension per head.
    num_pages : int
        Total physical pages in the pool.
    page_size : int
        Tokens per page (default 16).
    dtype : np.dtype
        Storage dtype (default float16).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_pages: int,
        page_size: int = 16,
        dtype: np.dtype = np.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_pages = num_pages
        self.page_size = page_size
        self.dtype = dtype

        # Pre-allocate page pool: (num_layers, 2, num_pages, num_heads, page_size, head_dim)
        # 2 = K and V
        self._page_pool = np.zeros(
            (num_layers, 2, num_pages, num_heads, page_size, head_dim),
            dtype=dtype,
        )

        # Per-layer page tables: logical_page_idx -> physical_page_idx
        self._page_tables: list[dict[int, int]] = [{} for _ in range(num_layers)]
        self._free_pages: set[int] = set(range(num_pages))
        self._seq_len = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def append(self, layer_idx: int, key: np.ndarray, value: np.ndarray,
               position: int | None = None) -> None:
        """Append K/V for one token to the cache.

        Parameters
        ----------
        layer_idx : int
        key : np.ndarray, shape ``(num_heads, head_dim)`` or ``(num_heads, 1, head_dim)``
        value : np.ndarray, same shape as *key*
        position : int or None
            Explicit position to write at. If None, uses self._seq_len
            (auto-increment mode for single-request generation).
        """
        if key.ndim == 3:
            key = key[:, 0, :]
            value = value[:, 0, :]

        if position is None:
            position = self._seq_len

        logical_page = position // self.page_size
        offset = position % self.page_size

        # Allocate a new page if needed
        if logical_page not in self._page_tables[layer_idx]:
            if not self._free_pages:
                raise MemoryError(
                    f"PagedKVCache exhausted all {self.num_pages} pages. "
                    f"Cannot allocate page for layer {layer_idx}."
                )
            phys_page = self._free_pages.pop()
            self._page_tables[layer_idx][logical_page] = phys_page

        phys = self._page_tables[layer_idx][logical_page]
        self._page_pool[layer_idx, 0, phys, :, offset, :] = key
        self._page_pool[layer_idx, 1, phys, :, offset, :] = value

        # Only auto-increment when using auto mode
        if position is None or position == self._seq_len:
            self._seq_len = max(self._seq_len, position + 1)

    def get(
        self, layer_idx: int, start: int = 0, end: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve cached K/V for *layer_idx* over position range ``[start, end)``.

        Returns
        -------
        (K, V) each with shape ``(num_heads, seq_len, head_dim)``
        """
        if end is None:
            end = self._seq_len
        if end <= start:
            return (
                np.empty((self.num_heads, 0, self.head_dim), dtype=self.dtype),
                np.empty((self.num_heads, 0, self.head_dim), dtype=self.dtype),
            )

        seq_len = end - start
        k_out = np.empty((self.num_heads, seq_len, self.head_dim), dtype=self.dtype)
        v_out = np.empty((self.num_heads, seq_len, self.head_dim), dtype=self.dtype)

        pos = start
        out_idx = 0
        while pos < end:
            logical_page = pos // self.page_size
            offset = pos % self.page_size
            # How many tokens remain in this page
            page_remaining = self.page_size - offset
            chunk = min(page_remaining, end - pos)

            if logical_page not in self._page_tables[layer_idx]:
                # Page not yet allocated — fill with zeros
                k_out[:, out_idx : out_idx + chunk, :] = 0
                v_out[:, out_idx : out_idx + chunk, :] = 0
            else:
                phys = self._page_tables[layer_idx][logical_page]
                k_out[:, out_idx : out_idx + chunk, :] = self._page_pool[
                    layer_idx, 0, phys, :, offset : offset + chunk, :
                ]
                v_out[:, out_idx : out_idx + chunk, :] = self._page_pool[
                    layer_idx, 1, phys, :, offset : offset + chunk, :
                ]

            pos += chunk
            out_idx += chunk

        return k_out, v_out

    # ------------------------------------------------------------------
    # Prefix sharing
    # ------------------------------------------------------------------

    def share_prefix(
        self,
        other: "PagedKVCache",
        layer_idx: int,
        prefix_len: int,
    ) -> None:
        """Share physical pages with *other* cache for a common prefix.

        Copies the page pool data and page table entries from *other* so that
        ``self`` can read the same KV data for the first *prefix_len* tokens.

        Parameters
        ----------
        other : PagedKVCache
            Cache whose prefix to share.
        layer_idx : int
        prefix_len : int
            Number of tokens in the shared prefix.
        """
        pages_needed = (prefix_len + self.page_size - 1) // self.page_size
        for lp in range(pages_needed):
            if lp not in other._page_tables[layer_idx]:
                continue
            other_phys = other._page_tables[layer_idx][lp]
            if lp in self._page_tables[layer_idx]:
                old_phys = self._page_tables[layer_idx][lp]
                self._free_pages.add(old_phys)
            # Allocate a new page in our pool and copy data
            if not self._free_pages:
                raise MemoryError(
                    f"PagedKVCache exhausted all {self.num_pages} pages. "
                    f"Cannot allocate page for prefix sharing."
                )
            our_phys = self._free_pages.pop()
            # Copy K and V data from other's pool to ours
            self._page_pool[layer_idx, 0, our_phys] = other._page_pool[layer_idx, 0, other_phys].copy()
            self._page_pool[layer_idx, 1, our_phys] = other._page_pool[layer_idx, 1, other_phys].copy()
            self._page_tables[layer_idx][lp] = our_phys

        # Update our sequence length to reflect the shared prefix
        self._seq_len = max(self._seq_len, prefix_len)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def free_pages(self, count: int) -> int:
        """Try to free up to *count* unused pages from the tail.

        Returns the actual number freed.
        """
        freed = 0
        # Walk backwards through logical pages, free if they belong only to
        # positions beyond current seq_len
        if self._seq_len == 0:
            return 0

        logical_pages_used = (self._seq_len + self.page_size - 1) // self.page_size
        for layer_idx in range(self.num_layers):
            to_free: list[int] = []
            for lp, pp in self._page_tables[layer_idx].items():
                if lp >= logical_pages_used:
                    to_free.append(lp)
            for lp in to_free:
                self._free_pages.add(self._page_tables[layer_idx].pop(lp))
                freed += 1
                if freed >= count:
                    return freed
        return freed

    @property
    def memory_usage_bytes(self) -> int:
        """Memory actually in use (pages allocated), not total pool."""
        pages_used = sum(len(pt) for pt in self._page_tables)
        element_bytes = np.dtype(self.dtype).itemsize
        return pages_used * self.num_heads * self.page_size * self.head_dim * 2 * element_bytes

    @property
    def total_pool_bytes(self) -> int:
        """Total memory allocated for the page pool."""
        return self._page_pool.nbytes

    @property
    def num_free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def num_used_pages(self) -> int:
        return self.num_pages - len(self._free_pages)

    def clear(self) -> None:
        """Reset all state."""
        self._page_pool.fill(0)
        self._page_tables = [{} for _ in range(self.num_layers)]
        self._free_pages = set(range(self.num_pages))
        self._seq_len = 0

    def __len__(self) -> int:
        return self._seq_len

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(layers={self.num_layers}, pages={self.num_pages}/{self.num_free_pages} free, "
            f"seq_len={self._seq_len}, page_size={self.page_size})"
        )

    def bulk_append(
        self, layer_idx: int, keys: np.ndarray, values: np.ndarray,
        start_pos: int = 0,
    ) -> None:
        """Append multiple K/V entries at once (used after prefill).

        Parameters
        ----------
        layer_idx : int
        keys : np.ndarray, shape ``(num_heads, seq_len, head_dim)``
        values : np.ndarray, same shape
        start_pos : int
            Starting position in the cache.
        """
        seq_len = keys.shape[1]
        for pos_offset in range(seq_len):
            self.append(
                layer_idx,
                keys[:, pos_offset, :],
                values[:, pos_offset, :],
                position=start_pos + pos_offset,
            )
        self._seq_len = max(self._seq_len, start_pos + seq_len)

    def to_flat_caches(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Export all KV data as flat per-layer arrays.

        Returns:
            (kv_caches_k, kv_caches_v) — each is a list of length num_layers.
            kv_caches_k[i] shape: (num_heads, seq_len, head_dim)
        """
        k_list: list[np.ndarray] = []
        v_list: list[np.ndarray] = []
        for layer_idx in range(self.num_layers):
            k, v = self.get(layer_idx)
            k_list.append(k)
            v_list.append(v)
        return k_list, v_list
