import torch
from dataclasses import dataclass, field
from typing import List, Optional


PAGE_SIZE = 16  # tokens per physical page


@dataclass
class PagedKVCache:
    """
    Manages a pool of physical KV pages shared across all concurrent sequences.

    Layout
    ------
    k_pool / v_pool : [num_pages, num_kv_heads, PAGE_SIZE, head_dim]
    block_table     : [max_seqs, max_blocks_per_seq]   logical → physical

    Allocation
    ----------
    Sequences are assigned physical pages on demand via a free-list.
    When a sequence is evicted / finishes, its pages are returned to the pool.

    Usage
    -----
    cache = PagedKVCache(config, num_pages=2048, max_seqs=8, max_blocks_per_seq=512, device='cuda')
    seq_id = cache.allocate_sequence()
    cache.write_page(seq_id, logical_page, layer_idx, k, v)
    seq_lens = cache.get_seq_lens()
    out = paged_flash_decode(q, cache.k_pool[layer_idx], cache.v_pool[layer_idx],
                             cache.block_table, seq_lens, PAGE_SIZE)
    cache.free_sequence(seq_id)
    """

    num_layers:         int
    num_pages:          int
    max_seqs:           int
    max_blocks_per_seq: int
    num_kv_heads:       int
    head_dim:           int
    dtype:              torch.dtype
    device:             str

    def __post_init__(self):
        # Physical page pools — one pool per layer for each K and V
        self.k_pool = torch.zeros(
            self.num_layers, self.num_pages, self.num_kv_heads, PAGE_SIZE, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        self.v_pool = torch.zeros_like(self.k_pool)

        # Logical → physical mapping: -1 = unallocated
        self.block_table = torch.full(
            (self.max_seqs, self.max_blocks_per_seq), -1,
            dtype=torch.int32, device=self.device
        )

        # Sequence metadata (CPU-side)
        self._seq_lens:    List[int]       = [0]   * self.max_seqs
        self._active:      List[bool]      = [False] * self.max_seqs
        self._free_pages:  List[int]       = list(range(self.num_pages))

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        num_pages:          int = 4096,
        max_seqs:           int = 16,
        max_blocks_per_seq: int = 512,
        device: str = "cuda",
    ) -> "PagedKVCache":
        return cls(
            num_layers         = config.num_hidden_layers,
            num_pages          = num_pages,
            max_seqs           = max_seqs,
            max_blocks_per_seq = max_blocks_per_seq,
            num_kv_heads       = config.num_key_value_heads,
            head_dim           = config.head_dim,
            dtype              = torch.bfloat16,
            device             = device,
        )

    # ------------------------------------------------------------------
    def allocate_sequence(self) -> int:
        for sid in range(self.max_seqs):
            if not self._active[sid]:
                self._active[sid] = True
                self._seq_lens[sid] = 0
                return sid
        raise RuntimeError("All sequence slots are occupied")

    def free_sequence(self, seq_id: int):
        n_pages = (self._seq_lens[seq_id] + PAGE_SIZE - 1) // PAGE_SIZE
        for lp in range(n_pages):
            pp = int(self.block_table[seq_id, lp].item())
            if pp >= 0:
                self._free_pages.append(pp)
                self.block_table[seq_id, lp] = -1
        self._active[seq_id] = False
        self._seq_lens[seq_id] = 0

    def _alloc_page(self) -> int:
        if not self._free_pages:
            raise RuntimeError("Out of KV pages")
        return self._free_pages.pop()

    # ------------------------------------------------------------------
    def append_kv(
        self,
        seq_id:    int,
        layer_idx: int,
        k: torch.Tensor,   # [B=1, KV_H, 1, D]
        v: torch.Tensor,
    ):
        """Append a single new token's K/V to the given sequence."""
        pos = self._seq_lens[seq_id]
        page_idx   = pos // PAGE_SIZE
        page_offset = pos % PAGE_SIZE

        # Allocate a new physical page at the start of each page
        if page_offset == 0:
            pp = self._alloc_page()
            self.block_table[seq_id, page_idx] = pp
        else:
            pp = int(self.block_table[seq_id, page_idx].item())

        # Write K/V into the physical page at the token offset
        self.k_pool[layer_idx, pp, :, page_offset, :] = k[0, :, 0, :]
        self.v_pool[layer_idx, pp, :, page_offset, :] = v[0, :, 0, :]

        self._seq_lens[seq_id] = pos + 1

    # ------------------------------------------------------------------
    def get_block_table_tensor(self, active_seq_ids: List[int]) -> torch.Tensor:
        idx = torch.tensor(active_seq_ids, dtype=torch.long, device=self.device)
        return self.block_table[idx]  # [n_active, max_blocks]

    def get_seq_lens_tensor(self, active_seq_ids: List[int]) -> torch.Tensor:
        lens = [self._seq_lens[sid] for sid in active_seq_ids]
        return torch.tensor(lens, dtype=torch.int32, device=self.device)

    # ------------------------------------------------------------------
    @property
    def pages_used(self) -> int:
        return self.num_pages - len(self._free_pages)

    @property
    def pages_free(self) -> int:
        return len(self._free_pages)

    def __repr__(self) -> str:
        return (f"PagedKVCache(pages={self.num_pages}, used={self.pages_used}, "
                f"seqs={sum(self._active)}/{self.max_seqs})")
