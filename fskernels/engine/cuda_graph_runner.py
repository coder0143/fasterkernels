import torch
from typing import Optional, Tuple
from transformers.cache_utils import Cache


def get_kv_from_hf_cache(cache, layer_idx: int):
    """
    Robust helper to extract key and value tensors from any HF Cache structure.
    Handles:
      1. Modern transformers (>=4.36+): cache.layers[layer_idx].keys / .values
      2. Legacy/Custom caches: cache.key_cache[layer_idx] / cache.value_cache[layer_idx]
      3. Subscriptable caches: cache[layer_idx] -> (k, v)
    """
    if hasattr(cache, "layers") and layer_idx < len(cache.layers):
        layer = cache.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    try:
        kv = cache[layer_idx]
        return kv[0], kv[1]
    except Exception:
        pass

    raise AttributeError(f"Could not extract K/V states from cache object of type {type(cache)}")


class CUDAGraphCache(Cache):
    """
    Static KV cache with a GPU-tensor write-position for CUDA graph compatibility.

    HF's StaticCache uses a Python integer (_seen_tokens) to track where to write
    new K/V values. That integer is frozen at graph-capture time and never updates
    during replay, so every decode step would overwrite the same cache slot.

    We replace the Python integer with a single-element CUDA tensor (_pos).
    The CUDAGraphRunner calls _pos.fill_(current_pos) before each graph.replay(),
    so the index_copy_ operations in update() correctly scatter to the right slot.
    """

    def __init__(self, config, max_batch_size: int, max_cache_len: int,
                 device: str, dtype: torch.dtype):
        nL = config.num_hidden_layers
        kv_h = config.num_key_value_heads
        d = config.head_dim

        # [num_layers, B, KV_H, max_len, D]
        self.key_cache   = torch.zeros(nL, max_batch_size, kv_h, max_cache_len, d, dtype=dtype, device=device)
        self.value_cache = torch.zeros_like(self.key_cache)

        # 1-element GPU tensor — the only position-tracking state in the graph
        self._pos = torch.zeros(1, dtype=torch.long, device=device)
        self._num_layers = nL

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, cache_kwargs=None):
        """
        key_states / value_states: [B, KV_H, q_len, D] during decode or verification.
        index_copy_ scatters them into slots along the seq dim.
        """
        cache_position = None
        if cache_kwargs is not None:
            if isinstance(cache_kwargs, dict):
                cache_position = cache_kwargs.get("cache_position", None)
            elif hasattr(cache_kwargs, "cache_position"):
                cache_position = cache_kwargs.cache_position

        # Fallback if key_states has > 1 tokens but we have no cache_position
        if cache_position is None and key_states.size(2) > 1:
            pos = int(self._pos.item())
            cache_position = torch.arange(pos, pos + key_states.size(2), device=key_states.device)

        if cache_position is not None:
            self.key_cache[layer_idx].index_copy_(2, cache_position, key_states)
            self.value_cache[layer_idx].index_copy_(2, cache_position, value_states)
        else:
            self.key_cache[layer_idx].index_copy_(2, self._pos, key_states)
            self.value_cache[layer_idx].index_copy_(2, self._pos, value_states)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self._pos.item())

    def get_usable_length(self, seq_len: int, layer_idx: int = 0) -> int:
        return seq_len

    def get_mask_sizes(self, cache_position: torch.Tensor, *args, **kwargs) -> Tuple[int, int]:
        mask_seq_len = int(cache_position[-1].item()) + 1
        seen_tokens = int(self._pos.item())
        if seen_tokens == 0:
            return mask_seq_len, mask_seq_len
        return mask_seq_len, seen_tokens

    def load_from_cache(self, cache, seq_len: int):
        """
        Copy prefill K/V from any HF Cache type into our static buffers.
        """
        for li in range(self._num_layers):
            k, v = get_kv_from_hf_cache(cache, li)
            self.key_cache[li, :, :, :seq_len, :]   = k
            self.value_cache[li, :, :, :seq_len, :] = v


class CUDAGraphRunner:
    """
    Captures the single-token decode forward pass as a CUDA graph.

    Profiled on L4 (SM89):
      - Python-loop decode   → ~172 ms / token (current native HF)
      - CUDA graph decode    → ~25–40 ms / token  (target)

    API:
        runner = CUDAGraphRunner(model, batch_size=1, max_seq_len=2048)
        prefill_logits = runner.prefill(input_ids)       # DynamicCache, variable len
        runner.build()                                   # warmup + capture
        for step in range(max_new_tokens):
            logits = runner.step(next_token_id)          # graph.replay()
    """

    def __init__(
        self,
        model,
        batch_size: int,
        max_seq_len: int,
        device: str = "cuda",
        num_splits: int = 8,
    ):
        self.model       = model
        self.batch_size  = batch_size
        self.max_seq_len = max_seq_len
        self.device      = device
        self.num_splits  = num_splits

        self.kv_cache: Optional[CUDAGraphCache] = None
        self._graph:   Optional[torch.cuda.CUDAGraph] = None

        self._static_input_ids:  Optional[torch.Tensor] = None
        self._static_cache_pos:  Optional[torch.Tensor] = None
        self._static_pos_ids:    Optional[torch.Tensor] = None
        self._static_logits:     Optional[torch.Tensor] = None

        self._position: int = 0
        self._attn_mask_dict: dict = {"full_attention": None}

    # ------------------------------------------------------------------
    # Prefill: standard forward with DynamicCache, then hand off to graph cache
    # ------------------------------------------------------------------
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape

        # Pass past_key_values=None so HF creates a fully-initialised cache
        # internally (DynamicCache in 4.51 only sets up key_cache/value_cache
        # when it receives the model config, which happens inside the model).
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
                logits_to_keep=1,
            )

        returned_cache = out.past_key_values  # fully populated by the forward pass

        self.kv_cache = CUDAGraphCache(
            self.model.config, bsz, self.max_seq_len, self.device, self.model.dtype
        )
        self.kv_cache.load_from_cache(returned_cache, seq_len)
        self._position = seq_len

        # Detect if the model has sliding window layers (Qwen3-8B: no)
        if getattr(self.model.model, "has_sliding_layers", False):
            self._attn_mask_dict["sliding_attention"] = None

        return out.logits[:, -1, :]  # [B, vocab]

    # ------------------------------------------------------------------
    # Build: warmup to compile Triton kernels, then capture the graph
    # ------------------------------------------------------------------
    def build(self, warmup_steps: int = 5):
        assert self.kv_cache is not None, "Call prefill() first"

        from fskernels.triton.triton_gqa_decode_hf import set_num_splits
        set_num_splits(self.num_splits)

        bsz = self.batch_size
        pos0 = self._position

        self._static_input_ids = torch.zeros(bsz, 1, dtype=torch.long, device=self.device)
        self._static_cache_pos = torch.tensor([pos0], dtype=torch.long, device=self.device)
        self._static_pos_ids   = torch.full((bsz, 1), pos0, dtype=torch.long, device=self.device)

        # Warmup: drives Triton JIT compilation for all kernel configs
        with torch.no_grad():
            for i in range(warmup_steps):
                p = pos0 + i
                self._sync_position(p)
                self.model(
                    input_ids=self._static_input_ids,
                    position_ids=self._static_pos_ids,
                    cache_position=self._static_cache_pos,
                    past_key_values=self.kv_cache,
                    attention_mask=self._attn_mask_dict,
                    use_cache=True,
                    logits_to_keep=1,
                )
        torch.cuda.synchronize()

        # Capture at the next fresh position
        capture_pos = pos0 + warmup_steps
        self._sync_position(capture_pos)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            with torch.no_grad():
                _out = self.model(
                    input_ids=self._static_input_ids,
                    position_ids=self._static_pos_ids,
                    cache_position=self._static_cache_pos,
                    past_key_values=self.kv_cache,
                    attention_mask=self._attn_mask_dict,
                    use_cache=True,
                    logits_to_keep=1,
                )
        self._static_logits = _out.logits  # static tensor; lives in graph memory pool
        torch.cuda.synchronize()

        # Reset target position back to pos0 so generation starts at the correct post-prefill index
        self._position = pos0

    # ------------------------------------------------------------------
    # Step: update position buffers, replay graph, return logits
    # ------------------------------------------------------------------
    def step(self, next_token: torch.Tensor) -> torch.Tensor:
        assert self._graph is not None, "Call build() before step()"
        self._static_input_ids.copy_(next_token.view(self.batch_size, 1))
        self._sync_position(self._position)
        self._graph.replay()
        self._position += 1
        return self._static_logits[:, -1, :]  # [B, vocab]

    def _sync_position(self, pos: int):
        self._static_cache_pos.fill_(pos)
        self._static_pos_ids.fill_(pos)
        self.kv_cache._pos.fill_(pos)
