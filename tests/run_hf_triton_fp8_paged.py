import os
import sys
import time
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

from fskernels.paged_attn import PagedKVCache, paged_flash_decode, PAGE_SIZE
from fskernels.engine import sample_token

# Global paged cache instance for the demo
_DEMO_PAGED_CACHE: Optional[PagedKVCache] = None
_DEMO_SEQ_ID: int = 0


def custom_paged_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    global _DEMO_PAGED_CACHE, _DEMO_SEQ_ID

    bsz, q_heads, q_len, head_dim = query_states.shape
    past_key_values = kwargs.get("past_key_values", None)

    # Decode step: write to and read from PagedKVCache
    if past_key_values is not None and q_len == 1 and _DEMO_PAGED_CACHE is not None:
        layer_idx = getattr(module, "layer_idx", 0)

        # 1. Append the new K/V projection to the paged block allocator
        _DEMO_PAGED_CACHE.append_kv(_DEMO_SEQ_ID, layer_idx, key_states, value_states)

        # 2. Get block mapping and sequence lengths for Triton paged decode kernel
        block_table = _DEMO_PAGED_CACHE.get_block_table_tensor([_DEMO_SEQ_ID])
        seq_lens = _DEMO_PAGED_CACHE.get_seq_lens_tensor([_DEMO_SEQ_ID])

        # 3. Execute paged decode attention kernel
        q_2d = query_states.squeeze(2)  # [B, Q_H, D]
        out_2d = paged_flash_decode(
            q_2d,
            _DEMO_PAGED_CACHE.k_pool[layer_idx],
            _DEMO_PAGED_CACHE.v_pool[layer_idx],
            block_table,
            seq_lens,
            PAGE_SIZE,
        )
        return out_2d.unsqueeze(2), None

    # Prefill fallback: use native eager attention
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as native
    return native(
        module, query_states, key_states, value_states, attention_mask, scaling, dropout, **kwargs
    )


def main():
    global _DEMO_PAGED_CACHE, _DEMO_SEQ_ID

    # Patch the model's eager attention implementation
    qwen3_mod.eager_attention_forward = custom_paged_attention_forward

    model_card = "Qwen/Qwen3-8B-FP8"
    device = "cuda"

    print(f"Loading tokenizer & model {model_card} …")
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    tokenizer.clean_up_tokenization_spaces = False

    model = AutoModelForCausalLM.from_pretrained(
        model_card,
        device_map=device,
    ).eval()

    # Allocate Paged KV Cache pool (e.g., 2048 blocks of 16 tokens each)
    _DEMO_PAGED_CACHE = PagedKVCache.from_config(
        model.config,
        num_pages=2048,
        max_seqs=4,
        max_blocks_per_seq=512,
        device=device,
    )
    _DEMO_SEQ_ID = _DEMO_PAGED_CACHE.allocate_sequence()

    prompt = "Explain the hardware roofline constraints of custom CUDA attention kernels in detail:"
    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    bsz, prefill_len = input_ids.shape

    # 1. Run Prefill (generates first token and populates a temporary DynamicCache)
    print("\nRunning Prefill pass …")
    dyn_cache = DynamicCache()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=dyn_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    # 2. Transfer the prefill keys and values into the PagedKVCache
    print("Transferring prefill KV states to Paged Block Allocator …")
    for li in range(model.config.num_hidden_layers):
        # Retrieve computed prefill key/value states from Hugging Face's cache
        from fskernels.engine.cuda_graph_runner import get_kv_from_hf_cache
        k, v = get_kv_from_hf_cache(dyn_cache, li)
        # Append each token's prefill projection to the paged cache
        for t in range(prefill_len):
            _DEMO_PAGED_CACHE.append_kv(_DEMO_SEQ_ID, li, k[:, :, t:t+1, :], v[:, :, t:t+1, :])

    # 3. Autoregressive decode loop
    next_tok = sample_token(out.logits[:, -1, :], temperature=0.0)  # greedy
    generated = [next_tok]

    print("\nDecoding with Triton Paged attention …")
    t0 = time.perf_counter()

    max_new_tokens = 400
    for _ in range(max_new_tokens - 1):
        pos_ids = torch.full((bsz, 1), prefill_len + len(generated), dtype=torch.long, device=device)
        # Pass dyn_cache to satisfy model checks; our patched forward intercepts it and routes to _DEMO_PAGED_CACHE
        with torch.no_grad():
            out = model(
                input_ids=next_tok,
                position_ids=pos_ids,
                past_key_values=dyn_cache,
                use_cache=True,
                logits_to_keep=1,
            )
        next_tok = sample_token(out.logits[:, -1, :], temperature=0.0)
        generated.append(next_tok)

        if int(next_tok.item()) in {tokenizer.eos_token_id}:
            break

    elapsed = time.perf_counter() - t0
    n = len(generated)

    # Decode and print output
    decoded_text = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True)
    print(f"\nAnswer:\n{decoded_text}")

    print("\n" + "=" * 50)
    print(f"Tokens Generated : {n}")
    print(f"Time Taken       : {elapsed:.2f} seconds")
    print(f"Throughput       : {n / elapsed:.2f} tokens/sec")
    print(f"Paged cache status: {_DEMO_PAGED_CACHE}")
    print("=" * 50)

    # Cleanup
    _DEMO_PAGED_CACHE.free_sequence(_DEMO_SEQ_ID)


if __name__ == "__main__":
    main()
