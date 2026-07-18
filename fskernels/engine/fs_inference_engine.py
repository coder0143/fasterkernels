import time
import torch
import torch.nn.functional as F
from typing import Optional

import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
from fskernels.triton.triton_gqa_decode_hf import custom_hf_decode_attention_forward
from fskernels.engine.cuda_graph_runner import CUDAGraphRunner
from transformers import DynamicCache


def _patch_attention():
    qwen3_mod.eager_attention_forward = custom_hf_decode_attention_forward


def _restore_attention():
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as _native
    qwen3_mod.eager_attention_forward = _native


def sample_token(logits: torch.Tensor, temperature: float = 0.6,
                 top_k: int = 20, top_p: float = 0.95) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float('-inf')
        logits = logits.scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


class FsInferenceEngine:
    """
    High-performance inference engine for Qwen3-8B-FP8 on L4.

    Strategies
    ----------
    use_graph=True  : CUDA-graph decode  (~4–6x less Python overhead)
    use_graph=False : Eager Triton attn  (still faster than native sdpa)

    Usage
    -----
    engine = FsInferenceEngine(model, tokenizer)
    result = engine.generate(prompt, max_new_tokens=400)
    print(result['tok_per_sec'])
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_seq_len: int = 8192,
        device: str = "cuda",
        num_splits: int = 8,
        patch_attention: bool = True,
    ):
        if patch_attention:
            _patch_attention()

        self.model       = model
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.device      = device
        self.num_splits  = num_splits

        eos_ids = getattr(model.config, "eos_token_id", tokenizer.eos_token_id)
        self.eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 400,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        use_graph: bool = True,
        warmup_steps: int = 5,
    ) -> dict:
        inputs   = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        bsz, prefill_len = input_ids.shape

        cap = min(self.max_seq_len, prefill_len + max_new_tokens + 64)

        if use_graph:
            return self._generate_graphed(
                input_ids, bsz, prefill_len, cap,
                max_new_tokens, temperature, top_k, top_p, warmup_steps
            )
        return self._generate_eager(
            input_ids, max_new_tokens, temperature, top_k, top_p
        )

    def _generate_graphed(
        self, input_ids, bsz, prefill_len, cap,
        max_new_tokens, temperature, top_k, top_p, warmup_steps
    ) -> dict:
        runner = CUDAGraphRunner(
            self.model, bsz, cap, self.device, self.num_splits
        )

        prefill_logits = runner.prefill(input_ids)
        runner.build(warmup_steps=warmup_steps)

        next_tok = sample_token(prefill_logits, temperature, top_k, top_p)
        generated = [next_tok]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            logits = runner.step(next_tok)
            next_tok = sample_token(logits, temperature, top_k, top_p)
            generated.append(next_tok)
            if int(next_tok.item()) in self.eos_ids:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return self._pack_result(generated, elapsed, prefill_len)

    def _generate_eager(
        self, input_ids, max_new_tokens, temperature, top_k, top_p
    ) -> dict:
        bsz, prefill_len = input_ids.shape
        dyn_cache = DynamicCache()
        attn_mask = {"full_attention": None}

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=dyn_cache,
                use_cache=True,
                logits_to_keep=1,
                attention_mask=attn_mask,
            )

        next_tok = sample_token(out.logits[:, -1, :], temperature, top_k, top_p)
        generated = [next_tok]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        pos = prefill_len
        for _ in range(max_new_tokens - 1):
            cache_pos = torch.tensor([pos], dtype=torch.long, device=self.device)
            pos_ids   = torch.full((bsz, 1), pos, dtype=torch.long, device=self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=next_tok,
                    position_ids=pos_ids,
                    cache_position=cache_pos,
                    past_key_values=dyn_cache,
                    attention_mask=attn_mask,
                    use_cache=True,
                    logits_to_keep=1,
                )
            next_tok = sample_token(out.logits[:, -1, :], temperature, top_k, top_p)
            generated.append(next_tok)
            pos += 1
            if int(next_tok.item()) in self.eos_ids:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return self._pack_result(generated, elapsed, prefill_len)

    def _pack_result(self, generated, elapsed, prefill_len) -> dict:
        ids = torch.cat(generated, dim=-1)[0]
        n   = len(generated)
        return {
            "text":        self.tokenizer.decode(ids, skip_special_tokens=True),
            "new_tokens":  n,
            "prefill_len": prefill_len,
            "elapsed_s":   elapsed,
            "tok_per_sec": n / elapsed,
            "ms_per_tok":  elapsed * 1000 / n,
        }
