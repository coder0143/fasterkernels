import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Optional, List

import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
from fskernels.triton.triton_gqa_decode_hf import custom_hf_decode_attention_forward
from fskernels.engine.cuda_graph_runner import CUDAGraphRunner
from fskernels.engine.fs_inference_engine import sample_token


class SpeculativeEngine:
    """
    Speculative decoding with a small Qwen3 draft model.

    Theory
    ------
    With acceptance rate α and k draft tokens per round, the effective
    speedup ≈ (1 + k·α) / (1 + k·cost_draft/cost_target).

    For Qwen3-0.6B draft (≈10x smaller) with k=5 and α≈0.75:
        speedup ≈ (1 + 3.75) / (1 + 0.5) ≈ 3.2×  on top of the target model

    Combined with CUDA graphs on the target (~5x) → ~16x vs. native HF.

    Usage
    -----
    engine = SpeculativeEngine.from_model_ids(
        target_id="Qwen/Qwen3-8B-FP8",
        draft_id="Qwen/Qwen3-0.6B",
        device="cuda",
    )
    result = engine.generate(prompt, max_new_tokens=400, gamma=5)
    """

    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        device: str = "cuda",
        gamma: int = 5,
        max_seq_len: int = 8192,
        target_num_splits: int = 8,
    ):
        # Patch attention on the target model only (draft runs eager sdpa — it's tiny)
        qwen3_mod.eager_attention_forward = custom_hf_decode_attention_forward

        self.target = target_model
        self.draft  = draft_model
        self.tok    = tokenizer
        self.device = device
        self.gamma  = gamma
        self.max_seq_len    = max_seq_len
        self.target_splits  = target_num_splits

        eos = getattr(target_model.config, "eos_token_id", tokenizer.eos_token_id)
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)

    @classmethod
    def from_model_ids(
        cls,
        target_id: str,
        draft_id:  str,
        device: str = "cuda",
        **kw,
    ) -> "SpeculativeEngine":
        tok = AutoTokenizer.from_pretrained(target_id)
        target = AutoModelForCausalLM.from_pretrained(
            target_id, dtype="auto", device_map=device
        ).eval()
        draft = AutoModelForCausalLM.from_pretrained(
            draft_id, dtype=torch.bfloat16, device_map=device
        ).eval()
        return cls(target, draft, tok, device=device, **kw)

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 400,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        warmup_steps: int = 5,
    ) -> dict:
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        bsz, prefill_len = input_ids.shape

        cap = min(self.max_seq_len, prefill_len + max_new_tokens + 64)

        # --- Prefill target (with CUDA graph cache) ---
        target_runner = CUDAGraphRunner(
            self.target, bsz, cap, self.device, self.target_splits
        )
        target_prefill_logits = target_runner.prefill(input_ids)
        target_runner.build(warmup_steps=warmup_steps)

        # --- Prefill draft ---
        draft_cache = DynamicCache()
        with torch.no_grad():
            draft_out = self.draft(
                input_ids=input_ids,
                past_key_values=draft_cache,
                use_cache=True,
                logits_to_keep=1,
            )

        # Current accepted token (end of prefill)
        cur_tok = sample_token(target_prefill_logits, temperature, top_k, top_p)

        generated: List[torch.Tensor] = [cur_tok]
        n_accepted_total = 0
        n_rounds = 0

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        draft_pos = prefill_len

        while len(generated) < max_new_tokens:
            gamma = min(self.gamma, max_new_tokens - len(generated))

            # --- Draft forward: generate γ candidate tokens ---
            draft_toks: List[torch.Tensor] = []
            draft_probs: List[torch.Tensor] = []

            cur_in = cur_tok
            for _ in range(gamma):
                cache_pos = torch.tensor([draft_pos], dtype=torch.long, device=self.device)
                pos_ids   = torch.full((bsz, 1), draft_pos, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    dout = self.draft(
                        input_ids=cur_in,
                        position_ids=pos_ids,
                        cache_position=cache_pos,
                        past_key_values=draft_cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                if temperature == 0.0:
                    next_d = dout.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    draft_toks.append(next_d)
                else:
                    lp = F.softmax(dout.logits[:, -1, :] / temperature, dim=-1)
                    draft_probs.append(lp)
                    next_d = torch.multinomial(lp, 1)
                    draft_toks.append(next_d)
                cur_in = next_d
                draft_pos += 1

            # --- Target verification: score γ+1 tokens in one forward pass ---
            # The concatenated input includes the last accepted token + γ drafts
            verify_ids = torch.cat([cur_tok] + draft_toks, dim=1)  # [B, γ+1]
            verify_pos = prefill_len + len(generated) - 1
            verify_positions = torch.arange(
                verify_pos, verify_pos + gamma + 1,
                dtype=torch.long, device=self.device
            ).unsqueeze(0).expand(bsz, -1)  # [B, γ+1]

            # Verification is variable-length → run outside graph
            # (still benefits from our Triton attention kernel for long sequences)
            with torch.no_grad():
                tout = self.target(
                    input_ids=verify_ids,
                    position_ids=verify_positions,
                    past_key_values=target_runner.kv_cache,
                    use_cache=True,
                    logits_to_keep=gamma + 1,
                )
            # target_runner.kv_cache._pos is now stale — the verify pass wrote
            # γ+1 slots starting at verify_pos. Advance the runner accordingly.

            target_logits = tout.logits  # [B, γ+1, V]

            # --- Rejection sampling ---
            n_accept = 0
            last_good_tok = cur_tok

            if temperature == 0.0:
                for i in range(gamma):
                    target_argmax = target_logits[:, i, :].argmax(dim=-1, keepdim=True)
                    tok_i = draft_toks[i]

                    if (target_argmax == tok_i).all():
                        generated.append(tok_i)
                        last_good_tok = tok_i
                        n_accept += 1
                        if int(tok_i.item()) in self.eos_ids:
                            break
                    else:
                        generated.append(target_argmax)
                        last_good_tok = target_argmax
                        break

                # Bonus token from target at γ+1 position (always accepted)
                if n_accept == gamma:
                    bonus = target_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated.append(bonus)
                    last_good_tok = bonus
                    n_accept += 1
            else:
                target_probs = F.softmax(target_logits / temperature, dim=-1)
                for i in range(gamma):
                    q = target_probs[:, i, :]   # target prob at position i
                    p = draft_probs[i]          # draft prob for draft_toks[i]
                    tok_i = draft_toks[i]       # the draft token

                    ratio = (q.gather(1, tok_i) / (p.gather(1, tok_i) + 1e-9)).clamp(max=1.0)
                    accept = torch.rand(bsz, 1, device=self.device) < ratio

                    if accept.all():
                        generated.append(tok_i)
                        last_good_tok = tok_i
                        n_accept += 1
                        if int(tok_i.item()) in self.eos_ids:
                            break
                    else:
                        # Resample from adjusted distribution
                        adj = (q - p).clamp(min=0)
                        adj_sum = adj.sum(-1, keepdim=True)
                        adj = adj / torch.clamp(adj_sum, min=1e-9)
                        adj = torch.where(adj_sum > 1e-9, adj, q)
                        bonus = torch.multinomial(adj, 1)
                        generated.append(bonus)
                        last_good_tok = bonus
                        break

                # Bonus token from target at γ+1 position (always accepted)
                if n_accept == gamma:
                    bonus_logits = target_logits[:, -1, :]
                    bonus = sample_token(bonus_logits, temperature, top_k, top_p)
                    generated.append(bonus)
                    last_good_tok = bonus
                    n_accept += 1

            n_accepted_total += n_accept
            n_rounds += 1

            # Roll back draft cache to the accepted position
            draft_pos = prefill_len + len(generated) - 1
            draft_cache = self._trim_dynamic_cache(draft_cache, draft_pos)

            # Sync target graph runner position to accepted position
            target_runner._position = draft_pos
            target_runner.kv_cache._pos.fill_(draft_pos - 1)

            cur_tok = last_good_tok

            if int(cur_tok.item()) in self.eos_ids:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ids = torch.cat(generated, dim=-1)[0]
        n = len(generated)
        alpha = n_accepted_total / max(n_rounds * self.gamma, 1)

        return {
            "text":            self.tok.decode(ids, skip_special_tokens=True),
            "new_tokens":      n,
            "elapsed_s":       elapsed,
            "tok_per_sec":     n / elapsed,
            "ms_per_tok":      elapsed * 1000 / n,
            "acceptance_rate": alpha,
            "n_rounds":        n_rounds,
        }

    @staticmethod
    def _trim_dynamic_cache(cache, keep_len: int):
        """Trim KV cache to keep_len tokens (drop rolled-back draft tokens)."""
        if hasattr(cache, "crop"):
            cache.crop(keep_len)
            if hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = keep_len
            return cache

        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            # Fallback for old style DynamicCache
            new_cache = type(cache)()
            for li in range(len(cache.key_cache)):
                k = cache.key_cache[li][:, :, :keep_len, :]
                v = cache.value_cache[li][:, :, :keep_len, :]
                new_cache.key_cache.append(k)
                new_cache.value_cache.append(v)
            new_cache._seen_tokens = keep_len
            return new_cache

        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if hasattr(layer, "crop"):
                    layer.crop(keep_len)
            if hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = keep_len
            return cache

        return cache
