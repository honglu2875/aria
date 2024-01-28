"""Contains generation/sampling code"""

# This file contains code from https://github.com/facebookresearch/llama which
# is available under the following license:

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU
# General Public License version 3.
import functools
import logging
import math
import torch

from typing import List
from tqdm import tqdm

from aria.model import TransformerLM
from aria.tokenizer import Tokenizer
from aria.model.cache import KVCache


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# TODO: Add which instruments were detected in the prompt


def _get_cfg_coeff(cfg_gamma, cfg_mode, cur_pos, start_pos, total_len):
    if cfg_mode is None:
        return cfg_gamma
    elif cfg_mode == "linear":
        p = (cur_pos - start_pos) / (total_len - start_pos)
        return cfg_gamma * p + (1 - p)
    elif cfg_mode == "hat":
        p = (cur_pos - start_pos) / (total_len - start_pos)
        if 2 * cur_pos < total_len + start_pos:
            return cfg_gamma * 2 * p + (1 - 2 * p)
        else:
            return cfg_gamma * 2 * (1 - p) + (2 * p - 1)
    elif cfg_mode == "sine":
        p = (cur_pos - start_pos) / (total_len - start_pos)
        return (cfg_gamma - 1) * math.sin(p * 3.14159) + 1
    else:
        raise ValueError(f"Unknown cfg_mode: {cfg_mode}")


def _process_prompts(
    prompts,
    pad_token="<P>",
    neg_prompts=None,
    use_cfg=False,
    neg_prompt_len=None,
) -> list:
    """
    Preprocess prompts for generation.
    If cfg is used,
        1. the prompts and negative prompts will be combined.
        2. the negative prompts will be truncated for at most as long as the longest prompt.
    Args:
        prompts: list of prompts
        pad_token: pad token ('<P>')
        neg_prompts: list of negative prompts
        use_cfg: whether to use cfg
        neg_prompt_len: max length of negative prompts. If more than the longest prompt,
                        pad to this length.
    Returns:
        list of padded prompts
    """
    processed_prompts = []
    max_len = max(len(t) for t in prompts)
    pad_len = max(max_len, neg_prompt_len or 0)

    if use_cfg:
        if neg_prompts is None:
            neg_prompts = [t[-1:] for t in prompts]
        assert len(prompts) == len(
            neg_prompts
        ), "Prompts and neg_prompts must have the same count."

        for prompt in prompts + neg_prompts:
            processed_prompts.append(
                [pad_token] * max(0, pad_len - len(prompt)) + prompt[:pad_len]
            )
    else:
        max_len = max(len(t) for t in prompts)
        for prompt in prompts:
            processed_prompts.append(
                [pad_token] * (max_len - len(prompt)) + prompt
            )

    return processed_prompts


def _batch_encode(tokenizer, prompts: list[list]) -> torch.Tensor:
    return torch.stack([tokenizer.encode(p) for p in prompts], dim=0)


class CUDAGraphRunner:
    """Capture and run a CUDA graph, à la vllm."""

    def __init__(self, fn: callable):
        self.fn = fn
        self.graph = None
        self.input_buffers = {}
        self.output_buffers = {}

    def capture(
        self,
        src: torch.Tensor,
        attn_mask: torch.Tensor,
        past_kv: list[KVCache],
    ):
        assert self.graph is None

        # Warm up CUDA stream
        next_pos = past_kv[0].next_pos
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.fn(src, attn_mask, past_kv)
        torch.cuda.current_stream().wait_stream(s)
        for cache in past_kv:
            cache.next_pos = next_pos

        # Perform CUDA graph capturing
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            logits = self.fn(
                src,
                attn_mask=attn_mask,
                past_kv=past_kv,
            )
        torch.cuda.synchronize()

        self.input_buffers = {
            "src": src,
            "attn_mask": attn_mask,
        }
        self.output_buffers = {
            "logits": logits,
        }
        return

    def __call__(
        self,
        src: torch.Tensor,
        attn_mask: torch.Tensor,
        past_kv: list[KVCache],
    ):
        del past_kv

        self.input_buffers["src"].copy_(src)
        if attn_mask.data_ptr() != self.input_buffers["attn_mask"].data_ptr():
            self.input_buffers["attn_mask"].copy_(attn_mask)

        self.graph.replay()

        return self.output_buffers["logits"]

    @property
    def batch_size(self):
        return self.input_buffers["src"].shape[0]


class TorchCompileRunner:
    def __init__(self, fn: callable):
        self.fn = fn
        self.compiled_fn = None

    def capture(
        self,
        src: torch.Tensor,
        attn_mask: torch.Tensor,
        past_kv: list[KVCache],
    ):
        assert self.compiled_fn is None
        self.compiled_fn = torch.compile(self.fn)

    def __call__(
        self,
        src: torch.Tensor,
        attn_mask: torch.Tensor,
        past_kv: list[KVCache],
    ):
        del past_kv
        return self.compiled_fn(src, attn_mask, past_kv)



def _wrap_callable(
    fn: callable,
    attn_mask_buffer: torch.Tensor,
    initial_cache: list[KVCache],
    prompt_len: int,
    token_shape: tuple,
    compilation_method: str = "compile",
) -> callable:
    """A custom wrapper for cudagraph callables that takes a kv cache with
    varying lengths."""
    # The dict maps sequence position to the corresponding cudagraph callable
    if compilation_method not in ["compile", "cudagraph"]:
        raise ValueError(f"`compilation_method` needs to be one of ['compile', 'cudagraph']")
    
    graph_map = {}

    total_len = token_shape[1]
    if compilation_method == "cudagraph":
        logger.info("Capturing CUDA Graph:")
    elif compilation_method == "compile":
        logger.info("Compiling model:")

    for cur_pos in (
        pbar := tqdm(
            range(prompt_len, total_len),
            total=total_len - prompt_len,
            leave=False,
        )
    ):
        if cur_pos == prompt_len:
            token = torch.zeros(
                token_shape[:1] + (cur_pos,), dtype=torch.int, device="cuda"
            )
        else:
            token = torch.zeros(
                token_shape[:1] + (1,), dtype=torch.int, device="cuda"
            )
        attn_mask = attn_mask_buffer[:, :cur_pos]

        if compilation_method == "cudagraph":
            graph_map[cur_pos] = CUDAGraphRunner(fn)
        elif compilation_method == "compile":
            graph_map[cur_pos] = TorchCompileRunner(fn)

        graph_map[cur_pos].capture(token, attn_mask, initial_cache)

    # Reset the KV cache for the real run
    for cache in initial_cache:
        cache.next_pos = 0

    @functools.wraps(fn)
    def _fn(src, attn_mask, past_kv):
        cur_pos = attn_mask.shape[1]
        return graph_map[cur_pos](src, attn_mask, past_kv)

    return _fn


# Some good settings:
# temp=0.85, top_p=0.9, cfg_gamma=1.4


@torch.no_grad()
def greedy_sample(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: List[list],
    max_new_tokens: int,
    device: torch.device | None = None,
    cfg_gamma: float | None = 1.4,
    cfg_mode: str | None = None,
    cfg_window: int | None = None,
    neg_prompts: List[list] | None = None,
    neg_prompt_len: int | None = None,
    alpha: float | None = 0.4,
    force_end=False,
    temperature: float = 0.95,
    top_p: float = 0.95,
):
    """Performs greedy (top_p) autoregressive sampling on a batch of prompts.

    Args:
        model (TransformerLM): Model to sample from.
        tokenizer (Tokenizer): Tokenizer corresponding to model.
        prompts (List[list]): A list of prompts to sample as a batch.
        max_new_tokens (int): Maximum number of new generated tokens.
        device (torch.device, optional): Device to use. Defaults to None.
        cfg_gamma (float, optional): CFG gamma parameter. Defaults to 1.2.
            This parameter *determines* whether parameters related to CFG are used.
            None: No CFG or interpolation. `cfg_mode, neg_prompts, neg_prompt_len, alpha` are ignored.
        cfg_mode (str, optional): CFG mode. Defaults to None (applying constant CFG strength).
            "linear": linearly increasing/decreasing CFG strength from 1.
            "hat": piecewise-linearly scale CFG gamma: 1 -> gamma -> 1
            "sine": sine curve from 1 -> gamma -> 1
        cfg_window (int, optional): If not None, we use a sliding window of given length for negative prompt.
        neg_prompts (List[list], optional): Alternative prompts to sample from.
            Defaults to None ("unconditioned" model is approximated using only the last tokens of prompts).
        neg_prompt_len (int, optional): Max length used for the negative prompts.
            Defaults to None (align to prompts).
            When set, if `neg_prompt_len > max(t for t in prompts)`, we pad to `neg_prompt_len`.
        alpha (float, optional): an alpha parameter during interpolation.
            Only takes effect when neg_prompt_len < minimal length of neg_prompts. Defaults to 0.4.
        force_end (bool, optional): Whether to force the end of the prompt. Defaults to False.
        temperature (float, optional): Sampling temperature. Defaults to 0.75.
        top_p (float, optional): Parameter for top-p sampling. Defaults to 0.95.

    Returns:
        List[list]: The list of samples, decoded by the tokenizer.
    """
    assert tokenizer.return_tensors is True, "tokenizer must return tensors."
    device = device or torch.device("cuda")
    model.eval()

    pad_id = tokenizer.pad_id
    pad_tok = tokenizer.pad_tok
    eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]

    padded_combined_prompts = _process_prompts(
        prompts,
        pad_tok,
        neg_prompts,
        cfg_gamma is not None,
        neg_prompt_len=neg_prompt_len,
    )
    if neg_prompts is not None:
        padded_negative_prompts = _process_prompts(
            neg_prompts, pad_tok, None, False
        )
    else:
        padded_negative_prompts = [t[-1:] for t in prompts]

    prompt_len = len(padded_combined_prompts[0])
    if neg_prompts is not None:
        neg_offset_len = max(0, prompt_len - max(len(t) for t in prompts))
    else:
        neg_offset_len = 0

    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    logger.info(
        f"Using hyperparams: temp={temperature}, top_p={top_p}, gamma={cfg_gamma}, gen_len={max_new_tokens}"
    )

    total_len = prompt_len + max_new_tokens
    tokens = torch.full(
        (len(padded_combined_prompts), total_len), pad_id, device=device
    )
    tokens[:, :prompt_len] = _batch_encode(
        tokenizer, padded_combined_prompts
    ).to(device)
    full_neg_tokens = _batch_encode(tokenizer, padded_negative_prompts).to(
        device
    )

    dim_tok_inserted = [False for _ in range(tokens.size(0))]
    attn_mask = torch.ones(
        (len(padded_combined_prompts), total_len),
        device=device,
        dtype=torch.bool,
    )
    attn_mask[:, :prompt_len] = tokens[:, :prompt_len] != pad_id
    start_pos = prompt_len

    past_kv = model.get_cache(
        max_batch_size=tokens.size(0), max_len=total_len, device=device
    )

    model.forward = _wrap_callable(
        model.forward,
        attn_mask_buffer=attn_mask,
        initial_cache=past_kv,
        prompt_len=prompt_len,
        token_shape=tokens.shape,
    )

    for cur_pos in (
        pbar := tqdm(
            range(start_pos, total_len),
            total=total_len - start_pos,
            leave=False,
        )
    ):
        if cur_pos == start_pos:
            token = tokens[:, :start_pos]
        else:
            token = tokens[:, cur_pos - 1 : cur_pos]

        logits = model.forward(
            token, attn_mask=attn_mask[:, :cur_pos], past_kv=past_kv
        )
        logits = logits[:, -1, :]

        if cfg_gamma is not None:
            coeff = _get_cfg_coeff(
                cfg_gamma, cfg_mode, cur_pos, start_pos, total_len
            )
            cond_logits = logits[: logits.size(0) // 2]
            uncond_logits = logits[logits.size(0) // 2 :]
            logits = uncond_logits + coeff * (cond_logits - uncond_logits)
            # Update the sliding window
            if cfg_window is not None and cur_pos - start_pos >= cfg_window:
                attn_mask[:, cur_pos - cfg_window].zero_()

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        # When alpha is used, in the first `max_new_tokens * alpha` generations, the negative
        # prompt completions still use its original content (if not exceeding). After that, the
        # negative prompt completions will be updated by the new tokens.
        if (
            alpha is not None
            and cur_pos - neg_offset_len < full_neg_tokens.size(0)
            and cur_pos - start_pos < max_new_tokens * alpha
        ):
            neg_slice = full_neg_tokens[:, cur_pos - neg_offset_len]
            next_token = torch.concat([next_token, neg_slice], dim=0)
        else:
            if cfg_gamma is not None:
                next_token = next_token.repeat(2)  # Also update neg prompts

        # Insert dim tokens
        if force_end and cur_pos >= total_len - 130:
            for _idx in range(tokens.size(0)):
                if dim_tok_inserted[_idx] is False and tokenizer.id_to_tok[
                    next_token[_idx].item()
                ][0] not in ("dur", "onset"):
                    next_token[_idx] = tokenizer.tok_to_id[tokenizer.dim_tok]

        # Update dim_tok_inserted
        for _idx in range(tokens.size(0)):
            if next_token[_idx] == tokenizer.tok_to_id[tokenizer.dim_tok]:
                dim_tok_inserted[_idx] = True

        tokens[:, cur_pos] = next_token

    decoded = []
    for idx, seq in enumerate(tokens.tolist()):
        if cfg_gamma is not None and 2 * idx >= tokens.size(0):
            break
        # Cut to eos tok if any
        try:
            seq = seq[: seq.index(eos_id)]
        except ValueError:
            pass
        decoded.append(tokenizer.decode(seq))

    for idx, seq in enumerate(decoded):
        if tokenizer.eos_tok in seq:
            eos_idx = seq.index(tokenizer.eos_tok)
            decoded[idx] = seq[:eos_idx]

    return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
