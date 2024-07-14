'''
Modified from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py
'''
import gc
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable, Optional

import torch
from transformers.generation import GreedySearchDecoderOnlyOutput


class TTTCache:

    def __init__(self, max_batch_size, model):
        self.config = model.config
        self.model = model
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.dtype = model.config.dtype
        self.seq_modeling_block = model.config.seq_modeling_block
        self.mini_batch_size = model.config.mini_batch_size
        self.params_dict = defaultdict(dict)
        if self.seq_modeling_block == 'ttt-linear' or self.seq_modeling_block == 'ttt-linear-fast':
            self.param_names = ["W1", "b1"]
        elif self.seq_modeling_block == 'ttt-mlp' or self.seq_modeling_block == 'ttt-mlp-fast':
            self.param_names = ["W1", "b1", "W2", "b2"]
        else:
            raise NotImplementedError(f"Sequence Modeling Block: {self.seq_modeling_block} Not Implemented!")

    def allocate_inference_cache(self):
        for layer_idx in range(self.model.config.num_hidden_layers):
            for name in self.param_names:
                weight = getattr(self.model.model.layers[layer_idx].seq_modeling_block, name)
                tiled_weight = torch.tile(weight, (self.max_batch_size,) + (1,) * (weight.dim() - 1))  # [B*nh,f,f]
                self.params_dict[f"{name}_init"][layer_idx] = tiled_weight
                self.params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)
            self.params_dict[f"conv_cache"][layer_idx] = torch.zeros(
                size=(self.max_batch_size, self.config.hidden_size, self.config.conv_kernel),
                dtype=self.dtype, device=weight.device
            )
            if self.config.conv_before_ttt:
                self.params_dict[f"pre_ttt_conv_cache"][layer_idx] = torch.zeros(
                    size=(self.max_batch_size, self.config.hidden_size, self.config.conv_kernel),
                    dtype=self.dtype, device=weight.device
                )

    def reset(self, max_seqlen, max_batch_size, model):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.model = model
        for layer_idx in range(self.model.config.num_hidden_layers):
            for name in self.param_names:
                weight = getattr(self.model.model.layers[layer_idx].seq_modeling_block, name)
                tiled_weight = torch.tile(weight, (max_batch_size,) + (1,) * (weight.dim() - 1))  # [B*nh,f,f]
                self.params_dict[f"{name}_init"][layer_idx].copy_(tiled_weight)
                self.params_dict[f"{name}_grad"][layer_idx].zero_()
            self.params_dict[f"conv_cache"][layer_idx].zero_()
            if self.config.conv_before_ttt:
                self.params_dict[f"pre_ttt_conv_cache"][layer_idx].zero_()


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    eos_token_id=None,
    vocab_size=None,
    tensor_parallel=1,
    cg=False,
):
    """Decoding
    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
    """
    batch_size, seqlen_og = input_ids.shape
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None

        ## Capture is_last_in_mini_batch = False
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            tensor_parallel=tensor_parallel,
            is_prefill=False,
            is_last_in_mini_batch=False,
        )

        inference_params = model._decoding_cache.inference_params

        inference_params.reset(max_length, batch_size, model)

        ## Capture is_last_in_mini_batch = True
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            tensor_parallel=tensor_parallel,
            is_prefill=False,
            is_last_in_mini_batch=True,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size, model)

    else:
        inference_params = TTTCache(max_batch_size=batch_size, model=model)
        inference_params.allocate_inference_cache()

    def get_logits(input_ids, inference_params):
        # If prompt=1, use decode mode directly
        decoding = inference_params.seqlen_offset > 0 or input_ids.shape[1] == 1

        if not cg or not decoding:
            if not decoding:
                # assume prompt is a multiple of ttt mini-batch size
                is_last_in_mini_batch = True
                is_prefill = True
                logits = model(
                    input_ids,
                    is_prefill=is_prefill,
                    is_last_in_mini_batch=is_last_in_mini_batch,
                    cache_params=inference_params,
                    num_last_tokens=1,
                ).logits.squeeze(dim=1)     # [BS,1,vocab] -> [BS,vocab]
            else:
                is_last_in_mini_batch = ((inference_params.seqlen_offset + 1) % inference_params.mini_batch_size == 0)
                is_prefill = False
                logits = model(
                    input_ids,
                    is_prefill=is_prefill,
                    is_last_in_mini_batch=is_last_in_mini_batch,
                    cache_params=inference_params,
                ).logits.squeeze(dim=1)  # [BS,1,vocab] -> [BS,vocab]
        else:
            ## cg and decoding
            is_prefill = False
            is_last_in_mini_batch = ((inference_params.seqlen_offset + 1) % inference_params.mini_batch_size == 0)
            logits = model._decoding_cache.run(
                input_ids, is_prefill, is_last_in_mini_batch
            ).squeeze(dim=1)  # [BS,decode_len,vocab_size]

        return logits[..., :vocab_size] if vocab_size is not None else logits

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    sequences = [input_ids]
    while not should_stop(sequences[-1], inference_params):
        logits = get_logits(sequences[-1], inference_params)  # [BS, V]
        inference_params.seqlen_offset += sequences[-1].shape[1]
        new_token = logits.argmax(dim=-1).unsqueeze(-1)  # greedy sampling for speed benchmark: [BS,1]
        sequences.append(new_token)

    output_cls = GreedySearchDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=None)


class GenerationMixin:

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        max_length,
        **kwargs,
    ):
        output = decode(
            input_ids, self, max_length, **kwargs
        )
        return output


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[TTTCache] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    decoding_seqlens=(1,),
    tensor_parallel=1,
    dtype=None,
    n_warmups=2,
    is_prefill=False,
    is_last_in_mini_batch=False,
):
    if cache is None:
        cache = DecodingCGCache()

    param_example = next(iter(model.parameters()))
    device = param_example.device

    if dtype is None:
        dtype = param_example.dtype

    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        cache.inference_params = TTTCache(
            max_batch_size=batch_size,
            model=model,
        )
        cache.inference_params.allocate_inference_cache()
        cache.mempool = torch.cuda.graphs.graph_pool_handle()

    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen, is_prefill, is_last_in_mini_batch) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen, is_prefill, is_last_in_mini_batch] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
                is_prefill=is_prefill,
                is_last_in_mini_batch=is_last_in_mini_batch,
            )

    def dispatch(input_ids, is_prefill, is_last_in_mini_batch):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen, is_prefill, is_last_in_mini_batch](input_ids)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
        model,
        inference_params,
        batch_size, max_seqlen, decoding_seqlen=1,
        mempool=None,
        n_warmups=2,
        is_prefill=False,
        is_last_in_mini_batch=False
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)

    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                cache_params=inference_params,
                is_prefill=is_prefill,
                is_last_in_mini_batch=is_last_in_mini_batch,
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    torch.cuda.current_stream().wait_stream(s)

    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            cache_params=inference_params,
            is_prefill=is_prefill,
            is_last_in_mini_batch=is_last_in_mini_batch,
        ).logits

    def run(new_input_ids):
        input_ids.copy_(new_input_ids)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og

    return run
