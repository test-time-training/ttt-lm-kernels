import pdb

import argparse
import time
import os
import os.path as osp
import logging

import torch

from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Copied from Mamba's repo: https://github.com/state-spaces/mamba/tree/main/mamba_ssm
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from ttt.configuration_ttt import TTT_STANDARD_CONFIGS, TTTConfig
from ttt.modeling_ttt import TTTForCausalLM


parser = argparse.ArgumentParser(description="Speed Benchmark for 1B TTT and Mamba")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--model-name", type=str, default="ttt-linear",
                    choices=[
                        'ttt-linear', 'ttt-linear-fast',
                        'ttt-mlp', 'ttt-mlp-fast',
                        'mamba', 'transformer-vllm'
                    ])
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--prompt_len", type=int, default=1)
parser.add_argument("--generate_len", type=int, default=128)
args = parser.parse_args()

def print_args(args):
    s = "Arguments: \n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

os.makedirs(args.logdir, mode=0o777, exist_ok=True)

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to the lowest level to catch all messages
file_handler = logging.FileHandler(osp.join(args.logdir, "logging.log"), mode='a')
console_handler = logging.StreamHandler()
# Create a formatter and set it on handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Test messages
logger.info(f"\n============== Model Name: {args.model_name} ==============")
config_str = print_args(args)
logger.info(config_str)

torch.random.manual_seed(0)
# follow Mamba's benchmark: https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/benchmarks/benchmark_generation_mamba_simple.py#L32
repeats = 3
device = "cuda"
dtype = torch.float16

is_mamba = args.model_name.startswith("mamba")
is_ttt = args.model_name.startswith("ttt")
is_transformer = args.model_name.startswith("transformer")
if is_mamba:
    # Copy Mamba 1.4b's config https://huggingface.co/state-spaces/mamba-1.4b/blob/main/config.json
    # except changing vocab size to llama2 tokenizer's vocab size
    config = {
        "d_model": 2048,
        "n_layer": 48,
        "vocab_size": 32000,
        "ssm_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8
    }
    model = MambaLMHeadModel(**config, device=device, dtype=dtype)
elif is_ttt:
    ttt_config = TTTConfig(**TTT_STANDARD_CONFIGS['1b'], vocab_size=32000)
    ttt_config.seq_modeling_block = args.model_name
    ttt_config.use_compile = True
    ttt_config.dtype = dtype
    ttt_config.fused_add_norm = True
    ttt_config.residual_in_fp32 = True
    model = TTTForCausalLM(ttt_config).to(device=device, dtype=dtype)
elif is_transformer:
    model = AutoModelForCausalLM.from_pretrained('LeoLearntoCode/llama-1.3b').to(dtype=dtype)
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}\n")
    del model
    # A modified llama model that has 1.3B param, since the smallest official llama is 7B.
    # This modified llama model is the same as the transformer baseline in our paper.
    # vLLM will stop generating once total_len exceeds `max_position_embeddings` in HF model config,
    # resulting in less time than actually needed.
    # On the other hand, `max_position_embeddings` could affect the size of static cache created by vLLM for CUDA Graph,
    # which could potentially affect its performance.
    # Therefore, we create seperate configs for benchmarking different range of total lengths as a workaround.
    total_len = args.prompt_len + args.generate_len
    if total_len < 8400:
        model_name = 'LeoLearntoCode/llama-1.3b'
    elif total_len < 16600:
        model_name = 'LeoLearntoCode/llama-1.3b-16k'
    elif total_len < 33000:
        model_name = 'LeoLearntoCode/llama-1.3b-32k'
    else:
        raise NotImplementedError(f"Please create your own config with `max_position_embeddings` >= {total_len}.")
    llm = LLM(model=model_name, skip_tokenizer_init=True)
else:
    raise NotImplementedError(f"Benchmark for {args.model_name} is not implemented.")

if is_mamba or is_ttt:
    model.eval()
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}\n")
    input_ids = torch.randint(1, 32000, (args.batch, args.prompt_len), dtype=torch.long, device=device)
    max_length = input_ids.shape[1] + args.generate_len
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=True
    )
else:
    sampling_params = SamplingParams(
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.generate_len,
        min_tokens=args.generate_len
    )
    input_ids = [
        torch.randint(low=1, high=32000, size=(args.prompt_len,)).tolist() for _ in range(args.batch)
    ]
    fn = lambda: llm.generate(
        prompt_token_ids=input_ids,
        sampling_params=sampling_params
    )

# Capture graph if cg=True.
# This is not timed, following Mamba's benchmark:
# https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/benchmarks/benchmark_generation_mamba_simple.py#L82
out = fn()
logger.info("Succeeded.")
prefill_len = len(input_ids[0])
if is_mamba or is_ttt:
    decode_len = len(out.sequences[0]) - prefill_len
else:
    decode_len = len(out[0].outputs[0].token_ids)
del out

torch.cuda.synchronize()
start = time.time()
for i in range(repeats):
    fn()
torch.cuda.synchronize()
avg_time = (time.time() - start) / repeats

logger.info(f"Prompt length: {prefill_len}, generation length: {decode_len}")
logger.info(f"Prompt processing + Decoding time: {avg_time * 1000:.0f}ms")
logger.info(f"Throughput (total tok = prefill): {args.batch * prefill_len / avg_time:.3f} tokens / s")
logger.info(f"Throughput (total tok = decode): {args.batch * decode_len / avg_time:.3f} tokens / s")
logger.info("==================================")
