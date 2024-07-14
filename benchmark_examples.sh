#!/bin/bash

set -x

### Prefill ####
for model in ttt-linear-fast ttt-mlp-fast mamba transformer-vllm
do
  for plen in 2048 4096 8192 16384 32768
  do
    for bs in 8 16 32 64 128 256 512
    do
      python benchmark_prefill_decode.py --model-name ${model} \
                                         --batch ${bs} \
                                         --prompt_len ${plen} \
                                         --generate_len 1
    done
  done
done

### Decode ###
for model in ttt-linear-fast ttt-mlp-fast mamba transformer-vllm
do
  for glen in 512 1024 2048 4096 8192
  do
    for bs in 64 128 256 512 1024 2048 4096
    do
      python benchmark_prefill_decode.py --model-name ${model} \
                                         --batch ${bs} \
                                         --prompt_len 1 \
                                         --generate_len ${glen}
    done
  done
done

### Prefill then Decode 128 ###
for model in ttt-linear-fast ttt-mlp-fast mamba transformer-vllm
do
  for plen in 2048 4096 8192 16384 32768
  do
    for bs in 8 16 32 64 128 256 512
    do
      python benchmark_prefill_decode.py --model-name ${model} \
                                         --batch ${bs} \
                                         --prompt_len ${plen} \
                                         --generate_len 128
    done
  done
done
