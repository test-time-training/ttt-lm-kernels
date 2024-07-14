#!/bin/bash

pip install packaging
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install flash-attn --no-build-isolation
pip install causal-conv1d
pip install mamba-ssm
pip install vllm

cd ThunderKittens;
source env.src;
cd examples;
cd ttt_linear_prefill;
bash build.sh;
cd ../ttt_mlp_prefill;
bash build.sh;
