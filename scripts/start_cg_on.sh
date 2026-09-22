#!/bin/bash
# startup script with CUDA graph ON (for testing)
export FLASHINFER_CUDA_ARCH_LIST='12.0'
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=/usr/local/cuda-13.2/bin:$HOME/.local/bin:$PATH
vllm serve ~/models/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 100000 --max-num-seqs 12 \
  --kv-cache-dtype int4_per_token_head \
  --gpu-memory-utilization 0.94 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature": 0.8}'
