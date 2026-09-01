#!/bin/sh
export TORCH_CUDA_ARCH_LIST=12.1a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_MATMUL_PRECISION=high
export NVIDIA_FORWARD_COMPAT=1
export NVIDIA_DISABLE_REQUIRE=1
export ENABLE_NVFP4_SM100=0
export VLLM_TEST_FORCE_FP8_MARLIN=0
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export WORKDIR=$(pwd)
vllm serve nvidia/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --runner generate \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --quantization modelopt \
  --max-num-batched-tokens 16384 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.65 \
  --enable-lora \
  --specialize-active-lora \
  --enable-tower-connector-lora \
  --lora-modules base-lora=$WORKDIR/adapters/gen1/base \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --load-format safetensors \
  --attention-backend flash_attn \
  --kv-cache-dtype bfloat16 \
  --moe-backend flashinfer_b12x \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":4,"video":2}' \
  --mm-processor-cache-type shm \
  --mm-shm-cache-max-object-size-mb 256 \
  --generation-config vllm \
  --chat-template $WORKDIR/adapters/gen1/base/chat_template.jinja \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'