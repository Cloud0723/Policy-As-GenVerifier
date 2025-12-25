#!/bin/bash

# Configuration parameters
CUDA_DEVICES=${1:-"0,1,2,3"}
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
HOST=${2:-"0.0.0.0"}
PORT=${3:-9000}
# MODEL=${4:-"Qwen/Qwen2.5-72B-Instruct"}
MODEL=${4:-"/mnt/data1/li003968/infinite_thinking/global_step_600_hf"}
MAX_MODEL_LEN=${5:-32768}


# Display system info and configuration
cat << EOF
CUDA Devices: $CUDA_VISIBLE_DEVICES
Current IP: $(hostname -I | cut -d' ' -f1)

Starting VLLM server:
  Model: $MODEL
  CUDA Devices: $CUDA_DEVICES
  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE
  Host: $HOST
  Port: $PORT
  Max Model Length: $MAX_MODEL_LEN

EOF

# Build vLLM command with conditional parameters
VLLM_CMD="vllm serve $MODEL \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --host $HOST \
    --port $PORT \
    --gpu-memory-utilization 0.9 \
    --max-model-len $MAX_MODEL_LEN"

# Add hf-overrides if max_model_len > 32k
if [ $MAX_MODEL_LEN -gt 32768 ]; then
    VLLM_CMD="$VLLM_CMD \
    --hf-overrides '{\"max_position_embeddings\": $MAX_MODEL_LEN}'"
    echo "Max model length > 32k, adding --hf-overrides with max_position_embeddings=$MAX_MODEL_LEN"
fi

VLLM_CMD="$VLLM_CMD \
    --disable-log-stats"

# Start VLLM server
eval $VLLM_CMD