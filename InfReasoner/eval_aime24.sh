#!/bin/bash
# AIME24 Single Turn Evaluation Script
# Reference: CURE/eval.sh

set -e

# Model and server configuration
HOST="localhost"
PORT=9001
MODEL_PATH="Qwen/Qwen2.5-Math-7B"

# Generation parameters (from CURE eval.sh)
context_length=16384
max_prompt_length=512
n_samples=8  # Number of samples per problem (for Avg@32)
top_p=0.95
temperature=0.6
top_k=-1  # Note: top_k is not supported by OpenAI-compatible API
max_response_length=$((context_length - max_prompt_length))

# Output configuration
timestamp=$(date +%Y%m%d_%H%M%S)
save_dir="./eval_results"
mkdir -p ${save_dir}

data_name="aime24"
output_file="${save_dir}/${data_name}_n${n_samples}_topp${top_p}_topk${top_k}_temp${temperature}_${timestamp}.json"

echo "=========================================="
echo "AIME24 Single Turn Evaluation"
echo "=========================================="
echo "Configuration:"
echo "  Dataset: math-ai/aime24"
echo "  Model: ${MODEL_PATH}"
echo "  Samples per problem: ${n_samples}"
echo "  Max response length: ${max_response_length}"
echo "  Temperature: ${temperature}"
echo "  Top-p: ${top_p}"
echo "  Top-k: ${top_k}"
echo "  Output: ${output_file}"
echo "=========================================="

# Check if vLLM server is running
if ! curl -s http://${HOST}:${PORT}/health > /dev/null 2>&1; then
    echo "❌ Error: vLLM server is not running at ${HOST}:${PORT}"
    echo "Please start the server first: bash vllm_serve.sh"
    exit 1
fi

echo "✓ vLLM server is running"
echo ""

# Run evaluation
python3 eval_aime24_single_turn.py \
    --host ${HOST} \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --n-samples ${n_samples} \
    --max-tokens ${max_response_length} \
    --temperature ${temperature} \
    --top-p ${top_p} \
    --top-k ${top_k} \
    --output ${output_file}

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: ${output_file}"
echo "=========================================="
