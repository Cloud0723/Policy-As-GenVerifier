#!/bin/bash

# Configuration
MODEL="Qwen/Qwen3-4B"
NUM_QUESTIONS=30
MODE="single-turn"

# Experiment parameters
SAMPLES_ARRAY=(1 2 4 8 16 32 64)
MAX_TOKENS_ARRAY=(1024 2048 4096 8192 16384 32000 65000)

# Run experiments
for SAMPLES in "${SAMPLES_ARRAY[@]}"; do
    for MAX_TOKENS in "${MAX_TOKENS_ARRAY[@]}"; do
        echo "Running: mode=${MODE}, samples=${SAMPLES}, max_tokens=${MAX_TOKENS}"
        python vllm_client.py \
            --mode "${MODE}" \
            --model "${MODEL}" \
            --num-questions ${NUM_QUESTIONS} \
            --num-samples ${SAMPLES} \
            --max-tokens-single ${MAX_TOKENS}
    done
done

echo "Batch experiments completed!"
