#!/bin/bash

# Configuration
MODEL="/mnt/data1/li003968/infinite_thinking/global_step_600_hf"
NUM_QUESTIONS=30
MODE="multi-turn"

# Experiment parameters
SAMPLES_ARRAY=(1)
MAX_TOKENS_ARRAY=(4096 8192 16384 32000)

# Run experiments
for SAMPLES in "${SAMPLES_ARRAY[@]}"; do
    for MAX_TOKENS in "${MAX_TOKENS_ARRAY[@]}"; do
        echo "Running: mode=${MODE}, samples=${SAMPLES}, max_tokens=${MAX_TOKENS}"
        python vllm_client.py \
            --mode "${MODE}" \
            --model "${MODEL}" \
            --num-questions ${NUM_QUESTIONS} \
            --num-samples ${SAMPLES} \
            --max-tokens-multi ${MAX_TOKENS}
    done
done

echo "Batch experiments completed!"
