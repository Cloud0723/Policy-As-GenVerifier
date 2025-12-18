#!/bin/bash
# 调试模式：使用前4张GPU运行
# 使用 torchrun 自动处理多GPU分布式设置

set -x

# 激活 conda 环境（pag）
# 如果已经在 conda 环境中，这行不会造成问题
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pag

# 验证 Python 环境
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Checking transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"

# 设置模型路径
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 设置使用前4张GPU (GPU 0, 1, 2, 3)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置 GPU 数量
NUM_GPUS=4

# 可选：设置 tensor_model_parallel_size（如果使用 tensor parallelism）
# 默认使用 data parallelism，如果需要 tensor parallelism，可以设置为 2 或 4
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"

# 设置其他配置参数
export N_SAMPLES="${N_SAMPLES:-8}"  # 每个问题的样本数（avg@8）
export MAX_PROBLEMS="${MAX_PROBLEMS:-0}"  # 0=全部问题
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"

# 设置 Python 未缓冲输出
export PYTHONUNBUFFERED=1

# 使用 conda 环境中的 Python 来运行 torchrun
# 方法1：使用 python -m torch.distributed.run（推荐，确保使用正确的环境）
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    --log_dir=./torchrun_logs \
    --tee=3 \
    run_vllm_spmd_direct_multiturn_rollout.py

# 如果上面的方法不行，可以尝试直接使用 torchrun（需要确保在 conda 环境中）
# torchrun \
#     --standalone \
#     --nproc_per_node=${NUM_GPUS} \
#     --log_dir=./torchrun_logs \
#     --tee=3 \
#     run_vllm_spmd_direct_multiturn_rollout.py
