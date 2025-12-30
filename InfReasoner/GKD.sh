#!/usr/bin/env bash
set -euo pipefail

# W&B（建议：不要把 API Key 写进脚本/仓库；请在 shell/环境里提前 export WANDB_API_KEY=...）
# 需要同时设置 `--report_to wandb` 才会真正上报。
export WANDB_ENTITY="${WANDB_ENTITY:-rl_agent}"
export WANDB_PROJECT="${WANDB_PROJECT:-Infinite-Think}"
# 目标：把启动方式改成 TRL/Transformers + accelerate 兼容的形式。
# 说明：
# - 你仍然可以用 CUDA_VISIBLE_DEVICES 控制可见卡；accelerate 会自动继承它
# - accelerate 的并行/分布式由 `--config_file` 决定

# 无论从哪里执行，都先切到本脚本所在目录（保证相对路径可用）
export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 默认：沿用你原脚本的 4 卡可见设置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

# 显存碎片/大块分配失败时的缓解（可按需关闭/覆盖）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 默认：使用 4 processes 的配置（对应 4 卡）。如果你要 8 卡，可改成 `trl/accelerate_configs/multi_gpu.yaml`
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/accelerate_configs/multi_gpu.yaml}"

# accelerate launch \
#   --main_process_port 29502 \
#   --config_file "${ACCELERATE_CONFIG}" \
#   examples/scripts/gkd_openr1_math220k.py \
#   --dtype bfloat16 \
#   --attn_implementation sdpa \
#   --model_name_or_path Qwen/Qwen3-4B \
#   --teacher_model_name_or_path Qwen/Qwen3-32B \
#   --dataset_name open-r1/OpenR1-Math-220k \
#   --dataset_train_split train \
#   --problem_column problem \
#   --prompt_template "Solve the following math problem. Give your final answer at the end.\n\n{problem}" \
#   --lmbda 1.0 \
#   --beta 0.0 \
#   --max_new_tokens 4096 \
#   --learning_rate 2e-7 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --logging_steps 10 \
#   --save_steps 200 \
#   --output_dir /mnt/data1/li003968/onpolicy_checkpoint/gkd-openr1-math220k \
#   --num_train_epochs 1 \
#   --report_to wandb \
#   --gradient_checkpointing

# 下面保留原来的参考命令（按需取消注释/修改）：
#

accelerate launch \
  --main_process_port 29502 \
  --config_file "${ACCELERATE_CONFIG}" \
  trl/experimental/gkd/gkd_openr1_math220k.py \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_name_or_path Qwen/Qwen3-4B \
  --teacher_model_name_or_path Qwen/Qwen3-32B \
  --dataset_name open-r1/OpenR1-Math-220k \
  --dataset_train_split train \
  --problem_column problem \
  --prompt_template "Solve the following math problem. Give your final answer at the end.\n\n{problem}" \
  --lmbda 1.0 \
  --beta 0.0 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 200 \
  --learning_rate 2e-7 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir /mnt/data1/li003968/onpolicy_checkpoint/gkd-openr1-math220k \
  --num_train_epochs 1 \
  --report_to wandb \
  --max_new_tokens 4096 \
  --gradient_checkpointing
