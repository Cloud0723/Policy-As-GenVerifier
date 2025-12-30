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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 默认：沿用你原脚本的 4 卡可见设置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

# 显存碎片/大块分配失败时的缓解（可按需关闭/覆盖）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 默认：使用 4 processes 的配置（对应 4 卡）。如果你要 8 卡，可改成 `trl/accelerate_configs/multi_gpu.yaml`
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/accelerate_configs/multi_gpu.yaml}"

accelerate launch \
  --main_process_port 29502 \
  --config_file "${ACCELERATE_CONFIG}" \
  trl/experimental/gold/gold.py \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_name_or_path Qwen/Qwen3-4B \
  --teacher_model_name_or_path Qwen/Qwen3-32B \
  --dataset_name open-thoughts/OpenThoughts3-1.2M \
  --learning_rate 2e-7 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir gold-model \
  --num_train_epochs 1 \
  --report_to wandb \
  --gradient_checkpointing

# 下面保留原来的参考命令（按需取消注释/修改）：
#
# accelerate launch \
#   --config_file examples/accelerate_configs/multi_gpu.yaml \
#   trl/experimental/gold/gold.py \
#   --model_name_or_path Qwen/Qwen3-4B \
#   --teacher_model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
#   --dtype auto \
#   --attn_implementation sdpa \
#   --dataset_name allenai/tulu-3-sft-mixture \
#   --dataset_train_split train \
#   --dataset_test_split train \
#   --learning_rate 1e-7 \
#   --gradient_checkpointing \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 64 \
#   --num_train_epochs 1 \
#   --eval_strategy steps \
#   --eval_steps 100 \
#   --temperature 1.0 \
#   --top_p 0.95 \
#   --top_k 0 \
#   --lmbda 0.25 \
#   --beta 0.0 \
#   --use_uld_loss \
#   --use_extended_uld \
#   --uld_use_hybrid_loss \
#   --uld_crossentropy_weight 0.0 \
#   --uld_distillation_weight 1.0 \
#   --uld_student_temperature 1.0 \
#   --uld_teacher_temperature 1.0 \
#   --uld_hybrid_unmatched_weight 1.0 \
#   --uld_hybrid_matched_weight 1.0 \
#   --logging_steps 1 \
#   --report_to trackio \
#   --seed 42 \
#   --warmup_ratio 0.05 \
#   --lr_scheduler_type cosine_with_min_lr
#
# accelerate launch \
#   --config_file examples/accelerate_configs/multi_gpu.yaml \
#   quick_GKD.py \
#   --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
#   --teacher_model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
#   --dtype auto \
#   --attn_implementation sdpa \
#   --dataset_name HuggingFaceTB/Countdown-Task-GOLD \
#   --dataset_config_name verified_Qwen2.5-0.5B-Instruct \
#   --dataset_train_split train \
#   --per_device_train_batch_size 1 \
#   --gradient_checkpointing \
#   --num_train_epochs 1 \
#   --use_uld_loss \
#   --uld_use_hybrid_loss \
#   --logging_steps 10 \
#   --save_steps 200 \
#   --output_dir gold-model \
#   --report_to none \
#   --seed 42


