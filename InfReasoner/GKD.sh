# # accelerate launch \
# #   --config_file examples/accelerate_configs/multi_gpu.yaml trl/experimental/gold/gold.py \
# #   --model_name_or_path Qwen/Qwen3-4B \
# #   --dtype auto \
# #   --attn_implementation sdpa \
# #   --dataset_name allenai/tulu-3-sft-mixture \
# #   --dataset_train_split train \
# #   --bf16 True \
# #   --learning_rate 1e-7 \
# #   --dataset_test_split train \
# #   --gradient_checkpointing \
# #   --per_device_train_batch_size 1 \
# #   --gradient_accumulation_steps 64 \
# #   --num_train_epochs 1 \
# #   --eval_strategy steps \
# #   --eval_steps 100 \
# #   --temperature 1.0 \
# #   --top_p 0.95 \
# #   --top_k 0 \
# #   --lmbda 0.25 \
# #   --beta 0.0 \
# #   --use_uld_loss \
# #   --use_extended_uld \
# #   --uld_use_hybrid_loss \
# #   --uld_crossentropy_weight 0.0 \
# #   --uld_distillation_weight 1.0 \
# #   --uld_student_temperature 1.0 \
# #   --uld_teacher_temperature 1.0 \
# #   --uld_hybrid_unmatched_weight 1.0 \
# #   --uld_hybrid_matched_weight 1.0 \
# #   --teacher_model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
# #   --logging_steps 1 \
# #   --report_to trackio \
# #   --seed 42 \
# #   --warmup_ratio 0.05 \
# #   --lr_scheduler_type cosine_with_min_lr

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


CUDA_VISIBLE_DEVICES=0,1,2,3 python trl/experimental/gold/gold.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-model \
    --num_train_epochs 1 \
    --gradient_checkpointing
