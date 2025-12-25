#!/bin/bash
# 运行数据预处理脚本

set -e

echo "开始预处理 InftyThink 数据集..."

# 设置输出目录
OUTPUT_DIR="/home/li003968/infinite_think/Policy-As-GenVerifier/datasets"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行预处理脚本
# 可选参数：
# --max_samples N  # 只处理前 N 个样本（用于测试）
# --val_ratio 0.05 # 验证集比例
# --no_split       # 不创建训练/验证集分割

python sftdataset_preclean.py \
    --dataset_name ZJU-REAL/InftyThink \
    --subset default \
    --split InftyThink__OpenR1__default__4k \
    --output_dir $OUTPUT_DIR \
    --val_ratio 0.05 
    # --max_samples 10

echo ""
echo "✅ 预处理完成！"
echo "数据已保存到: $OUTPUT_DIR"
echo ""
echo "生成的文件："
ls -lh $OUTPUT_DIR
echo ""
echo "现在可以使用以下命令开始训练："
echo "bash ../../../verl/SFT_qwen3-4B.sh 4 /path/to/save/model"

