# InftyThink 数据集预处理说明

## 概述

本脚本用于处理 ZJU-REAL/InftyThink 数据集，将其转换为适合 SFT 训练的格式。

## 数据处理逻辑

### 第一轮 Turn（history 为空）
- **Prompt**: 使用 `FIRST_TURN_REASONING_PROMPT`，插入 `question`
- **Response**: `reasoning_process` + `summary`（如果有 summary）

```
[PROMPT]
Question: {question}
Please reason step by step...

[RESPONSE]
{reasoning_process}
{summary}
```

### 后续 Turn（history 不为空）
- **Prompt**: 使用 `MIDDLE_TURN_REASONING_PROMPT`，插入 `question` 和 `history`
- **Response**: `reasoning_process` + `conclusion`（如果有 conclusion）

```
[PROMPT]
Question: {question}
Summary of previous reasoning: {history}
Please continue reasoning step by step...

[RESPONSE]
{reasoning_process}
{conclusion}
```

## 使用方法

### 1. 测试运行（处理少量样本）

```bash
cd /home/li003968/infinite_think/Policy-As-GenVerifier/InfReasoner
bash test_preprocess.sh
```

或直接运行：
```bash
python sftdataset_preclean.py \
    --dataset_name ZJU-REAL/InftyThink \
    --subset default \
    --split InftyThink__OpenR1__default__2k \
    --output_dir /path/to/output \
    --max_samples 10 \
    --val_ratio 0.1
```

### 2. 完整数据集处理

```bash
bash run_preprocess.sh
```

或直接运行：
```bash
python sftdataset_preclean.py \
    --dataset_name ZJU-REAL/InftyThink \
    --subset default \
    --split InftyThink__OpenR1__default__2k \
    --output_dir $HOME/data/InftyThink \
    --val_ratio 0.05
```

### 3. 参数说明

- `--dataset_name`: HuggingFace 数据集名称（默认：ZJU-REAL/InftyThink）
- `--subset`: 数据集子集（默认：default）
- `--split`: 数据集分割（默认：InftyThink__OpenR1__default__2k）
- `--output_dir`: 输出目录（默认：~/data/InftyThink）
- `--max_samples`: 最大样本数，用于测试（默认：None，处理所有数据）
- `--val_ratio`: 验证集比例（默认：0.05）
- `--no_split`: 不创建训练/验证集分割

## 输出文件

处理完成后会生成以下文件：

```
{output_dir}/
├── train.parquet                                    # 训练集
├── test.parquet                                     # 验证集
├── {split}_processed.parquet                        # 处理后的完整数据
└── {split}_processed_full.parquet                   # 包含所有字段的完整数据
```

## 数据字段

处理后的数据包含以下字段：

- `uuid`: 样本唯一标识符
- `question`: 原始问题
- `prompt`: 格式化后的提示（用于训练）
- `response`: 格式化后的回复（用于训练）
- `is_first_turn`: 是否为第一轮 turn
- `turn_number`: Turn 编号
- `has_summary`: 是否有 summary
- `has_conclusion`: 是否有 conclusion

**注意**：训练时只需要使用 `prompt` 和 `response` 字段。

## SFT 训练

预处理完成后，使用以下命令开始训练：

```bash
cd /home/li003968/infinite_think/verl
bash SFT_qwen3-4B.sh 4 /path/to/save/model
```

其中：
- `4` 是使用的 GPU 数量
- `/path/to/save/model` 是模型保存路径

## 依赖包

```bash
pip install datasets pandas pyarrow scikit-learn
```

## 示例输出

```
加载数据集: ZJU-REAL/InftyThink/default/InftyThink__OpenR1__default__2k
数据集大小: 279000

处理完成！总共 279000 条数据
  - 第一轮 turn: 150000 条
    - 有 summary: 145000 条
  - 后续 turn: 129000 条
    - 有 conclusion: 125000 条

数据已保存到: /home/li003968/data/InftyThink/InftyThink__OpenR1__default__2k_processed.parquet
完整数据已保存到: /home/li003968/data/InftyThink/InftyThink__OpenR1__default__2k_processed_full.parquet

================================================================================
示例数据（第一个 first turn）:
================================================================================

[PROMPT]
Question: Find all positive integers $n$ such that...
Please reason step by step...

[RESPONSE (前 800 字符)]
Let's start by analyzing the problem...
...
```

## 故障排除

### 问题：找不到 prompt_multi_turn_v2_lcl 模块

确保 `prompt/prompt_multi_turn_v2_lcl.py` 文件存在。

### 问题：内存不足

使用 `--max_samples` 参数先处理部分数据：
```bash
python sftdataset_preclean.py --max_samples 1000
```

### 问题：HuggingFace 下载速度慢

设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

