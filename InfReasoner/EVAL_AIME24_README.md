# AIME24 Single Turn Evaluation

参考CURE/eval.sh的评估流程，专门针对AIME24数据集的single turn评估。

## 快速开始

### 1. 启动vLLM服务器

```bash
bash vllm_serve.sh
```

### 2. 运行评估

```bash
bash eval_aime24.sh
```

## 配置说明

### 默认参数（参考CURE/eval.sh）

```bash
n_samples=32        # 每题生成32个样本 (Avg@32)
temperature=0.6     # 采样温度
top_p=0.95          # Top-p采样
top_k=-1            # Top-k (注意: OpenAI API不支持)
max_tokens=3584     # 最大响应长度 (4096-512)
```

### 自定义参数

修改`eval_aime24.sh`中的参数，或直接调用Python脚本：

```bash
python3 eval_aime24_single_turn.py \
    --host localhost \
    --port 9001 \
    --n-samples 32 \
    --max-tokens 3584 \
    --temperature 0.6 \
    --top-p 0.95 \
    --output ./results.json
```

## 评估模式

### Single Turn vs Multi-Turn

- **Single Turn** (此脚本): 
  - 一次生成，不使用PAG的自我验证和重新生成
  - 类似CURE的评估方式
  - 更快，适合baseline评估

- **Multi-Turn** (PAG): 
  - 使用PAG的验证和重新生成机制
  - 多轮自我修正
  - 参见`vllm_client_pag.py`

## 输出指标

评估完成后会输出：

```json
{
  "metrics": {
    "avg@32": 0.4521,    // 32个样本的平均准确率
    "best@32": 0.7834,   // 32个样本的最佳准确率  
    "pass@1": 0.3667,    // 单次尝试准确率
    "best@1": 0.3667,    // Best-of-1
    "best@2": 0.5234,    // Best-of-2
    "best@4": 0.6123,    // Best-of-4
    "best@8": 0.6891,    // Best-of-8
    "best@16": 0.7456    // Best-of-16
  }
}
```

## 预计时间

- 30题 × 32样本 = 960次生成
- 单GPU: 约4-8小时
- 取决于模型大小和硬件

## 常见问题

### Q: 为什么不支持top_k？

**A**: OpenAI兼容API（包括vLLM的OpenAI接口）不支持`top_k`参数。这是API设计限制。如果需要top_k，应该：
1. 直接使用vLLM的Python API（不通过OpenAI接口）
2. 或使用temperature和top_p来控制采样

### Q: 和CURE eval.sh的区别？

**A**: 
- CURE使用`verl.trainer.main_generation`（分布式框架）
- 此脚本使用vLLM的OpenAI接口（更简单）
- 评估逻辑相同：生成N个样本并计算Avg@N

### Q: 如何加速评估？

**A**: 
1. 减少n_samples（例如使用16或8）
2. 使用更小的模型
3. 使用CURE的分布式评估框架

## 参考

- CURE eval.sh: `/home/li003968/infinite_think/CURE/eval.sh`
- 数据集: https://huggingface.co/datasets/math-ai/aime24
- math_verify: https://github.com/MARIO-Math-Reasoning/math_verify
