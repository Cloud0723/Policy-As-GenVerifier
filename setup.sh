# 1. Conda 环境配置
# conda init
# conda create -n verl030p1-env python=3.10 -y
# conda activate verl030p1-env

# 2. 安装 PyTorch, Flash Attention, vLLM

# 注意：
# 1. PyTorch 版本已更新为 2.6.0，使用 cu121
# 2. vLLM 版本已更新为 0.8.2
# 3. Flash Attention 版本保持不变
# pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install vllm==0.8.2

pip install verl==0.3.0.post1

pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install vllm==0.8.2

# 4. 安装其他依赖
pip install wandb