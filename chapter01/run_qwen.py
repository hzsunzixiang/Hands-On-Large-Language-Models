"""
Chapter 1 - 语言模型入门
使用 Qwen 模型生成文本的示例
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ========== 配置区域 ==========
# 可选模型 (按大小排序，选择适合你硬件的版本):
# - "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B 参数，约 1GB，最轻量
# - "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B 参数，约 3GB
# - "Qwen/Qwen2.5-3B-Instruct"    # 3B 参数，约 6GB
# - "Qwen/Qwen2.5-7B-Instruct"    # 7B 参数，约 14GB
# - "microsoft/Phi-3-mini-4k-instruct"  # 原示例模型

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 推荐先用小模型测试
# ==============================

print('=== Chapter 1: 语言模型入门 ===')
print(f'PyTorch 版本: {torch.__version__}')
print(f'MPS 可用: {torch.backends.mps.is_available()}')
print(f'使用模型: {MODEL_NAME}')
print()

print('正在加载模型 (首次需下载)...')

# 加载模型 - MPS 有兼容性问题，使用 CPU + float32 更稳定
# 如果想用 MPS 加速，可以尝试 dtype=torch.float32 + device_map='mps'
device = 'cpu'  # MPS 在某些 PyTorch 版本有 matmul 兼容问题
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    dtype=torch.float32,  # 新版 transformers 用 dtype 替代 torch_dtype
    trust_remote_code=True,  # Qwen 需要设置为 True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f'模型已加载到: {device}')

# 创建 pipeline
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=200,
    do_sample=False
)

print('\n生成文本中...')

# 构建消息 (Qwen 支持中文)
messages = [
    {'role': 'user', 'content': '讲一个关于鸡的笑话。'}
]

# 生成输出
output = generator(messages)
print('\n=== 生成结果 ===')
print(output[0]['generated_text'])
