"""
Part 1: 加载 LLM 模型
演示如何加载 Transformer LLM 并进行文本生成
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("=" * 60)
print("Part 1: 加载 LLM 模型")
print("=" * 60)

# 检测设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\n使用设备: {device}")

# 使用较小的模型以便在 CPU/MPS 上运行
# 原 notebook 使用 Phi-3-mini-4k-instruct (3.8B 参数)
# 这里使用 GPT-2 small (124M 参数) 便于快速演示
MODEL_NAME = "gpt2"  # 也可以改为 "microsoft/Phi-3-mini-4k-instruct"

print(f"\n加载模型: {MODEL_NAME}")
print("-" * 60)

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)
model = model.to(device)

print(f"\n模型加载完成!")
print(f"  模型类型: {type(model).__name__}")
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

# 创建文本生成 pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device if device != "mps" else -1,  # pipeline 不直接支持 mps
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)

# 测试文本生成
print("\n" + "-" * 60)
print("测试文本生成")
print("-" * 60)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
print(f"\n输入提示:\n  {prompt}")

# 使用 model.generate() 直接生成（兼容 MPS）
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
generated_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

print(f"\n生成的文本:\n{generated_text}")

# 查看模型结构
print("\n" + "-" * 60)
print("模型结构概览")
print("-" * 60)
print(model)

print("\n" + "=" * 60)
print("Part 1 完成!")
print("=" * 60)
