"""
Part 1: Tokenizer 基础
演示 Tokenizer 的基本使用 - 文本如何被切分和编码
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer

print("=" * 60)
print("Part 1: Tokenizer 基础")
print("=" * 60)

# 加载 Phi-3 的 tokenizer
print("\n加载 Phi-3 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# 测试文本
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap."

# 将文本转换为 token IDs
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(f"\n原始文本: {prompt}")
print(f"\nToken IDs shape: {input_ids.shape}")
print(f"Token IDs: {input_ids[0].tolist()}")

# 逐个解码 token 查看分词结果 (带索引)
print("\n分词结果 (带索引):")
for i, token_id in enumerate(input_ids[0]):
    token = tokenizer.decode(token_id)
    print(f"  {i}: ID={token_id.item():5d} -> '{token}'")

# 逐个解码 token (简洁格式，每个词元占一行)
print("\n" + "-" * 40)
print("逐个词元输出 (每行一个):")
print("-" * 40)
for id in input_ids[0]:
    print(tokenizer.decode(id))

# 演示子词组合
print("\n" + "-" * 40)
print("子词组合示例:")
print("-" * 40)
print(f"  tokenizer.decode([3323, 622]) = '{tokenizer.decode([3323, 622])}'")

# 演示 decode 方法：单个 ID vs 多个 ID 组合
print("\n" + "-" * 40)
print("decode 方法演示 (单个ID vs 组合):")
print("-" * 40)
print(f"tokenizer.decode(3323)        -> '{tokenizer.decode(3323)}'")
print(f"tokenizer.decode(622)         -> '{tokenizer.decode(622)}'")
print(f"tokenizer.decode([3323, 622]) -> '{tokenizer.decode([3323, 622])}'")
print(f"tokenizer.decode(29901)       -> '{tokenizer.decode(29901)}'")
# 演示编码和解码的完整流程
print("\n" + "-" * 40)
print("编码/解码完整流程:")
print("-" * 40)
text = "Hello world!"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print(f"  原文: '{text}'")
print(f"  编码: {encoded}")
print(f"  解码: '{decoded}'")

# 文本生成演示
print("\n" + "-" * 40)
print("文本生成 (model.generate):")
print("-" * 40)

from transformers import AutoModelForCausalLM

# 使用 GPT-2 替代 Phi-3，避免版本兼容性问题
print("\n加载 GPT-2 模型...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

# 重新编码 prompt (使用 GPT-2 的 tokenizer)
input_ids_gpt2 = tokenizer_gpt2(prompt, return_tensors="pt").input_ids

print(f"\n输入 prompt: {prompt}")
print("生成中...")

# 设置 pad_token 避免警告
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

# 生成文本
generation_output = model.generate(
    input_ids=input_ids_gpt2,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer_gpt2.eos_token_id
)

# 打印生成的 token IDs
print(f"\n生成的 Token IDs:")
print(generation_output)

# 解码并打印输出
generated_text = tokenizer_gpt2.decode(generation_output[0], skip_special_tokens=True)
print(f"\n生成结果:\n{generated_text}")

# 演示输出端的 decode：逐个 token 解码
print("\n" + "-" * 40)
print("输出端逐个词元解码:")
print("-" * 40)
# 只显示新生成的 tokens (跳过输入部分)
input_len = input_ids_gpt2.shape[1]
new_tokens = generation_output[0][input_len:]
print(f"新生成的 Token IDs: {new_tokens.tolist()[:10]}...")  # 只显示前10个
print("\n逐个解码:")
for i, token_id in enumerate(new_tokens[:10]):  # 只显示前10个
    print(f"  {token_id.item():5d} -> '{tokenizer_gpt2.decode(token_id)}'")

print("\n" + "=" * 60)
print("Part 1 完成!")
print("=" * 60)
