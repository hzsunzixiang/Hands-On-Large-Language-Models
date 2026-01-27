"""
Part 3.1: 获取上下文词嵌入
演示如何从语言模型获取上下文感知的词嵌入向量
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 macOS OpenMP 冲突

import torch
from transformers import AutoModel, AutoTokenizer

print("=" * 60)
print("Part 3.1: 获取上下文词嵌入")
print("=" * 60)

# 加载 DeBERTa 模型和 tokenizer
print("\n加载 DeBERTa 模型...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# 处理文本
text = "Hello world"
tokens = tokenizer(text, return_tensors='pt')

print(f"\n输入文本: '{text}'")
print(f"\nTokenizer 输出:")
print(f"  input_ids: {tokens['input_ids']}")
print(f"  attention_mask: {tokens['attention_mask']}")

# 获取上下文嵌入
with torch.no_grad():
    output = model(**tokens)[0]

print(f"\n模型输出:")
print(f"  输出形状: {output.shape}")  # [batch, tokens, embedding_dim]
print(f"  batch size: {output.shape[0]}")
print(f"  Token 数量: {output.shape[1]}")
print(f"  嵌入维度: {output.shape[2]}")

# 展示每个 token 的信息
print("\n" + "-" * 60)
print("每个 Token 的嵌入向量:")
print("-" * 60)
for i, token_id in enumerate(tokens['input_ids'][0]):
    token = tokenizer.decode(token_id)
    embedding = output[0, i, :5].tolist()  # 只显示前5维
    print(f"  位置 {i}: ID={token_id.item():5d} -> '{token}'")
    print(f"          嵌入前5维: [{', '.join(f'{x:.4f}' for x in embedding)}, ...]")

print("\n" + "=" * 60)
print("Part 3.1 完成!")
print("=" * 60)
