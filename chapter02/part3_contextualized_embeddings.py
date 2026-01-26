"""
Part 3: 上下文词嵌入 (Contextualized Embeddings)
演示从语言模型获取上下文感知的词嵌入
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"

import torch
from transformers import AutoModel, AutoTokenizer

print("=" * 60)
print("Part 3: 上下文词嵌入 (Contextualized Embeddings)")
print("=" * 60)

# 加载 DeBERTa 模型和 tokenizer
print("\n加载 DeBERTa 模型...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# 处理文本
text = "Hello world"
tokens = tokenizer(text, return_tensors='pt')

# 获取上下文嵌入
with torch.no_grad():
    output = model(**tokens)[0]

print(f"\n输入文本: '{text}'")
print(f"Token 数量: {output.shape[1]}")
print(f"嵌入维度: {output.shape[2]}")
print(f"输出形状: {output.shape}")  # [batch, tokens, embedding_dim]

# 展示每个 token 的信息
print("\n每个 Token 的嵌入:")
for i, token_id in enumerate(tokens['input_ids'][0]):
    token = tokenizer.decode(token_id)
    embedding = output[0, i, :5].tolist()  # 只显示前5维
    print(f"  {i}: '{token}' -> [{', '.join(f'{x:.4f}' for x in embedding)}, ...]")

# 演示上下文相关性
print("\n" + "-" * 60)
print("上下文相关性演示:")
print("-" * 60)
print("同一个词在不同句子中的嵌入是不同的!")

sentences = [
    "The bank is by the river.",      # bank = 河岸
    "I went to the bank to deposit money."  # bank = 银行
]

for sent in sentences:
    tokens = tokenizer(sent, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)[0]
    
    # 找到 "bank" 的位置
    token_ids = tokens['input_ids'][0]
    for i, tid in enumerate(token_ids):
        if 'bank' in tokenizer.decode(tid).lower():
            embedding = output[0, i, :3].tolist()
            print(f"\n句子: '{sent}'")
            print(f"  'bank' 的嵌入前3维: [{', '.join(f'{x:.4f}' for x in embedding)}, ...]")

print("\n" + "=" * 60)
print("Part 3 完成!")
print("=" * 60)
