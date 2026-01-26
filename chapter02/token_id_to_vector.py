"""
词元ID 与 向量 的关系演示

本脚本展示：
1. 分词器如何将文本转为词元ID
2. 嵌入层如何将ID转为向量（本质是查表）
3. 模型内部的嵌入矩阵结构
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModel, AutoTokenizer

print("=" * 60)
print("词元ID 与 向量 的索引关系演示")
print("=" * 60)

# 加载模型和分词器
model_name = "gpt2"
print(f"\n加载模型: {model_name}")
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================
# Part 1: 分词 - 文本 → 词元ID
# ============================================
print("\n" + "=" * 60)
print("Part 1: 分词器将文本转为词元ID")
print("=" * 60)

text = "Hello world"
token_ids = tokenizer(text, return_tensors="pt")["input_ids"]

print(f"\n输入文本: '{text}'")
print(f"词元ID: {token_ids}")
print(f"形状: {token_ids.shape}  # [batch_size, seq_len]")

# 查看每个ID对应的词元
print("\nID → 词元 映射:")
for i, tid in enumerate(token_ids[0]):
    token = tokenizer.decode(tid)
    print(f"  ID {tid.item():>5} → '{token}'")

# ============================================
# Part 2: 嵌入矩阵 - 模型内部的"词典"
# ============================================
print("\n" + "=" * 60)
print("Part 2: 嵌入矩阵（Embedding Matrix）")
print("=" * 60)

# GPT-2 的词嵌入层
embed_layer = model.wte  # wte = word token embeddings
embed_matrix = embed_layer.weight

print(f"\n嵌入层: model.wte")
print(f"嵌入矩阵形状: {embed_matrix.shape}")
print(f"  - 词表大小 (vocab_size): {embed_matrix.shape[0]}")
print(f"  - 嵌入维度 (embed_dim): {embed_matrix.shape[1]}")
print(f"\n含义: 有 {embed_matrix.shape[0]} 个词，每个词对应 {embed_matrix.shape[1]} 维向量")

# ============================================
# Part 3: ID → 向量 (查表操作)
# ============================================
print("\n" + "=" * 60)
print("Part 3: 词元ID → 向量 (查表)")
print("=" * 60)

print("\n本质: vector = embedding_matrix[token_id]")
print("就像查字典一样，用ID作为索引，直接取出对应的向量行\n")

for tid in token_ids[0]:
    tid_val = tid.item()
    token = tokenizer.decode(tid)
    
    # 方法1: 直接索引嵌入矩阵
    vector = embed_matrix[tid_val]
    
    print(f"Token: '{token}' (ID={tid_val})")
    print(f"  向量形状: {vector.shape}")
    print(f"  向量前5维: [{', '.join([f'{v:.4f}' for v in vector[:5].tolist()])}]")
    print()

# ============================================
# Part 4: 验证 - 嵌入层的forward就是查表
# ============================================
print("=" * 60)
print("Part 4: 验证嵌入层的计算过程")
print("=" * 60)

# 方法1: 直接用索引
vector_by_index = embed_matrix[token_ids[0][0].item()]

# 方法2: 用嵌入层的forward (内部也是索引)
vector_by_embed = embed_layer(token_ids[0][0].unsqueeze(0)).squeeze()

# 验证两者相同
is_same = torch.allclose(vector_by_index, vector_by_embed)
print(f"\n直接索引 vs 嵌入层forward: {'✓ 完全相同' if is_same else '✗ 不同'}")
print("说明: 嵌入层的本质就是用ID查表!")

# ============================================
# Part 5: 完整流程图
# ============================================
print("\n" + "=" * 60)
print("Part 5: 完整流程总结")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│                    模型处理流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   "Hello world"                                             │
│        │                                                    │
│        ▼ 分词器 (Tokenizer)                                 │
│   [15496, 995]  ← 词元ID列表                                │
│        │                                                    │
│        ▼ 嵌入层 (Embedding Layer)                           │
│   ┌─────────────────────────────────────┐                   │
│   │  嵌入矩阵 [50257 x 768]             │                   │
│   │  ┌───┬───────────────────────┐      │                   │
│   │  │ 0 │ [0.01, -0.02, ...]    │      │                   │
│   │  │...│         ...           │      │                   │
│   │  │15496│ [0.12, -0.34, ...] │ ← Hello 的向量           │
│   │  │...│         ...           │      │                   │
│   │  │995│ [-0.05, 0.18, ...]   │ ← world 的向量           │
│   │  │...│         ...           │      │                   │
│   │  └───┴───────────────────────┘      │                   │
│   └─────────────────────────────────────┘                   │
│        │                                                    │
│        ▼ 输出: [2, 768] 的向量序列                          │
│   [[0.12, -0.34, ...],   ← Hello                           │
│    [-0.05, 0.18, ...]]   ← world                           │
│        │                                                    │
│        ▼ Transformer 层处理...                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

print("\n关键点:")
print("  1. 词元ID 是整数，便于存储和传输")
print("  2. 嵌入矩阵是模型的可学习参数")
print("  3. ID→向量 本质是 matrix[id] 的索引操作")
print("  4. 模型输出的也是ID（预测下一个词的概率分布）")
