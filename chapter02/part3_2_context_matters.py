"""
Part 3.2: 上下文相关性演示
演示同一个词在不同上下文中会得到不同的嵌入向量
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 macOS OpenMP 冲突

import torch
from transformers import AutoModel, AutoTokenizer

print("=" * 60)
print("Part 3.2: 上下文相关性演示")
print("=" * 60)

# 加载 DeBERTa 模型和 tokenizer
print("\n加载 DeBERTa 模型...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

print("\n" + "-" * 60)
print("核心概念: 同一个词在不同句子中的嵌入是不同的!")
print("-" * 60)

# 两个包含 "bank" 但含义不同的句子
sentences = [
    "The bank is by the river.",           # bank = 河岸
    "I went to the bank to deposit money." # bank = 银行
]

bank_embeddings = []

for sent in sentences:
    print(f"\n句子: '{sent}'")
    
    # 分词
    tokens = tokenizer(sent, return_tensors='pt')
    token_ids = tokens['input_ids'][0]
    
    # 显示分词结果
    print("  分词结果:")
    for i, tid in enumerate(token_ids):
        token_text = tokenizer.decode(tid)
        print(f"    位置 {i}: ID={tid.item():5d} -> '{token_text}'")
    
    # 获取上下文嵌入
    with torch.no_grad():
        output = model(**tokens)[0]
    
    # 找到 "bank" 的位置并提取嵌入
    for i, tid in enumerate(token_ids):
        token_text = tokenizer.decode(tid)
        if 'bank' in token_text.lower():
            embedding = output[0, i, :]  # 完整嵌入 [hidden_dim]
            bank_embeddings.append(embedding)
            
            print(f"\n  ★ 找到 'bank' 在位置 {i}")
            print(f"    嵌入维度: {embedding.shape}")
            print(f"    嵌入前5维: [{', '.join(f'{x:.4f}' for x in embedding[:5].tolist())}]")

# 计算两个 "bank" 嵌入的相似度
print("\n" + "-" * 60)
print("两个 'bank' 嵌入的比较:")
print("-" * 60)

if len(bank_embeddings) == 2:
    emb1, emb2 = bank_embeddings
    
    # 余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    print(f"  余弦相似度: {cos_sim.item():.4f}")
    print(f"  (如果完全相同=1.0, 完全不同=-1.0)")
    
    # 欧氏距离
    euclidean_dist = torch.dist(emb1, emb2)
    print(f"  欧氏距离: {euclidean_dist.item():.4f}")
    
    print("\n结论:")
    if cos_sim < 0.95:
        print("  → 两个 'bank' 的嵌入向量明显不同!")
        print("  → 模型能够根据上下文区分 '河岸' 和 '银行' 的含义")
    else:
        print("  → 两个 'bank' 的嵌入向量非常相似")

print("\n" + "=" * 60)
print("Part 3.2 完成!")
print("=" * 60)
