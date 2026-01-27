"""
Word2Vec Skip-gram 模型演示
演示 One-hot → 隐藏层 → Softmax 的完整过程
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 macOS OpenMP 冲突

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("Word2Vec Skip-gram 模型演示")
print("=" * 60)

# ============================================================
# 1. 定义词汇表
# ============================================================
vocab = ["the", "a", "king", "queen", "man", "woman", "cat", "dog"]
vocab_size = len(vocab)  # V = 8
embedding_dim = 4        # D = 4 (通常是 100-300，这里简化)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

print(f"\n词汇表: {vocab}")
print(f"词汇表大小 V = {vocab_size}")
print(f"嵌入维度 D = {embedding_dim}")

# ============================================================
# 2. 创建 One-hot 编码
# ============================================================
print("\n" + "-" * 60)
print("Step 1: One-hot 编码")
print("-" * 60)

def create_one_hot(word, vocab_size):
    """创建词的 One-hot 向量"""
    idx = word_to_idx[word]
    one_hot = torch.zeros(vocab_size)
    one_hot[idx] = 1.0
    return one_hot

# 以 "king" 为例
center_word = "king"
one_hot_king = create_one_hot(center_word, vocab_size)

print(f"\n'{center_word}' 的索引: {word_to_idx[center_word]}")
print(f"'{center_word}' 的 One-hot 向量:")
for i, val in enumerate(one_hot_king):
    marker = " ←" if val == 1 else ""
    print(f"  [{i}] {val:.0f}  ({idx_to_word[i]}){marker}")

# ============================================================
# 3. 输入层 → 隐藏层 (获取词嵌入)
# ============================================================
print("\n" + "-" * 60)
print("Step 2: 输入层 → 隐藏层 (词嵌入矩阵)")
print("-" * 60)

# 词嵌入矩阵 W_in: [V × D]
# 每一行是一个词的嵌入向量
torch.manual_seed(42)  # 固定随机种子
W_in = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)

print(f"\n词嵌入矩阵 W_in 形状: {W_in.shape} [V×D]")
print(f"\nW_in 矩阵内容:")
print("        dim0    dim1    dim2    dim3")
for i in range(vocab_size):
    row = W_in[i].detach().numpy()
    print(f"{idx_to_word[i]:8s} [{row[0]:6.3f}, {row[1]:6.3f}, {row[2]:6.3f}, {row[3]:6.3f}]")

# One-hot × W_in = 选出对应行（词嵌入）
# 这就是为什么说 "实际上就是从中选出对应的那一行向量"
embedding_king = one_hot_king @ W_in  # [V] × [V×D] = [D]

print(f"\n计算: One-hot × W_in")
print(f"  {one_hot_king.tolist()} × W_in")
print(f"  = W_in[{word_to_idx[center_word]}] (选出第 {word_to_idx[center_word]} 行)")
print(f"  = {embedding_king.detach().numpy()}")
print(f"\n'{center_word}' 的嵌入向量 (D={embedding_dim}维): {embedding_king.detach().tolist()}")

# 验证：直接索引和矩阵乘法结果相同
direct_lookup = W_in[word_to_idx[center_word]]
print(f"\n验证 - 直接索引 W_in[{word_to_idx[center_word]}]: {direct_lookup.detach().tolist()}")
print(f"两种方式结果相同: {torch.allclose(embedding_king, direct_lookup)}")

# ============================================================
# 4. 隐藏层 → 输出层 (预测上下文词)
# ============================================================
print("\n" + "-" * 60)
print("Step 3: 隐藏层 → 输出层 (预测上下文词)")
print("-" * 60)

# 输出权重矩阵 W_out: [D × V]
# 用于计算中心词与所有词的相似度
W_out = nn.Parameter(torch.randn(embedding_dim, vocab_size) * 0.1)

print(f"\n输出权重矩阵 W_out 形状: {W_out.shape} [D×V]")

# 计算相似度分数 (logits)
# embedding_king: [D]
# W_out: [D × V]
# scores: [V]
scores = embedding_king @ W_out  # [D] × [D×V] = [V]

print(f"\n相似度分数 (logits):")
print(f"  embedding × W_out = {scores.detach().numpy()}")

# ============================================================
# 5. Softmax 转换为概率
# ============================================================
print("\n" + "-" * 60)
print("Step 4: Softmax 转换为概率")
print("-" * 60)

probabilities = F.softmax(scores, dim=0)

print(f"\nSoftmax 公式: P(word_i) = exp(score_i) / Σexp(score_j)")
print(f"\n'{center_word}' 的上下文词预测概率:")
print("-" * 40)

# 按概率排序
sorted_probs = sorted(enumerate(probabilities.detach().numpy()), 
                      key=lambda x: x[1], reverse=True)

for idx, prob in sorted_probs:
    word = idx_to_word[idx]
    bar = "█" * int(prob * 50)
    print(f"  {word:8s}: {prob:.4f} {bar}")

print(f"\n概率和: {probabilities.sum().item():.4f} (应该 ≈ 1.0)")

# ============================================================
# 6. 完整的 Word2Vec 模型类
# ============================================================
print("\n" + "-" * 60)
print("完整的 Word2Vec Skip-gram 模型")
print("-" * 60)

class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # 输入嵌入矩阵 (中心词)
        self.W_in = nn.Embedding(vocab_size, embedding_dim)
        # 输出嵌入矩阵 (上下文词)
        self.W_out = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, center_word_idx):
        """
        center_word_idx: 中心词的索引
        返回: 所有词作为上下文词的概率
        """
        # 1. 查找中心词的嵌入 (相当于 One-hot × W_in)
        embedding = self.W_in(center_word_idx)  # [D]
        
        # 2. 计算与所有词的相似度
        scores = self.W_out(embedding)  # [V]
        
        # 3. Softmax 得到概率
        probs = F.softmax(scores, dim=-1)  # [V]
        
        return probs, embedding

# 创建模型
model = Word2VecSkipGram(vocab_size, embedding_dim)

# 测试
center_idx = torch.tensor(word_to_idx["king"])
probs, emb = model(center_idx)

print(f"\n模型输入: 'king' (idx={center_idx.item()})")
print(f"嵌入向量: {emb.detach().tolist()}")
print(f"输出概率分布: {probs.detach().numpy()}")

# ============================================================
# 7. 训练过程演示 (简化版)
# ============================================================
print("\n" + "-" * 60)
print("训练过程演示")
print("-" * 60)

# 假设训练样本: ("king", "queen") - king 的上下文中有 queen
# 目标: 让 P("queen" | "king") 变高

center = torch.tensor(word_to_idx["king"])
context = torch.tensor(word_to_idx["queen"])

print(f"\n训练样本: ('{idx_to_word[center.item()]}', '{idx_to_word[context.item()]}')")
print(f"目标: 让 P('queen' | 'king') 提高")

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
criterion = nn.CrossEntropyLoss()

print(f"\n训练前:")
probs_before, _ = model(center)
print(f"  P('queen' | 'king') = {probs_before[context].item():.4f}")

# 训练几轮
for epoch in range(100):
    optimizer.zero_grad()
    probs, _ = model(center)
    loss = criterion(probs.unsqueeze(0), context.unsqueeze(0))
    loss.backward()
    optimizer.step()

print(f"\n训练后 (100轮):")
probs_after, _ = model(center)
print(f"  P('queen' | 'king') = {probs_after[context].item():.4f}")

print(f"\n概率变化: {probs_before[context].item():.4f} → {probs_after[context].item():.4f}")
print(f"提升了 {probs_after[context].item() / probs_before[context].item():.1f} 倍!")

print("\n" + "=" * 60)
print("演示完成!")
print("=" * 60)
