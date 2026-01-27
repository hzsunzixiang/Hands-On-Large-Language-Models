"""
Word2Vec 逐步可视化演示
详细展示 One-hot → 嵌入 → 相似度 → Softmax 的完整过程
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 macOS OpenMP 冲突

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 第一步：定义词汇表
# ============================================================
print("=" * 60)
print("第一步：定义词汇表和参数")
print("=" * 60)

vocab = ["king", "queen", "man", "woman"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)
embedding_dim = 3  # 为了可视化，使用3维向量

print(f"词汇表: {vocab}")
print(f"词汇表大小: {vocab_size}")
print(f"嵌入维度: {embedding_dim}")

# ============================================================
# 第二步：创建 One-hot 编码
# ============================================================
print("\n" + "=" * 60)
print("第二步：One-hot 编码")
print("=" * 60)

def get_one_hot(word, vocab_size):
    """创建词的 One-hot 编码"""
    vec = np.zeros(vocab_size)
    vec[word_to_idx[word]] = 1
    return vec

# 展示所有词的 One-hot 编码
print("\n所有词的 One-hot 编码:")
for word in vocab:
    onehot = get_one_hot(word, vocab_size)
    print(f"  {word:6s}: {onehot}")

# ============================================================
# 第三步：实现 Word2Vec 简化神经网络
# ============================================================
print("\n" + "=" * 60)
print("第三步：定义 Word2Vec 模型")
print("=" * 60)

class SimpleWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleWord2Vec, self).__init__()
        
        # 输入层 → 隐藏层：这就是我们要学习的词向量矩阵！
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 隐藏层 → 输出层：计算与所有词的相似度
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, word_idx):
        """
        word_idx: 词的索引 (相当于 One-hot 的简化表示)
        """
        # Step 1: 通过 embedding 层得到词向量
        # 这一步等价于: One-hot × Embedding矩阵
        embedded = self.embeddings(word_idx)  # [batch_size, embedding_dim]
        
        # Step 2: 通过线性层计算与所有词的相似度
        output = self.linear(embedded)  # [batch_size, vocab_size]
        
        # Step 3: 通过 Softmax 得到概率分布
        probs = torch.softmax(output, dim=-1)
        
        return embedded, output, probs

print("""
模型结构:
┌─────────────────────────────────────────────────────────────┐
│  输入层          隐藏层              输出层                   │
│  (One-hot)      (词嵌入)            (概率)                   │
│                                                             │
│  [1,0,0,0]  →  Embedding  →  [3维向量]  →  Linear  →  Softmax │
│   "king"        矩阵                      计算相似度    概率   │
│                [4×3]                      [3×4]              │
└─────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 第四步：可视化网络如何工作
# ============================================================
print("=" * 60)
print("第四步：跟踪 'king' 在网络中的处理过程")
print("=" * 60)

# 创建模型
torch.manual_seed(42)  # 固定随机种子，便于复现
model = SimpleWord2Vec(vocab_size, embedding_dim)

# 获取 "king" 的索引
king_idx = torch.tensor([word_to_idx["king"]])

# 前向传播
embedded_vector, similarity_scores, probabilities = model(king_idx)

# 1. 查看 Embedding 矩阵
print("\n1. Embedding 矩阵（词向量表，随机初始化）:")
print("   这就是我们要学习的核心参数！")
print("-" * 50)
for i, word in enumerate(vocab):
    vec = model.embeddings.weight[i].detach().numpy()
    print(f"   {word:6s}: [{vec[0]:7.4f}, {vec[1]:7.4f}, {vec[2]:7.4f}]")

# 2. 查看 "king" 的词向量
print(f"\n2. 'king' 的词向量（从 Embedding 矩阵第 0 行取出）:")
print(f"   {embedded_vector.detach().numpy()}")

# 3. 查看线性层权重
print(f"\n3. 线性层权重矩阵形状: {model.linear.weight.shape}")
print("   用于计算中心词与所有词的相似度")

# 4. 查看相似度分数
print(f"\n4. 'king' 与词汇表中所有词的相似度分数 (logits):")
print("-" * 50)
for i, word in enumerate(vocab):
    score = similarity_scores[0, i].item()
    print(f"   king vs {word:6s}: {score:8.4f}")

# 5. 查看 Softmax 后的概率
print(f"\n5. Softmax 后的概率分布:")
print("-" * 50)
for i, word in enumerate(vocab):
    prob = probabilities[0, i].item()
    bar = "█" * int(prob * 40)
    print(f"   P({word:6s} | king) = {prob:.4f} {bar}")

# ============================================================
# 第五步：演示 One-hot 如何选择词向量
# ============================================================
print("\n" + "=" * 60)
print("第五步：One-hot 编码如何选择词向量")
print("=" * 60)

print("""
核心原理:
  One-hot 向量 × Embedding 矩阵 = 选择矩阵的某一行

例如 "king" 的 One-hot = [1, 0, 0, 0]

        ┌─────────────────────┐
        │  E[0] = king 向量    │ ← One-hot[0]=1 选中这行
        │  E[1] = queen 向量   │
   E =  │  E[2] = man 向量     │
        │  E[3] = woman 向量   │
        └─────────────────────┘

  [1,0,0,0] × E = E[0] (king 的词向量)
""")

# 获取 Embedding 矩阵
E = model.embeddings.weight.detach().numpy()

print("验证 One-hot × E = E[i]:")
print("-" * 50)
for i, word in enumerate(vocab):
    onehot = np.zeros(vocab_size)
    onehot[i] = 1
    
    # One-hot × Embedding矩阵 = 选择第 i 行
    selected_vector = np.dot(onehot, E)
    direct_lookup = E[i, :]
    
    print(f"\n'{word}':")
    print(f"  One-hot: {onehot}")
    print(f"  One-hot × E = {selected_vector}")
    print(f"  E[{i}]       = {direct_lookup}")
    print(f"  相等: {np.allclose(selected_vector, direct_lookup)}")

# ============================================================
# 第六步：模拟训练过程
# ============================================================
print("\n" + "=" * 60)
print("第六步：模拟训练过程")
print("=" * 60)

print("""
训练目标:
  假设 "king" 的上下文经常出现 "queen" 和 "man"
  我们希望: P("queen" | "king") ↑  和  P("man" | "king") ↑
""")

# 创建新模型
torch.manual_seed(123)
model2 = SimpleWord2Vec(vocab_size, embedding_dim)

# 定义优化器和损失函数
optimizer = optim.SGD(model2.parameters(), lr=0.5)
criterion = nn.CrossEntropyLoss()

# 训练数据: "king" 的上下文是 "queen" 和 "man"
center_word = torch.tensor([word_to_idx["king"]])
target_words = [word_to_idx["queen"], word_to_idx["man"]]

print("\n训练前的词向量:")
print("-" * 50)
for i, word in enumerate(vocab):
    vec = model2.embeddings.weight[i].detach().numpy()
    print(f"  {word:6s}: [{vec[0]:7.4f}, {vec[1]:7.4f}, {vec[2]:7.4f}]")

# 计算训练前的相似度
king_vec_before = model2.embeddings.weight[word_to_idx["king"]].detach()
queen_vec_before = model2.embeddings.weight[word_to_idx["queen"]].detach()
sim_before = torch.cosine_similarity(king_vec_before.unsqueeze(0), 
                                      queen_vec_before.unsqueeze(0))
print(f"\n训练前 'king' 与 'queen' 的余弦相似度: {sim_before.item():.4f}")

# 训练
n_iterations = 100
print(f"\n开始训练 {n_iterations} 轮...")
print("-" * 50)

for epoch in range(n_iterations):
    optimizer.zero_grad()
    
    # 前向传播
    embedded, output, probs = model2(center_word)
    
    # 计算损失 - 希望 "queen" 和 "man" 的概率高
    loss = 0
    for target_idx in target_words:
        target = torch.tensor([target_idx])
        loss += criterion(output, target)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 打印进度
    if (epoch + 1) % 20 == 0:
        _, _, probs_now = model2(center_word)
        p_queen = probs_now[0, word_to_idx["queen"]].item()
        p_man = probs_now[0, word_to_idx["man"]].item()
        print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, "
              f"P(queen|king)={p_queen:.4f}, P(man|king)={p_man:.4f}")

print("\n训练后的词向量:")
print("-" * 50)
for i, word in enumerate(vocab):
    vec = model2.embeddings.weight[i].detach().numpy()
    print(f"  {word:6s}: [{vec[0]:7.4f}, {vec[1]:7.4f}, {vec[2]:7.4f}]")

# 计算训练后的相似度
king_vec_after = model2.embeddings.weight[word_to_idx["king"]].detach()
queen_vec_after = model2.embeddings.weight[word_to_idx["queen"]].detach()
man_vec_after = model2.embeddings.weight[word_to_idx["man"]].detach()
woman_vec_after = model2.embeddings.weight[word_to_idx["woman"]].detach()

print("\n训练后的词向量相似度:")
print("-" * 50)
pairs = [("king", "queen"), ("king", "man"), ("king", "woman"), ("queen", "woman")]
for w1, w2 in pairs:
    v1 = model2.embeddings.weight[word_to_idx[w1]].detach()
    v2 = model2.embeddings.weight[word_to_idx[w2]].detach()
    sim = torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    print(f"  {w1:6s} vs {w2:6s}: {sim.item():.4f}")

# 最终概率分布
print("\n训练后 P(word | king) 的概率分布:")
print("-" * 50)
_, _, final_probs = model2(center_word)
for i, word in enumerate(vocab):
    prob = final_probs[0, i].item()
    bar = "█" * int(prob * 40)
    marker = " ← 上下文词" if word in ["queen", "man"] else ""
    print(f"  P({word:6s} | king) = {prob:.4f} {bar}{marker}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("关键理解要点总结")
print("=" * 60)

print("""
1. One-hot 编码是选择器:
   One-hot 向量与 Embedding 矩阵相乘，实际上是从矩阵中
   选择出对应行（即该词的向量）。

2. Embedding 矩阵是核心:
   模型训练完成后，Embedding.weight 就是我们要的词向量表。

3. 线性层计算相似度:
   线性层的权重矩阵用于计算中心词向量与所有词向量的相似度。

4. Softmax 得到概率:
   相似度分数通过 Softmax 转化为概率分布，表示在给定中心词时，
   每个词出现在其上下文的概率。

5. 训练目标:
   通过调整 Embedding 矩阵，让实际出现在上下文中的词（如 "queen"）
   的概率变高，不出现的词的概率变低。

这就是 Word2Vec 的魔力:
   通过一个简单的分类任务，间接学习到了有语义信息的词向量！
""")

print("=" * 60)
print("演示完成!")
print("=" * 60)
