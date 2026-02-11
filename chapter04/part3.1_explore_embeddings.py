"""
Chapter 4 - Part 2.1: 探索嵌入向量
读取保存的嵌入向量文件，展示向量数据的特性
"""

import numpy as np
from pathlib import Path

# 嵌入缓存目录
CACHE_DIR = Path(__file__).parent / "embeddings_cache"
MODEL_NAME = "all-mpnet-base-v2"


def load_embeddings():
    """加载保存的嵌入向量"""
    prefix = MODEL_NAME.replace("/", "_")
    
    train_embeddings = np.load(CACHE_DIR / f"{prefix}_train_embeddings.npy")
    test_embeddings = np.load(CACHE_DIR / f"{prefix}_test_embeddings.npy")
    train_labels = np.load(CACHE_DIR / f"{prefix}_train_labels.npy")
    test_labels = np.load(CACHE_DIR / f"{prefix}_test_labels.npy")
    
    return train_embeddings, test_embeddings, train_labels, test_labels


def explore_embeddings():
    """探索嵌入向量的特性"""
    print("=" * 60)
    print("Part 2.1: 探索嵌入向量")
    print("=" * 60)
    
    # 加载数据
    print(f"\n从 {CACHE_DIR} 加载嵌入向量...")
    train_emb, test_emb, train_labels, test_labels = load_embeddings()
    
    # 基本信息
    print("\n" + "-" * 40)
    print("1. 基本信息")
    print("-" * 40)
    print(f"训练集嵌入形状: {train_emb.shape} (样本数 x 向量维度)")
    print(f"测试集嵌入形状: {test_emb.shape}")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"测试集标签形状: {test_labels.shape}")
    print(f"向量维度: {train_emb.shape[1]}")
    print(f"数据类型: {train_emb.dtype}")
    
    # 向量统计
    print("\n" + "-" * 40)
    print("2. 向量统计信息")
    print("-" * 40)
    print(f"训练集嵌入值范围: [{train_emb.min():.4f}, {train_emb.max():.4f}]")
    print(f"训练集嵌入均值: {train_emb.mean():.6f}")
    print(f"训练集嵌入标准差: {train_emb.std():.4f}")
    
    # 单个向量的 L2 范数
    norms = np.linalg.norm(train_emb, axis=1)
    print(f"向量 L2 范数范围: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"向量 L2 范数均值: {norms.mean():.4f}")
    
    # 展示具体向量示例
    print("\n" + "-" * 40)
    print("3. 向量示例")
    print("-" * 40)
    
    for i in range(3):
        label_text = "正面" if train_labels[i] == 1 else "负面"
        print(f"\n样本 {i} (标签: {train_labels[i]} - {label_text}):")
        print(f"  向量前10维: {train_emb[i, :10]}")
        print(f"  向量后10维: {train_emb[i, -10:]}")
        print(f"  向量 L2 范数: {np.linalg.norm(train_emb[i]):.4f}")
    
    # 类别分析
    print("\n" + "-" * 40)
    print("4. 按类别分析")
    print("-" * 40)
    
    pos_mask = train_labels == 1
    neg_mask = train_labels == 0
    
    pos_emb = train_emb[pos_mask]
    neg_emb = train_emb[neg_mask]
    
    print(f"正面评论数量: {pos_emb.shape[0]}")
    print(f"负面评论数量: {neg_emb.shape[0]}")
    
    # 计算类别中心
    pos_center = pos_emb.mean(axis=0)
    neg_center = neg_emb.mean(axis=0)
    
    print(f"\n正面评论中心向量前10维: {pos_center[:10]}")
    print(f"负面评论中心向量前10维: {neg_center[:10]}")
    
    # 类别中心之间的余弦相似度
    cos_sim = np.dot(pos_center, neg_center) / (np.linalg.norm(pos_center) * np.linalg.norm(neg_center))
    print(f"\n正负类别中心的余弦相似度: {cos_sim:.4f}")
    
    # 类别中心之间的欧氏距离
    euclidean_dist = np.linalg.norm(pos_center - neg_center)
    print(f"正负类别中心的欧氏距离: {euclidean_dist:.4f}")
    
    # 随机选取几对样本计算相似度
    print("\n" + "-" * 40)
    print("5. 样本对相似度示例")
    print("-" * 40)
    
    np.random.seed(42)
    
    # 同类样本
    pos_indices = np.where(train_labels == 1)[0]
    neg_indices = np.where(train_labels == 0)[0]
    
    i, j = np.random.choice(pos_indices, 2, replace=False)
    sim = np.dot(train_emb[i], train_emb[j]) / (np.linalg.norm(train_emb[i]) * np.linalg.norm(train_emb[j]))
    print(f"两个正面样本 ({i}, {j}) 的余弦相似度: {sim:.4f}")
    
    i, j = np.random.choice(neg_indices, 2, replace=False)
    sim = np.dot(train_emb[i], train_emb[j]) / (np.linalg.norm(train_emb[i]) * np.linalg.norm(train_emb[j]))
    print(f"两个负面样本 ({i}, {j}) 的余弦相似度: {sim:.4f}")
    
    # 异类样本
    i = np.random.choice(pos_indices)
    j = np.random.choice(neg_indices)
    sim = np.dot(train_emb[i], train_emb[j]) / (np.linalg.norm(train_emb[i]) * np.linalg.norm(train_emb[j]))
    print(f"一正一负样本 ({i}, {j}) 的余弦相似度: {sim:.4f}")
    
    print("\n" + "=" * 60)
    print("探索完成!")
    print("=" * 60)
    
    return train_emb, test_emb, train_labels, test_labels


if __name__ == "__main__":
    explore_embeddings()
