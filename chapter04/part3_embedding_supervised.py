"""
Chapter 4 - Part 3: 嵌入 + 监督学习分类
使用 Sentence Transformer 生成嵌入，再用逻辑回归分类
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from common import load_data, evaluate_performance

# 嵌入缓存目录
CACHE_DIR = Path(__file__).parent / "embeddings_cache"


def save_embeddings(train_embeddings, test_embeddings, train_labels, test_labels, model_name="all-mpnet-base-v2"):
    """保存嵌入向量到本地文件"""
    CACHE_DIR.mkdir(exist_ok=True)
    
    # 使用模型名作为文件前缀
    prefix = model_name.replace("/", "_")
    
    np.save(CACHE_DIR / f"{prefix}_train_embeddings.npy", train_embeddings)
    np.save(CACHE_DIR / f"{prefix}_test_embeddings.npy", test_embeddings)
    np.save(CACHE_DIR / f"{prefix}_train_labels.npy", train_labels)
    np.save(CACHE_DIR / f"{prefix}_test_labels.npy", test_labels)
    
    print(f"嵌入向量已保存到: {CACHE_DIR}")


def load_embeddings(model_name="all-mpnet-base-v2"):
    """从本地文件加载嵌入向量"""
    prefix = model_name.replace("/", "_")
    
    train_file = CACHE_DIR / f"{prefix}_train_embeddings.npy"
    test_file = CACHE_DIR / f"{prefix}_test_embeddings.npy"
    train_labels_file = CACHE_DIR / f"{prefix}_train_labels.npy"
    test_labels_file = CACHE_DIR / f"{prefix}_test_labels.npy"
    
    if all(f.exists() for f in [train_file, test_file, train_labels_file, test_labels_file]):
        train_embeddings = np.load(train_file)
        test_embeddings = np.load(test_file)
        train_labels = np.load(train_labels_file)
        test_labels = np.load(test_labels_file)
        print(f"从缓存加载嵌入向量: {CACHE_DIR}")
        return train_embeddings, test_embeddings, train_labels, test_labels
    
    return None


def embedding_supervised_classification(data, use_cache=True):
    """
    使用嵌入 + 监督学习分类
    """
    print("\n" + "=" * 60)
    print("Part 3: 嵌入 + 监督学习分类")
    print("=" * 60)
    
    model_name = "all-mpnet-base-v2"
    
    # 尝试从缓存加载
    if use_cache:
        cached = load_embeddings(model_name)
        if cached is not None:
            train_embeddings, test_embeddings, train_labels, test_labels = cached
            print(f"\n训练集嵌入维度: {train_embeddings.shape}")
            print(f"测试集嵌入维度: {test_embeddings.shape}")
            
            # 训练逻辑回归分类器
            print("\n训练逻辑回归分类器...")
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(train_embeddings, train_labels)
            
            # 预测
            y_pred = clf.predict(test_embeddings)
            
            print("\n监督分类结果:")
            evaluate_performance(test_labels, y_pred)
            
            return None, train_embeddings, test_embeddings, y_pred, train_labels, test_labels
    
    # 加载 Sentence Transformer 模型
    print("\n加载 Sentence Transformer 模型...")
    model = SentenceTransformer(f'sentence-transformers/{model_name}')
    
    # 生成嵌入
    print("生成训练集嵌入...")
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    print("生成测试集嵌入...")
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
    
    # 获取标签
    train_labels = np.array(data["train"]["label"])
    test_labels = np.array(data["test"]["label"])
    
    print(f"\n训练集嵌入维度: {train_embeddings.shape}")
    print(f"测试集嵌入维度: {test_embeddings.shape}")
    
    # 保存嵌入向量
    save_embeddings(train_embeddings, test_embeddings, train_labels, test_labels, model_name)
    
    # 训练逻辑回归分类器
    print("\n训练逻辑回归分类器...")
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(train_embeddings, train_labels)
    
    # 预测
    y_pred = clf.predict(test_embeddings)
    
    print("\n监督分类结果:")
    evaluate_performance(test_labels, y_pred)
    
    return model, train_embeddings, test_embeddings, y_pred, train_labels, test_labels


def average_embedding_classification(train_embeddings, test_embeddings, train_labels, test_labels):
    """
    额外实验: 不使用分类器，直接用平均嵌入 + 余弦相似度
    """
    print("\n" + "-" * 40)
    print("额外实验: 平均嵌入 + 余弦相似度 (无分类器)")
    print("-" * 40)
    
    # 计算每个类别的平均嵌入
    df = pd.DataFrame(np.hstack([
        train_embeddings, 
        train_labels.reshape(-1, 1)
    ]))
    averaged_target_embeddings = df.groupby(768).mean().values
    
    # 计算相似度并预测
    sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
    y_pred_avg = np.argmax(sim_matrix, axis=1)
    
    print("\n平均嵌入分类结果:")
    evaluate_performance(test_labels, y_pred_avg)
    
    return y_pred_avg


def main():
    # 加载数据
    data = load_data()
    
    # 运行嵌入 + 监督学习分类 (use_cache=True 自动使用缓存)
    result = embedding_supervised_classification(data, use_cache=True)
    model, train_embeddings, test_embeddings, y_pred, train_labels, test_labels = result
    
    # 运行平均嵌入实验
    y_pred_avg = average_embedding_classification(train_embeddings, test_embeddings, train_labels, test_labels)
    
    return model, test_embeddings


if __name__ == "__main__":
    main()
