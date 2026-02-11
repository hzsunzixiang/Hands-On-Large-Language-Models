"""
Chapter 4 - Part 3: 嵌入 + 监督学习分类
使用 Sentence Transformer 生成嵌入，再用逻辑回归分类
参考: Hands-On Large Language Models - Chapter 4
"""

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
    prefix = model_name.replace("/", "_")
    
    np.save(CACHE_DIR / f"{prefix}_train_embeddings.npy", train_embeddings)
    np.save(CACHE_DIR / f"{prefix}_test_embeddings.npy", test_embeddings)
    np.save(CACHE_DIR / f"{prefix}_train_labels.npy", train_labels)
    np.save(CACHE_DIR / f"{prefix}_test_labels.npy", test_labels)
    
    print(f"嵌入向量已保存到: {CACHE_DIR}")


def load_cached_embeddings(model_name="all-mpnet-base-v2"):
    """从本地文件加载嵌入向量"""
    prefix = model_name.replace("/", "_")
    
    files = [
        CACHE_DIR / f"{prefix}_train_embeddings.npy",
        CACHE_DIR / f"{prefix}_test_embeddings.npy",
        CACHE_DIR / f"{prefix}_train_labels.npy",
        CACHE_DIR / f"{prefix}_test_labels.npy"
    ]
    
    if all(f.exists() for f in files):
        train_embeddings = np.load(files[0])
        test_embeddings = np.load(files[1])
        train_labels = np.load(files[2])
        test_labels = np.load(files[3])
        print(f"从缓存加载嵌入向量: {CACHE_DIR}")
        return train_embeddings, test_embeddings, train_labels, test_labels
    
    return None


def embedding_supervised_classification(data, use_cache=True):
    """
    使用嵌入 + 监督学习分类 (Supervised Classification)
    """
    print("\n" + "=" * 60)
    print("Part 3: 嵌入 + 监督学习分类 (Supervised Classification)")
    print("=" * 60)
    
    model_name = "all-mpnet-base-v2"
    
    # 尝试从缓存加载
    if use_cache:
        cached = load_cached_embeddings(model_name)
        if cached is not None:
            train_embeddings, test_embeddings, train_labels, test_labels = cached
            print(f"\n训练集嵌入形状: {train_embeddings.shape}")
            print(f"测试集嵌入形状: {test_embeddings.shape}")
            return None, train_embeddings, test_embeddings, train_labels, test_labels
    
    # 加载 Sentence Transformer 模型
    print("\n加载 Sentence Transformer 模型...")
    model = SentenceTransformer(f'sentence-transformers/{model_name}')
    
    # 生成嵌入 (Convert text to embeddings)
    print("生成训练集嵌入...")
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    print("生成测试集嵌入...")
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
    
    # 获取标签
    train_labels = np.array(data["train"]["label"])
    test_labels = np.array(data["test"]["label"])
    
    print(f"\n训练集嵌入形状: {train_embeddings.shape}")
    print(f"测试集嵌入形状: {test_embeddings.shape}")
    
    # 保存嵌入向量
    save_embeddings(train_embeddings, test_embeddings, train_labels, test_labels, model_name)
    
    return model, train_embeddings, test_embeddings, train_labels, test_labels


def supervised_classification(train_embeddings, test_embeddings, train_labels, test_labels):
    """
    训练逻辑回归分类器并预测
    """
    print("\n" + "-" * 40)
    print("监督学习分类 (Logistic Regression)")
    print("-" * 40)
    
    # 训练逻辑回归分类器
    print("训练逻辑回归分类器...")
    clf = LogisticRegression(random_state=42)
    clf.fit(train_embeddings, train_labels)
    
    # 预测
    y_pred = clf.predict(test_embeddings)
    
    print("\n分类结果:")
    evaluate_performance(test_labels, y_pred)
    
    return y_pred


def average_embedding_classification(train_embeddings, test_embeddings, train_labels, test_labels):
    """
    不使用分类器，直接用平均嵌入 + 余弦相似度
    (What would happen if we would not use a classifier at all?)
    """
    print("\n" + "-" * 40)
    print("平均嵌入 + 余弦相似度 (无分类器)")
    print("-" * 40)
    
    # 计算每个类别的平均嵌入
    # Average the embeddings of all documents in each target label
    df = pd.DataFrame(np.hstack([
        train_embeddings, 
        train_labels.reshape(-1, 1)
    ]))
    averaged_target_embeddings = df.groupby(768).mean().values
    
    # 计算相似度并预测
    # Find the best matching embeddings between evaluation documents and target embeddings
    sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)
    
    print("\n分类结果:")
    evaluate_performance(test_labels, y_pred)
    
    return y_pred


def main():
    # 加载数据
    data = load_data()
    
    # 生成/加载嵌入
    result = embedding_supervised_classification(data, use_cache=True)
    model, train_embeddings, test_embeddings, train_labels, test_labels = result
    
    # 方法1: 监督学习分类
    y_pred_supervised = supervised_classification(
        train_embeddings, test_embeddings, train_labels, test_labels
    )
    
    # 方法2: 平均嵌入 + 余弦相似度 (无分类器)
    y_pred_avg = average_embedding_classification(
        train_embeddings, test_embeddings, train_labels, test_labels
    )
    
    return train_embeddings, test_embeddings


if __name__ == "__main__":
    main()
