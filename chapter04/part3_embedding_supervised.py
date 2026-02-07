"""
Chapter 4 - Part 3: 嵌入 + 监督学习分类
使用 Sentence Transformer 生成嵌入，再用逻辑回归分类
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from common import load_data, evaluate_performance


def embedding_supervised_classification(data):
    """
    使用嵌入 + 监督学习分类
    """
    print("\n" + "=" * 60)
    print("Part 3: 嵌入 + 监督学习分类")
    print("=" * 60)
    
    # 加载 Sentence Transformer 模型
    print("\n加载 Sentence Transformer 模型...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # 生成嵌入
    print("生成训练集嵌入...")
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    print("生成测试集嵌入...")
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
    
    print(f"\n训练集嵌入维度: {train_embeddings.shape}")
    print(f"测试集嵌入维度: {test_embeddings.shape}")
    
    # 训练逻辑回归分类器
    print("\n训练逻辑回归分类器...")
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(train_embeddings, data["train"]["label"])
    
    # 预测
    y_pred = clf.predict(test_embeddings)
    
    print("\n监督分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return model, train_embeddings, test_embeddings, y_pred


def average_embedding_classification(data, train_embeddings, test_embeddings):
    """
    额外实验: 不使用分类器，直接用平均嵌入 + 余弦相似度
    """
    print("\n" + "-" * 40)
    print("额外实验: 平均嵌入 + 余弦相似度 (无分类器)")
    print("-" * 40)
    
    # 计算每个类别的平均嵌入
    df = pd.DataFrame(np.hstack([
        train_embeddings, 
        np.array(data["train"]["label"]).reshape(-1, 1)
    ]))
    averaged_target_embeddings = df.groupby(768).mean().values
    
    # 计算相似度并预测
    sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
    y_pred_avg = np.argmax(sim_matrix, axis=1)
    
    print("\n平均嵌入分类结果:")
    evaluate_performance(data["test"]["label"], y_pred_avg)
    
    return y_pred_avg


def main():
    # 加载数据
    data = load_data()
    
    # 运行嵌入 + 监督学习分类
    model, train_embeddings, test_embeddings, y_pred = embedding_supervised_classification(data)
    
    # 运行平均嵌入实验
    y_pred_avg = average_embedding_classification(data, train_embeddings, test_embeddings)
    
    return model, test_embeddings


if __name__ == "__main__":
    main()
