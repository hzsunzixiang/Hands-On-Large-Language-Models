"""
Chapter 4 - Part 4: 零样本分类
使用标签描述的嵌入与文档嵌入比较，无需训练数据
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from common import load_data, evaluate_performance


def zero_shot_classification(data, model=None, test_embeddings=None):
    """
    零样本分类: 使用标签描述的嵌入与文档嵌入比较
    """
    print("\n" + "=" * 60)
    print("Part 4: 零样本分类 (Zero-shot)")
    print("=" * 60)
    
    # 如果没有提供模型和嵌入，则加载/生成
    if model is None:
        print("\n加载 Sentence Transformer 模型...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    if test_embeddings is None:
        print("生成测试集嵌入...")
        test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
    
    # 为标签创建嵌入
    print("\n创建标签嵌入...")
    label_embeddings = model.encode(["A negative review", "A positive review"])
    
    # 计算相似度
    sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)
    
    print("\n零样本分类结果 (简单标签描述):")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return model, test_embeddings, y_pred


def zero_shot_detailed_labels(data, model, test_embeddings):
    """
    尝试更详细的标签描述
    """
    print("\n" + "-" * 40)
    print("尝试更详细的标签描述:")
    print("-" * 40)
    
    label_embeddings_v2 = model.encode([
        "A very negative movie review",
        "A very positive movie review"
    ])
    
    sim_matrix_v2 = cosine_similarity(test_embeddings, label_embeddings_v2)
    y_pred_v2 = np.argmax(sim_matrix_v2, axis=1)
    
    print("\n使用详细标签描述的结果:")
    evaluate_performance(data["test"]["label"], y_pred_v2)
    
    return y_pred_v2


def main():
    # 加载数据
    data = load_data()
    
    # 运行零样本分类
    model, test_embeddings, y_pred = zero_shot_classification(data)
    
    # 尝试详细标签
    y_pred_v2 = zero_shot_detailed_labels(data, model, test_embeddings)
    
    return y_pred


if __name__ == "__main__":
    main()
