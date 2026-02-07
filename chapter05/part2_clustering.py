"""
Chapter 5 - Part 2: 文本聚类流程
嵌入 -> 降维 -> 聚类
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from common import load_data, get_device


def create_embeddings(abstracts, device=None):
    """使用 Sentence Transformer 生成嵌入"""
    print("\n" + "=" * 60)
    print("Part 2.1: 生成文档嵌入")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    print("\n加载嵌入模型: thenlper/gte-small")
    embedding_model = SentenceTransformer('thenlper/gte-small', device=device)
    
    print("生成嵌入中...")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
    
    print(f"\n嵌入维度: {embeddings.shape}")
    print(f"  - 文档数: {embeddings.shape[0]}")
    print(f"  - 嵌入维度: {embeddings.shape[1]}")
    
    return embeddings, embedding_model


def reduce_dimensions(embeddings):
    """使用 UMAP 降维"""
    print("\n" + "=" * 60)
    print("Part 2.2: 降维 (UMAP)")
    print("=" * 60)
    
    # 从 384 维降到 5 维 (用于聚类)
    print("\n将嵌入从 384 维降至 5 维...")
    umap_model = UMAP(
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    print(f"降维后维度: {reduced_embeddings.shape}")
    
    # 降到 2 维 (用于可视化)
    print("\n降至 2 维 (用于可视化)...")
    umap_2d = UMAP(
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    embeddings_2d = umap_2d.fit_transform(embeddings)
    
    return reduced_embeddings, embeddings_2d


def cluster_documents(reduced_embeddings):
    """使用 HDBSCAN 聚类"""
    print("\n" + "=" * 60)
    print("Part 2.3: 聚类 (HDBSCAN)")
    print("=" * 60)
    
    print("\n使用 HDBSCAN 聚类...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        metric='euclidean',
        cluster_selection_method='eom'
    ).fit(reduced_embeddings)
    
    clusters = hdbscan_model.labels_
    n_clusters = len(set(clusters))
    
    print(f"生成的簇数量: {n_clusters}")
    print(f"噪声点数量 (label=-1): {sum(clusters == -1)}")
    
    return clusters


def inspect_clusters(abstracts, clusters, cluster_id=0, n_docs=3):
    """检查特定簇中的文档"""
    print(f"\n" + "-" * 40)
    print(f"簇 {cluster_id} 的前 {n_docs} 个文档:")
    print("-" * 40)
    
    indices = np.where(clusters == cluster_id)[0][:n_docs]
    for idx in indices:
        print(f"\n{abstracts[idx][:250]}...\n")


def main():
    # 加载数据
    abstracts, titles = load_data(sample_size=5000)
    
    # 生成嵌入
    embeddings, embedding_model = create_embeddings(abstracts)
    
    # 降维
    reduced_embeddings, embeddings_2d = reduce_dimensions(embeddings)
    
    # 聚类
    clusters = cluster_documents(reduced_embeddings)
    
    # 检查簇内容
    inspect_clusters(abstracts, clusters, cluster_id=0, n_docs=2)
    inspect_clusters(abstracts, clusters, cluster_id=1, n_docs=2)
    
    return embeddings, embedding_model, reduced_embeddings, embeddings_2d, clusters


if __name__ == "__main__":
    main()
