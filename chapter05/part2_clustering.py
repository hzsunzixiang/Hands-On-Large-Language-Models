"""
Chapter 5 - Part 2: 文本聚类流程
嵌入 -> 降维 -> 聚类
"""
import os
# 解决 macOS 上 OpenMP 库冲突问题 (UMAP + numba)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from common import load_data, get_device

# 缓存目录
CACHE_DIR = Path(__file__).parent / "embeddings_cache"


def save_embeddings(embeddings: np.ndarray, abstracts: list, titles: list, 
                    model_name: str, sample_size: int):
    """保存嵌入到缓存"""
    CACHE_DIR.mkdir(exist_ok=True)
    
    # 使用模型名和样本数作为缓存键
    safe_model_name = model_name.replace("/", "_")
    prefix = f"{safe_model_name}_{sample_size}"
    
    np.save(CACHE_DIR / f"{prefix}_embeddings.npy", embeddings)
    np.save(CACHE_DIR / f"{prefix}_abstracts.npy", np.array(abstracts, dtype=object))
    np.save(CACHE_DIR / f"{prefix}_titles.npy", np.array(titles, dtype=object))
    
    print(f"  已保存嵌入到: {CACHE_DIR / prefix}_*.npy")


def load_embeddings(model_name: str, sample_size: int):
    """从缓存加载嵌入"""
    safe_model_name = model_name.replace("/", "_")
    prefix = f"{safe_model_name}_{sample_size}"
    
    embeddings_path = CACHE_DIR / f"{prefix}_embeddings.npy"
    abstracts_path = CACHE_DIR / f"{prefix}_abstracts.npy"
    titles_path = CACHE_DIR / f"{prefix}_titles.npy"
    
    if embeddings_path.exists() and abstracts_path.exists() and titles_path.exists():
        embeddings = np.load(embeddings_path)
        abstracts = np.load(abstracts_path, allow_pickle=True).tolist()
        titles = np.load(titles_path, allow_pickle=True).tolist()
        return embeddings, abstracts, titles
    
    return None, None, None


def create_embeddings(abstracts, model_name="thenlper/gte-small", 
                      sample_size=5000, device=None):
    """使用 Sentence Transformer 生成嵌入 (带缓存)"""
    print("\n" + "=" * 60)
    print("Part 2.1: 生成文档嵌入")
    print("=" * 60)
    
    # 尝试从缓存加载
    cached_embeddings, _, _ = load_embeddings(model_name, sample_size)
    if cached_embeddings is not None:
        print(f"\n从缓存加载嵌入: {model_name} (样本数: {sample_size})")
        print(f"嵌入维度: {cached_embeddings.shape}")
        return cached_embeddings
    
    # 缓存未命中，生成新嵌入
    if device is None:
        device = get_device()
    
    print(f"\n加载嵌入模型: {model_name}")
    embedding_model = SentenceTransformer(model_name, device=device)
    
    print("生成嵌入中...")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
    
    print(f"\n嵌入维度: {embeddings.shape}")
    print(f"  - 文档数: {embeddings.shape[0]}")
    print(f"  - 嵌入维度: {embeddings.shape[1]}")
    
    return embeddings


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
    model_name = "thenlper/gte-small"
    sample_size = 5000
    
    # 加载数据
    abstracts, titles = load_data(sample_size=sample_size)
    
    # 生成嵌入 (带缓存)
    embeddings = create_embeddings(abstracts, model_name=model_name, 
                                   sample_size=sample_size)
    
    # 首次运行时保存缓存
    cache_file = CACHE_DIR / f"{model_name.replace('/', '_')}_{sample_size}_embeddings.npy"
    if not cache_file.exists():
        save_embeddings(embeddings, abstracts, titles, model_name, sample_size)
    
    # 降维
    reduced_embeddings, embeddings_2d = reduce_dimensions(embeddings)
    
    # 聚类
    clusters = cluster_documents(reduced_embeddings)
    
    # 检查簇内容
    inspect_clusters(abstracts, clusters, cluster_id=0, n_docs=2)
    inspect_clusters(abstracts, clusters, cluster_id=1, n_docs=2)
    
    return embeddings, reduced_embeddings, embeddings_2d, clusters


if __name__ == "__main__":
    main()
