"""
Chapter 5 - Part 3: BERTopic 主题建模
整合了: 嵌入 + UMAP + HDBSCAN + c-TF-IDF
"""

from copy import deepcopy
from pathlib import Path
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from common import load_data, get_device

# 缓存目录 (与 part2 共享)
CACHE_DIR = Path(__file__).parent / "embeddings_cache"


def save_embeddings(embeddings: np.ndarray, abstracts: list, titles: list, 
                    model_name: str, sample_size: int):
    """保存嵌入到缓存"""
    CACHE_DIR.mkdir(exist_ok=True)
    
    safe_model_name = model_name.replace("/", "_")
    prefix = f"{safe_model_name}_{sample_size}"
    
    np.save(CACHE_DIR / f"{prefix}_embeddings.npy", embeddings)
    np.save(CACHE_DIR / f"{prefix}_abstracts.npy", np.array(abstracts, dtype=object))
    np.save(CACHE_DIR / f"{prefix}_titles.npy", np.array(titles, dtype=object))
    
    print(f"  已保存嵌入到: {CACHE_DIR / prefix}_*.npy")


def load_cached_embeddings(model_name: str, sample_size: int):
    """从缓存加载嵌入"""
    safe_model_name = model_name.replace("/", "_")
    prefix = f"{safe_model_name}_{sample_size}"
    
    embeddings_path = CACHE_DIR / f"{prefix}_embeddings.npy"
    
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        return embeddings
    
    return None


def create_embeddings(abstracts, titles=None, model_name="thenlper/gte-small", 
                      sample_size=5000, device=None):
    """生成嵌入 (带缓存)"""
    print("\n" + "=" * 60)
    print("Part 3.1: 生成文档嵌入")
    print("=" * 60)
    
    # 尝试从缓存加载
    cached_embeddings = load_cached_embeddings(model_name, sample_size)
    if cached_embeddings is not None:
        print(f"\n从缓存加载嵌入: {model_name} (样本数: {sample_size})")
        print(f"嵌入维度: {cached_embeddings.shape}")
        
        # 仍需加载模型供 BERTopic 使用
        if device is None:
            device = get_device()
        embedding_model = SentenceTransformer(model_name, device=device)
        
        return cached_embeddings, embedding_model
    
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
    
    # 保存缓存
    if titles is not None:
        save_embeddings(embeddings, abstracts, titles, model_name, sample_size)
    
    return embeddings, embedding_model


def bertopic_modeling(abstracts, embeddings, embedding_model):
    """使用 BERTopic 进行主题建模"""
    print("\n" + "=" * 60)
    print("Part 3: BERTopic 主题建模")
    print("=" * 60)
    
    print("\n训练 BERTopic 模型...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        verbose=True
    )
    
    # 使用预计算的嵌入来训练
    topics, probs = topic_model.fit_transform(abstracts, embeddings)
    
    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    print(f"\n发现 {len(topic_info)} 个主题")
    print("\n前 10 个主题:")
    print(topic_info.head(10)[['Topic', 'Count', 'Name']])
    
    return topic_model, topics


def explore_topics(topic_model, abstracts, titles):
    """探索和搜索主题"""
    print("\n" + "-" * 40)
    print("探索主题")
    print("-" * 40)
    
    # 查看特定主题的关键词
    print("\n主题 0 的关键词:")
    topic_0_keywords = topic_model.get_topic(0)
    for word, score in topic_0_keywords[:5]:
        print(f"  {word}: {score:.4f}")
    
    # 搜索主题
    print("\n搜索 'topic modeling' 相关主题:")
    similar_topics, similarities = topic_model.find_topics("topic modeling")
    print(f"  最相关主题: {similar_topics[0]} (相似度: {similarities[0]:.4f})")
    
    # 检查该主题的关键词
    if similar_topics[0] >= 0:
        print(f"\n主题 {similar_topics[0]} 的关键词:")
        for word, score in topic_model.get_topic(similar_topics[0])[:5]:
            print(f"  {word}: {score:.4f}")
    
    # 搜索其他主题
    for query in ["machine translation", "sentiment analysis", "question answering"]:
        similar_topics, similarities = topic_model.find_topics(query)
        print(f"\n'{query}' 相关主题: {similar_topics[0]} (相似度: {similarities[0]:.4f})")


def main():
    model_name = "thenlper/gte-small"
    sample_size = 5000
    
    # 加载数据
    abstracts, titles = load_data(sample_size=sample_size)
    
    # 生成嵌入 (带缓存)
    embeddings, embedding_model = create_embeddings(
        abstracts, titles=titles, 
        model_name=model_name, sample_size=sample_size
    )
    
    # BERTopic 主题建模
    topic_model, topics = bertopic_modeling(abstracts, embeddings, embedding_model)
    
    # 保存原始主题表示 (供后续对比)
    original_topics = deepcopy(topic_model.topic_representations_)
    
    # 探索主题
    explore_topics(topic_model, abstracts, titles)
    
    return topic_model, original_topics, abstracts, titles, embeddings


if __name__ == "__main__":
    main()
