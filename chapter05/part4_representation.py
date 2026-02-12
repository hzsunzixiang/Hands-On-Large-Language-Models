"""
Chapter 5 - Part 4: 主题表示模型
使用 KeyBERT 和 MMR 改进主题描述
"""

from copy import deepcopy
from pathlib import Path
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer

from common import load_data, get_device, topic_differences

# 缓存目录 (与其他 part 共享)
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


def create_topic_model(abstracts, titles=None, model_name="thenlper/gte-small", 
                       sample_size=5000):
    """创建并训练 BERTopic 模型 (带缓存)"""
    device = get_device()
    
    # 尝试从缓存加载
    cached_embeddings = load_cached_embeddings(model_name, sample_size)
    
    if cached_embeddings is not None:
        print(f"\n从缓存加载嵌入: {model_name} (样本数: {sample_size})")
        print(f"嵌入维度: {cached_embeddings.shape}")
        embeddings = cached_embeddings
        
        print("加载嵌入模型...")
        embedding_model = SentenceTransformer(model_name, device=device)
    else:
        print("\n加载嵌入模型...")
        embedding_model = SentenceTransformer(model_name, device=device)
        
        print("生成嵌入...")
        embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
        
        # 保存缓存
        if titles is not None:
            save_embeddings(embeddings, abstracts, titles, model_name, sample_size)
    
    print("\n训练 BERTopic 模型...")
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
    topic_model.fit_transform(abstracts, embeddings)
    
    return topic_model


def keybert_representation(topic_model, abstracts, original_topics):
    """使用 KeyBERTInspired 更新主题表示"""
    print("\n" + "=" * 60)
    print("Part 4.1: KeyBERTInspired")
    print("  - 使用嵌入相似度选择更具代表性的关键词")
    print("=" * 60)
    
    representation_model = KeyBERTInspired()
    topic_model.update_topics(abstracts, representation_model=representation_model)
    
    print("\n主题表示对比:")
    df = topic_differences(topic_model, original_topics)
    print(df.to_string(index=False))


def mmr_representation(topic_model, abstracts, original_topics):
    """使用 MMR 增加关键词多样性"""
    print("\n" + "=" * 60)
    print("Part 4.2: Maximal Marginal Relevance (MMR)")
    print("  - 在相关性和多样性之间取得平衡")
    print("=" * 60)
    
    representation_model = MaximalMarginalRelevance(diversity=0.5)
    topic_model.update_topics(abstracts, representation_model=representation_model)
    
    print("\n主题表示对比:")
    df = topic_differences(topic_model, original_topics)
    print(df.to_string(index=False))


def main():
    model_name = "thenlper/gte-small"
    sample_size = 5000
    
    # 加载数据
    abstracts, titles = load_data(sample_size=sample_size)
    
    # 创建模型 (带缓存)
    topic_model = create_topic_model(abstracts, titles=titles, 
                                     model_name=model_name, sample_size=sample_size)
    
    # 保存原始主题表示
    original_topics = deepcopy(topic_model.topic_representations_)
    
    print("\n原始主题表示 (c-TF-IDF):")
    for topic in range(5):
        words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        print(f"  Topic {topic}: {words}")
    
    # KeyBERT
    keybert_representation(topic_model, abstracts, original_topics)
    
    # MMR
    mmr_representation(topic_model, abstracts, original_topics)


if __name__ == "__main__":
    main()
