"""
Chapter 5 - Part 3: BERTopic 主题建模
整合了: 嵌入 + UMAP + HDBSCAN + c-TF-IDF
"""

from copy import deepcopy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from common import load_data, get_device


def create_embeddings(abstracts, device=None):
    """生成嵌入"""
    if device is None:
        device = get_device()
    
    print("\n加载嵌入模型: thenlper/gte-small")
    embedding_model = SentenceTransformer('thenlper/gte-small', device=device)
    
    print("生成嵌入中...")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
    
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
    # 加载数据
    abstracts, titles = load_data(sample_size=5000)
    
    # 生成嵌入
    embeddings, embedding_model = create_embeddings(abstracts)
    
    # BERTopic 主题建模
    topic_model, topics = bertopic_modeling(abstracts, embeddings, embedding_model)
    
    # 保存原始主题表示 (供后续对比)
    original_topics = deepcopy(topic_model.topic_representations_)
    
    # 探索主题
    explore_topics(topic_model, abstracts, titles)
    
    return topic_model, original_topics, abstracts, titles, embeddings


if __name__ == "__main__":
    main()
