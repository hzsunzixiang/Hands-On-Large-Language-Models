"""
Chapter 5 - Part 4: 主题表示模型
使用 KeyBERT 和 MMR 改进主题描述
"""

from copy import deepcopy
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer

from common import load_data, get_device, topic_differences


def create_topic_model(abstracts):
    """创建并训练 BERTopic 模型"""
    device = get_device()
    
    print("\n加载嵌入模型...")
    embedding_model = SentenceTransformer('thenlper/gte-small', device=device)
    
    print("生成嵌入...")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
    
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
    # 加载数据
    abstracts, titles = load_data(sample_size=5000)
    
    # 创建模型
    topic_model = create_topic_model(abstracts)
    
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
