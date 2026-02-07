"""
Chapter 5 - Part 5: 使用 LLM 生成主题标签
使用 Flan-T5 或 DeepSeek 生成人类可读的主题描述
"""

from copy import deepcopy
from pathlib import Path
from bertopic import BERTopic
from bertopic.representation import TextGeneration
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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


def llm_representation_flant5(topic_model, abstracts, original_topics):
    """使用 Flan-T5 生成主题标签"""
    print("\n" + "=" * 60)
    print("Part 5.1: 使用 Flan-T5 生成主题标签")
    print("=" * 60)
    
    prompt = """I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the documents and keywords, what is this topic about?"""
    
    print("\n加载 Flan-T5-small 模型...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    generator = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        device="cpu"  # T5 在 MPS 上可能有问题
    )
    
    representation_model = TextGeneration(
        generator,
        prompt=prompt,
        doc_length=50,
        tokenizer="whitespace"
    )
    
    print("使用 LLM 更新主题表示...")
    topic_model.update_topics(abstracts, representation_model=representation_model)
    
    print("\n主题表示对比 (原始 vs Flan-T5):")
    df = topic_differences(topic_model, original_topics)
    print(df.to_string(index=False))


def llm_representation_deepseek(topic_model, abstracts, original_topics):
    """使用 DeepSeek 生成主题标签"""
    import requests
    from bertopic.representation import BaseRepresentation
    
    print("\n" + "=" * 60)
    print("Part 5.2: 使用 DeepSeek 生成主题标签")
    print("=" * 60)
    
    # 检查 API key
    key_file = Path.home() / ".deepseek"
    if not key_file.exists():
        print(f"跳过: 未找到 {key_file}")
        return
    
    api_key = key_file.read_text().strip()
    
    class DeepSeekRepresentation(BaseRepresentation):
        def __init__(self, api_key):
            self.api_key = api_key
            self.api_url = "https://api.deepseek.com/chat/completions"
        
        def extract_topics(self, topic_model, documents, c_tf_idf, topics):
            updated_topics = {}
            
            for topic_id in set(topics):
                if topic_id == -1:
                    continue
                
                # 获取该主题的关键词
                keywords = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                keywords_str = ", ".join(keywords)
                
                # 获取该主题的代表性文档
                topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id][:3]
                docs_str = "\n".join(topic_docs)[:500]
                
                prompt = f"""Based on these keywords: {keywords_str}
And sample documents: {docs_str}

Give a short (2-5 words) topic label. Just the label, nothing else."""
                
                try:
                    response = requests.post(
                        self.api_url,
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0}
                    )
                    label = response.json()["choices"][0]["message"]["content"].strip()
                    updated_topics[topic_id] = [(label, 1.0)] + [(w, s) for w, s in topic_model.get_topic(topic_id)[:4]]
                except Exception as e:
                    print(f"  Topic {topic_id} 失败: {e}")
                    updated_topics[topic_id] = topic_model.get_topic(topic_id)
            
            return updated_topics
    
    print("\n使用 DeepSeek 生成主题标签 (前5个主题)...")
    representation_model = DeepSeekRepresentation(api_key)
    
    # 只处理前5个主题以节省 API 调用
    topic_model.update_topics(abstracts, representation_model=representation_model)
    
    print("\n主题表示对比 (原始 vs DeepSeek):")
    df = topic_differences(topic_model, original_topics)
    print(df.to_string(index=False))


def main():
    # 加载数据
    abstracts, titles = load_data(sample_size=3000)  # 使用较少数据加快速度
    
    # 创建模型
    topic_model = create_topic_model(abstracts)
    
    # 保存原始主题表示
    original_topics = deepcopy(topic_model.topic_representations_)
    
    print("\n原始主题表示 (c-TF-IDF):")
    for topic in range(5):
        words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        print(f"  Topic {topic}: {words}")
    
    # Flan-T5
    try:
        llm_representation_flant5(topic_model, abstracts, original_topics)
    except Exception as e:
        print(f"\nFlan-T5 失败: {e}")
    
    # DeepSeek (可选)
    try:
        # 重新创建模型以使用原始表示
        topic_model2 = create_topic_model(abstracts)
        original_topics2 = deepcopy(topic_model2.topic_representations_)
        llm_representation_deepseek(topic_model2, abstracts, original_topics2)
    except Exception as e:
        print(f"\nDeepSeek 失败: {e}")


if __name__ == "__main__":
    main()
