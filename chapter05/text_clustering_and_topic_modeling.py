"""
Chapter 5 - Text Clustering and Topic Modeling
使用多种语言模型对文档进行聚类和主题建模

本章内容:
1. 加载数据 (ArXiv NLP 论文摘要)
2. 文本聚类通用流程: 嵌入 -> 降维 -> 聚类
3. 使用 BERTopic 进行主题建模
4. 主题表示模型 (KeyBERT, MMR, LLM)
5. 可视化
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from copy import deepcopy
import torch


def get_device():
    """
    自动检测最佳可用设备
    优先级: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"使用设备: CUDA ({device_name})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("使用设备: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("使用设备: CPU")
    return device


def load_data():
    """加载 ArXiv NLP 论文数据集"""
    from datasets import load_dataset
    
    print("=" * 60)
    print("Part 1: 加载数据")
    print("=" * 60)
    
    dataset = load_dataset("maartengr/arxiv_nlp")["train"]
    
    # 提取摘要和标题
    abstracts = list(dataset["Abstracts"])
    titles = list(dataset["Titles"])
    
    print(f"\n论文数量: {len(abstracts)}")
    print(f"\n示例标题: {titles[0]}")
    print(f"\n示例摘要 (前200字): {abstracts[0][:200]}...")
    
    return abstracts, titles


def create_embeddings(abstracts, device=None):
    """
    Part 2.1: 使用 Sentence Transformer 为文档生成嵌入
    支持 CUDA / MPS / CPU 加速
    """
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "=" * 60)
    print("Part 2.1: 生成文档嵌入")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 使用轻量级嵌入模型
    print("\n加载嵌入模型: thenlper/gte-small")
    embedding_model = SentenceTransformer('thenlper/gte-small', device=device)
    
    # 生成嵌入
    print("生成嵌入中...")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
    
    print(f"\n嵌入维度: {embeddings.shape}")
    print(f"  - 文档数: {embeddings.shape[0]}")
    print(f"  - 嵌入维度: {embeddings.shape[1]}")
    
    return embeddings, embedding_model, device


def reduce_dimensions(embeddings):
    """
    Part 2.2: 使用 UMAP 降维
    """
    from umap import UMAP
    
    print("\n" + "=" * 60)
    print("Part 2.2: 降维 (UMAP)")
    print("=" * 60)
    
    # 从 384 维降到 5 维
    print("\n将嵌入从 384 维降至 5 维...")
    umap_model = UMAP(
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    print(f"降维后维度: {reduced_embeddings.shape}")
    
    # 为可视化再降到 2 维
    umap_2d = UMAP(
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    embeddings_2d = umap_2d.fit_transform(embeddings)
    
    return reduced_embeddings, embeddings_2d


def cluster_documents(reduced_embeddings):
    """
    Part 2.3: 使用 HDBSCAN 聚类
    """
    from hdbscan import HDBSCAN
    
    print("\n" + "=" * 60)
    print("Part 2.3: 聚类 (HDBSCAN)")
    print("=" * 60)
    
    # HDBSCAN 聚类
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


def bertopic_modeling(abstracts, embeddings, embedding_model):
    """
    Part 3: 使用 BERTopic 进行主题建模
    BERTopic 整合了: 嵌入 + UMAP + HDBSCAN + c-TF-IDF
    """
    from bertopic import BERTopic
    
    print("\n" + "=" * 60)
    print("Part 3: BERTopic 主题建模")
    print("=" * 60)
    
    # 创建 BERTopic 模型
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
    
    # 查找特定论文的主题
    if 'BERTopic' in str(titles):
        try:
            bertopic_idx = titles.index('BERTopic: Neural topic modeling with a class-based TF-IDF procedure')
            bertopic_topic = topic_model.topics_[bertopic_idx]
            print(f"\nBERTopic 论文所属主题: {bertopic_topic}")
        except:
            pass


def topic_differences(model, original_topics, nr_topics=5):
    """展示主题表示更新前后的差异"""
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    return df


def representation_models_demo(topic_model, abstracts, original_topics):
    """
    Part 4: 主题表示模型演示
    展示如何使用不同的表示模型改进主题描述
    """
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    
    print("\n" + "=" * 60)
    print("Part 4: 主题表示模型")
    print("=" * 60)
    
    # 4.1 KeyBERTInspired
    print("\n" + "-" * 40)
    print("4.1 KeyBERTInspired - 使用嵌入相似度选择关键词")
    print("-" * 40)
    
    representation_model = KeyBERTInspired()
    topic_model.update_topics(abstracts, representation_model=representation_model)
    
    df = topic_differences(topic_model, original_topics)
    print(df.to_string(index=False))
    
    # 4.2 Maximal Marginal Relevance (MMR)
    print("\n" + "-" * 40)
    print("4.2 MMR - 增加关键词多样性")
    print("-" * 40)
    
    representation_model = MaximalMarginalRelevance(diversity=0.5)
    topic_model.update_topics(abstracts, representation_model=representation_model)
    
    df = topic_differences(topic_model, original_topics)
    print(df.to_string(index=False))


def llm_representation_demo(topic_model, abstracts, original_topics, device=None):
    """
    Part 5: 使用 LLM (Flan-T5) 生成主题标签
    支持 CUDA / MPS / CPU
    """
    from transformers import pipeline
    from bertopic.representation import TextGeneration
    
    print("\n" + "=" * 60)
    print("Part 5: 使用 LLM 生成主题标签 (Flan-T5)")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 定义提示模板
    prompt = """I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the documents and keywords, what is this topic about?"""
    
    print("\n加载 Flan-T5-small 模型...")
    # MPS 设备需要特殊处理
    if device == "mps":
        # 某些模型在 MPS 上可能有问题，使用 CPU 作为后备
        try:
            generator = pipeline('text2text-generation', model='google/flan-t5-small', device=device)
        except:
            print("MPS 不兼容，回退到 CPU...")
            generator = pipeline('text2text-generation', model='google/flan-t5-small', device="cpu")
    elif device == "cuda":
        generator = pipeline('text2text-generation', model='google/flan-t5-small', device=0)
    else:
        generator = pipeline('text2text-generation', model='google/flan-t5-small', device="cpu")
    
    representation_model = TextGeneration(
        generator,
        prompt=prompt,
        doc_length=50,
        tokenizer="whitespace"
    )
    
    print("使用 LLM 更新主题表示...")
    topic_model.update_topics(abstracts, representation_model=representation_model)
    
    print("\n主题表示对比 (原始 vs LLM 生成):")
    df = topic_differences(topic_model, original_topics)
    print(df.to_string(index=False))


def visualize_topics(topic_model, abstracts, titles, embeddings_2d):
    """
    Part 6: 可视化
    """
    print("\n" + "=" * 60)
    print("Part 6: 可视化")
    print("=" * 60)
    
    try:
        # 1. 文档可视化
        print("\n生成文档可视化...")
        fig_docs = topic_model.visualize_documents(
            titles,
            reduced_embeddings=embeddings_2d,
            width=1200,
            hide_annotations=True
        )
        fig_docs.write_html("topic_documents.html")
        print("  -> 已保存: topic_documents.html")
        
        # 2. 主题关键词柱状图
        print("\n生成主题关键词柱状图...")
        fig_bar = topic_model.visualize_barchart(top_n_topics=10)
        fig_bar.write_html("topic_barchart.html")
        print("  -> 已保存: topic_barchart.html")
        
        # 3. 主题相似度热力图
        print("\n生成主题相似度热力图...")
        fig_heat = topic_model.visualize_heatmap(n_clusters=30)
        fig_heat.write_html("topic_heatmap.html")
        print("  -> 已保存: topic_heatmap.html")
        
        # 4. 层次聚类树
        print("\n生成层次聚类树...")
        fig_hier = topic_model.visualize_hierarchy()
        fig_hier.write_html("topic_hierarchy.html")
        print("  -> 已保存: topic_hierarchy.html")
        
    except Exception as e:
        print(f"可视化时出错: {e}")
        print("(可能需要安装 plotly)")


def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("Chapter 5 总结")
    print("=" * 60)
    
    print("""
文本聚类与主题建模流程:

┌──────────────────────────────────────────────────────────┐
│  1. 文档嵌入 (Sentence Transformer)                      │
│     text -> [384维向量]                                  │
├──────────────────────────────────────────────────────────┤
│  2. 降维 (UMAP)                                          │
│     [384维] -> [5维] (用于聚类)                          │
│     [384维] -> [2维] (用于可视化)                        │
├──────────────────────────────────────────────────────────┤
│  3. 聚类 (HDBSCAN)                                       │
│     基于密度的聚类，自动发现簇数量                       │
├──────────────────────────────────────────────────────────┤
│  4. 主题表示 (c-TF-IDF / KeyBERT / MMR / LLM)           │
│     为每个簇生成描述性关键词或标签                       │
└──────────────────────────────────────────────────────────┘

主题表示方法对比:

| 方法               | 特点                           |
|--------------------|--------------------------------|
| c-TF-IDF (默认)    | 基于词频的统计方法             |
| KeyBERTInspired    | 使用嵌入相似度选择关键词       |
| MMR                | 增加关键词多样性               |
| Flan-T5            | LLM 生成人类可读的主题标签     |
| OpenAI             | 更高质量的主题标签 (需要API)   |

BERTopic 优势:
- 模块化设计，可替换各组件
- 支持多种表示模型
- 丰富的可视化功能
- 可进行主题搜索和层次分析
""")


def main():
    """主函数"""
    # Part 1: 加载数据
    abstracts, titles = load_data()
    
    # 为了演示，使用部分数据 (完整数据 ~45k 条)
    sample_size = min(5000, len(abstracts))
    abstracts = abstracts[:sample_size]
    titles = titles[:sample_size]
    print(f"\n使用样本数据: {sample_size} 条")
    
    # Part 2.1: 生成嵌入
    embeddings, embedding_model, device = create_embeddings(abstracts)
    
    # Part 2.2: 降维
    reduced_embeddings, embeddings_2d = reduce_dimensions(embeddings)
    
    # Part 2.3: 聚类
    clusters = cluster_documents(reduced_embeddings)
    
    # 检查簇内容
    inspect_clusters(abstracts, clusters, cluster_id=0, n_docs=2)
    
    # Part 3: BERTopic 主题建模
    topic_model, topics = bertopic_modeling(abstracts, embeddings, embedding_model)
    
    # 保存原始主题表示
    original_topics = deepcopy(topic_model.topic_representations_)
    
    # 探索主题
    explore_topics(topic_model, abstracts, titles)
    
    # Part 4: 主题表示模型
    representation_models_demo(topic_model, abstracts, original_topics)
    
    # Part 5: LLM 生成主题标签 (可选，需要更多时间)
    try:
        llm_representation_demo(topic_model, abstracts, original_topics, device)
    except Exception as e:
        print(f"\nLLM 主题生成跳过: {e}")
    
    # Part 6: 可视化
    visualize_topics(topic_model, abstracts, titles, embeddings_2d)
    
    # 总结
    print_summary()


if __name__ == "__main__":
    main()
