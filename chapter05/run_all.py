"""
Chapter 5 - 运行全部
便捷脚本，用于运行所有部分或指定部分

使用方法:
    python run_all.py              # 运行全部
    python run_all.py --parts 1,2  # 只运行指定部分
    python run_all.py --summary    # 只显示总结
"""

import argparse
from copy import deepcopy

from common import load_data, get_device, topic_differences


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
| DeepSeek           | 更高质量的主题标签             |

BERTopic 优势:
- 模块化设计，可替换各组件
- 支持多种表示模型
- 丰富的可视化功能
- 可进行主题搜索和层次分析
""")


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 5 - 文本聚类与主题建模",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
各部分说明:
  Part 1: 加载数据
  Part 2: 文本聚类 (嵌入 -> 降维 -> 聚类)
  Part 3: BERTopic 主题建模
  Part 4: 主题表示模型 (KeyBERT, MMR)
  Part 5: LLM 主题标签 (Flan-T5, DeepSeek)
  Part 6: 可视化

示例:
  python run_all.py              # 运行 Part 1-4, 6
  python run_all.py --parts 1,2  # 只运行 Part 1 和 2
  python run_all.py --parts all  # 运行全部包括 LLM
  python run_all.py --summary    # 只显示总结
        """
    )
    parser.add_argument("--parts", "-p", type=str, default="1,2,3,4,6",
                        help="运行哪些部分 (默认: 1,2,3,4,6)")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="只显示总结")
    parser.add_argument("--sample-size", "-n", type=int, default=5000,
                        help="样本数量 (默认: 5000)")
    
    args = parser.parse_args()
    
    if args.summary:
        print_summary()
        return
    
    # 解析要运行的部分
    if args.parts == "all":
        parts = [1, 2, 3, 4, 5, 6]
    else:
        parts = [int(p.strip()) for p in args.parts.split(",")]
    
    print(f"将运行部分: {parts}")
    print(f"样本数量: {args.sample_size}")
    
    # Part 1: 加载数据
    abstracts, titles = load_data(sample_size=args.sample_size)
    
    if parts == [1]:
        return
    
    # Part 2: 聚类
    if 2 in parts:
        from part2_clustering import create_embeddings, reduce_dimensions, cluster_documents, inspect_clusters
        embeddings, embedding_model = create_embeddings(abstracts)
        reduced_embeddings, embeddings_2d = reduce_dimensions(embeddings)
        clusters = cluster_documents(reduced_embeddings)
        inspect_clusters(abstracts, clusters, cluster_id=0, n_docs=2)
    
    # Part 3: BERTopic
    topic_model = None
    original_topics = None
    if 3 in parts or 4 in parts:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        
        device = get_device()
        if 'embedding_model' not in dir():
            embedding_model = SentenceTransformer('thenlper/gte-small', device=device)
            embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
        
        print("\n训练 BERTopic 模型...")
        topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
        topic_model.fit_transform(abstracts, embeddings)
        original_topics = deepcopy(topic_model.topic_representations_)
        
        topic_info = topic_model.get_topic_info()
        print(f"\n发现 {len(topic_info)} 个主题")
        print(topic_info.head(10)[['Topic', 'Count', 'Name']])
    
    # Part 4: 主题表示
    if 4 in parts and topic_model:
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
        
        print("\n使用 KeyBERTInspired...")
        topic_model.update_topics(abstracts, representation_model=KeyBERTInspired())
        print(topic_differences(topic_model, original_topics).to_string(index=False))
        
        print("\n使用 MMR...")
        topic_model.update_topics(abstracts, representation_model=MaximalMarginalRelevance(diversity=0.5))
        print(topic_differences(topic_model, original_topics).to_string(index=False))
    
    # Part 5: LLM (可选)
    if 5 in parts:
        try:
            from part5_llm_topics import llm_representation_flant5
            if topic_model is None:
                from part3_bertopic import create_topic_model
                topic_model = create_topic_model(abstracts)
                original_topics = deepcopy(topic_model.topic_representations_)
            llm_representation_flant5(topic_model, abstracts, original_topics)
        except Exception as e:
            print(f"\nLLM 主题生成跳过: {e}")
    
    # Part 6: 可视化
    if 6 in parts:
        from part6_visualization import (visualize_documents, visualize_barchart, 
                                         visualize_heatmap, visualize_hierarchy)
        if topic_model and 'embeddings_2d' in dir():
            try:
                visualize_documents(topic_model, titles, embeddings_2d)
                visualize_barchart(topic_model)
                visualize_heatmap(topic_model)
                visualize_hierarchy(topic_model)
            except Exception as e:
                print(f"可视化失败: {e}")
    
    # 总结
    print_summary()


if __name__ == "__main__":
    main()
