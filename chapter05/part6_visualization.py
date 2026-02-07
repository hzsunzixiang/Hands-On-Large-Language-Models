"""
Chapter 5 - Part 6: 可视化
生成主题模型的各种可视化
"""

from copy import deepcopy
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from common import load_data, get_device


def create_topic_model_with_embeddings(abstracts):
    """创建模型并返回嵌入"""
    device = get_device()
    
    print("\n加载嵌入模型...")
    embedding_model = SentenceTransformer('thenlper/gte-small', device=device)
    
    print("生成嵌入...")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device=device)
    
    # 降维到 2D 用于可视化
    print("降维到 2D...")
    umap_2d = UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    embeddings_2d = umap_2d.fit_transform(embeddings)
    
    print("\n训练 BERTopic 模型...")
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
    topic_model.fit_transform(abstracts, embeddings)
    
    return topic_model, embeddings_2d


def visualize_documents(topic_model, titles, embeddings_2d):
    """文档主题分布可视化"""
    print("\n" + "-" * 40)
    print("生成文档可视化...")
    print("-" * 40)
    
    fig = topic_model.visualize_documents(
        titles,
        reduced_embeddings=embeddings_2d,
        width=1200,
        hide_annotations=True
    )
    fig.write_html("topic_documents.html")
    print("  -> 已保存: topic_documents.html")
    return fig


def visualize_barchart(topic_model):
    """主题关键词柱状图"""
    print("\n" + "-" * 40)
    print("生成主题关键词柱状图...")
    print("-" * 40)
    
    fig = topic_model.visualize_barchart(top_n_topics=10)
    fig.write_html("topic_barchart.html")
    print("  -> 已保存: topic_barchart.html")
    return fig


def visualize_heatmap(topic_model):
    """主题相似度热力图"""
    print("\n" + "-" * 40)
    print("生成主题相似度热力图...")
    print("-" * 40)
    
    fig = topic_model.visualize_heatmap(n_clusters=30)
    fig.write_html("topic_heatmap.html")
    print("  -> 已保存: topic_heatmap.html")
    return fig


def visualize_hierarchy(topic_model):
    """层次聚类树"""
    print("\n" + "-" * 40)
    print("生成层次聚类树...")
    print("-" * 40)
    
    fig = topic_model.visualize_hierarchy()
    fig.write_html("topic_hierarchy.html")
    print("  -> 已保存: topic_hierarchy.html")
    return fig


def visualize_topics_overview(topic_model):
    """主题概览图"""
    print("\n" + "-" * 40)
    print("生成主题概览图...")
    print("-" * 40)
    
    fig = topic_model.visualize_topics()
    fig.write_html("topic_overview.html")
    print("  -> 已保存: topic_overview.html")
    return fig


def main():
    print("=" * 60)
    print("Part 6: 可视化")
    print("=" * 60)
    
    # 加载数据
    abstracts, titles = load_data(sample_size=5000)
    
    # 创建模型
    topic_model, embeddings_2d = create_topic_model_with_embeddings(abstracts)
    
    # 生成各种可视化
    try:
        visualize_documents(topic_model, titles, embeddings_2d)
    except Exception as e:
        print(f"  文档可视化失败: {e}")
    
    try:
        visualize_barchart(topic_model)
    except Exception as e:
        print(f"  柱状图失败: {e}")
    
    try:
        visualize_heatmap(topic_model)
    except Exception as e:
        print(f"  热力图失败: {e}")
    
    try:
        visualize_hierarchy(topic_model)
    except Exception as e:
        print(f"  层次图失败: {e}")
    
    try:
        visualize_topics_overview(topic_model)
    except Exception as e:
        print(f"  概览图失败: {e}")
    
    print("\n" + "=" * 60)
    print("可视化文件已生成，用浏览器打开 .html 文件查看")
    print("=" * 60)


if __name__ == "__main__":
    main()
