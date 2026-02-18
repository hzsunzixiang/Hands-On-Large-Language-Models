"""
9.3 SBERT-CLIP 简化接口
========================

本节内容:
- Sentence-Transformers 库的 CLIP 封装
- 统一的编码接口
- 批量处理和相似度计算
- 实用工具函数

Sentence-Transformers 提供了更简洁的 API 来使用 CLIP，
让多模态嵌入的使用变得更加便捷。
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt


def get_device():
    """自动检测最佳可用设备"""
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


# 示例数据
IMAGE_URLS = {
    "puppy": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png",
    "beach": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/beach.png",
    "car": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png",
}


def load_image_from_url(url):
    """从 URL 加载图片"""
    return Image.open(urlopen(url)).convert("RGB")


def sbert_clip_overview():
    """SBERT-CLIP 概览"""
    print("=" * 60)
    print("SBERT-CLIP 概览")
    print("=" * 60)
    
    overview = """
Sentence-Transformers CLIP 封装的优势:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 统一接口:
   ✓ model.encode() - 统一编码图像和文本
   ✓ 与文本嵌入模型 API 一致
   ✓ 无需分别处理图像和文本

2. 便捷工具:
   ✓ util.cos_sim() - 余弦相似度计算
   ✓ util.semantic_search() - 语义搜索
   ✓ util.paraphrase_mining() - 释义挖掘

3. 批量处理:
   ✓ 自动批处理优化
   ✓ 内存管理
   ✓ GPU 加速支持

4. 多种模型:
   ✓ clip-ViT-B-32 (标准版)
   ✓ clip-ViT-L-14 (大模型)
   ✓ multilingual-clip (多语言)

API 对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原始 Transformers API:
```python
# 需要分别处理
tokenizer = CLIPTokenizer.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

text_inputs = tokenizer(texts, return_tensors="pt")
image_inputs = processor(images=images, return_tensors="pt")

text_embeds = model.get_text_features(**text_inputs)
image_embeds = model.get_image_features(**image_inputs)
```

SBERT-CLIP API:
```python
# 统一处理
model = SentenceTransformer('clip-ViT-B-32')

text_embeds = model.encode(texts)
image_embeds = model.encode(images)
```

更简洁，更易用！
"""
    print(overview)


def sbert_clip_basic_demo(device=None):
    """
    SBERT-CLIP 基础演示
    展示统一的编码接口
    """
    from sentence_transformers import SentenceTransformer, util
    
    print("\n" + "=" * 60)
    print("9.3 SBERT-CLIP 基础演示")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 1. 加载模型
    print("\n[步骤 1] 加载 SBERT-CLIP 模型...")
    model_name = 'clip-ViT-B-32'
    model = SentenceTransformer(model_name, device=device)
    
    print(f"✓ 模型: {model_name}")
    print(f"✓ 设备: {device}")
    print(f"✓ 嵌入维度: {model.get_sentence_embedding_dimension()}")
    
    # 2. 准备数据
    print("\n[步骤 2] 准备测试数据...")
    
    # 加载图像
    images = []
    image_names = []
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            images.append(img)
            image_names.append(name)
            print(f"✓ 图像: {name}")
        except Exception as e:
            print(f"✗ 跳过: {name} - {e}")
    
    # 准备文本
    texts = [
        "a puppy playing in the snow",
        "a sandy beach with ocean waves", 
        "a sports car on the road"
    ]
    
    print(f"✓ 文本数量: {len(texts)}")
    
    # 3. 统一编码
    print("\n[步骤 3] 统一编码...")
    
    # 编码图像 (自动批处理)
    print("  编码图像...")
    image_embeddings = model.encode(images, convert_to_tensor=True)
    print(f"✓ 图像嵌入形状: {image_embeddings.shape}")
    
    # 编码文本
    print("  编码文本...")
    text_embeddings = model.encode(texts, convert_to_tensor=True)
    print(f"✓ 文本嵌入形状: {text_embeddings.shape}")
    
    # 4. 相似度计算
    print("\n[步骤 4] 计算相似度...")
    
    # 使用 util.cos_sim() 计算余弦相似度
    similarity_matrix = util.cos_sim(image_embeddings, text_embeddings)
    
    print(f"✓ 相似度矩阵形状: {similarity_matrix.shape}")
    print(f"✓ 数据类型: {type(similarity_matrix)}")
    
    # 5. 结果展示
    print("\n[步骤 5] 结果展示...")
    
    sim_np = similarity_matrix.cpu().numpy()
    
    print("\n📊 相似度矩阵:")
    print("-" * 70)
    print(f"{'图像\\文本':>10}", end="")
    for text in texts:
        short_text = text[:20] + "..." if len(text) > 20 else text
        print(f"{short_text:>25}", end="")
    print()
    
    for i, img_name in enumerate(image_names):
        print(f"{img_name:>10}", end="")
        for j in range(len(texts)):
            value = sim_np[i, j]
            if i == j:  # 对角线高亮
                print(f"    [{value:>6.3f}]    ", end="")
            else:
                print(f"     {value:>6.3f}     ", end="")
        print()
    
    return model, similarity_matrix


def advanced_similarity_operations(model, device):
    """高级相似度操作"""
    print("\n" + "=" * 60)
    print("高级相似度操作")
    print("=" * 60)
    
    from sentence_transformers import util
    
    # 1. 语义搜索演示
    print("\n[操作 1] 语义搜索...")
    
    # 准备图像库
    images = []
    image_descriptions = []
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            images.append(img)
            image_descriptions.append(f"{name} image")
        except:
            pass
    
    # 编码图像库
    image_embeddings = model.encode(images)
    
    # 搜索查询
    queries = [
        "cute animal",
        "nature landscape", 
        "transportation vehicle",
        "winter scene"
    ]
    
    print(f"✓ 图像库: {len(images)} 张")
    print(f"✓ 查询: {len(queries)} 个")
    
    for query in queries:
        # 编码查询
        query_embedding = model.encode([query])
        
        # 语义搜索
        search_results = util.semantic_search(
            query_embedding, 
            image_embeddings, 
            top_k=len(images)
        )[0]
        
        print(f"\n🔍 查询: '{query}'")
        for i, result in enumerate(search_results):
            idx = result['corpus_id']
            score = result['score']
            desc = image_descriptions[idx]
            print(f"  {i+1}. {desc}: {score:.4f}")
    
    # 2. 批量相似度计算
    print(f"\n[操作 2] 批量相似度计算...")
    
    # 创建更多测试文本
    extended_texts = [
        "a dog playing outside",
        "puppy in snow",
        "ocean waves",
        "beach vacation",
        "red sports car",
        "fast vehicle",
        "mountain landscape",
        "city street"
    ]
    
    # 批量编码
    text_embeddings = model.encode(extended_texts)
    
    # 计算图像与所有文本的相似度
    all_similarities = util.cos_sim(image_embeddings, text_embeddings)
    
    print(f"✓ 扩展相似度矩阵: {all_similarities.shape}")
    
    # 找到每张图像的最佳文本匹配
    for i, img_desc in enumerate(image_descriptions):
        similarities = all_similarities[i].cpu().numpy()
        best_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"\n🏆 {img_desc} 的最佳匹配:")
        for j, idx in enumerate(best_indices):
            text = extended_texts[idx]
            score = similarities[idx]
            print(f"  {j+1}. '{text}': {score:.4f}")
    
    return all_similarities


def clustering_and_visualization(model, device):
    """聚类和可视化"""
    print("\n" + "=" * 60)
    print("聚类和可视化")
    print("=" * 60)
    
    from sentence_transformers import util
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # 1. 准备混合数据
    print("\n[步骤 1] 准备混合数据...")
    
    # 图像数据
    images = []
    image_labels = []
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            images.append(img)
            image_labels.append(f"img_{name}")
        except:
            pass
    
    # 文本数据
    texts = [
        "cute puppy playing",
        "dog in winter",
        "beautiful beach scene",
        "ocean and sand",
        "sports car racing",
        "red vehicle"
    ]
    text_labels = [f"text_{i}" for i in range(len(texts))]
    
    # 2. 编码所有数据
    print("\n[步骤 2] 编码混合数据...")
    
    image_embeddings = model.encode(images)
    text_embeddings = model.encode(texts)
    
    # 合并嵌入
    all_embeddings = np.vstack([image_embeddings, text_embeddings])
    all_labels = image_labels + text_labels
    
    print(f"✓ 总嵌入数量: {all_embeddings.shape[0]}")
    print(f"✓ 嵌入维度: {all_embeddings.shape[1]}")
    
    # 3. 降维可视化
    print("\n[步骤 3] PCA 降维...")
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    print(f"✓ 降维后形状: {embeddings_2d.shape}")
    print(f"✓ 解释方差比: {pca.explained_variance_ratio_}")
    
    # 4. 聚类分析
    print("\n[步骤 4] K-means 聚类...")
    
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    print(f"✓ 聚类数量: {n_clusters}")
    
    # 5. 可视化
    print("\n[步骤 5] 生成可视化...")
    
    plt.figure(figsize=(12, 8))
    
    # 绘制聚类结果
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        plt.scatter(
            embeddings_2d[cluster_mask, 0],
            embeddings_2d[cluster_mask, 1],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=100,
            alpha=0.7,
            label=f'Cluster {i}'
        )
    
    # 添加标签
    for i, (x, y) in enumerate(embeddings_2d):
        label = all_labels[i]
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    plt.title('CLIP 嵌入空间可视化 (PCA降维)', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('/Users/ericksun/workspace/deeplearning/Hands-On-Large-Language-Models/chapter09/clip_embedding_visualization.png',
                dpi=300, bbox_inches='tight')
    print("✓ 可视化已保存: clip_embedding_visualization.png")
    
    try:
        plt.show()
    except:
        print("✓ 可视化生成完成")
    
    plt.close()
    
    # 6. 聚类分析
    print("\n[步骤 6] 聚类结果分析...")
    
    for i in range(n_clusters):
        cluster_items = [all_labels[j] for j in range(len(all_labels)) if cluster_labels[j] == i]
        print(f"\n🔍 Cluster {i}: {cluster_items}")


def practical_applications(model, device):
    """实际应用示例"""
    print("\n" + "=" * 60)
    print("实际应用示例")
    print("=" * 60)
    
    from sentence_transformers import util
    
    # 1. 图像标注生成
    print("\n[应用 1] 自动图像标注...")
    
    # 预定义标签库
    label_categories = {
        "animals": ["dog", "cat", "puppy", "kitten", "pet", "animal"],
        "nature": ["beach", "ocean", "sea", "sand", "water", "landscape"],
        "vehicles": ["car", "automobile", "vehicle", "transportation", "sports car"],
        "weather": ["snow", "winter", "cold", "sunny", "cloudy"],
        "activities": ["playing", "running", "sleeping", "driving", "swimming"]
    }
    
    # 展平所有标签
    all_labels = []
    for category, labels in label_categories.items():
        all_labels.extend(labels)
    
    # 编码标签
    label_embeddings = model.encode(all_labels)
    
    # 为每张图像生成标注
    for name, url in IMAGE_URLS.items():
        try:
            image = load_image_from_url(url)
            image_embedding = model.encode([image])
            
            # 计算与所有标签的相似度
            similarities = util.cos_sim(image_embedding, label_embeddings)[0]
            
            # 获取top-5标签
            top_indices = similarities.argsort(descending=True)[:5]
            
            print(f"\n🏷️  {name} 图像的自动标注:")
            for i, idx in enumerate(top_indices):
                label = all_labels[idx]
                score = similarities[idx].item()
                print(f"  {i+1}. {label}: {score:.4f}")
                
        except Exception as e:
            print(f"✗ 跳过 {name}: {e}")
    
    # 2. 内容推荐系统
    print(f"\n[应用 2] 基于图像的内容推荐...")
    
    # 模拟用户偏好
    user_preferences = [
        "I love cute animals",
        "I enjoy beach vacations",
        "I'm interested in fast cars"
    ]
    
    # 编码用户偏好
    preference_embeddings = model.encode(user_preferences)
    
    # 图像库
    images = []
    image_names = []
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            images.append(img)
            image_names.append(name)
        except:
            pass
    
    image_embeddings = model.encode(images)
    
    # 为每个用户偏好推荐图像
    for i, preference in enumerate(user_preferences):
        similarities = util.cos_sim(preference_embeddings[i:i+1], image_embeddings)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx].item()
        
        print(f"\n👤 用户偏好: '{preference}'")
        print(f"   推荐图像: {image_names[best_idx]} (相似度: {best_score:.4f})")
        
        # 显示所有匹配度
        sorted_indices = similarities.argsort(descending=True)
        print(f"   完整排序: ", end="")
        for j, idx in enumerate(sorted_indices):
            print(f"{image_names[idx]}({similarities[idx]:.3f})", end="")
            if j < len(sorted_indices) - 1:
                print(", ", end="")
        print()
    
    # 3. 多模态搜索引擎
    print(f"\n[应用 3] 多模态搜索引擎...")
    
    # 构建混合索引 (图像 + 文本)
    search_corpus = []
    corpus_types = []
    corpus_items = []
    
    # 添加图像
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            search_corpus.append(img)
            corpus_types.append("image")
            corpus_items.append(f"image_{name}")
        except:
            pass
    
    # 添加文本描述
    text_descriptions = [
        "A cute puppy playing in the snow during winter",
        "Beautiful sandy beach with clear blue ocean waves",
        "Red sports car driving on an empty road"
    ]
    
    search_corpus.extend(text_descriptions)
    corpus_types.extend(["text"] * len(text_descriptions))
    corpus_items.extend([f"text_{i}" for i in range(len(text_descriptions))])
    
    # 编码整个语料库
    corpus_embeddings = model.encode(search_corpus)
    
    # 搜索查询
    search_queries = [
        "winter animals",
        "vacation destination",
        "fast transportation"
    ]
    
    print(f"✓ 搜索语料库: {len(search_corpus)} 项")
    print(f"  - 图像: {corpus_types.count('image')} 个")
    print(f"  - 文本: {corpus_types.count('text')} 个")
    
    for query in search_queries:
        query_embedding = model.encode([query])
        
        # 搜索最相关的内容
        search_results = util.semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=3
        )[0]
        
        print(f"\n🔍 搜索: '{query}'")
        for j, result in enumerate(search_results):
            idx = result['corpus_id']
            score = result['score']
            item_type = corpus_types[idx]
            item_name = corpus_items[idx]
            
            print(f"  {j+1}. [{item_type}] {item_name}: {score:.4f}")


def main():
    """主函数"""
    print("🚀 开始 SBERT-CLIP 学习...")
    
    # 概览
    sbert_clip_overview()
    
    # 设备检测
    device = get_device()
    
    try:
        # 基础演示
        model, sim_matrix = sbert_clip_basic_demo(device)
        
        # 高级操作
        advanced_similarity_operations(model, device)
        
        # 聚类和可视化
        clustering_and_visualization(model, device)
        
        # 实际应用
        practical_applications(model, device)
        
        print("\n" + "=" * 60)
        print("✅ 9.3 SBERT-CLIP 学习完成!")
        print("=" * 60)
        print("\n🎯 关键收获:")
        print("  • 统一的多模态编码接口")
        print("  • 便捷的相似度计算工具")
        print("  • 丰富的实际应用场景")
        print("  • 高效的批量处理能力")
        print("\n下一步: 运行 9.4_blip2_vision_qa.py 学习视觉问答")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查 sentence-transformers 库是否正确安装")
        print("安装命令: pip install sentence-transformers")


if __name__ == "__main__":
    main()