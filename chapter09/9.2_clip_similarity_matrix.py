import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
9.2 CLIP 相似度矩阵分析
=========================

本节内容:
- 多图像与多文本的相似度计算
- 相似度矩阵可视化
- 零样本分类原理
- 跨模态检索应用

通过相似度矩阵，我们可以理解 CLIP 如何在图文之间建立对应关系，
这是零样本分类和跨模态检索的基础。
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from urllib.request import urlopen

# 配置 matplotlib 中文字体 (macOS)
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


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


# 示例图片和描述
IMAGE_URLS = {
    "puppy": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png",
    "cat": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/cat.png", 
    "car": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png",
}

CAPTIONS = [
    "a puppy playing in the snow",
    "a cat sitting comfortably", 
    "a sports car on the road"
]


def load_image_from_url(url):
    """从 URL 加载图片"""
    return Image.open(urlopen(url)).convert("RGB")


def similarity_matrix_concept():
    """相似度矩阵概念解释"""
    print("=" * 60)
    print("相似度矩阵概念")
    print("=" * 60)
    
    concept = """
相似度矩阵是什么?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

相似度矩阵是一个 M×N 的矩阵，其中:
• M = 图像数量
• N = 文本描述数量  
• 矩阵[i,j] = 第i张图像与第j个文本的相似度

示例 (3×3 矩阵):
                文本1      文本2      文本3
                puppy      cat       car
图像1 puppy    [0.85]     0.12      0.08
图像2 cat       0.15     [0.92]     0.11  
图像3 car       0.09      0.13     [0.88]

理想情况: 对角线值最高 (正确匹配)

应用场景:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 零样本图像分类:
   • 将类别名作为文本描述
   • 计算图像与所有类别的相似度
   • 选择相似度最高的类别

2. 图像检索:
   • 用文本查询描述想要的图像
   • 计算查询与所有图像的相似度
   • 返回相似度最高的图像

3. 文本检索:
   • 用图像查询相关的文本描述
   • 计算图像与所有文本的相似度
   • 返回相似度最高的文本

4. 多模态推荐:
   • 基于用户的图像偏好推荐相关文本
   • 基于用户的文本偏好推荐相关图像
"""
    print(concept)


def clip_similarity_matrix_demo(device=None):
    """
    CLIP 相似度矩阵演示
    计算多张图片和多个描述之间的相似度矩阵
    """
    from transformers import CLIPProcessor, CLIPModel
    
    print("\n" + "=" * 60)
    print("9.2 CLIP 相似度矩阵分析")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 1. 加载模型
    print("\n[步骤 1] 加载 CLIP 模型...")
    model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    print(f"✓ 模型加载完成: {model_id}")
    
    # 2. 加载图像数据
    print("\n[步骤 2] 加载图像数据...")
    images = []
    image_names = []
    
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            images.append(img)
            image_names.append(name)
            print(f"✓ 加载图像: {name} ({img.size})")
        except Exception as e:
            print(f"✗ 跳过图像 {name}: {e}")
    
    print(f"✓ 成功加载 {len(images)} 张图像")
    
    # 3. 准备文本描述
    print("\n[步骤 3] 准备文本描述...")
    captions = CAPTIONS[:len(images)]  # 确保数量匹配
    
    for i, caption in enumerate(captions):
        print(f"✓ 文本 {i+1}: '{caption}'")
    
    # 4. 批量计算嵌入
    print("\n[步骤 4] 批量计算嵌入...")
    
    # 使用 CLIP 的批量处理功能
    inputs = clip_processor(
        text=captions,
        images=images, 
        return_tensors="pt",
        padding=True
    ).to(device)
    
    print(f"✓ 文本输入形状: {inputs['input_ids'].shape}")
    print(f"✓ 图像输入形状: {inputs['pixel_values'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds  # [num_images, 512]
        text_embeds = outputs.text_embeds    # [num_texts, 512]
    
    print(f"✓ 图像嵌入形状: {image_embeds.shape}")
    print(f"✓ 文本嵌入形状: {text_embeds.shape}")
    
    # 5. 归一化嵌入
    print("\n[步骤 5] 归一化嵌入...")
    image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    print("✓ 嵌入已归一化 (L2范数 = 1)")
    
    # 6. 计算相似度矩阵
    print("\n[步骤 6] 计算相似度矩阵...")
    similarity_matrix = (image_embeds_norm @ text_embeds_norm.T).cpu().numpy()
    
    print(f"✓ 相似度矩阵形状: {similarity_matrix.shape}")
    print(f"✓ 相似度范围: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    return similarity_matrix, image_names, captions


def visualize_similarity_matrix(sim_matrix, image_names, captions):
    """可视化相似度矩阵"""
    print("\n[步骤 7] 可视化相似度矩阵...")
    
    # 1. 文本表格显示
    print("\n📊 相似度矩阵 (图像 × 文本):")
    print("-" * 80)
    
    # 表头
    print(f"{'图像\\文本':>12}", end="")
    for i, caption in enumerate(captions):
        short_caption = caption[:20] + "..." if len(caption) > 20 else caption
        print(f"{short_caption:>25}", end="")
    print()
    
    # 数据行
    for i, img_name in enumerate(image_names):
        print(f"{img_name:>12}", end="")
        for j in range(len(captions)):
            value = sim_matrix[i, j]
            # 高亮对角线元素
            if i == j:
                print(f"    [{value:>6.3f}]    ", end="")
            else:
                print(f"     {value:>6.3f}     ", end="")
        print()
    
    # 2. 热力图可视化
    plt.figure(figsize=(10, 8))
    
    # 创建热力图
    ax = sns.heatmap(
        sim_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        xticklabels=[cap[:30] + "..." if len(cap) > 30 else cap for cap in captions],
        yticklabels=image_names,
        cbar_kws={'label': '余弦相似度'},
        square=True
    )
    
    plt.title('CLIP 图文相似度矩阵', fontsize=16, pad=20)
    plt.xlabel('文本描述', fontsize=12)
    plt.ylabel('图像', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('/Users/ericksun/workspace/deeplearning/Hands-On-Large-Language-Models/chapter09/similarity_matrix.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 热力图已保存: similarity_matrix.png")
    
    # 显示图片 (如果在支持的环境中)
    try:
        plt.show()
    except:
        print("✓ 热力图生成完成 (无法显示，但已保存)")
    
    plt.close()


def analyze_similarity_results(sim_matrix, image_names, captions):
    """分析相似度结果"""
    print("\n[步骤 8] 分析相似度结果...")
    
    # 1. 对角线分析 (正确匹配)
    diagonal_values = np.diag(sim_matrix)
    print(f"\n🎯 对角线相似度 (正确匹配):")
    for i, (img_name, caption, score) in enumerate(zip(image_names, captions, diagonal_values)):
        print(f"  {img_name} ↔ '{caption}': {score:.4f}")
    
    print(f"✓ 平均对角线相似度: {diagonal_values.mean():.4f}")
    
    # 2. 最佳匹配分析
    print(f"\n🏆 每张图像的最佳文本匹配:")
    for i, img_name in enumerate(image_names):
        best_text_idx = np.argmax(sim_matrix[i])
        best_score = sim_matrix[i, best_text_idx]
        is_correct = (best_text_idx == i)
        
        status = "✓ 正确" if is_correct else "✗ 错误"
        print(f"  {img_name}: '{captions[best_text_idx]}' ({best_score:.4f}) {status}")
    
    print(f"\n🏆 每个文本的最佳图像匹配:")
    for j, caption in enumerate(captions):
        best_img_idx = np.argmax(sim_matrix[:, j])
        best_score = sim_matrix[best_img_idx, j]
        is_correct = (best_img_idx == j)
        
        status = "✓ 正确" if is_correct else "✗ 错误"
        print(f"  '{caption}': {image_names[best_img_idx]} ({best_score:.4f}) {status}")
    
    # 3. 准确率计算
    img_to_text_correct = sum(1 for i in range(len(image_names)) 
                             if np.argmax(sim_matrix[i]) == i)
    text_to_img_correct = sum(1 for j in range(len(captions)) 
                             if np.argmax(sim_matrix[:, j]) == j)
    
    img_to_text_acc = img_to_text_correct / len(image_names)
    text_to_img_acc = text_to_img_correct / len(captions)
    
    print(f"\n📈 检索准确率:")
    print(f"  图像→文本: {img_to_text_correct}/{len(image_names)} = {img_to_text_acc:.1%}")
    print(f"  文本→图像: {text_to_img_correct}/{len(captions)} = {text_to_img_acc:.1%}")
    print(f"  平均准确率: {(img_to_text_acc + text_to_img_acc) / 2:.1%}")
    
    # 4. 相似度分布分析
    print(f"\n📊 相似度分布统计:")
    print(f"  最高相似度: {sim_matrix.max():.4f}")
    print(f"  最低相似度: {sim_matrix.min():.4f}")
    print(f"  平均相似度: {sim_matrix.mean():.4f}")
    print(f"  标准差: {sim_matrix.std():.4f}")
    
    return {
        'diagonal_scores': diagonal_values,
        'img_to_text_acc': img_to_text_acc,
        'text_to_img_acc': text_to_img_acc,
        'avg_accuracy': (img_to_text_acc + text_to_img_acc) / 2
    }


def zero_shot_classification_demo(device):
    """零样本分类演示"""
    print("\n" + "=" * 60)
    print("零样本分类演示")
    print("=" * 60)
    
    from transformers import CLIPProcessor, CLIPModel
    
    # 加载模型
    model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    
    # 定义分类类别
    class_names = [
        "dog", "cat", "car", "airplane", "ship",
        "truck", "bird", "horse", "bicycle", "motorcycle"
    ]
    
    # 创建分类模板
    templates = [f"a photo of a {class_name}" for class_name in class_names]
    
    print(f"✓ 分类类别: {len(class_names)} 个")
    print(f"✓ 模板示例: '{templates[0]}'")
    
    # 测试图像
    test_image = load_image_from_url(IMAGE_URLS["car"])
    
    # 计算嵌入
    inputs = clip_processor(
        text=templates,
        images=[test_image],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        image_embed = outputs.image_embeds[0:1]  # 只有一张图
        text_embeds = outputs.text_embeds
    
    # 归一化
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarities = (image_embed @ text_embeds.T).cpu().numpy()[0]
    
    # 排序结果
    sorted_indices = np.argsort(similarities)[::-1]
    
    print(f"\n🔍 零样本分类结果 (测试图像: car):")
    print("-" * 40)
    for i, idx in enumerate(sorted_indices[:5]):
        class_name = class_names[idx]
        score = similarities[idx]
        confidence = (score + 1) / 2 * 100  # 转换为百分比
        
        if i == 0:
            print(f"🏆 {class_name:>12}: {score:.4f} ({confidence:.1f}%)")
        else:
            print(f"   {class_name:>12}: {score:.4f} ({confidence:.1f}%)")
    
    predicted_class = class_names[sorted_indices[0]]
    print(f"\n✓ 预测类别: {predicted_class}")
    print(f"✓ 置信度: {similarities[sorted_indices[0]]:.4f}")


def cross_modal_retrieval_demo(device):
    """跨模态检索演示"""
    print("\n" + "=" * 60)
    print("跨模态检索演示")
    print("=" * 60)
    
    from transformers import CLIPProcessor, CLIPModel
    
    # 加载模型
    model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    
    # 准备图像库
    images = []
    image_names = []
    for name, url in IMAGE_URLS.items():
        try:
            images.append(load_image_from_url(url))
            image_names.append(name)
        except:
            pass
    
    # 查询文本
    query_texts = [
        "cute animal in winter",
        "a fluffy cat",
        "fast vehicle",
        "something blue"
    ]
    
    print(f"✓ 图像库: {len(images)} 张图像")
    print(f"✓ 查询: {len(query_texts)} 个文本")
    
    # 计算图像嵌入
    image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    print(f"\n🔍 文本检索图像结果:")
    print("-" * 50)
    
    for query in query_texts:
        # 计算查询嵌入
        text_inputs = clip_processor(text=[query], return_tensors="pt").to(device)
        with torch.no_grad():
            text_embed = model.get_text_features(**text_inputs)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarities = (text_embed @ image_embeds.T).cpu().numpy()[0]
        
        # 找到最佳匹配
        best_idx = np.argmax(similarities)
        best_image = image_names[best_idx]
        best_score = similarities[best_idx]
        
        print(f"查询: '{query}'")
        print(f"  → 最佳匹配: {best_image} (相似度: {best_score:.4f})")
        
        # 显示所有结果
        sorted_indices = np.argsort(similarities)[::-1]
        print(f"  → 排序结果: ", end="")
        for i, idx in enumerate(sorted_indices):
            print(f"{image_names[idx]}({similarities[idx]:.3f})", end="")
            if i < len(sorted_indices) - 1:
                print(", ", end="")
        print("\n")


def main():
    """主函数"""
    print("🚀 开始 CLIP 相似度矩阵学习...")
    
    # 概念解释
    similarity_matrix_concept()
    
    # 设备检测
    device = get_device()
    
    try:
        # 相似度矩阵演示
        sim_matrix, image_names, captions = clip_similarity_matrix_demo(device)
        
        # 可视化
        visualize_similarity_matrix(sim_matrix, image_names, captions)
        
        # 结果分析
        results = analyze_similarity_results(sim_matrix, image_names, captions)
        
        # 零样本分类演示
        zero_shot_classification_demo(device)
        
        # 跨模态检索演示
        cross_modal_retrieval_demo(device)
        
        print("\n" + "=" * 60)
        print("✅ 9.2 CLIP 相似度矩阵学习完成!")
        print("=" * 60)
        print(f"📊 关键指标:")
        print(f"  • 平均检索准确率: {results['avg_accuracy']:.1%}")
        print(f"  • 对角线平均相似度: {results['diagonal_scores'].mean():.4f}")
        print("\n下一步: 运行 9.3_sbert_clip.py 学习简化接口")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查网络连接和依赖安装")


if __name__ == "__main__":
    main()