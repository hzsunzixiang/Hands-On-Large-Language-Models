import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
9.1 CLIP 基础 - 图文嵌入对齐
=================================

本节内容:
- CLIP 模型架构理解
- 图像和文本嵌入生成
- 统一嵌入空间的概念
- 余弦相似度计算

CLIP (Contrastive Language-Image Pre-training) 是 OpenAI 开发的多模态模型，
通过对比学习让图像和文本共享同一个嵌入空间。
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from urllib.request import urlopen


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


# 示例图片 URL
IMAGE_URLS = {
    "puppy": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png",
    "cat": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/cat.png",
    "car": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png",
}


def load_image_from_url(url):
    """从 URL 加载图片"""
    return Image.open(urlopen(url)).convert("RGB")


def clip_architecture_overview():
    """CLIP 架构概览"""
    print("=" * 60)
    print("CLIP 架构概览")
    print("=" * 60)
    
    architecture = """
CLIP 双塔架构:
┌─────────────────┐         ┌─────────────────┐
│     图像        │         │     文本        │
│   "puppy.jpg"   │         │   "a puppy"     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Vision Encoder │         │  Text Encoder   │
│   (ViT-B/32)    │         │ (Transformer)   │
│                 │         │                 │
│ • 图像分块      │         │ • 文本分词      │
│ • Patch嵌入     │         │ • 位置编码      │
│ • Transformer   │         │ • 自注意力      │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   图像嵌入      │◄──相似度──►│   文本嵌入      │
│    [512维]      │   计算    │    [512维]      │
│                 │         │                 │
│ • L2 归一化     │         │ • L2 归一化     │
│ • 余弦相似度    │         │ • 余弦相似度    │
└─────────────────┘         └─────────────────┘

关键特点:
• 统一嵌入空间: 图像和文本映射到相同的512维空间
• 对比学习: 匹配的图文对相似度高，不匹配的相似度低
• 零样本能力: 无需额外训练即可进行图文匹配
• 预训练数据: 4亿个图文对 (WIT数据集)
"""
    print(architecture)


def clip_embeddings_demo(device=None):
    """
    CLIP 图文嵌入演示
    展示如何使用 CLIP 生成图像和文本的统一嵌入
    """
    from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
    
    print("\n" + "=" * 60)
    print("9.1 CLIP 基础 - 图文嵌入对齐")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 1. 模型加载
    print("\n[步骤 1] 加载 CLIP 模型...")
    model_id = "openai/clip-vit-base-patch32"
    
    # 分别加载各个组件
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    
    print(f"✓ 模型: {model_id}")
    print(f"✓ 文本嵌入维度: 512")
    print(f"✓ 图像嵌入维度: 512 (统一空间)")
    print(f"✓ 参数量: ~151M")
    
    # 2. 加载示例数据
    print("\n[步骤 2] 加载示例图像和文本...")
    image = load_image_from_url(IMAGE_URLS["puppy"])
    caption = "a puppy playing in the snow"
    
    print(f"✓ 图像: 雪地小狗")
    print(f"✓ 描述: '{caption}'")
    print(f"✓ 图像尺寸: {image.size}")
    
    # 3. 文本处理和嵌入
    print("\n[步骤 3] 生成文本嵌入...")
    text_inputs = clip_tokenizer(caption, return_tensors="pt").to(device)
    
    # 展示分词结果
    tokens = clip_tokenizer.convert_ids_to_tokens(text_inputs["input_ids"][0])
    print(f"✓ 分词结果: {tokens}")
    print(f"✓ Token数量: {len(tokens)}")
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)
    
    print(f"✓ 文本嵌入形状: {text_embedding.shape}")
    print(f"✓ 嵌入范围: [{text_embedding.min():.3f}, {text_embedding.max():.3f}]")
    
    # 4. 图像处理和嵌入
    print("\n[步骤 4] 生成图像嵌入...")
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    
    print(f"✓ 图像张量形状: {image_inputs['pixel_values'].shape}")
    print("  → [batch_size, channels, height, width] = [1, 3, 224, 224]")
    print(f"✓ 像素值范围: [{image_inputs['pixel_values'].min():.3f}, {image_inputs['pixel_values'].max():.3f}]")
    
    with torch.no_grad():
        image_embedding = model.get_image_features(**image_inputs)
    
    print(f"✓ 图像嵌入形状: {image_embedding.shape}")
    print(f"✓ 嵌入范围: [{image_embedding.min():.3f}, {image_embedding.max():.3f}]")
    
    # 5. 相似度计算
    print("\n[步骤 5] 计算图文相似度...")
    
    # L2 归一化 (CLIP 的标准做法)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    
    # 余弦相似度 (归一化后的点积)
    similarity = (text_embedding_norm @ image_embedding_norm.T).item()
    
    print(f"✓ 原始嵌入模长:")
    print(f"  - 文本: {text_embedding.norm():.3f}")
    print(f"  - 图像: {image_embedding.norm():.3f}")
    print(f"✓ 归一化后模长: 1.000 (标准化)")
    print(f"✓ 余弦相似度: {similarity:.4f}")
    
    # 6. 相似度解释
    print("\n[步骤 6] 相似度解释...")
    if similarity > 0.3:
        print(f"🎯 高相似度 ({similarity:.4f}) - 图文匹配良好!")
    elif similarity > 0.1:
        print(f"🔍 中等相似度 ({similarity:.4f}) - 图文有一定关联")
    else:
        print(f"❌ 低相似度 ({similarity:.4f}) - 图文不匹配")
    
    print("\n相似度范围说明:")
    print("• [0.8, 1.0]: 完美匹配")
    print("• [0.5, 0.8]: 强相关")
    print("• [0.2, 0.5]: 中等相关")
    print("• [0.0, 0.2]: 弱相关")
    print("• [-1.0, 0.0]: 负相关")
    
    return model, clip_processor, clip_tokenizer


def demonstrate_embedding_properties(model, clip_processor, clip_tokenizer, device):
    """演示嵌入的性质"""
    print("\n" + "=" * 60)
    print("嵌入空间性质演示")
    print("=" * 60)
    
    # 测试多个文本描述
    texts = [
        "a puppy playing in the snow",
        "a dog in winter",
        "a cat sleeping",
        "a car on the road",
        "snow and animals"
    ]
    
    # 加载图像
    image = load_image_from_url(IMAGE_URLS["puppy"])
    
    print("\n计算图像与多个文本的相似度...")
    
    # 生成图像嵌入
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**image_inputs)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    
    # 计算与每个文本的相似度
    similarities = []
    for text in texts:
        text_inputs = clip_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embedding = model.get_text_features(**text_inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        sim = (image_embedding @ text_embedding.T).item()
        similarities.append(sim)
        print(f"'{text}': {sim:.4f}")
    
    # 找到最佳匹配
    best_idx = np.argmax(similarities)
    print(f"\n🏆 最佳匹配: '{texts[best_idx]}' (相似度: {similarities[best_idx]:.4f})")
    
    return similarities


def technical_details():
    """CLIP 技术细节"""
    print("\n" + "=" * 60)
    print("CLIP 技术细节")
    print("=" * 60)
    
    details = """
1. 模型架构:
   • Vision Encoder: Vision Transformer (ViT-B/32)
     - 输入: 224×224 RGB图像
     - Patch大小: 32×32 (196个patches)
     - 层数: 12层 Transformer
     - 注意力头: 12个
     - 隐藏维度: 768
   
   • Text Encoder: Transformer
     - 最大序列长度: 77 tokens
     - 层数: 12层
     - 注意力头: 8个
     - 隐藏维度: 512
   
   • 投影层: 将两个编码器输出投影到512维空间

2. 训练过程:
   • 数据: 4亿个图文对 (从互联网收集)
   • 损失函数: 对比损失 (Contrastive Loss)
   • 批次大小: 32,768
   • 训练时间: 12天 (592个V100 GPU)
   
3. 对比学习原理:
   • 正样本: 匹配的图文对，相似度最大化
   • 负样本: 不匹配的图文对，相似度最小化
   • 温度参数: 控制分布的锐度
   
4. 零样本能力:
   • 图像分类: 将类别名转换为文本，计算相似度
   • 图像检索: 用文本查询找到最相似的图像
   • 文本检索: 用图像查询找到最相似的文本

5. 优势:
   ✓ 无需标注数据进行分类
   ✓ 泛化能力强
   ✓ 多语言支持
   ✓ 鲁棒性好
   
6. 局限性:
   ✗ 细粒度理解有限
   ✗ 复杂推理能力弱
   ✗ 生成能力缺失
   ✗ 对抽象概念理解不足
"""
    print(details)


def main():
    """主函数"""
    print("🚀 开始 CLIP 基础学习...")
    
    # 架构概览
    clip_architecture_overview()
    
    # 设备检测
    device = get_device()
    
    try:
        # 基础演示
        model, processor, tokenizer = clip_embeddings_demo(device)
        
        # 嵌入性质演示
        demonstrate_embedding_properties(model, processor, tokenizer, device)
        
        # 技术细节
        technical_details()
        
        print("\n" + "=" * 60)
        print("✅ 9.1 CLIP 基础学习完成!")
        print("=" * 60)
        print("\n下一步: 运行 9.2_clip_similarity_matrix.py 学习相似度矩阵")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查网络连接和依赖安装")


if __name__ == "__main__":
    main()