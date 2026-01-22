"""
Chapter 9 - Multimodal Large Language Models
多模态大语言模型

本章内容:
1. CLIP - 对比学习图文模型，联合理解图像和文本
2. 图文嵌入对齐 - 同一空间中的图文表示
3. 图像-文本相似度计算
4. BLIP-2 - 视觉问答和图像描述生成
5. 多轮对话式视觉问答

注意: 本章需要 GPU 支持以获得最佳性能
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


# ============================================================
# 示例图片 URL
# ============================================================
IMAGE_URLS = {
    "puppy": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png",
    "beach": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/beach.png",
    "car": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png",
}


def load_image_from_url(url):
    """从 URL 加载图片"""
    return Image.open(urlopen(url)).convert("RGB")


# ============================================================
# Part 1: CLIP 基础 - 图文嵌入
# ============================================================
def clip_embeddings_demo(device=None):
    """
    Part 1: CLIP 模型基础
    展示如何使用 CLIP 生成图像和文本的统一嵌入
    """
    from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
    
    print("\n" + "=" * 60)
    print("Part 1: CLIP - 图文嵌入 (Contrastive Language-Image Pre-training)")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 1.1 加载 CLIP 模型
    print("\n[1.1] 加载 CLIP 模型...")
    model_id = "openai/clip-vit-base-patch32"
    
    # 文本分词器
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    # 图像处理器
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    # 主模型
    model = CLIPModel.from_pretrained(model_id).to(device)
    
    print(f"模型: {model_id}")
    print(f"文本嵌入维度: 512")
    print(f"图像嵌入维度: 512 (统一空间)")
    
    # 1.2 加载示例图像
    print("\n[1.2] 加载示例图像...")
    image = load_image_from_url(IMAGE_URLS["puppy"])
    caption = "a puppy playing in the snow"
    print(f"图像描述: '{caption}'")
    
    # 1.3 生成文本嵌入
    print("\n[1.3] 生成文本嵌入...")
    text_inputs = clip_tokenizer(caption, return_tensors="pt").to(device)
    
    # 查看分词结果
    tokens = clip_tokenizer.convert_ids_to_tokens(text_inputs["input_ids"][0])
    print(f"分词结果: {tokens}")
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)
    print(f"文本嵌入形状: {text_embedding.shape}")
    
    # 1.4 生成图像嵌入
    print("\n[1.4] 生成图像嵌入...")
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    print(f"图像张量形状: {image_inputs['pixel_values'].shape}")
    print("  -> [batch, channels, height, width] = [1, 3, 224, 224]")
    
    with torch.no_grad():
        image_embedding = model.get_image_features(**image_inputs)
    print(f"图像嵌入形状: {image_embedding.shape}")
    
    # 1.5 计算相似度
    print("\n[1.5] 计算图文相似度...")
    # 归一化
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    
    # 余弦相似度
    similarity = (text_embedding @ image_embedding.T).item()
    print(f"图文相似度: {similarity:.4f}")
    
    return model, clip_processor, clip_tokenizer


# ============================================================
# Part 2: CLIP 相似度矩阵
# ============================================================
def clip_similarity_matrix_demo(device=None):
    """
    Part 2: CLIP 相似度矩阵
    展示多张图片和多个描述之间的相似度
    """
    from transformers import CLIPProcessor, CLIPModel
    
    print("\n" + "=" * 60)
    print("Part 2: CLIP 相似度矩阵")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 加载模型
    model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    
    # 加载多张图片
    print("\n加载图片...")
    images = []
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            images.append(img)
            print(f"  加载: {name}")
        except Exception as e:
            print(f"  跳过 {name}: {e}")
    
    # 定义描述
    captions = [
        "a puppy playing in the snow",
        "a sandy beach with ocean waves",
        "a sports car on the road"
    ]
    
    print(f"\n图片数量: {len(images)}")
    print(f"描述数量: {len(captions)}")
    
    # 生成嵌入
    print("\n计算嵌入...")
    inputs = clip_processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    
    # 归一化
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # 计算相似度矩阵
    sim_matrix = (image_embeds @ text_embeds.T).cpu().numpy()
    
    print("\n相似度矩阵 (图像 x 文本):")
    print("-" * 60)
    print(f"{'':20}", end="")
    for i, cap in enumerate(captions):
        print(f"{cap[:15]:>18}", end="")
    print()
    
    for i, (name, _) in enumerate(IMAGE_URLS.items()):
        if i < len(sim_matrix):
            print(f"{name:20}", end="")
            for j in range(len(captions)):
                print(f"{sim_matrix[i, j]:18.3f}", end="")
            print()
    
    print("\n解读: 对角线上的值最高，说明 CLIP 正确匹配了图文对")
    
    return sim_matrix


# ============================================================
# Part 3: 使用 Sentence-Transformers 的 CLIP
# ============================================================
def sbert_clip_demo(device=None):
    """
    Part 3: SBERT-CLIP
    使用 Sentence-Transformers 库简化 CLIP 使用
    """
    from sentence_transformers import SentenceTransformer, util
    
    print("\n" + "=" * 60)
    print("Part 3: SBERT-CLIP (Sentence-Transformers 封装)")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 加载 SBERT 兼容的 CLIP 模型
    print("\n加载 SBERT-CLIP 模型...")
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    
    # 加载图片
    images = []
    for name, url in IMAGE_URLS.items():
        try:
            images.append(load_image_from_url(url))
        except:
            pass
    
    captions = [
        "a puppy playing in the snow",
        "a sandy beach with ocean waves",
        "a sports car on the road"
    ]
    
    # 编码
    print("编码图像和文本...")
    image_embeddings = model.encode(images)
    text_embeddings = model.encode(captions)
    
    # 计算相似度
    sim_matrix = util.cos_sim(image_embeddings, text_embeddings)
    
    print("\n相似度矩阵:")
    print(sim_matrix.numpy().round(3))
    
    return model


# ============================================================
# Part 4: BLIP-2 视觉问答
# ============================================================
def blip2_demo(device=None):
    """
    Part 4: BLIP-2 - 视觉语言模型
    展示图像描述生成和视觉问答
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    print("\n" + "=" * 60)
    print("Part 4: BLIP-2 - 视觉语言模型")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 检查是否有足够内存
    print("\n注意: BLIP-2 模型较大 (~15GB), 需要足够的 GPU/内存")
    
    # 加载处理器和模型
    print("\n加载 BLIP-2 模型...")
    model_id = "Salesforce/blip2-opt-2.7b"
    
    try:
        blip_processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "mps":
            model = model.to(device)
        
        print(f"模型加载成功: {model_id}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用较小的模型...")
        return None
    
    # 加载图像
    print("\n加载测试图像...")
    image = load_image_from_url(IMAGE_URLS["car"])
    
    # 4.1 图像描述生成
    print("\n[4.1] 图像描述生成 (Image Captioning)")
    prompt = "Question: Write down what you see in this picture. Answer:"
    
    inputs = blip_processor(image, text=prompt, return_tensors="pt")
    if device != "cpu":
        inputs = inputs.to(device, torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=30)
        generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print(f"提示: {prompt}")
    print(f"回答: {generated_text}")
    
    # 4.2 视觉问答
    print("\n[4.2] 视觉问答 (Visual Question Answering)")
    questions = [
        "What color is the car?",
        "Is the car moving?",
        "What time of day is it?"
    ]
    
    for question in questions:
        prompt = f"Question: {question} Answer:"
        inputs = blip_processor(image, text=prompt, return_tensors="pt")
        if device != "cpu":
            inputs = inputs.to(device, torch.float16)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            answer = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print(f"Q: {question}")
        print(f"A: {answer}\n")
    
    return model, blip_processor


# ============================================================
# Part 5: 轻量级替代方案 (LLaVA 风格)
# ============================================================
def lightweight_vlm_demo(device=None):
    """
    Part 5: 轻量级视觉语言模型
    使用较小的模型进行视觉问答
    """
    from transformers import AutoProcessor, AutoModelForVision2Seq
    
    print("\n" + "=" * 60)
    print("Part 5: 轻量级视觉语言模型")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 尝试使用较小的 BLIP 模型
    print("\n加载轻量级模型...")
    model_id = "Salesforce/blip-image-captioning-base"
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        
        print(f"模型加载成功: {model_id}")
        
        # 图像描述
        image = load_image_from_url(IMAGE_URLS["puppy"])
        
        # 无条件生成
        print("\n[无条件图像描述]")
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"生成的描述: {caption}")
        
        # 条件生成
        print("\n[条件图像描述]")
        text = "a photo of"
        inputs = processor(image, text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"条件前缀: '{text}'")
        print(f"生成的描述: {caption}")
        
        return model, processor
        
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None


def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("Chapter 9 总结")
    print("=" * 60)
    
    summary = """
┌─────────────────────────────────────────────────────────────┐
│                  多模态大语言模型 (MLLM)                     │
├─────────────────────────────────────────────────────────────┤
│  1. CLIP (Contrastive Language-Image Pre-training)         │
│     - 对比学习: 让图像和文本共享同一嵌入空间                   │
│     - 架构: Vision Transformer (图像) + Text Transformer     │
│     - 训练: 4 亿图文对，对比损失函数                          │
│     - 应用: 图像检索、零样本分类、图文匹配                     │
│                                                             │
│  2. BLIP-2 (Bootstrapping Language-Image Pre-training)     │
│     - 三阶段架构:                                            │
│       ① Vision Encoder (冻结的图像编码器)                    │
│       ② Q-Former (可学习的查询变换器)                        │
│       ③ LLM (冻结的语言模型)                                │
│     - Q-Former 充当图像和文本的"桥梁"                        │
│     - 支持: 图像描述、视觉问答、多轮对话                      │
│                                                             │
│  3. 关键概念                                                │
│     - 模态对齐: 让不同模态的表示可以直接比较                   │
│     - 对比学习: 正样本(匹配的图文对)靠近，负样本远离           │
│     - 指令微调: 让模型理解视觉问答指令                        │
│     - 上下文学习: 通过示例引导模型输出                        │
└─────────────────────────────────────────────────────────────┘

CLIP 架构:
  ┌─────────────┐      ┌─────────────┐
  │   Image     │      │   Text      │
  │ "puppy.jpg" │      │  "a puppy"  │
  └──────┬──────┘      └──────┬──────┘
         │                    │
  ┌──────▼──────┐      ┌──────▼──────┐
  │   Vision    │      │   Text      │
  │  Encoder    │      │  Encoder    │
  │ (ViT-B/32)  │      │(Transformer)│
  └──────┬──────┘      └──────┬──────┘
         │                    │
  ┌──────▼──────┐      ┌──────▼──────┐
  │   Image     │      │   Text      │
  │ Embedding   │◄────►│ Embedding   │
  │   [512]     │  相  │   [512]     │
  └─────────────┘  似  └─────────────┘
                  度

BLIP-2 架构:
  ┌─────────────┐
  │   Image     │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │   Vision    │ (冻结)
  │  Encoder    │
  └──────┬──────┘
         │ 图像特征
  ┌──────▼──────┐
  │  Q-Former   │ (可学习)
  │  32 queries │
  └──────┬──────┘
         │ 视觉 tokens
  ┌──────▼──────┐
  │    LLM      │ (冻结)
  │ (OPT/FlanT5)│
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ 生成的文本   │
  └─────────────┘
"""
    print(summary)


def main():
    """主函数"""
    device = get_device()
    
    # Part 1: CLIP 基础
    try:
        clip_embeddings_demo(device)
    except Exception as e:
        print(f"\nPart 1 跳过: {e}")
    
    # Part 2: CLIP 相似度矩阵
    try:
        clip_similarity_matrix_demo(device)
    except Exception as e:
        print(f"\nPart 2 跳过: {e}")
    
    # Part 3: SBERT-CLIP
    try:
        sbert_clip_demo(device)
    except Exception as e:
        print(f"\nPart 3 跳过: {e}")
    
    # Part 4: BLIP-2 (可选，需要大量内存)
    print("\n" + "=" * 60)
    print("是否运行 BLIP-2 演示? (需要 ~15GB 内存)")
    print("=" * 60)
    
    try:
        run_blip = input("输入 'y' 运行 BLIP-2 演示，其他跳过: ").strip().lower()
        if run_blip == 'y':
            blip2_demo(device)
        else:
            # 运行轻量级替代
            lightweight_vlm_demo(device)
    except EOFError:
        print("非交互模式，运行轻量级模型...")
        lightweight_vlm_demo(device)
    
    # 打印总结
    print_summary()


if __name__ == "__main__":
    main()
