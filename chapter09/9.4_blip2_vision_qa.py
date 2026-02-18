import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
9.4 BLIP-2 视觉问答系统
========================

本节内容:
- BLIP-2 架构深入理解
- 图像描述生成 (Image Captioning)
- 视觉问答 (Visual Question Answering)
- 多轮对话式视觉问答
- 模型优化和部署考虑

BLIP-2 是 Salesforce 开发的先进视觉语言模型，
通过 Q-Former 架构实现了强大的图文理解和生成能力。
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from urllib.request import urlopen
import gc


def get_device():
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"使用设备: CUDA ({device_name})")
        print(f"GPU 内存: {memory_gb:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("使用设备: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("使用设备: CPU")
    return device


# 示例图片
IMAGE_URLS = {
    "puppy": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png",
    "beach": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/beach.png",
    "car": "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png",
}


def load_image_from_url(url):
    """从 URL 加载图片"""
    return Image.open(urlopen(url)).convert("RGB")


def blip2_architecture_overview():
    """BLIP-2 架构详解"""
    print("=" * 60)
    print("BLIP-2 架构详解")
    print("=" * 60)
    
    architecture = """
BLIP-2 三阶段架构设计:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────┐
│                        输入图像                              │
│                     (224×224×3)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Vision Encoder                              │
│                   (冻结参数)                                 │
│  • ViT-L/14 或 ViT-g/14                                    │
│  • 输出: 图像特征 [batch, 257, 1408]                        │
│  • 预训练权重保持不变                                        │
└────────────────────┬────────────────────────────────────────┘
                     │ 图像特征
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Q-Former                                  │
│                 (可学习参数)                                 │
│  • 32 个可学习查询向量 (Learnable Queries)                   │
│  • 12 层 Transformer                                       │
│  • 自注意力 + 交叉注意力                                     │
│  • 输出: 视觉 tokens [batch, 32, 768]                      │
└────────────────────┬────────────────────────────────────────┘
                     │ 视觉 tokens
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                大语言模型 (LLM)                              │
│                   (冻结参数)                                 │
│  • OPT-2.7B / FlanT5-XL                                   │
│  • 接收视觉 tokens 作为前缀                                 │
│  • 生成文本回答                                             │
└─────────────────────────────────────────────────────────────┘

核心创新 - Q-Former:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 可学习查询 (Learnable Queries):
   • 32 个固定数量的查询向量
   • 通过交叉注意力从图像中提取信息
   • 压缩图像信息为固定长度表示

2. 三种注意力机制:
   • 自注意力: 查询之间的交互
   • 交叉注意力: 查询与图像特征的交互  
   • 因果注意力: 文本生成时的掩码注意力

3. 训练策略:
   • 阶段1: 图文对比学习 + 图文匹配 + 图像描述生成
   • 阶段2: 指令微调，对齐视觉和语言理解

优势:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 模块化设计: 各组件可独立优化
✓ 参数效率: 只训练 Q-Former，其他组件冻结
✓ 强大生成: 利用大语言模型的生成能力
✓ 多任务支持: 图像描述、视觉问答、对话

技术规格:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 总参数量: ~15B (OPT-2.7B 版本)
• 可训练参数: ~188M (仅 Q-Former)
• 内存需求: ~15GB GPU 内存
• 推理速度: 中等 (受 LLM 大小影响)
"""
    print(architecture)


def check_system_requirements():
    """检查系统要求"""
    print("\n" + "=" * 60)
    print("系统要求检查")
    print("=" * 60)
    
    device = get_device()
    
    # 检查 GPU 内存
    if device == "cuda":
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n💾 GPU 内存检查:")
        print(f"  可用内存: {memory_gb:.1f} GB")
        
        if memory_gb >= 16:
            print("  ✅ 内存充足，可运行 BLIP-2")
        elif memory_gb >= 8:
            print("  ⚠️  内存较少，建议使用 FP16 精度")
        else:
            print("  ❌ 内存不足，建议使用较小模型")
    
    elif device == "mps":
        print(f"\n💾 MPS 设备:")
        print("  ⚠️  Apple Silicon GPU，内存共享")
        print("  建议监控内存使用情况")
    
    else:
        print(f"\n💾 CPU 模式:")
        print("  ⚠️  推理速度较慢")
        print("  建议使用较小的模型")
    
    # 检查依赖
    print(f"\n📦 依赖检查:")
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        print("  ✅ transformers 库已安装")
    except ImportError:
        print("  ❌ 需要安装: pip install transformers")
        return False
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ 需要安装 PyTorch")
        return False
    
    return True


def load_blip2_model(device, model_size="2.7b"):
    """加载 BLIP-2 模型"""
    print(f"\n[模型加载] 加载 BLIP-2 模型...")
    
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    # 模型选择
    model_configs = {
        "2.7b": "Salesforce/blip2-opt-2.7b",
        "6.7b": "Salesforce/blip2-opt-6.7b", 
        "flan-t5-xl": "Salesforce/blip2-flan-t5-xl"
    }
    
    model_id = model_configs.get(model_size, model_configs["2.7b"])
    print(f"✓ 选择模型: {model_id}")
    
    try:
        # 加载处理器
        processor = Blip2Processor.from_pretrained(model_id)
        print("✓ 处理器加载完成")
        
        # 加载模型 (优化内存使用)
        print("  正在加载模型权重...")
        
        model_kwargs = {}
        
        if device == "cuda":
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_8bit": False  # 可设为 True 进一步节省内存
            })
        elif device == "mps":
            model_kwargs.update({
                "torch_dtype": torch.float16
            })
        else:
            model_kwargs.update({
                "torch_dtype": torch.float32
            })
        
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        if device == "mps":
            model = model.to(device)
        
        print("✅ 模型加载成功")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型信息:")
        print(f"  总参数量: {total_params/1e9:.1f}B")
        print(f"  可训练参数: {trainable_params/1e6:.1f}M")
        print(f"  冻结比例: {(1-trainable_params/total_params)*100:.1f}%")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n💡 建议:")
        print("  1. 检查网络连接")
        print("  2. 确保有足够的内存/存储空间")
        print("  3. 尝试使用较小的模型")
        return None, None


def image_captioning_demo(model, processor, device):
    """图像描述生成演示"""
    print("\n" + "=" * 60)
    print("图像描述生成演示")
    print("=" * 60)
    
    # 测试不同类型的图像描述
    captioning_tasks = [
        {
            "image_key": "puppy",
            "prompts": [
                "Question: What do you see in this image? Answer:",
                "Question: Describe this image in detail. Answer:",
                "Question: Write a caption for this photo. Answer:"
            ]
        },
        {
            "image_key": "beach", 
            "prompts": [
                "Question: What is the setting of this image? Answer:",
                "Question: Describe the landscape. Answer:"
            ]
        },
        {
            "image_key": "car",
            "prompts": [
                "Question: What vehicle is shown? Answer:",
                "Question: Describe the car and its surroundings. Answer:"
            ]
        }
    ]
    
    for task in captioning_tasks:
        image_key = task["image_key"]
        prompts = task["prompts"]
        
        print(f"\n🖼️  测试图像: {image_key}")
        print("-" * 40)
        
        try:
            # 加载图像
            image = load_image_from_url(IMAGE_URLS[image_key])
            print(f"✓ 图像尺寸: {image.size}")
            
            for i, prompt in enumerate(prompts):
                print(f"\n[任务 {i+1}] {prompt}")
                
                # 预处理
                inputs = processor(image, text=prompt, return_tensors="pt")
                
                # 移动到设备
                if device != "cpu":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    if device == "cuda":
                        inputs = {k: v.half() if v.dtype == torch.float32 else v 
                                for k, v in inputs.items()}
                
                # 生成
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=3,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # 解码
                generated_text = processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0].strip()
                
                # 清理输出 (移除提示部分)
                if "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                else:
                    answer = generated_text
                
                print(f"🤖 回答: {answer}")
                
        except Exception as e:
            print(f"❌ 处理 {image_key} 时出错: {e}")


def visual_question_answering_demo(model, processor, device):
    """视觉问答演示"""
    print("\n" + "=" * 60)
    print("视觉问答演示")
    print("=" * 60)
    
    # 定义问答任务
    vqa_tasks = [
        {
            "image_key": "puppy",
            "questions": [
                "What animal is in the image?",
                "What is the dog doing?",
                "What season does this appear to be?",
                "Is the dog indoors or outdoors?",
                "What color is the dog's fur?"
            ]
        },
        {
            "image_key": "beach",
            "questions": [
                "What type of landscape is this?",
                "Is this a natural or artificial environment?",
                "What time of day might this be?",
                "Are there any people visible?",
                "What's the weather like?"
            ]
        },
        {
            "image_key": "car",
            "questions": [
                "What type of vehicle is shown?",
                "What color is the car?",
                "Is the car moving or stationary?",
                "What kind of road is the car on?",
                "How many cars are visible?"
            ]
        }
    ]
    
    for task in vqa_tasks:
        image_key = task["image_key"]
        questions = task["questions"]
        
        print(f"\n🖼️  图像: {image_key}")
        print("=" * 40)
        
        try:
            # 加载图像
            image = load_image_from_url(IMAGE_URLS[image_key])
            
            for i, question in enumerate(questions):
                prompt = f"Question: {question} Answer:"
                
                # 预处理
                inputs = processor(image, text=prompt, return_tensors="pt")
                
                # 移动到设备
                if device != "cpu":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    if device == "cuda":
                        inputs = {k: v.half() if v.dtype == torch.float32 else v 
                                for k, v in inputs.items()}
                
                # 生成回答
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        num_beams=2,
                        temperature=0.5,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # 解码
                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                # 提取答案
                if "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                else:
                    answer = generated_text
                
                print(f"❓ Q{i+1}: {question}")
                print(f"🤖 A{i+1}: {answer}\n")
                
        except Exception as e:
            print(f"❌ 处理 {image_key} 时出错: {e}")


def conversational_vqa_demo(model, processor, device):
    """多轮对话式视觉问答"""
    print("\n" + "=" * 60)
    print("多轮对话式视觉问答")
    print("=" * 60)
    
    # 选择一张图像进行深入对话
    image_key = "puppy"
    image = load_image_from_url(IMAGE_URLS[image_key])
    
    print(f"🖼️  对话图像: {image_key}")
    print("💬 开始多轮对话...")
    
    # 定义对话流程
    conversation_flow = [
        "What do you see in this image?",
        "What is the dog doing specifically?", 
        "What might the weather be like?",
        "Is this a good environment for the dog?",
        "What breed might this dog be?",
        "Would you recommend any activities for this dog?"
    ]
    
    conversation_history = []
    
    for i, question in enumerate(conversation_flow):
        print(f"\n--- 轮次 {i+1} ---")
        print(f"👤 用户: {question}")
        
        # 构建带历史的提示
        if conversation_history:
            # 包含之前的对话历史
            history_text = " ".join([
                f"Q: {q} A: {a}" for q, a in conversation_history[-2:]  # 保留最近2轮
            ])
            prompt = f"{history_text} Question: {question} Answer:"
        else:
            prompt = f"Question: {question} Answer:"
        
        try:
            # 预处理
            inputs = processor(image, text=prompt, return_tensors="pt")
            
            # 移动到设备
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if device == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v 
                            for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=3,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # 解码
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            # 提取答案
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text
            
            print(f"🤖 BLIP-2: {answer}")
            
            # 添加到对话历史
            conversation_history.append((question, answer))
            
        except Exception as e:
            print(f"❌ 生成回答时出错: {e}")
            break
    
    print(f"\n✅ 对话完成，共 {len(conversation_history)} 轮")


def performance_analysis(model, processor, device):
    """性能分析"""
    print("\n" + "=" * 60)
    print("性能分析")
    print("=" * 60)
    
    import time
    
    # 测试图像
    image = load_image_from_url(IMAGE_URLS["car"])
    
    # 测试不同长度的生成
    test_configs = [
        {"max_tokens": 10, "description": "短回答"},
        {"max_tokens": 30, "description": "中等回答"},
        {"max_tokens": 50, "description": "长回答"}
    ]
    
    prompt = "Question: Describe this image in detail. Answer:"
    
    print("🔬 生成长度对性能的影响:")
    print("-" * 50)
    
    for config in test_configs:
        max_tokens = config["max_tokens"]
        desc = config["description"]
        
        # 预处理
        inputs = processor(image, text=prompt, return_tensors="pt")
        
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v 
                        for k, v in inputs.items()}
        
        # 测量时间
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=2,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        end_time = time.time()
        
        # 解码
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text
        
        # 计算统计
        generation_time = end_time - start_time
        tokens_generated = len(processor.tokenizer.encode(answer))
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        print(f"\n{desc} (max_tokens={max_tokens}):")
        print(f"  生成时间: {generation_time:.2f}s")
        print(f"  实际tokens: {tokens_generated}")
        print(f"  生成速度: {tokens_per_second:.1f} tokens/s")
        print(f"  回答: {answer[:100]}...")
    
    # 内存使用分析
    if device == "cuda":
        print(f"\n💾 GPU 内存使用:")
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  已分配: {memory_allocated:.1f} GB")
        print(f"  已保留: {memory_reserved:.1f} GB")


def cleanup_memory():
    """清理内存"""
    print("\n🧹 清理内存...")
    
    # 清理 Python 垃圾回收
    gc.collect()
    
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ CUDA 缓存已清理")
    
    print("✓ 内存清理完成")


def main():
    """主函数"""
    print("🚀 开始 BLIP-2 视觉问答学习...")
    
    # 架构概览
    blip2_architecture_overview()
    
    # 系统检查
    if not check_system_requirements():
        print("❌ 系统要求不满足，退出")
        return
    
    # 设备检测
    device = get_device()
    
    # 询问是否继续
    print(f"\n⚠️  BLIP-2 模型较大 (~15GB)，确认继续？")
    try:
        response = input("输入 'y' 继续，其他退出: ").strip().lower()
        if response != 'y':
            print("👋 用户取消，退出")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n👋 非交互模式或用户中断，退出")
        return
    
    try:
        # 加载模型
        model, processor = load_blip2_model(device, "2.7b")
        
        if model is None or processor is None:
            print("❌ 模型加载失败，退出")
            return
        
        # 图像描述演示
        image_captioning_demo(model, processor, device)
        
        # 视觉问答演示
        visual_question_answering_demo(model, processor, device)
        
        # 多轮对话演示
        conversational_vqa_demo(model, processor, device)
        
        # 性能分析
        performance_analysis(model, processor, device)
        
        print("\n" + "=" * 60)
        print("✅ 9.4 BLIP-2 视觉问答学习完成!")
        print("=" * 60)
        print("\n🎯 关键收获:")
        print("  • BLIP-2 三阶段架构设计")
        print("  • Q-Former 的桥梁作用")
        print("  • 强大的图像理解和生成能力")
        print("  • 多轮对话的实现方式")
        print("\n下一步: 运行 9.5_lightweight_vlm.py 学习轻量级替代方案")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查网络连接、内存和依赖安装")
    
    finally:
        # 清理内存
        cleanup_memory()


if __name__ == "__main__":
    main()