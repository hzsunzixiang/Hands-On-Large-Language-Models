import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
9.5 轻量级视觉语言模型
======================

本节内容:
- 轻量级 VLM 模型选择
- BLIP-base 图像描述
- 资源友好的部署方案
- 边缘设备适配
- 性能与资源的权衡

当 BLIP-2 等大模型资源需求过高时，轻量级模型提供了
实用的替代方案，适合资源受限的环境。
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from urllib.request import urlopen
import time
import gc


def get_device():
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"使用设备: CUDA ({device_name}, {memory_gb:.1f}GB)")
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


def lightweight_vlm_overview():
    """轻量级 VLM 概览"""
    print("=" * 60)
    print("轻量级视觉语言模型概览")
    print("=" * 60)
    
    overview = """
轻量级 VLM 的必要性:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 资源限制场景:
   • 边缘设备 (手机、嵌入式系统)
   • 云服务成本控制
   • 实时应用需求
   • 离线部署需求

2. 大模型的挑战:
   • BLIP-2: ~15GB 内存需求
   • 推理速度慢
   • 部署成本高
   • 能耗大

轻量级模型对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│     模型        │   大小   │   内存   │   速度   │   能力   │
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ BLIP-2-OPT-2.7B │  ~15GB   │  ~15GB   │    慢    │   强大   │
│ BLIP-base       │  ~1GB    │  ~2GB    │    快    │   中等   │
│ CLIP            │  ~600MB  │  ~1GB    │   很快   │  仅嵌入  │
│ MiniGPT-4       │  ~7GB    │  ~8GB    │   中等   │   较强   │
│ LLaVA-7B        │  ~13GB   │  ~14GB   │   中等   │   强大   │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘

轻量级模型的优势:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 快速部署: 模型下载和加载速度快
✓ 低延迟: 推理速度快，适合实时应用
✓ 低成本: 硬件要求低，运行成本低
✓ 易集成: 简单的 API，易于集成到应用中
✓ 离线友好: 可在无网络环境下运行

应用场景:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 移动应用:
   • 图片自动标注
   • 实时图像描述
   • 辅助功能 (视觉辅助)

2. 边缘计算:
   • IoT 设备图像分析
   • 监控系统
   • 自动驾驶辅助

3. 原型开发:
   • 快速概念验证
   • 教学演示
   • 算法研究

4. 批量处理:
   • 大规模图像标注
   • 内容审核
   • 数据预处理
"""
    print(overview)


def model_comparison_analysis():
    """模型对比分析"""
    print("\n" + "=" * 60)
    print("模型详细对比分析")
    print("=" * 60)
    
    analysis = """
BLIP-base vs BLIP-2 详细对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BLIP-base (Salesforce/blip-image-captioning-base):
┌─────────────────────────────────────────────────────────────┐
│ 架构: Vision Transformer + BERT-base                       │
│ 参数量: ~385M                                               │
│ 内存需求: ~2GB                                              │
│ 推理速度: 快 (~100ms/image)                                 │
│ 支持任务:                                                   │
│   ✓ 无条件图像描述                                          │
│   ✓ 条件图像描述 (带前缀)                                   │
│   ✗ 复杂视觉问答                                            │
│   ✗ 多轮对话                                                │
└─────────────────────────────────────────────────────────────┘

BLIP-2-OPT-2.7B:
┌─────────────────────────────────────────────────────────────┐
│ 架构: ViT + Q-Former + OPT-2.7B                           │
│ 参数量: ~15B                                                │
│ 内存需求: ~15GB                                             │
│ 推理速度: 慢 (~2s/image)                                    │
│ 支持任务:                                                   │
│   ✓ 复杂图像描述                                            │
│   ✓ 视觉问答                                                │
│   ✓ 多轮对话                                                │
│   ✓ 指令跟随                                                │
└─────────────────────────────────────────────────────────────┘

选择建议:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用 BLIP-base 的场景:
• 只需要基础图像描述
• 资源受限 (< 4GB GPU内存)
• 需要快速响应 (< 200ms)
• 批量处理大量图像
• 移动端/边缘设备部署

使用 BLIP-2 的场景:
• 需要复杂视觉理解
• 支持自然语言问答
• 多轮对话需求
• 有充足计算资源 (> 16GB GPU)
• 对准确性要求高于速度

混合策略:
• 用 CLIP 做初步筛选
• 用 BLIP-base 做基础描述
• 用 BLIP-2 做复杂分析
"""
    print(analysis)


def blip_base_demo(device):
    """BLIP-base 演示"""
    print("\n" + "=" * 60)
    print("BLIP-base 轻量级图像描述")
    print("=" * 60)
    
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    # 1. 加载模型
    print("\n[步骤 1] 加载 BLIP-base 模型...")
    model_id = "Salesforce/blip-image-captioning-base"
    
    try:
        start_time = time.time()
        
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        
        if device != "cpu":
            model = model.to(device)
        
        load_time = time.time() - start_time
        
        print(f"✅ 模型加载成功")
        print(f"⏱️  加载时间: {load_time:.1f}s")
        
        # 模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 参数量: {total_params/1e6:.1f}M")
        
        if device == "cuda":
            memory_mb = torch.cuda.memory_allocated() / 1e6
            print(f"💾 GPU内存使用: {memory_mb:.0f}MB")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None
    
    # 2. 无条件图像描述
    print(f"\n[步骤 2] 无条件图像描述...")
    
    for name, url in IMAGE_URLS.items():
        try:
            print(f"\n🖼️  测试图像: {name}")
            
            # 加载图像
            image = load_image_from_url(url)
            print(f"   图像尺寸: {image.size}")
            
            # 预处理 (无文本输入)
            inputs = processor(image, return_tensors="pt")
            if device != "cpu":
                inputs = inputs.to(device)
            
            # 生成描述
            start_time = time.time()
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True
                )
            
            generation_time = time.time() - start_time
            
            # 解码
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            print(f"   🤖 描述: {caption}")
            print(f"   ⏱️  生成时间: {generation_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    # 3. 条件图像描述
    print(f"\n[步骤 3] 条件图像描述...")
    
    conditional_tasks = [
        {
            "image_key": "puppy",
            "prefixes": [
                "a photo of",
                "this image shows",
                "in this picture"
            ]
        },
        {
            "image_key": "beach", 
            "prefixes": [
                "a beautiful",
                "this landscape shows",
                "the scene depicts"
            ]
        }
    ]
    
    for task in conditional_tasks:
        image_key = task["image_key"]
        prefixes = task["prefixes"]
        
        print(f"\n🖼️  图像: {image_key}")
        
        try:
            image = load_image_from_url(IMAGE_URLS[image_key])
            
            for prefix in prefixes:
                print(f"\n   前缀: '{prefix}'")
                
                # 带文本前缀的预处理
                inputs = processor(image, text=prefix, return_tensors="pt")
                if device != "cpu":
                    inputs = inputs.to(device)
                
                # 生成
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=2,
                        temperature=0.6
                    )
                
                # 解码
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                print(f"   🤖 完整描述: {caption}")
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    return model, processor


def performance_benchmark(model, processor, device):
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)
    
    # 测试配置
    test_configs = [
        {"batch_size": 1, "max_length": 20, "num_beams": 1},
        {"batch_size": 1, "max_length": 30, "num_beams": 2},
        {"batch_size": 1, "max_length": 50, "num_beams": 3},
    ]
    
    test_image = load_image_from_url(IMAGE_URLS["car"])
    
    print("🔬 不同配置的性能测试:")
    print("-" * 50)
    
    for i, config in enumerate(test_configs):
        batch_size = config["batch_size"]
        max_length = config["max_length"]
        num_beams = config["num_beams"]
        
        print(f"\n[测试 {i+1}] max_length={max_length}, num_beams={num_beams}")
        
        # 多次测试取平均
        times = []
        captions = []
        
        for _ in range(3):  # 3次测试
            inputs = processor(test_image, return_tensors="pt")
            if device != "cpu":
                inputs = inputs.to(device)
            
            start_time = time.time()
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            end_time = time.time()
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            times.append(end_time - start_time)
            captions.append(caption)
        
        # 统计结果
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   平均时间: {avg_time:.3f}s (±{std_time:.3f}s)")
        print(f"   示例输出: {captions[0]}")
        
        # 计算吞吐量
        throughput = 1 / avg_time
        print(f"   吞吐量: {throughput:.1f} images/s")
    
    # 内存使用统计
    if device == "cuda":
        print(f"\n💾 内存使用统计:")
        memory_allocated = torch.cuda.memory_allocated() / 1e6
        memory_reserved = torch.cuda.memory_reserved() / 1e6
        print(f"   已分配: {memory_allocated:.0f}MB")
        print(f"   已保留: {memory_reserved:.0f}MB")


def batch_processing_demo(model, processor, device):
    """批量处理演示"""
    print("\n" + "=" * 60)
    print("批量处理演示")
    print("=" * 60)
    
    # 准备多张图像
    images = []
    image_names = []
    
    for name, url in IMAGE_URLS.items():
        try:
            img = load_image_from_url(url)
            images.append(img)
            image_names.append(name)
        except:
            pass
    
    print(f"📦 批量处理 {len(images)} 张图像...")
    
    # 方法1: 逐个处理
    print(f"\n[方法 1] 逐个处理:")
    
    start_time = time.time()
    individual_captions = []
    
    for i, (image, name) in enumerate(zip(images, image_names)):
        inputs = processor(image, return_tensors="pt")
        if device != "cpu":
            inputs = inputs.to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=30, num_beams=2)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        individual_captions.append(caption)
        
        print(f"   {name}: {caption}")
    
    individual_time = time.time() - start_time
    print(f"   总时间: {individual_time:.2f}s")
    print(f"   平均: {individual_time/len(images):.2f}s/image")
    
    # 方法2: 批量处理 (如果支持)
    print(f"\n[方法 2] 批量处理:")
    
    try:
        start_time = time.time()
        
        # 注意: BLIP 可能不支持真正的批量处理，这里演示概念
        batch_captions = []
        
        # 模拟批量处理 (实际上还是逐个，但可以优化预处理)
        for image, name in zip(images, image_names):
            inputs = processor(image, return_tensors="pt")
            if device != "cpu":
                inputs = inputs.to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_length=30, num_beams=2)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            batch_captions.append(caption)
        
        batch_time = time.time() - start_time
        
        print(f"   批量时间: {batch_time:.2f}s")
        print(f"   加速比: {individual_time/batch_time:.1f}x")
        
    except Exception as e:
        print(f"   批量处理不支持: {e}")


def deployment_considerations():
    """部署考虑因素"""
    print("\n" + "=" * 60)
    print("部署考虑因素")
    print("=" * 60)
    
    considerations = """
1. 硬件要求:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最低配置:
• CPU: 4核心以上
• 内存: 4GB RAM
• 存储: 2GB 可用空间
• GPU: 可选 (加速推理)

推荐配置:
• CPU: 8核心以上
• 内存: 8GB RAM  
• GPU: 4GB+ VRAM (GTX 1660 或更好)
• 存储: SSD 存储

2. 软件环境:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

依赖项:
```bash
pip install torch torchvision
pip install transformers
pip install pillow
pip install numpy
```

Docker 部署:
```dockerfile
FROM python:3.9-slim
RUN pip install torch transformers pillow
COPY app.py /app/
WORKDIR /app
CMD ["python", "app.py"]
```

3. 性能优化:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

模型优化:
• 使用 FP16 精度 (减少内存使用)
• 模型量化 (INT8)
• 动态批处理
• 缓存机制

代码优化:
```python
# FP16 推理
model = model.half()
inputs = {k: v.half() if v.dtype == torch.float32 else v 
          for k, v in inputs.items()}

# 批量预处理
def preprocess_batch(images):
    return processor(images, return_tensors="pt", padding=True)
```

4. 扩展性考虑:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

水平扩展:
• 负载均衡
• 多实例部署
• 队列系统 (Redis/RabbitMQ)

垂直扩展:
• GPU 集群
• 模型并行
• 流水线并行

5. 监控和维护:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键指标:
• 推理延迟 (P50, P95, P99)
• 吞吐量 (QPS)
• 内存使用率
• GPU 利用率
• 错误率

日志记录:
• 请求/响应日志
• 性能指标
• 错误追踪
• 资源使用情况

6. 成本分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BLIP-base vs BLIP-2 成本对比:

┌─────────────┬──────────┬──────────┬──────────┬──────────┐
│    指标     │ BLIP-base│  BLIP-2  │   节省   │   说明   │
├─────────────┼──────────┼──────────┼──────────┼──────────┤
│ GPU 内存    │   2GB    │   15GB   │   87%    │ 硬件成本 │
│ 推理延迟    │  100ms   │   2000ms │   95%    │ 用户体验 │
│ 服务器成本  │  $50/月  │ $400/月  │   87%    │ 运营成本 │
│ 电力消耗    │   低     │    高    │   80%    │ 环保考虑 │
└─────────────┴──────────┴──────────┴──────────┴──────────┘
"""
    print(considerations)


def edge_device_simulation():
    """边缘设备模拟"""
    print("\n" + "=" * 60)
    print("边缘设备部署模拟")
    print("=" * 60)
    
    print("🔧 模拟资源受限环境...")
    
    # 模拟不同的资源限制
    scenarios = [
        {
            "name": "移动设备",
            "cpu_cores": 4,
            "memory_gb": 2,
            "gpu": False,
            "description": "智能手机/平板"
        },
        {
            "name": "边缘服务器", 
            "cpu_cores": 8,
            "memory_gb": 4,
            "gpu": True,
            "description": "小型边缘计算节点"
        },
        {
            "name": "嵌入式设备",
            "cpu_cores": 2, 
            "memory_gb": 1,
            "gpu": False,
            "description": "Raspberry Pi 等"
        }
    ]
    
    for scenario in scenarios:
        name = scenario["name"]
        cpu_cores = scenario["cpu_cores"]
        memory_gb = scenario["memory_gb"]
        has_gpu = scenario["gpu"]
        desc = scenario["description"]
        
        print(f"\n📱 场景: {name} ({desc})")
        print(f"   CPU: {cpu_cores} 核心")
        print(f"   内存: {memory_gb}GB")
        print(f"   GPU: {'是' if has_gpu else '否'}")
        
        # 评估适用性
        if memory_gb >= 2 and cpu_cores >= 4:
            print("   ✅ 适合 BLIP-base")
            print("   ❌ 不适合 BLIP-2")
            
            # 估算性能
            if has_gpu:
                estimated_time = "100-200ms"
                throughput = "5-10 images/s"
            else:
                estimated_time = "500-1000ms"  
                throughput = "1-2 images/s"
                
            print(f"   ⏱️  预估延迟: {estimated_time}")
            print(f"   📊 预估吞吐: {throughput}")
            
        elif memory_gb >= 1:
            print("   ⚠️  仅适合 CLIP (嵌入)")
            print("   ❌ 不适合生成模型")
        else:
            print("   ❌ 资源不足")
    
    # 优化建议
    print(f"\n💡 边缘部署优化建议:")
    print("   1. 使用模型量化 (INT8)")
    print("   2. 启用模型缓存")
    print("   3. 批量处理优化")
    print("   4. 异步推理队列")
    print("   5. 结果缓存机制")


def main():
    """主函数"""
    print("🚀 开始轻量级 VLM 学习...")
    
    # 概览
    lightweight_vlm_overview()
    
    # 模型对比
    model_comparison_analysis()
    
    # 设备检测
    device = get_device()
    
    try:
        # BLIP-base 演示
        model, processor = blip_base_demo(device)
        
        if model is not None and processor is not None:
            # 性能测试
            performance_benchmark(model, processor, device)
            
            # 批量处理
            batch_processing_demo(model, processor, device)
        
        # 部署考虑
        deployment_considerations()
        
        # 边缘设备模拟
        edge_device_simulation()
        
        print("\n" + "=" * 60)
        print("✅ 9.5 轻量级 VLM 学习完成!")
        print("=" * 60)
        print("\n🎯 关键收获:")
        print("  • 轻量级模型的优势和适用场景")
        print("  • BLIP-base 的实际性能表现")
        print("  • 资源与性能的权衡考虑")
        print("  • 边缘设备部署的实践指导")
        print("\n下一步: 运行 9.6_multimodal_summary.py 查看章节总结")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查网络连接和依赖安装")
    
    finally:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()