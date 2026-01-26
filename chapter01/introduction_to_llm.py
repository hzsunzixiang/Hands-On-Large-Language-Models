#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 1 - 语言模型入门 (Introduction to Language Models)

本章演示如何使用 HuggingFace Transformers 加载和运行大语言模型。

核心内容:
1. 模型加载 - 使用 AutoModelForCausalLM 加载预训练模型
2. 分词器 - 使用 AutoTokenizer 处理文本
3. Pipeline - 使用高级 API 简化推理流程
4. 设备适配 - 支持 CPU/CUDA/MPS 多种运行环境

支持的模型:
- microsoft/Phi-3-mini-4k-instruct (3.8B 参数)
- Qwen/Qwen2.5-0.5B-Instruct ~ Qwen2.5-7B-Instruct
- meta-llama/Llama-3.2-1B-Instruct 等

运行方式:
    python introduction_to_llm.py
    python introduction_to_llm.py --model Qwen/Qwen2.5-0.5B-Instruct
    python introduction_to_llm.py --prompt "写一首关于春天的诗"
"""

import argparse
import gc
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
# 可选模型 (按参数量排序):
AVAILABLE_MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",   # ~1GB, 最轻量
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",   # ~3GB
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",       # ~6GB
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",       # ~14GB
    "phi3": "microsoft/Phi-3-mini-4k-instruct",  # ~7GB, 原书示例
}

DEFAULT_MODEL = "qwen-0.5b"  # 默认使用小模型方便测试


# ============================================================
# 工具函数
# ============================================================
def get_device():
    """
    检测可用的计算设备
    
    优先级: CUDA > MPS > CPU
    
    Returns:
        str: 设备名称 ("cuda", "mps", "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_config(device: str):
    """
    根据设备返回最佳配置
    
    Args:
        device: 设备名称
    
    Returns:
        tuple: (device_map, dtype)
    """
    if device == "cuda":
        # CUDA: 使用 float16 节省显存
        return "cuda", torch.float16
    elif device == "mps":
        # MPS: Apple Silicon，某些操作有兼容问题，用 CPU 更稳定
        # 如果想尝试 MPS 加速，可改为 ("mps", torch.float32)
        return "cpu", torch.float32
    else:
        # CPU: 使用 float32
        return "cpu", torch.float32


def clear_memory():
    """清理内存和显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 核心功能
# ============================================================
def load_model(model_name: str, device: str = None):
    """
    加载语言模型和分词器
    
    Args:
        model_name: 模型名称 (HuggingFace model ID)
        device: 目标设备，None 则自动检测
    
    Returns:
        tuple: (model, tokenizer, device_map)
    """
    if device is None:
        device = get_device()
    
    device_map, dtype = get_device_config(device)
    
    print(f"  模型: {model_name}")
    print(f"  设备: {device_map}")
    print(f"  精度: {dtype}")
    print()
    
    print("  正在加载模型 (首次运行需下载)...")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",  # 避免 flash_attn 兼容问题
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    print("  ✓ 模型加载完成")
    
    return model, tokenizer, device_map


def create_generator(model, tokenizer, max_new_tokens: int = 200):
    """
    创建文本生成 Pipeline
    
    Pipeline 封装了:
    1. 文本 -> Token IDs (分词)
    2. Token IDs -> 模型推理
    3. 输出 Token IDs -> 文本 (解码)
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        max_new_tokens: 最大生成 token 数
    
    Returns:
        pipeline: 文本生成 pipeline
    """
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,   # 只返回生成的新内容
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 贪婪解码，结果确定
    )
    return generator


def generate_text(generator, prompt: str, system_prompt: str = None):
    """
    使用 Pipeline 生成文本
    
    Args:
        generator: 文本生成 pipeline
        prompt: 用户输入
        system_prompt: 系统提示词 (可选)
    
    Returns:
        str: 生成的文本
    """
    # 构建消息格式 (Chat Template)
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # 生成
    output = generator(messages)
    
    return output[0]["generated_text"]


# ============================================================
# 演示示例
# ============================================================
def demo_basic_generation(generator):
    """
    演示 1: 基础文本生成
    
    展示最简单的问答交互
    """
    print("\n" + "=" * 60)
    print("  演示 1: 基础文本生成")
    print("=" * 60)
    
    prompts = [
        "Create a funny joke about chickens.",
        "讲一个关于程序员的笑话。",
    ]
    
    for prompt in prompts:
        print(f"\n  [用户] {prompt}")
        response = generate_text(generator, prompt)
        print(f"  [模型] {response}")


def demo_with_system_prompt(generator):
    """
    演示 2: 使用系统提示词
    
    System Prompt 可以设定模型的角色和行为规范
    """
    print("\n" + "=" * 60)
    print("  演示 2: 使用系统提示词 (System Prompt)")
    print("=" * 60)
    
    system_prompt = "你是一位专业的技术助手，回答简洁明了，使用中文。"
    prompt = "什么是大语言模型？"
    
    print(f"\n  [系统] {system_prompt}")
    print(f"  [用户] {prompt}")
    
    response = generate_text(generator, prompt, system_prompt)
    print(f"  [模型] {response}")


def demo_code_generation(generator):
    """
    演示 3: 代码生成
    
    LLM 可以生成代码并解释
    """
    print("\n" + "=" * 60)
    print("  演示 3: 代码生成")
    print("=" * 60)
    
    prompt = "写一个 Python 函数，计算斐波那契数列的第 n 项"
    
    print(f"\n  [用户] {prompt}")
    response = generate_text(generator, prompt)
    print(f"  [模型] {response}")


# ============================================================
# 教学总结
# ============================================================
def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("  Chapter 1 总结")
    print("=" * 60)
    
    summary = """
    ┌─────────────────────────────────────────────────────────┐
    │           语言模型入门 - 核心概念                        │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  1. 模型加载 (Model Loading)                            │
    │     AutoModelForCausalLM.from_pretrained()              │
    │     - 自动下载和缓存模型权重                             │
    │     - 支持多种精度 (float32, float16, bfloat16)         │
    │     - device_map 控制设备分配                           │
    │                                                         │
    │  2. 分词器 (Tokenizer)                                  │
    │     AutoTokenizer.from_pretrained()                     │
    │     - 文本 <-> Token IDs 双向转换                       │
    │     - 处理特殊 token ([CLS], [SEP], <pad>)              │
    │     - 支持 Chat Template (对话格式)                     │
    │                                                         │
    │  3. Pipeline (高级 API)                                 │
    │     pipeline("text-generation", model=...)              │
    │     - 封装完整的推理流程                                │
    │     - 简化代码，适合快速原型                            │
    │     - 支持 batch 处理                                   │
    │                                                         │
    │  4. 生成参数                                            │
    │     - max_new_tokens: 最大生成长度                      │
    │     - do_sample: 是否采样 (False=贪婪解码)              │
    │     - temperature: 采样温度 (越高越随机)                │
    │     - top_p: 核采样阈值                                 │
    │                                                         │
    ├─────────────────────────────────────────────────────────┤
    │  关键洞见:                                              │
    │  • LLM 本质是「下一个 Token 预测器」                    │
    │  • 同一模型可用于多种任务 (问答/翻译/代码...)           │
    │  • 模型越大效果越好，但资源消耗也越大                   │
    │  • Pipeline 适合原型，生产环境需更细粒度控制            │
    └─────────────────────────────────────────────────────────┘
    """
    print(summary)


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Chapter 1 - 语言模型入门",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python introduction_to_llm.py                          # 使用默认模型运行演示
  python introduction_to_llm.py --model phi3             # 使用 Phi-3 模型
  python introduction_to_llm.py --prompt "你好"          # 自定义提示词
  python introduction_to_llm.py --list-models            # 列出可用模型
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"模型名称或别名 (默认: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="自定义提示词 (不设置则运行演示)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="最大生成 token 数 (默认: 200)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出可用模型别名"
    )
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        print("\n可用模型:")
        for alias, model_id in AVAILABLE_MODELS.items():
            print(f"  {alias:12s} -> {model_id}")
        return
    
    print("=" * 60)
    print("  Chapter 1 - 语言模型入门")
    print("=" * 60)
    
    # 环境信息
    device = get_device()
    print(f"\n  PyTorch 版本: {torch.__version__}")
    print(f"  检测到设备: {device}")
    print()
    
    # 解析模型名称
    model_name = AVAILABLE_MODELS.get(args.model, args.model)
    
    # 加载模型
    print("【步骤 1】加载模型...")
    model, tokenizer, device_map = load_model(model_name, device)
    
    # 创建 Pipeline
    print("\n【步骤 2】创建文本生成 Pipeline...")
    generator = create_generator(model, tokenizer, args.max_tokens)
    print("  ✓ Pipeline 创建完成")
    
    # 生成文本
    if args.prompt:
        # 使用自定义提示词
        print("\n【步骤 3】生成文本...")
        print(f"\n  [用户] {args.prompt}")
        response = generate_text(generator, args.prompt)
        print(f"  [模型] {response}")
    else:
        # 运行演示
        print("\n【步骤 3】运行演示...")
        demo_basic_generation(generator)
        demo_with_system_prompt(generator)
        demo_code_generation(generator)
    
    # 总结
    print_summary()
    
    # 清理
    clear_memory()
    
    print("\n" + "=" * 60)
    print("  运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
