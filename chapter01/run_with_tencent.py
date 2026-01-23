#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 1 - 语言模型入门 (使用腾讯 HAI API)

本脚本演示如何使用腾讯内部 HAI 大模型 API 进行文本生成。
HAI API 兼容 OpenAI 格式，可直接使用 openai 库。

支持的模型:
- DeepSeek-V3-0324: 128k 上下文，支持 function calling
- DeepSeek-V3.1: 128k 上下文，支持思维链和 function calling
- Qwen3-235B-A22B: 40k 上下文，支持思维链
- Kimi-K2-Instruct: 128k 上下文，支持 function calling
- Qwen3-32B-FP8: 50k 上下文，支持思维链和 function calling

使用前配置:
    export TENCENT_API_KEY=your_api_key

运行方式:
    python run_with_tencent.py
    python run_with_tencent.py --model DeepSeek-V3.1
    python run_with_tencent.py --prompt "写一首关于春天的诗"

API 文档: https://hai.woa.com
"""

import os
import argparse
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# ============================================================
# 配置
# ============================================================
# API 地址
# 外网地址: http://api.haihub.cn/v1
# IDC内网地址: http://21.126.30.10:10170/v1 (推荐，稳定性更高)
TENCENT_BASE_URL = "http://api.haihub.cn/v1"

# 可用模型
AVAILABLE_MODELS = {
    "deepseek-v3": "DeepSeek-V3-0324",      # 128k, function calling
    "deepseek-v3.1": "DeepSeek-V3.1",       # 128k, 思维链 + function calling
    "qwen3-235b": "Qwen3-235B-A22B",        # 40k, 思维链
    "kimi-k2": "Kimi-K2-Instruct",          # 128k, function calling
    "qwen3-32b": "Qwen3-32B-FP8",           # 50k, 思维链 + function calling
}

DEFAULT_MODEL = "DeepSeek-V3-0324"


# ============================================================
# 核心功能
# ============================================================
def generate_with_tencent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    use_thinking: bool = False,
) -> str:
    """
    使用腾讯 HAI API 生成文本
    
    Args:
        prompt: 用户输入
        model: 模型名称
        system_prompt: 系统提示词 (可选)
        temperature: 采样温度 (0-2, 越高越随机)
        max_tokens: 最大生成 token 数
        use_thinking: 是否启用思维链模式 (仅 DeepSeek-V3.1 支持)
    
    Returns:
        生成的文本
    """
    from openai import OpenAI
    
    # 初始化客户端
    client = OpenAI(
        api_key=os.getenv("TENCENT_API_KEY"),
        base_url=TENCENT_BASE_URL,
    )
    
    # 构建消息
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # 构建请求参数
    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # DeepSeek-V3.1 支持思维链模式
    if use_thinking and model == "DeepSeek-V3.1":
        request_params["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
    
    # 调用 API
    response = client.chat.completions.create(**request_params)
    
    result = response.choices[0].message.content
    
    # 如果有思维链内容，也返回
    if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
        reasoning = response.choices[0].message.reasoning_content
        return f"[思维链]\n{reasoning}\n\n[回答]\n{result}"
    
    return result


def check_api_key() -> bool:
    """检查 API Key 是否配置"""
    if not os.getenv("TENCENT_API_KEY"):
        print("\n❌ 错误: 未设置 TENCENT_API_KEY")
        print("\n请通过以下方式之一设置:")
        print("  1. 创建 .env 文件: TENCENT_API_KEY=your_api_key")
        print("  2. 设置环境变量: export TENCENT_API_KEY=your_api_key")
        print("\n获取 API Key: 联系 @chengjieyu 申请")
        print("API 文档: https://hai.woa.com")
        return False
    return True


# ============================================================
# 演示示例
# ============================================================
def demo_basic_generation(model: str):
    """演示 1: 基础文本生成"""
    print("\n" + "=" * 60)
    print("  演示 1: 基础文本生成")
    print("=" * 60)
    
    prompts = [
        "Create a funny joke about chickens.",
        "讲一个关于程序员的笑话。",
    ]
    
    for prompt in prompts:
        print(f"\n  [用户] {prompt}")
        response = generate_with_tencent(prompt, model=model)
        print(f"  [模型] {response}")


def demo_with_system_prompt(model: str):
    """演示 2: 使用系统提示词"""
    print("\n" + "=" * 60)
    print("  演示 2: 使用系统提示词 (System Prompt)")
    print("=" * 60)
    
    system_prompt = "你是一位专业的技术助手，回答简洁明了，使用中文。"
    prompt = "什么是大语言模型？用三句话解释。"
    
    print(f"\n  [系统] {system_prompt}")
    print(f"  [用户] {prompt}")
    
    response = generate_with_tencent(prompt, model=model, system_prompt=system_prompt)
    print(f"  [模型] {response}")


def demo_code_generation(model: str):
    """演示 3: 代码生成"""
    print("\n" + "=" * 60)
    print("  演示 3: 代码生成")
    print("=" * 60)
    
    prompt = "写一个 Python 函数，计算斐波那契数列的第 n 项，要求使用递归和记忆化。"
    
    print(f"\n  [用户] {prompt}")
    response = generate_with_tencent(prompt, model=model, max_tokens=512)
    print(f"  [模型]\n{response}")


def demo_thinking_mode():
    """演示 4: 思维链模式 (仅 DeepSeek-V3.1)"""
    print("\n" + "=" * 60)
    print("  演示 4: 思维链模式 (DeepSeek-V3.1)")
    print("=" * 60)
    
    prompt = "小明有5个苹果，给了小红2个，又买了3个，最后还剩几个？请一步步推理。"
    
    print(f"\n  [用户] {prompt}")
    response = generate_with_tencent(
        prompt, 
        model="DeepSeek-V3.1", 
        use_thinking=True
    )
    print(f"  [模型] {response}")


# ============================================================
# 教学总结
# ============================================================
def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("  Chapter 1 总结 - 使用腾讯 HAI API")
    print("=" * 60)
    
    summary = """
    ┌─────────────────────────────────────────────────────────┐
    │           使用云端 API 访问大语言模型                    │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  优势:                                                  │
    │  • 无需本地 GPU，适合资源受限环境                       │
    │  • 可访问超大模型 (如 235B 参数的 Qwen3)                │
    │  • 部署简单，只需 API Key                               │
    │                                                         │
    │  API 调用流程:                                          │
    │  1. 初始化 OpenAI 客户端 (兼容接口)                     │
    │  2. 构建 messages (system + user)                       │
    │  3. 调用 chat.completions.create()                      │
    │  4. 解析 response.choices[0].message.content            │
    │                                                         │
    │  关键参数:                                              │
    │  • model: 模型名称                                      │
    │  • temperature: 0-2, 越高越随机                         │
    │  • max_tokens: 最大生成长度                             │
    │  • messages: 对话历史                                   │
    │                                                         │
    │  对比本地部署:                                          │
    │  ┌─────────────┬─────────────┬─────────────┐            │
    │  │             │ 本地部署     │ 云端 API    │            │
    │  ├─────────────┼─────────────┼─────────────┤            │
    │  │ 硬件要求     │ 需要 GPU    │ 无需        │            │
    │  │ 模型大小     │ 受限于显存   │ 无限制      │            │
    │  │ 延迟        │ 较低        │ 依赖网络     │            │
    │  │ 隐私        │ 数据本地    │ 数据上云     │            │
    │  │ 成本        │ 一次性      │ 按量付费     │            │
    │  └─────────────┴─────────────┴─────────────┘            │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """
    print(summary)


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Chapter 1 - 使用腾讯 HAI API 进行文本生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_with_tencent.py                              # 运行演示
  python run_with_tencent.py --model deepseek-v3.1        # 使用 DeepSeek-V3.1
  python run_with_tencent.py --prompt "你好"              # 自定义提示词
  python run_with_tencent.py --list-models                # 列出可用模型
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
        "--system", "-s",
        default=None,
        help="系统提示词"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="最大生成 token 数 (默认: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度 (默认: 0.7)"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="启用思维链模式 (仅 DeepSeek-V3.1)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出可用模型"
    )
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        print("\n可用模型:")
        for alias, model_id in AVAILABLE_MODELS.items():
            print(f"  {alias:15s} -> {model_id}")
        return
    
    print("=" * 60)
    print("  Chapter 1 - 语言模型入门 (腾讯 HAI API)")
    print("=" * 60)
    
    # 检查 API Key
    if not check_api_key():
        return
    
    print(f"\n✓ API Key 已配置")
    print(f"  API 地址: {TENCENT_BASE_URL}")
    
    # 解析模型名称
    model = AVAILABLE_MODELS.get(args.model.lower(), args.model)
    print(f"  使用模型: {model}")
    
    if args.prompt:
        # 使用自定义提示词
        print("\n【生成文本】")
        print(f"\n  [用户] {args.prompt}")
        
        try:
            response = generate_with_tencent(
                prompt=args.prompt,
                model=model,
                system_prompt=args.system,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                use_thinking=args.thinking,
            )
            print(f"  [模型] {response}")
        except Exception as e:
            print(f"\n❌ API 调用失败: {e}")
    else:
        # 运行演示
        print("\n【运行演示】")
        try:
            demo_basic_generation(model)
            demo_with_system_prompt(model)
            demo_code_generation(model)
            
            # 如果使用 DeepSeek-V3.1，演示思维链
            if model == "DeepSeek-V3.1" or args.thinking:
                demo_thinking_mode()
                
        except Exception as e:
            print(f"\n❌ API 调用失败: {e}")
            print("请检查:")
            print("  1. API Key 是否正确")
            print("  2. 网络是否可达")
            print("  3. 模型名称是否正确")
    
    # 总结
    print_summary()
    
    print("\n" + "=" * 60)
    print("  运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
