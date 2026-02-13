"""
Chapter 7 - 公共模块
提供设备检测、模型加载等通用功能

本章使用 llama.cpp (GGUF 量化模型) + LangChain 框架
"""

import warnings
warnings.filterwarnings("ignore")

import os
import torch


def get_device():
    """
    自动检测最佳可用设备
    返回 (device, n_gpu_layers) 元组
    
    - CUDA: n_gpu_layers=-1 (全部层卸载到 GPU)
    - MPS (Apple Silicon): n_gpu_layers=-1 (使用 Metal 加速)
    - CPU: n_gpu_layers=0
    """
    if torch.cuda.is_available():
        device = "cuda"
        n_gpu_layers = -1
        print(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        n_gpu_layers = -1  # llama.cpp 支持 Metal
        print("使用设备: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        n_gpu_layers = 0
        print("使用设备: CPU")
    return device, n_gpu_layers


def load_llm(model_path=None, n_gpu_layers=-1, verbose=True):
    """
    使用 LangChain + llama.cpp 加载 LLM
    
    Args:
        model_path: GGUF 模型路径，为 None 时自动从 HuggingFace 下载
        n_gpu_layers: GPU 层数，-1 表示全部
        verbose: 是否打印详细信息
    
    Returns:
        LangChain LlamaCpp LLM 实例
    """
    try:
        from langchain_community.llms import LlamaCpp
    except ImportError:
        from langchain.llms import LlamaCpp
    
    if verbose:
        print("\n" + "=" * 60)
        print("加载 LLM (LangChain + llama.cpp)")
        print("=" * 60)
    
    if model_path is None:
        # 使用 HuggingFace 自动下载
        from llama_cpp import Llama
        llm_raw = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False
        )
        model_path = llm_raw.model_path
        del llm_raw
    
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        max_tokens=500,
        n_ctx=2048,
        seed=42,
        verbose=False
    )
    
    if verbose:
        print("模型加载完成!")
    
    return llm


# Phi-3 的 Chat Template
PHI3_TEMPLATE = """<s><|user|>
    {input_prompt}<|end|>
    <|assistant|>"""

PHI3_TEMPLATE_WITH_HISTORY = """<s><|user|>Current conversation:{chat_history}

    {input_prompt}<|end|>
    <|assistant|>"""


def print_section(title, level=1):
    """打印章节标题"""
    if level == 1:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
    else:
        print(f"\n{title}")
        print("-" * 40)
