"""
Chapter 6 - 公共模块
提供模型加载、设备检测和资源清理等通用功能
"""

import warnings
warnings.filterwarnings("ignore")

import os
import gc
import torch


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


def load_model(device=None, verbose=True):
    """
    加载 Phi-3-mini-4k-instruct 模型
    支持离线模式和镜像源
    
    Args:
        device: 指定设备，默认自动检测
        verbose: 是否打印详细信息
    
    Returns:
        (pipe, tokenizer, model, device) 元组
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    if verbose:
        print("\n" + "=" * 60)
        print("加载模型")
        print("=" * 60)
    
    if device is None:
        device = get_device()
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    if verbose:
        print(f"\n加载 {model_name} 模型...")
    
    # 检查是否有本地缓存，自动启用离线模式
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, "models--microsoft--Phi-3-mini-4k-instruct")
    
    # 设置加载参数
    # 注意: 新版 transformers 使用 dtype 替代 torch_dtype
    load_kwargs = {
        "dtype": "auto",
        "trust_remote_code": False,
    }
    
    # 如果本地有缓存，优先使用离线模式
    if os.path.exists(model_cache):
        if verbose:
            print("  ✓ 检测到本地缓存，使用离线模式")
        load_kwargs["local_files_only"] = True
    else:
        # 没有本地缓存，尝试使用镜像源
        if verbose:
            print("  → 本地无缓存，尝试从网络下载...")
        if os.environ.get("HF_ENDPOINT"):
            if verbose:
                print(f"  → 使用镜像源: {os.environ.get('HF_ENDPOINT')}")
    
    # 根据设备选择配置
    if device == "cuda":
        load_kwargs["device_map"] = "cuda"
    elif device == "mps":
        load_kwargs["device_map"] = "mps"
    else:
        load_kwargs["device_map"] = "cpu"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            local_files_only=load_kwargs.get("local_files_only", False)
        )
    except Exception as e:
        print(f"\n  ✗ 加载失败: {e}")
        print("\n  解决方案:")
        print("  1. 从其他机器拷贝模型缓存:")
        print(f"     rsync -avP ~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct user@本机:~/.cache/huggingface/hub/")
        print("  2. 或设置镜像源后重试:")
        print("     export HF_ENDPOINT=https://hf-mirror.com")
        raise
    
    # 创建 pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False,
    )
    
    if verbose:
        print("模型加载完成!")
    return pipe, tokenizer, model, device


def cleanup(model=None, tokenizer=None, pipe=None, verbose=True):
    """
    清理 GPU 内存
    
    Args:
        model: 模型对象
        tokenizer: 分词器对象
        pipe: pipeline 对象
        verbose: 是否打印详细信息
    """
    if verbose:
        print("\n" + "=" * 60)
        print("清理资源")
        print("=" * 60)
    
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if pipe is not None:
        del pipe
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print("CUDA 缓存已清理")
    elif torch.backends.mps.is_available():
        if verbose:
            print("MPS 资源已释放")
    else:
        if verbose:
            print("CPU 资源已释放")


def print_section(title, level=1):
    """打印章节标题"""
    if level == 1:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
    else:
        print(f"\n{title}")
        print("-" * 40)
