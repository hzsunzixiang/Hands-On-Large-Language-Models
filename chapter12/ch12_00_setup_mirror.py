"""
Chapter 12 - Part 0: 环境配置和模型预下载
使用 HuggingFace 镜像加速下载
"""

import os
import sys

# 配置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("=" * 60)
print("Chapter 12 环境配置和模型预下载")
print("=" * 60)
print()

# 检查 Python 版本
print(f"Python 版本: {sys.version}")
print()

# 检查必要的包
print("检查依赖包...")
required_packages = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "huggingface_hub",
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (未安装)")
        missing_packages.append(package)

if missing_packages:
    print()
    print("缺少以下依赖包，请先安装:")
    print(f"  pip install {' '.join(missing_packages)}")
    sys.exit(1)

print()
print("=" * 60)
print("开始预下载模型和数据集...")
print("=" * 60)
print()

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import huggingface_hub

# 设置更长的超时时间
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 600  # 10分钟

models_to_download = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
]

datasets_to_download = [
    ("HuggingFaceH4/ultrachat_200k", "test_sft"),
]

# 1. 下载 Tokenizers
print("1️⃣  下载 Tokenizers...")
for model_name in models_to_download:
    try:
        print(f"   正在下载: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
        )
        print(f"   ✓ {model_name} tokenizer 下载完成")
    except Exception as e:
        print(f"   ✗ {model_name} tokenizer 下载失败: {e}")

print()

# 2. 下载模型
print("2️⃣  下载模型 (这可能需要一些时间)...")
import torch

# 检测设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"   检测到设备: {device}")
print()

for model_name in models_to_download:
    try:
        print(f"   正在下载: {model_name}")
        if device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
        print(f"   ✓ {model_name} 模型下载完成")
        del model  # 释放内存
    except Exception as e:
        print(f"   ✗ {model_name} 模型下载失败: {e}")
        print(f"   提示: 可以稍后重试，或手动从 https://hf-mirror.com/{model_name} 下载")

print()

# 3. 下载数据集
print("3️⃣  下载数据集...")
for dataset_name, split in datasets_to_download:
    try:
        print(f"   正在下载: {dataset_name} (split: {split})")
        dataset = load_dataset(
            dataset_name,
            split=split,
            trust_remote_code=True,
        )
        print(f"   ✓ {dataset_name} 数据集下载完成 ({len(dataset)} 条)")
    except Exception as e:
        print(f"   ✗ {dataset_name} 数据集下载失败: {e}")
        print(f"   提示: 可以稍后重试")

print()
print("=" * 60)
print("✨ 环境配置完成！")
print("=" * 60)
print()
print("现在可以运行训练脚本:")
print("  python ch12_02_sft_train.py")
print()
