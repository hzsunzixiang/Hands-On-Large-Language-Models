"""
Chapter 5 - 共用工具函数
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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


def load_data(sample_size=5000):
    """加载 ArXiv NLP 论文数据集"""
    from datasets import load_dataset
    
    print("=" * 60)
    print("加载数据: ArXiv NLP 论文")
    print("=" * 60)
    
    dataset = load_dataset("maartengr/arxiv_nlp")["train"]
    
    # 提取摘要和标题
    abstracts = list(dataset["Abstracts"])
    titles = list(dataset["Titles"])
    
    # 采样
    if sample_size and sample_size < len(abstracts):
        abstracts = abstracts[:sample_size]
        titles = titles[:sample_size]
        print(f"\n使用样本数据: {sample_size} 条 (共 {len(dataset)} 条)")
    else:
        print(f"\n论文数量: {len(abstracts)}")
    
    print(f"\n示例标题: {titles[0]}")
    print(f"\n示例摘要 (前200字): {abstracts[0][:200]}...")
    
    return abstracts, titles


def topic_differences(model, original_topics, nr_topics=5):
    """展示主题表示更新前后的差异"""
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    return df
