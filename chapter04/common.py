"""
Chapter 4 - 共用工具函数
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import classification_report


def evaluate_performance(y_true, y_pred):
    """创建并打印分类报告"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)
    return performance


def get_device():
    """检测并返回可用的计算设备"""
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"使用设备: {device}")
    return device


def load_data():
    """加载 Rotten Tomatoes 数据集"""
    from datasets import load_dataset
    
    print("=" * 60)
    print("加载数据: Rotten Tomatoes 数据集")
    print("=" * 60)
    
    data = load_dataset("rotten_tomatoes")
    print(f"\n数据集结构:")
    print(data)
    
    print(f"\n样本示例:")
    print(f"训练集第一条: {data['train'][0]}")
    print(f"训练集最后一条: {data['train'][-1]}")
    
    return data
