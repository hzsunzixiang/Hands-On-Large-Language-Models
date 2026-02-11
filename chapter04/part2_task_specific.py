"""
Chapter 4 - Part 2: 使用任务特定模型进行情感分类
使用 Twitter RoBERTa 模型，该模型专门为 Twitter 情感分析训练
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common import load_data, get_device, evaluate_performance


def task_specific_classification(data, device="cpu"):
    """
    使用任务特定模型 (Twitter RoBERTa) 进行情感分类
    """
    print("\n" + "=" * 60)
    print("Part 2: 使用任务特定模型 (Twitter-RoBERTa)")
    print("=" * 60)
    
    # 加载预训练的情感分析模型
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"\n加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # 模型标签映射: 0=negative, 1=neutral, 2=positive
    print(f"模型标签: {model.config.id2label}")
    
    # 运行推理
    print("\n在测试集上运行推理...")
    y_pred = []
    
    with torch.no_grad():
        for text in tqdm(data["test"]["text"], total=len(data["test"])):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推理
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]
            
            # 模型输出: 0=negative, 1=neutral, 2=positive
            # 我们只关心 negative (index 0) 和 positive (index 2)
            negative_score = scores[0].item()
            positive_score = scores[2].item()
            
            # 0=负面, 1=正面 (与数据集标签对应)
            assignment = 0 if negative_score > positive_score else 1
            y_pred.append(assignment)
    
    print("\n分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return y_pred


def main():
    # 加载数据
    data = load_data()
    
    # 获取设备
    device = get_device()
    # Twitter-RoBERTa 在 MPS 上可能有问题，使用 CPU
    if device == "mps":
        print("注意: 在 MPS 设备上可能有兼容性问题，使用 CPU")
        device = "cpu"
    
    # 运行分类
    y_pred = task_specific_classification(data, device=device)
    
    return y_pred


if __name__ == "__main__":
    main()
