"""
Chapter 4 - Part 2: 使用任务特定模型进行情感分类
使用 Twitter RoBERTa 模型，该模型专门为 Twitter 情感分析训练
参考: Hands-On Large Language Models - Chapter 4
"""

import numpy as np
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

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
    
    # 使用 pipeline 加载模型
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        top_k=None,  # 返回所有类别的分数 (替代已废弃的 return_all_scores=True)
        device=device
    )
    
    # 模型标签: 0=negative, 1=neutral, 2=positive
    print(f"模型标签: negative, neutral, positive")
    
    # 运行推理
    print("\n在测试集上运行推理...")
    y_pred = []
    
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        # output 是一个列表，包含每个类别的分数
        # 找到 negative 和 positive 的分数
        scores = {item["label"]: item["score"] for item in output}
        negative_score = scores.get("negative", 0)
        positive_score = scores.get("positive", 0)
        
        # 0=负面, 1=正面 (与数据集标签对应)
        assignment = np.argmax([negative_score, positive_score])
        y_pred.append(assignment)
    
    print("\n分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return y_pred


def main():
    # 加载数据
    data = load_data()
    
    # 获取设备
    device = get_device()
    
    # 运行分类
    y_pred = task_specific_classification(data, device=device)
    
    return y_pred


if __name__ == "__main__":
    main()
