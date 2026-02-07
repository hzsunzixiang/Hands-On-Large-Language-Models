"""
Chapter 4 - Part 2: 使用任务特定模型进行情感分类
使用 Twitter RoBERTa 模型，该模型专门为 Twitter 情感分析训练
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
    
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        top_k=None,  # 返回所有类别的分数
        device=device
    )
    
    # 运行推理
    print("\n在测试集上运行推理...")
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        # 模型输出: negative, neutral, positive
        # 我们只关心 negative (index 0) 和 positive (index 2)
        scores = {item['label']: item['score'] for item in output}
        negative_score = scores.get('negative', 0)
        positive_score = scores.get('positive', 0)
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
    # Twitter-RoBERTa 在 MPS 上可能有问题，使用 CPU
    if device == "mps":
        device = "cpu"
    
    # 运行分类
    y_pred = task_specific_classification(data, device=device)
    
    return y_pred


if __name__ == "__main__":
    main()
