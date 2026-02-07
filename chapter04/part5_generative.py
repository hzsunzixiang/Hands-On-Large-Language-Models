"""
Chapter 4 - Part 5: 使用生成模型 (Flan-T5) 进行分类
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import load_data, get_device, evaluate_performance


def generative_classification(data, device="cpu"):
    """
    使用生成模型 (Flan-T5) 进行分类
    """
    print("\n" + "=" * 60)
    print("Part 5: 生成模型分类 (Flan-T5)")
    print("=" * 60)
    
    # 加载 Flan-T5 模型
    model_name = "google/flan-t5-small"
    print(f"\n加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # 准备数据: 添加提示词
    prompt = "Is the following sentence positive or negative? "
    
    print(f"\n提示词模板: '{prompt}'")
    print(f"示例输入: {prompt + data['test'][0]['text'][:80]}...")
    
    # 运行推理
    print("\n在测试集上运行推理...")
    y_pred = []
    
    with torch.no_grad():
        for example in tqdm(data["test"], total=len(data["test"])):
            input_text = prompt + example['text']
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, max_new_tokens=10)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            
            y_pred.append(0 if "negative" in generated_text else 1)
    
    print("\nFlan-T5 分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return y_pred


def main():
    # 加载数据
    data = load_data()
    
    # 获取设备
    device = get_device()
    # Flan-T5 在 MPS 上可能有问题，使用 CPU
    if device == "mps":
        device = "cpu"
    
    # 运行分类
    y_pred = generative_classification(data, device=device)
    
    return y_pred


if __name__ == "__main__":
    main()
