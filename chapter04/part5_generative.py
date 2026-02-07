"""
Chapter 4 - Part 5: 使用生成模型 (Flan-T5) 进行分类
"""

from tqdm import tqdm
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset

from common import load_data, get_device, evaluate_performance


def generative_classification(data, device="cpu"):
    """
    使用生成模型 (Flan-T5) 进行分类
    """
    print("\n" + "=" * 60)
    print("Part 5: 生成模型分类 (Flan-T5)")
    print("=" * 60)
    
    # 加载 Flan-T5 模型 (T5 是 seq2seq 模型，使用 text2text-generation)
    model_name = "google/flan-t5-small"
    print(f"\n加载模型: {model_name}")
    
    # 手动加载模型和 tokenizer 以避免警告
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    
    # 准备数据: 添加提示词
    prompt = "Is the following sentence positive or negative? "
    data_with_prompt = data.map(lambda example: {"t5": prompt + example['text']})
    
    print(f"\n提示词模板: '{prompt}'")
    print(f"示例输入: {data_with_prompt['test'][0]['t5'][:100]}...")
    
    # 运行推理
    print("\n在测试集上运行推理...")
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data_with_prompt["test"], "t5")), total=len(data_with_prompt["test"])):
        text = output[0]["generated_text"].lower()
        y_pred.append(0 if "negative" in text else 1)
    
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
