"""
Chapter 12 - Part 1: SFT 数据预处理
从 HuggingFace 加载 ultrachat_200k 数据集，使用 TinyLlama 的 chat template 格式化 prompt。
"""

from transformers import AutoTokenizer
from datasets import load_dataset


# Load a tokenizer to use its chat template
template_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def format_prompt(example):
    """Format the prompt to using the <|user|> template TinyLLama is using"""
    chat = example["messages"]
    prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}


# Load and format the data using the template TinyLLama is using
dataset = (
    load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    .shuffle(seed=42)
    .select(range(3_000))
)
dataset = dataset.map(format_prompt)

# Example of formatted prompt
print("=" * 60)
print("SFT 数据样例 (index=2576):")
print("=" * 60)
print(dataset["text"][2576])
print(f"\n数据集大小: {len(dataset)} 条")
print(f"字段: {dataset.column_names}")
