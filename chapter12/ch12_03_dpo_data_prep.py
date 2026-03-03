"""
Chapter 12 - Part 3: DPO 数据预处理
从 argilla/distilabel-intel-orca-dpo-pairs 加载偏好数据集，
格式化为 DPO 训练所需的 prompt / chosen / rejected 三元组。
"""

from datasets import load_dataset


def format_prompt(example):
    """Format the prompt to using the <|user|> template TinyLLama is using"""
    system = "<|system|>\n" + example['system'] + "</s>\n"
    prompt = "<|user|>\n" + example['input'] + "</s>\n<|assistant|>\n"
    chosen = example['chosen'] + "</s>\n"
    rejected = example['rejected'] + "</s>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


# Apply formatting to the dataset and select relatively short answers
dpo_dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
dpo_dataset = dpo_dataset.filter(
    lambda r:
        r["status"] != "tie" and
        r["chosen_score"] >= 8 and
        not r["in_gsm8k_train"]
)
dpo_dataset = dpo_dataset.map(format_prompt, remove_columns=dpo_dataset.column_names)

print("=" * 60)
print("DPO 数据集信息:")
print("=" * 60)
print(dpo_dataset)
print(f"\n样本数: {len(dpo_dataset)}")
print(f"字段: {dpo_dataset.column_names}")
print("\n--- prompt 样例 (index=0) ---")
print(dpo_dataset[0]["prompt"][:300] + "...")
print("\n--- chosen 样例 (index=0) ---")
print(dpo_dataset[0]["chosen"][:300] + "...")
print("\n--- rejected 样例 (index=0) ---")
print(dpo_dataset[0]["rejected"][:300] + "...")
