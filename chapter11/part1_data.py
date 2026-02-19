"""
Part 1: 数据准备
加载 Rotten Tomatoes 电影评论数据集（二分类：正面/负面）

这是后续所有实验的数据基础，但每个 part 文件都可以独立运行。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datasets import load_dataset

print("=" * 60)
print("Part 1: 加载 Rotten Tomatoes 数据集")
print("=" * 60)

# 加载数据
tomatoes = load_dataset("rotten_tomatoes")
train_data = tomatoes["train"]
test_data = tomatoes["test"]

print(f"\n数据集概览:")
print(f"  训练集大小: {len(train_data)}")
print(f"  测试集大小: {len(test_data)}")
print(f"  标签: 0=负面, 1=正面")

# 展示几个样本
print("\n" + "-" * 60)
print("样本展示")
print("-" * 60)

for i in range(3):
    text = train_data[i]["text"]
    label = train_data[i]["label"]
    sentiment = "正面" if label == 1 else "负面"
    print(f"\n  [{sentiment}] {text[:80]}...")

# 统计标签分布
from collections import Counter

train_labels = Counter(train_data["label"])
test_labels = Counter(test_data["label"])

print("\n" + "-" * 60)
print("标签分布")
print("-" * 60)
print(f"  训练集: 正面={train_labels[1]}, 负面={train_labels[0]}")
print(f"  测试集: 正面={test_labels[1]}, 负面={test_labels[0]}")

print("\n" + "=" * 60)
print("Part 1 完成!")
print("=" * 60)
