"""
Part 4: Few-shot 分类 - SetFit
使用 SetFit 进行少样本分类（每类仅 16 个样本）

SetFit 原理:
1. 从少量样本生成句子对 (正/负样本对)
2. 使用对比学习 (Contrastive Learning) 微调 Sentence Transformer
3. 训练一个简单分类头 (如 LogisticRegression)

优势:
- 只需每类 8-16 个样本
- 无需 prompt 工程
- 训练快速（对比全参数微调）
- 效果接近全监督

依赖: pip install setfit
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import torch
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


print("=" * 60)
print("Part 4: Few-shot 分类 (SetFit)")
print("=" * 60)

# ============================================================
# Step 1: 加载数据并采样
# ============================================================
print("\n" + "-" * 60)
print("Step 1: 加载数据并采样 (模拟 few-shot)")
print("-" * 60)

tomatoes = load_dataset("rotten_tomatoes")
train_data = tomatoes["train"]
test_data = tomatoes["test"]
print(f"完整训练集: {len(train_data)} 条")

from setfit import sample_dataset

# 每类采样 16 个样本
sampled_train_data = sample_dataset(tomatoes["train"], num_samples=16)
print(f"Few-shot 训练样本: {len(sampled_train_data)} (每类 16 个)")
print(f"测试集: {len(test_data)} 条 (完整)")

# 展示采样数据
print("\n采样数据示例:")
for i in range(3):
    text = sampled_train_data[i]["text"]
    label = sampled_train_data[i]["label"]
    sentiment = "正面" if label == 1 else "负面"
    print(f"  [{sentiment}] {text[:70]}...")

# ============================================================
# Step 2: 加载预训练 Sentence Transformer
# ============================================================
print("\n" + "-" * 60)
print("Step 2: 加载 SetFit 模型")
print("-" * 60)

from setfit import SetFitModel

model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
print("模型: sentence-transformers/all-mpnet-base-v2")
print(f"分类头类型: {type(model.model_head).__name__}")

# ============================================================
# Step 3: 训练
# ============================================================
print("\n" + "-" * 60)
print("Step 3: 训练 (对比学习 + 分类头)")
print("-" * 60)

from setfit import TrainingArguments as SetFitTrainingArguments
from setfit import Trainer as SetFitTrainer

args = SetFitTrainingArguments(
    num_epochs=3,       # 对比学习 epochs
    num_iterations=20,  # 每类生成的句子对数量
)
args.eval_strategy = args.evaluation_strategy

trainer = SetFitTrainer(
    model=model,
    args=args,
    train_dataset=sampled_train_data,
    eval_dataset=test_data,
    metric="f1"
)

print("训练参数:")
print(f"  对比学习 epochs: 3")
print(f"  每类句子对数: 20")
print(f"  训练样本: {len(sampled_train_data)}")

print("\n开始训练...")
trainer.train()

# ============================================================
# Step 4: 评估
# ============================================================
print("\n" + "-" * 60)
print("Step 4: 评估结果")
print("-" * 60)

results = trainer.evaluate()
print(f"  F1 Score: {results['f1']:.4f}")

# SetFit 使用 LogisticRegression 作为分类头
print(f"\n分类头: {model.model_head}")

clear_memory()

# ============================================================
# 总结
# ============================================================
print("\n" + "-" * 60)
print("SetFit Few-shot 分类总结")
print("-" * 60)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  SetFit 工作流程                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 句子对生成                                          │
│    从 16 个样本 × 20 iterations = 320 个句子对               │
│    正样本对: 同类别的两个句子                                 │
│    负样本对: 不同类别的两个句子                               │
│                                                             │
│  Step 2: 对比学习微调 Sentence Transformer                   │
│    拉近同类句子的 embedding                                  │
│    推远不同类句子的 embedding                                │
│                                                             │
│  Step 3: 训练分类头                                          │
│    用微调后的 embedding + LogisticRegression                 │
│    简单高效                                                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  关键优势:                                                   │
│  • 每类仅 16 个样本，F1 就能达到不错效果                      │
│  • 无需 GPU (小模型)                                         │
│  • 无需 prompt engineering                                  │
│  • 训练速度快 (对比全参数微调)                                │
└─────────────────────────────────────────────────────────────┘
""")

print("=" * 60)
print("Part 4 完成!")
print("=" * 60)
