"""
Part 2: 监督分类 - 全参数微调
使用 HuggingFace Trainer 对 BERT 进行全参数微调

流程:
1. 加载预训练 BERT + 随机初始化的分类头
2. Tokenize 数据
3. 使用 Trainer API 训练（所有参数都更新）
4. 评估 F1 Score
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import evaluate


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


print("=" * 60)
print("Part 2: 监督分类 - 全参数微调")
print("=" * 60)

device = get_device()
print(f"\n使用设备: {device}")

# ============================================================
# Step 1: 加载数据
# ============================================================
print("\n" + "-" * 60)
print("Step 1: 加载数据")
print("-" * 60)

tomatoes = load_dataset("rotten_tomatoes")
train_data = tomatoes["train"]
test_data = tomatoes["test"]
print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")

# ============================================================
# Step 2: 加载模型和 Tokenizer
# ============================================================
print("\n" + "-" * 60)
print("Step 2: 加载模型和 Tokenizer")
print("-" * 60)

model_id = "bert-base-cased"
print(f"模型: {model_id}")

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 打印模型结构概览
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,} (全部)")

# ============================================================
# Step 3: Tokenize 数据
# ============================================================
print("\n" + "-" * 60)
print("Step 3: Tokenize 数据")
print("-" * 60)


def preprocess_function(examples):
    """Tokenize 输入文本"""
    return tokenizer(examples["text"], truncation=True)


tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

# Data collator - 动态 padding（每个 batch 内 pad 到最长）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 展示 tokenize 效果
example = train_data[0]["text"]
tokens = tokenizer(example)
print(f"原文: {example[:60]}...")
print(f"Token IDs (前10个): {tokens['input_ids'][:10]}...")
print(f"Token 数量: {len(tokens['input_ids'])}")

# ============================================================
# Step 4: 定义评估指标
# ============================================================
print("\n" + "-" * 60)
print("Step 4: 定义评估指标")
print("-" * 60)


def compute_metrics(eval_pred):
    """计算 F1 Score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    load_f1 = evaluate.load("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"f1": f1}


print("评估指标: F1 Score")

# ============================================================
# Step 5: 训练
# ============================================================
print("\n" + "-" * 60)
print("Step 5: 训练 (全参数微调)")
print("-" * 60)

training_args = TrainingArguments(
    "output_supervised",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none",
    fp16=False,  # MPS 不支持 fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("开始训练...")
trainer.train()

# ============================================================
# Step 6: 评估
# ============================================================
print("\n" + "-" * 60)
print("Step 6: 评估结果")
print("-" * 60)

results = trainer.evaluate()
print(f"  F1 Score: {results['eval_f1']:.4f}")
print(f"  Loss: {results['eval_loss']:.4f}")

clear_memory()

print("\n" + "=" * 60)
print("Part 2 完成!")
print("=" * 60)
