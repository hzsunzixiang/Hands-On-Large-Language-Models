"""
Part 3: 冻结层策略
探索冻结不同 BERT 层对分类效果的影响

BERT 层结构 (bert-base-cased):
  - embeddings (word, position, token_type, LayerNorm)
  - encoder.layer.0 ~ encoder.layer.11 (12个 Transformer 块)
  - pooler
  - classifier (分类头，随机初始化)

策略:
  实验1: 只训练分类头 (冻结所有 BERT 层) - 最快但效果一般
  实验2: 冻结 layer 0-9，训练 layer 10-11 + 分类头 - 平衡方案
  实验3: [BONUS] 逐层冻结实验 - 观察各层对性能的影响
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
print("Part 3: 冻结层策略实验")
print("=" * 60)

device = get_device()
print(f"\n使用设备: {device}")

# ============================================================
# 准备数据 (与 Part 2 相同)
# ============================================================
print("\n" + "-" * 60)
print("准备数据")
print("-" * 60)

model_id = "bert-base-cased"
tomatoes = load_dataset("rotten_tomatoes")
train_data = tomatoes["train"]
test_data = tomatoes["test"]

tokenizer = AutoTokenizer.from_pretrained(model_id)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)
print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    load_f1 = evaluate.load("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"f1": f1}


training_args = TrainingArguments(
    "output_freeze",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
)

# ============================================================
# 实验1: 只训练分类头 (冻结所有 BERT 层)
# ============================================================
print("\n" + "-" * 60)
print("实验1: 只训练分类头 (冻结所有 BERT 层)")
print("-" * 60)

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# 冻结除 classifier 外的所有参数
for name, param in model.named_parameters():
    if name.startswith("classifier"):
        param.requires_grad = True
    else:
        param.requires_grad = False

# 检查冻结效果
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# 打印参数冻结状态
print("\n参数冻结状态 (前5个 + classifier):")
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 5 or name.startswith("classifier"):
        status = "可训练" if param.requires_grad else "已冻结"
        print(f"  {name:50s} [{status}]")
    elif i == 5:
        print(f"  ... (省略中间层) ...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n开始训练 (只训练分类头)...")
trainer.train()
results1 = trainer.evaluate()
print(f"  F1 Score: {results1['eval_f1']:.4f}")

clear_memory()

# ============================================================
# 实验2: 冻结 layer 0-9，训练 layer 10-11 + 分类头
# ============================================================
print("\n" + "-" * 60)
print("实验2: 冻结 layer 0-9，训练 layer 10-11 + 分类头")
print("-" * 60)

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# 打印参数索引，找到 layer 10 的起始位置
print("\n参数索引映射 (layer 10 附近):")
for index, (name, param) in enumerate(model.named_parameters()):
    if 160 <= index <= 170 or name.startswith("classifier"):
        print(f"  index={index:3d}: {name}")

# Encoder block 10 starts at index 165
# 冻结 index < 165 的所有参数
for index, (name, param) in enumerate(model.named_parameters()):
    if index < 165:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n开始训练 (layer 10-11 + 分类头)...")
trainer.train()
results2 = trainer.evaluate()
print(f"  F1 Score: {results2['eval_f1']:.4f}")

clear_memory()

# ============================================================
# [BONUS] 逐层冻结实验 (可选，耗时较长)
# ============================================================
print("\n" + "-" * 60)
print("[BONUS] 逐层冻结实验")
print("-" * 60)

RUN_BONUS = False  # 设为 True 运行完整实验（耗时较长）

if RUN_BONUS:
    scores = []
    for freeze_up_to in range(12):
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 冻结 encoder blocks 0 ~ freeze_up_to
        for name, param in model.named_parameters():
            if "layer" in name:
                layer_nr = int(name.split("layer")[1].split(".")[1])
                if layer_nr <= freeze_up_to:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        score = trainer.evaluate()["eval_f1"]
        scores.append(score)
        print(f"  冻结 layer 0-{freeze_up_to:2d}: F1 = {score:.4f}")
        clear_memory()

    print("\n逐层冻结结果:")
    for i, s in enumerate(scores):
        bar = "█" * int(s * 50)
        print(f"  冻结 0-{i:2d}: {s:.4f} {bar}")
else:
    # 使用书中预计算的结果展示
    print("\n(使用书中预计算结果，设 RUN_BONUS=True 可自行运行)")
    precomputed = [
        0.8542, 0.8526, 0.8515, 0.8507, 0.8398,
        0.8391, 0.8377, 0.8434, 0.8259, 0.8162,
        0.7917, 0.7019
    ]
    print("\n冻结不同层的 F1 Score:")
    for i, s in enumerate(precomputed):
        frozen = f"0-{i}" if i > 0 else "None"
        bar = "█" * int(s * 50)
        print(f"  冻结 {frozen:5s}: {s:.4f} {bar}")

# ============================================================
# 实验总结
# ============================================================
print("\n" + "-" * 60)
print("实验总结")
print("-" * 60)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  策略                            │  可训练参数  │  F1 Score     │
├─────────────────────────────────────────────────────────────────┤
│  只训练分类头                    │  {trainable:>8,}    │  {results1['eval_f1']:.4f}        │
│  冻结 layer 0-9 (训练 10-11)     │  较多        │  {results2['eval_f1']:.4f}        │
│  全参数微调 (见 Part 2)          │  全部        │  (见 Part 2)  │
├─────────────────────────────────────────────────────────────────┤
│  关键洞见:                                                      │
│  • 底层学通用语法特征，冻结影响小                                │
│  • 顶层学任务相关语义，冻结影响大                                │
│  • 冻结前几层是性价比最高的策略                                  │
└─────────────────────────────────────────────────────────────────┘
""")

print("=" * 60)
print("Part 3 完成!")
print("=" * 60)
