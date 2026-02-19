"""
Part 5: 掩码语言模型 (MLM) 领域适应
在领域数据上继续 MLM 预训练，让 BERT 学习领域特定的语言模式

原理:
1. BERT 预训练时使用了通用语料 (Wikipedia, BookCorpus)
2. 在领域数据上继续 MLM 预训练 = 领域适应 (Domain Adaptation)
3. 模型学习到领域特定的词汇关联
4. 再用于下游任务效果更好

流程:
  预训练 BERT → MLM 领域适应 (电影评论) → 下游任务微调
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import torch
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline
)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


print("=" * 60)
print("Part 5: 掩码语言模型 (MLM) 领域适应")
print("=" * 60)

# ============================================================
# Step 1: 加载数据
# ============================================================
print("\n" + "-" * 60)
print("Step 1: 加载数据")
print("-" * 60)

tomatoes = load_dataset("rotten_tomatoes")
train_data = tomatoes["train"]
test_data = tomatoes["test"]
print(f"训练集: {len(train_data)} 条电影评论")

# ============================================================
# Step 2: 加载 MLM 模型
# ============================================================
print("\n" + "-" * 60)
print("Step 2: 加载 MLM 模型")
print("-" * 60)

model_id = "bert-base-cased"
print(f"模型: {model_id}")

model = AutoModelForMaskedLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"模型类型: {type(model).__name__}")
print("注意: AutoModelForMaskedLM (不是 ForSequenceClassification)")

# ============================================================
# Step 3: Tokenize 数据
# ============================================================
print("\n" + "-" * 60)
print("Step 3: Tokenize 数据")
print("-" * 60)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Tokenize 并移除 label 列 (MLM 不需要分类标签)
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_train = tokenized_train.remove_columns("label")
tokenized_test = test_data.map(preprocess_function, batched=True)
tokenized_test = tokenized_test.remove_columns("label")

print("已移除 label 列 (MLM 是自监督任务，不需要分类标签)")

# ============================================================
# Step 4: MLM Data Collator
# ============================================================
print("\n" + "-" * 60)
print("Step 4: 设置 MLM Data Collator")
print("-" * 60)

# 随机 mask 15% 的 token
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

print("MLM 策略: 随机 mask 15% 的 token")
print("模型任务: 预测被 mask 的原始 token")

# 展示 mask 效果
example_text = train_data[0]["text"]
example_tokens = tokenizer(example_text, return_tensors="pt")
# collator 需要 list of dict
collated = data_collator([{k: v.squeeze() for k, v in example_tokens.items()}])
masked_ids = collated["input_ids"][0]
labels = collated["labels"][0]

masked_count = (labels != -100).sum().item()
total_count = (masked_ids != tokenizer.pad_token_id).sum().item()
print(f"\n示例: '{example_text[:50]}...'")
print(f"  总 token 数: {total_count}")
print(f"  被 mask 的: {masked_count} ({100 * masked_count / total_count:.1f}%)")

# ============================================================
# Step 5: MLM 训练
# ============================================================
print("\n" + "-" * 60)
print("Step 5: MLM 领域适应训练")
print("-" * 60)

training_args = TrainingArguments(
    "output_mlm",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,  # MLM 需要更多 epochs
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("训练参数:")
print(f"  epochs: 10 (MLM 需要更多轮次)")
print(f"  learning_rate: 2e-5")
print(f"  batch_size: 16")

# 先保存 tokenizer
tokenizer.save_pretrained("mlm_movie")

print("\n开始 MLM 训练...")
trainer.train()

# 保存模型
model.save_pretrained("mlm_movie")
print("模型已保存到 mlm_movie/")

# ============================================================
# Step 6: 对比 [MASK] 预测效果
# ============================================================
print("\n" + "-" * 60)
print("Step 6: 对比 [MASK] 预测效果")
print("-" * 60)

test_sentence = "What a horrible [MASK]!"
print(f"\n输入: '{test_sentence}'\n")

# 原始 BERT
print("【原始 BERT (通用预训练)】:")
mask_filler_original = pipeline("fill-mask", model="bert-base-cased")
preds_original = mask_filler_original(test_sentence)
for pred in preds_original:
    print(f"  >>> {pred['sequence']:40s}  (score: {pred['score']:.4f})")

# 领域适应后的 BERT
print("\n【领域适应后 (电影评论 MLM)】:")
mask_filler_domain = pipeline("fill-mask", model="mlm_movie")
preds_domain = mask_filler_domain(test_sentence)
for pred in preds_domain:
    print(f"  >>> {pred['sequence']:40s}  (score: {pred['score']:.4f})")

clear_memory()

# ============================================================
# 总结
# ============================================================
print("\n" + "-" * 60)
print("MLM 领域适应总结")
print("-" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│  MLM 领域适应流程                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  通用预训练 BERT (Wikipedia + BookCorpus)                    │
│          ↓                                                  │
│  MLM 领域适应 (电影评论数据，10 epochs)                      │
│          ↓                                                  │
│  模型学到: "horrible [MASK]" → "movie" / "film"              │
│  而不是通用的: "horrible [MASK]" → "thing" / "man"           │
│          ↓                                                  │
│  用于下游任务 (情感分类) 效果更好                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  关键洞见:                                                   │
│  • MLM 是自监督任务，不需要标注数据                           │
│  • 领域数据量越大，适应效果越好                               │
│  • 这是一种有效的迁移学习策略                                │
│  • 也可以用 Whole Word Masking 代替随机 token masking        │
└─────────────────────────────────────────────────────────────┘
""")

print("=" * 60)
print("Part 5 完成!")
print("=" * 60)
