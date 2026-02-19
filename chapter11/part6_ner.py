"""
Part 6: 命名实体识别 (NER)
使用 BERT 进行 Token Classification

数据集: CoNLL-2003
标签体系: BIO 格式
  - B-XXX: 实体开始 (Begin)
  - I-XXX: 实体内部 (Inside)
  - O: 非实体 (Outside)

实体类型: PER (人名), ORG (组织), LOC (地点), MISC (其他)

关键挑战: 子词标签对齐
  原始: "Maarten" → 标签 B-PER
  WordPiece: ["Ma", "##arte", "##n"]
  对齐后:   [B-PER, I-PER, I-PER]

  特殊 token ([CLS], [SEP]) 标签设为 -100 (训练时忽略)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

_start_time = time.perf_counter()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


print("=" * 60)
print("Part 6: 命名实体识别 (NER)")
print("=" * 60)

# ============================================================
# Step 1: 加载 CoNLL-2003 数据集
# ============================================================
print("\n" + "-" * 60)
print("Step 1: 加载 CoNLL-2003 数据集")
print("-" * 60)

use_wnut = False
try:
    # 方案1: 尝试社区维护版本
    dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
    print("使用 eriktks/conll2003 数据集")
except Exception as e1:
    try:
        # 方案2: 使用 HuggingFace 镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        dataset = load_dataset("conll2003", trust_remote_code=True)
        print("使用 HuggingFace 镜像加载 conll2003")
    except Exception as e2:
        # 方案3: 使用 wnut_17 作为替代
        print(f"CoNLL-2003 加载失败，使用替代数据集: wnut_17")
        dataset = load_dataset("wnut_17", trust_remote_code=True)
        use_wnut = True

print(f"训练集: {len(dataset['train'])} 条")
print(f"测试集: {len(dataset['test'])} 条")

# ============================================================
# Step 2: 标签映射
# ============================================================
print("\n" + "-" * 60)
print("Step 2: 标签体系")
print("-" * 60)

if use_wnut:
    label_names = dataset["train"].features["ner_tags"].feature.names
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for i, name in enumerate(label_names)}
    print(f"WNUT-17 标签: {label_names}")
else:
    label2id = {
        'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
        'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
    }
    id2label = {v: k for k, v in label2id.items()}
    print(f"CoNLL-2003 标签: {list(label2id.keys())}")

print(f"\nBIO 标注规则:")
print(f"  B-XXX: 实体开始 (Begin)")
print(f"  I-XXX: 实体内部 (Inside)")
print(f"  O:     非实体 (Outside)")

# 展示样本
example_idx = min(848, len(dataset["train"]) - 1)
example = dataset["train"][example_idx]
print(f"\n示例 (index={example_idx}):")
print(f"  Tokens:   {example['tokens']}")
print(f"  NER Tags: {[id2label[t] for t in example['ner_tags']]}")

# ============================================================
# Step 3: 加载模型
# ============================================================
print("\n" + "-" * 60)
print("Step 3: 加载模型")
print("-" * 60)

model_id = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
print(f"模型: {model_id}")
print(f"分类标签数: {len(id2label)}")
print(f"模型类型: AutoModelForTokenClassification")

# ============================================================
# Step 4: 子词标签对齐 (核心难点)
# ============================================================
print("\n" + "-" * 60)
print("Step 4: 子词标签对齐 (核心难点)")
print("-" * 60)

# 演示子词切分问题
print("\n子词切分示例:")
demo_tokens = example["tokens"]
token_ids = tokenizer(demo_tokens, is_split_into_words=True)["input_ids"]
sub_tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(f"  原始 tokens: {demo_tokens}")
print(f"  子词 tokens: {sub_tokens}")
print(f"  原始长度: {len(demo_tokens)}, 子词长度: {len(sub_tokens)}")


def align_labels(examples):
    """
    对齐子词和标签

    规则:
    1. 每个词的第一个子词: 使用原始标签
    2. 后续子词: B-XXX → I-XXX (实体内部)
    3. 特殊 token ([CLS], [SEP]): 标签 = -100 (忽略)
    """
    token_ids = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = examples["ner_tags"]

    updated_labels = []
    for index, label in enumerate(labels):
        # word_ids() 映射每个子词到它的原始词索引
        word_ids = token_ids.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # 新词的第一个子词
            if word_idx != previous_word_idx:
                previous_word_idx = word_idx
                # 特殊 token (None) → -100
                updated_label = -100 if word_idx is None else label[word_idx]
                label_ids.append(updated_label)

            # 特殊 token
            elif word_idx is None:
                label_ids.append(-100)

            # 后续子词: B-XXX → I-XXX
            else:
                updated_label = label[word_idx]
                if updated_label % 2 == 1:  # B-XXX (奇数)
                    updated_label += 1      # → I-XXX (偶数)
                label_ids.append(updated_label)

        updated_labels.append(label_ids)

    token_ids["labels"] = updated_labels
    return token_ids


print("\n对齐标签中...")
tokenized = dataset.map(align_labels, batched=True)

# 展示对齐效果
print(f"\n对齐前后对比 (index={example_idx}):")
print(f"  原始标签: {example['ner_tags']}")
print(f"  对齐标签: {tokenized['train'][example_idx]['labels']}")
print(f"  (-100 表示特殊 token，训练时忽略)")

# ============================================================
# Step 5: 训练
# ============================================================
print("\n" + "-" * 60)
print("Step 5: 训练 NER 模型")
print("-" * 60)

# 评估指标: seqeval (NER 专用)
seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        for token_pred, token_label in zip(prediction, label):
            # 忽略特殊 token
            if token_label != -100:
                true_predictions.append([id2label[token_pred]])
                true_labels.append([id2label[token_label]])

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {"f1": results["overall_f1"]}


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    "output_ner",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
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

# ============================================================
# Step 7: 推理示例
# ============================================================
print("\n" + "-" * 60)
print("Step 7: 推理示例")
print("-" * 60)

trainer.save_model("ner_model_final")
token_classifier = pipeline(
    "token-classification",
    model="ner_model_final",
)

test_sentences = [
    "My name is Maarten and I work at Hugging Face in New York.",
    "Apple was founded by Steve Jobs in Cupertino, California.",
    "The United Nations headquarters is located in Manhattan.",
]

for sentence in test_sentences:
    print(f"\n输入: {sentence}")
    preds = token_classifier(sentence)
    if preds:
        print("识别的实体:")
        for p in preds:
            print(f"  {p['word']:15s} → {p['entity']:8s} (score: {p['score']:.3f})")
    else:
        print("  (未识别到实体)")

clear_memory()

# ============================================================
# 总结
# ============================================================
print("\n" + "-" * 60)
print("NER 总结")
print("-" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│  NER 核心流程                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: "My name is Maarten ."                               │
│        ↓                                                    │
│  WordPiece: [CLS] My name is Ma ##arte ##n . [SEP]          │
│        ↓                                                    │
│  BERT Encoder: 每个子词得到一个上下文向量                     │
│        ↓                                                    │
│  分类头: 每个子词预测 BIO 标签                               │
│        ↓                                                    │
│  标签:  -100  O  O  O  B-PER I-PER I-PER O  -100           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  关键难点: 子词标签对齐                                      │
│  • 一个词可能被拆成多个子词                                  │
│  • 第一个子词用原始标签 (B-XXX)                              │
│  • 后续子词改为 I-XXX                                       │
│  • 特殊 token ([CLS], [SEP]) 用 -100 忽略                   │
│                                                             │
│  与序列分类的区别:                                           │
│  • 序列分类: 取 [CLS] 向量 → 一个标签                       │
│  • Token 分类: 取每个词的向量 → 每个词一个标签               │
└─────────────────────────────────────────────────────────────┘
""")

_elapsed = time.perf_counter() - _start_time
print("=" * 60)
print(f"Part 6 完成! 总耗时: {_elapsed:.2f} 秒 ({_elapsed/60:.1f} 分钟)")
print("=" * 60)
