"""10.4 有监督微调 + Augmented SBERT
Part A: 在预训练的 all-MiniLM-L6-v2 上用 MNRL 微调，对比微调前后效果。
Part B: Augmented SBERT — 先训练 cross-encoder 标注 silver 数据，
         再用 gold+silver 训练 bi-encoder，对比 gold-only 的效果。
"""
import gc
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers import (
    InputExample, SentenceTransformer, losses, models,
)
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ============================================================
# 共用评估器 — STS-B
# ============================================================
total_start = time.time()

val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine"
)

# ================================================================
# Part A: 微调预训练 Sentence-Transformer (all-MiniLM-L6-v2)
# ================================================================
print("=" * 60)
print("Part A: 微调 all-MiniLM-L6-v2 (MNRL)")
print("=" * 60)
part_a_start = time.time()

# --- 数据准备 (与 10.3 相同的三元组格式) ---
mnli = load_dataset("glue", "mnli", split="train").select(range(50_000))
mnli = mnli.remove_columns("idx")
mapping = {2: 0, 1: 0, 0: 1}

# 为 Part B 准备 gold 数据 (前 10000 条)
gold_dataset = load_dataset("glue", "mnli", split="train").select(range(10_000))
gold = pd.DataFrame({
    'sentence1': gold_dataset['premise'],
    'sentence2': gold_dataset['hypothesis'],
    'label': [mapping[label] for label in gold_dataset['label']]
})

# Part A 数据: entailment 三元组
import random
mnli_entailment = mnli.filter(lambda x: True if x['label'] == 0 else False)
train_data_a = {"anchor": [], "positive": [], "negative": []}
# 🔧 修复：先转换为 Python 列表，再打乱
soft_negatives = list(mnli_entailment["hypothesis"])  # 转换为 Python 列表
random.shuffle(soft_negatives)                        # 现在可以安全打乱
for row, soft_negative in zip(mnli_entailment, soft_negatives):
    train_data_a["anchor"].append(row["premise"])
    train_data_a["positive"].append(row["hypothesis"])
    train_data_a["negative"].append(soft_negative)
train_dataset_a = Dataset.from_dict(train_data_a)

# --- 微调 ---
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

args = SentenceTransformerTrainingArguments(
    output_dir="finetuned_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=False,  # MPS (Apple Silicon) 不支持 fp16
    eval_steps=100,
    logging_steps=100,
    report_to="none",
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset_a,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

print("\n微调后 STS-B 评估:")
results = evaluator(embedding_model)
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

# --- 对比: 原始预训练模型 ---
print("\n原始 all-MiniLM-L6-v2 STS-B 评估:")
original_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
results_orig = evaluator(original_model)
for k, v in results_orig.items():
    print(f"  {k}: {v:.4f}")

del embedding_model, original_model, trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ================================================================
# Part B: Augmented SBERT
# ================================================================
print("\n" + "=" * 60)
print("Part B: Augmented SBERT")
print("=" * 60)
part_b_start = time.time()

# --- Step 1: 训练 Cross-Encoder ---
print("\nStep 1: 训练 Cross-Encoder (bert-base-uncased, 10k gold)")
gold_examples = [
    InputExample(texts=[row["premise"], row["hypothesis"]], label=mapping[row["label"]])
    for row in tqdm(gold_dataset)
]
gold_dataloader = NoDuplicatesDataLoader(gold_examples, batch_size=32)

cross_encoder = CrossEncoder('bert-base-uncased', num_labels=2)
cross_encoder.fit(
    train_dataloader=gold_dataloader,
    epochs=1,
    show_progress_bar=True,
    warmup_steps=100,
    use_amp=False
)

# --- Step 2: 创建新的句对 ---
print("\nStep 2: 准备 silver 句对 (10k~50k)")
silver_raw = load_dataset("glue", "mnli", split="train").select(range(10_000, 50_000))
pairs = list(zip(silver_raw['premise'], silver_raw['hypothesis']))

# --- Step 3: 用 Cross-Encoder 打标 (silver dataset) ---
print("\nStep 3: Cross-Encoder 打标 silver 数据")
output = cross_encoder.predict(pairs, apply_softmax=True, show_progress_bar=True)
silver = pd.DataFrame({
    "sentence1": silver_raw["premise"],
    "sentence2": silver_raw["hypothesis"],
    "label": np.argmax(output, axis=1)
})
print(f"Silver 数据: {len(silver)} 条")
print(f"  标注为 entailment: {(silver['label'] == 1).sum()}")
print(f"  标注为 non-entailment: {(silver['label'] == 0).sum()}")

# --- Step 4: Gold + Silver 训练 Bi-Encoder ---
print("\nStep 4: Gold + Silver 训练 Bi-Encoder (CosineSimilarityLoss)")
data = pd.concat([gold, silver], ignore_index=True, axis=0)
data = data.drop_duplicates(subset=['sentence1', 'sentence2'], keep="first")
train_dataset_b = Dataset.from_pandas(data, preserve_index=False)
print(f"Gold+Silver 合并后: {len(train_dataset_b)} 条")

embedding_model = SentenceTransformer('bert-base-uncased')
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

args = SentenceTransformerTrainingArguments(
    output_dir="augmented_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=False,  # MPS (Apple Silicon) 不支持 fp16
    eval_steps=100,
    logging_steps=100,
    report_to="none",
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset_b,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

print("\nGold+Silver STS-B 评估:")
results_aug = evaluator(embedding_model)
for k, v in results_aug.items():
    print(f"  {k}: {v:.4f}")

trainer.accelerator.clear()
del embedding_model, trainer

# --- Step 5: 仅 Gold 训练对比 ---
print("\nStep 5: 仅 Gold 训练 (对比)")
data_gold_only = gold.drop_duplicates(subset=['sentence1', 'sentence2'], keep="first")
train_dataset_gold = Dataset.from_pandas(data_gold_only, preserve_index=False)
print(f"Gold-only: {len(train_dataset_gold)} 条")

embedding_model = SentenceTransformer('bert-base-uncased')
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

args = SentenceTransformerTrainingArguments(
    output_dir="gold_only_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=False,  # MPS (Apple Silicon) 不支持 fp16
    eval_steps=100,
    logging_steps=100,
    report_to="none",
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset_gold,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

print("\nGold-only STS-B 评估:")
results_gold = evaluator(embedding_model)
for k, v in results_gold.items():
    print(f"  {k}: {v:.4f}")

print("\n结论: 相比仅用 gold 数据，加入 silver 数据 (Augmented SBERT) 可以提升模型性能!")
part_b_elapsed = time.time() - part_b_start
print(f"\nPart B 耗时: {part_b_elapsed:.1f}s ({part_b_elapsed/60:.1f}min)")

# ============================================================
# 清理显存
# ============================================================
total_elapsed = time.time() - total_start
print(f"\n总运行时间: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()
print("\n显存已清理")
