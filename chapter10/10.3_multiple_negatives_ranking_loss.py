"""10.3 Multiple Negatives Ranking Loss (MNRL) 训练 Embedding 模型
仅使用 entailment 对 (premise→hypothesis) 作为正样本，
batch 内其他样本的 hypothesis 作为 in-batch negatives，
再额外构造 soft negative (打乱的 hypothesis)。
MNRL 是目前最有效的 embedding 训练损失之一。
"""
import gc
import random
import time
import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ============================================================
# 1. 数据准备 — 构造 (anchor, positive, negative) 三元组
# ============================================================
total_start = time.time()

print("=" * 60)
print("1. 构造三元组数据 (anchor, positive, soft_negative)")
print("=" * 60)

# Load MNLI, 只保留 entailment (label=0)
mnli = load_dataset("glue", "mnli", split="train").select(range(50_000))
mnli = mnli.remove_columns("idx")
mnli = mnli.filter(lambda x: True if x['label'] == 0 else False)
print(f"Entailment 样本数: {len(mnli)}")

# Prepare data and add a soft negative (打乱的 hypothesis)
train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negatives = mnli["hypothesis"]
random.shuffle(soft_negatives)

for row, soft_negative in tqdm(zip(mnli, soft_negatives)):
    train_dataset["anchor"].append(row["premise"])
    train_dataset["positive"].append(row["hypothesis"])
    train_dataset["negative"].append(soft_negative)

train_dataset = Dataset.from_dict(train_dataset)
print(f"训练集大小: {len(train_dataset)}")
print(f"示例:")
print(f"  anchor:   {train_dataset[0]['anchor'][:80]}...")
print(f"  positive: {train_dataset[0]['positive'][:80]}...")
print(f"  negative: {train_dataset[0]['negative'][:80]}...")

# ============================================================
# 2. 评估器 — STS-B
# ============================================================
print("\n" + "=" * 60)
print("2. 创建 STS-B 评估器")
print("=" * 60)

val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine"
)

# ============================================================
# 3. 模型 + MNRL 训练
# ============================================================
print("\n" + "=" * 60)
print("3. MultipleNegativesRankingLoss 训练")
print("=" * 60)
train_start = time.time()

embedding_model = SentenceTransformer('bert-base-uncased')
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

args = SentenceTransformerTrainingArguments(
    output_dir="mnrloss_embedding_model",
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
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()
train_elapsed = time.time() - train_start
print(f"\n训练耗时: {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")

# ============================================================
# 4. 评估
# ============================================================
print("\n" + "=" * 60)
print("4. STS-B 评估结果")
print("=" * 60)

results = evaluator(embedding_model)
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

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
