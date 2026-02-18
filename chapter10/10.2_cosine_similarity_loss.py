"""10.2 Cosine Similarity Loss 训练 Embedding 模型
将 MNLI 三分类标签转换为二分类 (entailment=1, neutral/contradiction=0)，
使用 CosineSimilarityLoss 训练，目标是让蕴含对的嵌入余弦相似度趋近 1，
非蕴含对的嵌入余弦相似度趋近 0。
"""
import gc
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ============================================================
# 1. 数据准备 — MNLI → 二分类
# ============================================================
print("=" * 60)
print("1. 加载 MNLI 并转换为二分类标签")
print("=" * 60)

# Load MNLI dataset from GLUE
# 0 = entailment, 1 = neutral, 2 = contradiction
train_dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

# (neutral/contradiction)=0 and (entailment)=1
mapping = {2: 0, 1: 0, 0: 1}
train_dataset = Dataset.from_dict({
    "sentence1": train_dataset["premise"],
    "sentence2": train_dataset["hypothesis"],
    "label": [float(mapping[label]) for label in train_dataset["label"]]
})

print(f"训练集大小: {len(train_dataset)}")
print(f"正样本 (entailment): {sum(1 for l in train_dataset['label'] if l == 1.0)}")
print(f"负样本 (neutral/contradiction): {sum(1 for l in train_dataset['label'] if l == 0.0)}")

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
# 3. 模型 + 损失函数 + 训练
# ============================================================
print("\n" + "=" * 60)
print("3. CosineSimilarityLoss 训练")
print("=" * 60)

embedding_model = SentenceTransformer('bert-base-uncased')
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

args = SentenceTransformerTrainingArguments(
    output_dir="cosineloss_embedding_model",
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
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()
print("\n显存已清理")
