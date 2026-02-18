"""10.1 从零训练 Embedding 模型 — SoftmaxLoss + MTEB 评估
使用 MNLI 数据集 (entailment/neutral/contradiction 三分类) 训练 BERT embedding 模型，
用 SoftmaxLoss 作为损失函数，然后在 STS-B 和 MTEB Banking77 上评估。
"""
import gc
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ============================================================
# 1. 数据准备 — MNLI 数据集
# ============================================================
print("=" * 60)
print("1. 加载 MNLI 数据集")
print("=" * 60)

# Load MNLI dataset from GLUE
# 0 = entailment, 1 = neutral, 2 = contradiction
train_dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

print(f"训练集大小: {len(train_dataset)}")
print(f"示例: {train_dataset[2]}")

# ============================================================
# 2. 模型 — 基于 bert-base-uncased
# ============================================================
print("\n" + "=" * 60)
print("2. 加载基座模型 bert-base-uncased")
print("=" * 60)

# Use a base model (未经过 sentence-transformers 训练，会自动添加 mean pooling)
embedding_model = SentenceTransformer('bert-base-uncased')

# ============================================================
# 3. 损失函数 — SoftmaxLoss (三分类)
# ============================================================
print("\n" + "=" * 60)
print("3. 定义 SoftmaxLoss (3类: entailment/neutral/contradiction)")
print("=" * 60)

# SoftmaxLoss: 将 (sentence1, sentence2) 的嵌入拼接后送入分类头
# 需要显式指定标签数量
train_loss = losses.SoftmaxLoss(
    model=embedding_model,
    sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
    num_labels=3
)
print(f"Embedding 维度: {embedding_model.get_sentence_embedding_dimension()}")

# ============================================================
# 4. 评估器 — STS-B
# ============================================================
print("\n" + "=" * 60)
print("4. 创建 STS-B 评估器")
print("=" * 60)

# Create an embedding similarity evaluator for stsb
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine",
)
print(f"STS-B 验证集大小: {len(val_sts)}")

# ============================================================
# 5. 训练
# ============================================================
print("\n" + "=" * 60)
print("5. 开始训练 (1 epoch)")
print("=" * 60)

args = SentenceTransformerTrainingArguments(
    output_dir="base_embedding_model",
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
# 6. 评估 — STS-B
# ============================================================
print("\n" + "=" * 60)
print("6. STS-B 评估结果")
print("=" * 60)

results = evaluator(embedding_model)
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# 7. MTEB 基准评估 — Banking77Classification
# ============================================================
print("\n" + "=" * 60)
print("7. MTEB Banking77 分类评估")
print("=" * 60)

try:
    from mteb import MTEB

    evaluation = MTEB(tasks=["Banking77Classification"])
    mteb_results = evaluation.run(embedding_model)
    print(f"MTEB 结果: {mteb_results}")
except ImportError:
    print("mteb 未安装，跳过 MTEB 评估 (pip install mteb)")

# ============================================================
# 清理显存
# ============================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()
print("\n显存已清理")
