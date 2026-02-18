"""10.5 无监督学习 — TSDAE (Transformer-based Denoising AutoEncoder)
无需任何标签，通过给输入文本添加噪声 (随机删除 token)，
让模型学习从损坏的文本重建原文，从而学到高质量的句子嵌入。
"""
import gc
import time
import torch
import nltk
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ============================================================
# 0. 下载 NLTK 分词器
# ============================================================
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============================================================
# 1. 数据准备 — 构造去噪数据
# ============================================================
total_start = time.time()

print("=" * 60)
print("1. 构造 TSDAE 去噪训练数据")
print("=" * 60)

# Create a flat list of sentences (premise + hypothesis)
mnli = load_dataset("glue", "mnli", split="train").select(range(25_000))
# 🔧 修复：将 Column 对象转换为列表后合并
flat_sentences = list(mnli["premise"]) + list(mnli["hypothesis"])
print(f"原始句子数: {len(flat_sentences)}")

# Add noise to our input data (默认 del_ratio=0.6)
flat_sentences_unique = list(set(flat_sentences))
damaged_data = DenoisingAutoEncoderDataset(flat_sentences_unique)
print(f"去重后句子数: {len(flat_sentences_unique)}")

# Create dataset
train_dataset = {"damaged_sentence": [], "original_sentence": []}
for data in tqdm(damaged_data, desc="构造去噪数据"):
    train_dataset["damaged_sentence"].append(data.texts[0])
    train_dataset["original_sentence"].append(data.texts[1])
train_dataset = Dataset.from_dict(train_dataset)

print(f"训练集大小: {len(train_dataset)}")
print(f"\n示例:")
print(f"  损坏: {train_dataset[0]['damaged_sentence'][:80]}...")
print(f"  原文: {train_dataset[0]['original_sentence'][:80]}...")

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
# 3. 模型 — 手动构建 (Transformer + CLS Pooling)
# ============================================================
print("\n" + "=" * 60)
print("3. 构建模型 (bert-base-uncased + CLS pooling)")
print("=" * 60)

# TSDAE 通常使用 CLS pooling 而非 mean pooling
word_embedding_model = models.Transformer('bert-base-uncased')
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), 'cls'
)
embedding_model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model]
)
print(f"Embedding 维度: {word_embedding_model.get_word_embedding_dimension()}")

# ============================================================
# 4. 损失函数 — DenoisingAutoEncoderLoss
# ============================================================
print("\n" + "=" * 60)
print("4. 定义 DenoisingAutoEncoderLoss")
print("=" * 60)

train_loss = losses.DenoisingAutoEncoderLoss(
    embedding_model, tie_encoder_decoder=True
)

# 将 decoder 移到正确的设备上
device = "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
train_loss.decoder = train_loss.decoder.to(device)
print(f"Decoder 设备: {device}")

# ============================================================
# 5. 训练
# ============================================================
print("\n" + "=" * 60)
print("5. 开始 TSDAE 训练 (1 epoch)")
print("=" * 60)

train_start = time.time()  # 🔧 添加缺失的计时变量

args = SentenceTransformerTrainingArguments(
    output_dir="tsdae_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
# 6. 评估
# ============================================================
print("\n" + "=" * 60)
print("6. STS-B 评估结果")
print("=" * 60)

results = evaluator(embedding_model)
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

print("\n注意: TSDAE 是无监督方法，不需要任何标注数据!")
print("适用场景: 领域适应 (domain adaptation)、冷启动时预训练")

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
