"""
Chapter 10 - Creating Text Embedding Models
创建文本嵌入模型

本章内容:
1. 从头训练嵌入模型 (Training from Scratch)
2. 多种损失函数对比 (Loss Functions)
3. 微调预训练模型 (Fine-tuning)
4. Augmented SBERT - 使用 Cross-Encoder 扩展数据
5. MTEB 评估基准
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_device():
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("使用设备: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("使用设备: CPU")
    return device


def clear_memory():
    """清理 GPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Part 1: 数据准备
# ============================================================
def load_mnli_data(num_samples=50000):
    """
    加载 MNLI 数据集 (Multi-Genre Natural Language Inference)
    标签: 0=蕴含(entailment), 1=中性(neutral), 2=矛盾(contradiction)
    """
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Part 1: 加载数据")
    print("=" * 60)
    
    train_dataset = load_dataset("glue", "mnli", split="train").select(range(num_samples))
    train_dataset = train_dataset.remove_columns("idx")
    
    print(f"数据集大小: {len(train_dataset)}")
    print(f"示例: {train_dataset[2]}")
    
    return train_dataset


def load_stsb_evaluator():
    """加载 STS-B 评估器 (Semantic Textual Similarity Benchmark)"""
    from datasets import load_dataset
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    
    val_sts = load_dataset('glue', 'stsb', split='validation')
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=val_sts["sentence1"],
        sentences2=val_sts["sentence2"],
        scores=[score/5 for score in val_sts["label"]],  # 归一化到 [0,1]
        main_similarity="cosine",
    )
    print("STS-B 评估器已加载")
    return evaluator


# ============================================================
# Part 2: 从头训练嵌入模型 (Softmax Loss)
# ============================================================
def train_with_softmax_loss(train_dataset, evaluator):
    """
    使用 Softmax Loss 从头训练嵌入模型
    适用于: 有明确类别标签的句子对 (如 NLI 任务)
    """
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    
    print("\n" + "=" * 60)
    print("Part 2: 使用 Softmax Loss 训练")
    print("=" * 60)
    
    # 从 BERT 基础模型开始
    print("\n加载基础模型: bert-base-uncased")
    embedding_model = SentenceTransformer('bert-base-uncased')
    
    # Softmax Loss: 适用于分类任务
    # 输入: (premise, hypothesis, label)
    train_loss = losses.SoftmaxLoss(
        model=embedding_model,
        sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
        num_labels=3  # entailment, neutral, contradiction
    )
    
    # 训练参数
    args = SentenceTransformerTrainingArguments(
        output_dir="softmax_embedding_model",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        fp16=False,  # MPS (Apple Silicon) 不支持 fp16
        eval_steps=100,
        logging_steps=100,
    )
    
    # 训练
    print("\n开始训练...")
    trainer = SentenceTransformerTrainer(
        model=embedding_model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator
    )
    trainer.train()
    
    # 评估
    print("\n评估结果:")
    results = evaluator(embedding_model)
    print(f"  Spearman Cosine: {results['spearman_cosine']:.4f}")
    
    clear_memory()
    return embedding_model, results


# ============================================================
# Part 3: 使用 Cosine Similarity Loss 训练
# ============================================================
def train_with_cosine_loss(evaluator):
    """
    使用 Cosine Similarity Loss 训练
    适用于: 有相似度分数的句子对
    """
    from datasets import Dataset, load_dataset
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    
    print("\n" + "=" * 60)
    print("Part 3: 使用 Cosine Similarity Loss 训练")
    print("=" * 60)
    
    # 准备数据: 将 3 分类转为 2 分类 (相似/不相似)
    train_data = load_dataset("glue", "mnli", split="train").select(range(50000))
    train_data = train_data.remove_columns("idx")
    
    # 映射: entailment=1 (相似), neutral/contradiction=0 (不相似)
    mapping = {0: 1, 1: 0, 2: 0}
    train_dataset = Dataset.from_dict({
        "sentence1": train_data["premise"],
        "sentence2": train_data["hypothesis"],
        "label": [float(mapping[label]) for label in train_data["label"]]
    })
    
    # 模型和损失函数
    embedding_model = SentenceTransformer('bert-base-uncased')
    train_loss = losses.CosineSimilarityLoss(model=embedding_model)
    
    # 训练
    args = SentenceTransformerTrainingArguments(
        output_dir="cosine_embedding_model",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        warmup_steps=100,
        fp16=False,  # MPS (Apple Silicon) 不支持 fp16
        logging_steps=100,
    )
    
    print("\n开始训练...")
    trainer = SentenceTransformerTrainer(
        model=embedding_model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator
    )
    trainer.train()
    
    # 评估
    print("\n评估结果:")
    results = evaluator(embedding_model)
    print(f"  Spearman Cosine: {results['spearman_cosine']:.4f}")
    
    clear_memory()
    return embedding_model, results


# ============================================================
# Part 4: 使用 Multiple Negatives Ranking Loss 训练
# ============================================================
def train_with_mnrl(evaluator):
    """
    使用 Multiple Negatives Ranking Loss (MNRL) 训练
    最常用的对比学习损失函数
    适用于: (anchor, positive) 或 (anchor, positive, negative) 三元组
    """
    import random
    from datasets import Dataset, load_dataset
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    
    print("\n" + "=" * 60)
    print("Part 4: 使用 Multiple Negatives Ranking Loss 训练")
    print("=" * 60)
    
    # 准备数据: 只使用 entailment 对作为正样本
    mnli = load_dataset("glue", "mnli", split="train").select(range(50000))
    mnli = mnli.remove_columns("idx")
    mnli = mnli.filter(lambda x: x['label'] == 0)  # 只保留 entailment
    
    # 创建三元组: (anchor, positive, soft_negative)
    train_data = {"anchor": [], "positive": [], "negative": []}
    soft_negatives = list(mnli["hypothesis"])
    random.shuffle(soft_negatives)
    
    for row, soft_neg in tqdm(zip(mnli, soft_negatives), desc="准备数据"):
        train_data["anchor"].append(row["premise"])
        train_data["positive"].append(row["hypothesis"])
        train_data["negative"].append(soft_neg)
    
    train_dataset = Dataset.from_dict(train_data)
    print(f"训练样本数: {len(train_dataset)}")
    
    # 模型和损失函数
    embedding_model = SentenceTransformer('bert-base-uncased')
    train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)
    
    # 训练
    args = SentenceTransformerTrainingArguments(
        output_dir="mnrl_embedding_model",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        warmup_steps=100,
        fp16=False,  # MPS (Apple Silicon) 不支持 fp16
        logging_steps=100,
    )
    
    print("\n开始训练...")
    trainer = SentenceTransformerTrainer(
        model=embedding_model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator
    )
    trainer.train()
    
    # 评估
    print("\n评估结果:")
    results = evaluator(embedding_model)
    print(f"  Spearman Cosine: {results['spearman_cosine']:.4f}")
    
    clear_memory()
    return embedding_model, results


# ============================================================
# Part 5: 微调预训练模型 (Fine-tuning)
# ============================================================
def finetune_pretrained_model(evaluator):
    """
    微调已有的高质量嵌入模型
    比从头训练 BERT 效果更好
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    
    print("\n" + "=" * 60)
    print("Part 5: 微调预训练嵌入模型")
    print("=" * 60)
    
    # 加载数据
    train_data = load_dataset("glue", "mnli", split="train").select(range(50000))
    train_data = train_data.remove_columns("idx")
    
    # 使用预训练的 Sentence Transformer
    print("\n加载预训练模型: all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 原始模型性能
    print("\n原始模型性能:")
    original_results = evaluator(embedding_model)
    print(f"  Spearman Cosine: {original_results['spearman_cosine']:.4f}")
    
    # 微调
    train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)
    
    args = SentenceTransformerTrainingArguments(
        output_dir="finetuned_embedding_model",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        warmup_steps=100,
        fp16=False,  # MPS (Apple Silicon) 不支持 fp16
        logging_steps=100,
    )
    
    print("\n开始微调...")
    trainer = SentenceTransformerTrainer(
        model=embedding_model,
        args=args,
        train_dataset=train_data,
        loss=train_loss,
        evaluator=evaluator
    )
    trainer.train()
    
    # 微调后性能
    print("\n微调后性能:")
    finetuned_results = evaluator(embedding_model)
    print(f"  Spearman Cosine: {finetuned_results['spearman_cosine']:.4f}")
    
    clear_memory()
    return embedding_model, original_results, finetuned_results


# ============================================================
# Part 6: Augmented SBERT - 数据增强
# ============================================================
def augmented_sbert_demo(evaluator):
    """
    Augmented SBERT: 使用 Cross-Encoder 扩展训练数据
    
    流程:
    1. 用少量标注数据训练 Cross-Encoder
    2. Cross-Encoder 对大量无标注数据打标签
    3. 用扩展后的数据训练 Bi-Encoder (SBERT)
    """
    from datasets import Dataset, load_dataset
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    
    print("\n" + "=" * 60)
    print("Part 6: Augmented SBERT")
    print("=" * 60)
    
    # Step 1: 准备少量 "金标准" 数据
    print("\n[Step 1] 准备金标准数据 (10,000 样本)...")
    gold_data = load_dataset("glue", "mnli", split="train").select(range(10000))
    mapping = {0: 1, 1: 0, 2: 0}
    
    gold_examples = [
        InputExample(
            texts=[row["premise"], row["hypothesis"]],
            label=mapping[row["label"]]
        )
        for row in tqdm(gold_data, desc="处理金标准数据")
    ]
    gold_dataloader = NoDuplicatesDataLoader(gold_examples, batch_size=32)
    
    gold_df = pd.DataFrame({
        'sentence1': gold_data['premise'],
        'sentence2': gold_data['hypothesis'],
        'label': [mapping[label] for label in gold_data['label']]
    })
    
    # Step 2: 训练 Cross-Encoder
    print("\n[Step 2] 训练 Cross-Encoder...")
    cross_encoder = CrossEncoder('bert-base-uncased', num_labels=2)
    cross_encoder.fit(
        train_dataloader=gold_dataloader,
        epochs=1,
        show_progress_bar=True,
        warmup_steps=100,
        use_amp=False
    )
    
    # Step 3: 用 Cross-Encoder 标注更多数据 ("银标准")
    print("\n[Step 3] 使用 Cross-Encoder 标注银标准数据 (40,000 样本)...")
    silver_data = load_dataset("glue", "mnli", split="train").select(range(10000, 50000))
    pairs = list(zip(silver_data['premise'], silver_data['hypothesis']))
    
    output = cross_encoder.predict(pairs, apply_softmax=True, show_progress_bar=True)
    silver_df = pd.DataFrame({
        "sentence1": silver_data["premise"],
        "sentence2": silver_data["hypothesis"],
        "label": np.argmax(output, axis=1)
    })
    
    # Step 4: 合并数据训练 Bi-Encoder
    print("\n[Step 4] 使用金+银数据训练 Bi-Encoder...")
    combined_data = pd.concat([gold_df, silver_df], ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset=['sentence1', 'sentence2'])
    train_dataset = Dataset.from_pandas(combined_data, preserve_index=False)
    print(f"合并后数据量: {len(train_dataset)}")
    
    # 训练
    embedding_model = SentenceTransformer('bert-base-uncased')
    train_loss = losses.CosineSimilarityLoss(model=embedding_model)
    
    args = SentenceTransformerTrainingArguments(
        output_dir="augmented_embedding_model",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        warmup_steps=100,
        fp16=False,  # MPS (Apple Silicon) 不支持 fp16
        logging_steps=100,
    )
    
    trainer = SentenceTransformerTrainer(
        model=embedding_model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator
    )
    trainer.train()
    
    print("\n使用金+银数据的结果:")
    combined_results = evaluator(embedding_model)
    print(f"  Spearman Cosine: {combined_results['spearman_cosine']:.4f}")
    
    clear_memory()
    return combined_results


# ============================================================
# Part 7: MTEB 评估
# ============================================================
def mteb_evaluation(embedding_model):
    """
    使用 MTEB (Massive Text Embedding Benchmark) 评估模型
    包含多种任务: 分类、聚类、检索、重排序等
    """
    from mteb import MTEB
    
    print("\n" + "=" * 60)
    print("Part 7: MTEB 评估")
    print("=" * 60)
    
    # 选择一个分类任务进行评估
    evaluation = MTEB(tasks=["Banking77Classification"])
    results = evaluation.run(embedding_model)
    
    print("\nMTEB 评估结果:")
    print(results)
    
    return results


# ============================================================
# 损失函数对比总结
# ============================================================
def print_loss_comparison():
    """打印损失函数对比"""
    print("\n" + "=" * 60)
    print("损失函数对比")
    print("=" * 60)
    
    comparison = """
┌─────────────────────────────────────────────────────────────┐
│                     损失函数对比                             │
├─────────────────────────────────────────────────────────────┤
│  1. SoftmaxLoss                                             │
│     数据格式: (sentence1, sentence2, class_label)           │
│     适用场景: NLI 分类任务                                   │
│     特点: 需要明确的类别标签                                 │
│                                                             │
│  2. CosineSimilarityLoss                                    │
│     数据格式: (sentence1, sentence2, similarity_score)      │
│     适用场景: 有相似度分数的句子对                           │
│     特点: 直接优化余弦相似度                                 │
│                                                             │
│  3. MultipleNegativesRankingLoss (MNRL) ⭐ 推荐             │
│     数据格式: (anchor, positive) 或 (anchor, pos, neg)      │
│     适用场景: 对比学习，检索任务                             │
│     特点: 利用 batch 内负样本，数据效率高                    │
│                                                             │
│  4. TripletLoss                                             │
│     数据格式: (anchor, positive, negative)                  │
│     适用场景: 需要明确负样本的场景                           │
│     特点: 经典三元组损失                                     │
└─────────────────────────────────────────────────────────────┘
"""
    print(comparison)


def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("Chapter 10 总结")
    print("=" * 60)
    
    summary = """
┌─────────────────────────────────────────────────────────────┐
│               创建文本嵌入模型的方法                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方法 1: 从头训练 (Training from Scratch)                   │
│  ────────────────────────────────────────                   │
│  • 基础模型: BERT, RoBERTa 等                               │
│  • 添加 Pooling 层: Mean Pooling                            │
│  • 需要大量数据                                              │
│                                                             │
│  方法 2: 微调预训练模型 (Fine-tuning) ⭐ 推荐               │
│  ────────────────────────────────────────                   │
│  • 基础模型: all-MiniLM-L6-v2, BGE, GTE 等                  │
│  • 在特定领域数据上继续训练                                  │
│  • 效果通常更好，训练更快                                    │
│                                                             │
│  方法 3: Augmented SBERT                                    │
│  ────────────────────────────────────────                   │
│  • 步骤: Cross-Encoder → 标注数据 → Bi-Encoder              │
│  • 适用于标注数据稀缺的场景                                  │
│  • 可以显著扩展训练数据                                      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     关键洞见                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 损失函数很重要                                          │
│     • MNRL 通常是最佳选择                                    │
│     • 利用 batch 内负样本提高效率                            │
│                                                             │
│  2. 预训练模型是好的起点                                    │
│     • 直接微调比从头训练效果好                               │
│     • 选择适合任务的预训练模型                               │
│                                                             │
│  3. 数据质量 > 数据数量                                     │
│     • 高质量的 (anchor, positive) 对最重要                  │
│     • Hard negatives 可以提升效果                           │
│                                                             │
│  4. 使用 MTEB 进行全面评估                                  │
│     • 多任务评估更能反映模型能力                             │
│     • 关注目标任务的性能                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""
    print(summary)


def main():
    """主函数 - 演示各种训练方法"""
    device = get_device()
    
    print("\n" + "=" * 60)
    print("Chapter 10 - Creating Text Embedding Models")
    print("=" * 60)
    
    # 加载评估器
    evaluator = load_stsb_evaluator()
    
    # 选择要运行的部分
    print("\n选择要运行的演示:")
    print("1. Softmax Loss 训练")
    print("2. Cosine Similarity Loss 训练")
    print("3. Multiple Negatives Ranking Loss 训练")
    print("4. 微调预训练模型")
    print("5. Augmented SBERT")
    print("6. 损失函数对比总结")
    print("0. 全部运行")
    
    try:
        choice = input("\n请输入选项 (默认 6): ").strip() or "6"
    except EOFError:
        choice = "6"
    
    if choice == "1" or choice == "0":
        train_dataset = load_mnli_data()
        train_with_softmax_loss(train_dataset, evaluator)
    
    if choice == "2" or choice == "0":
        train_with_cosine_loss(evaluator)
    
    if choice == "3" or choice == "0":
        train_with_mnrl(evaluator)
    
    if choice == "4" or choice == "0":
        finetune_pretrained_model(evaluator)
    
    if choice == "5" or choice == "0":
        augmented_sbert_demo(evaluator)
    
    if choice == "6" or choice == "0":
        print_loss_comparison()
    
    # 打印总结
    print_summary()


if __name__ == "__main__":
    main()
