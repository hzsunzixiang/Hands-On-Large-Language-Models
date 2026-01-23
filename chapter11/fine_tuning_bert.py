#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 11 - Fine-Tuning Representation Models for Classification

本章探索表征模型（如 BERT）在分类任务中的性能表现。

核心内容:
1. 使用 HuggingFace Trainer 进行监督分类
2. 冻结层策略 - 平衡效率与性能
3. Few-shot 分类 (SetFit)
4. 掩码语言模型 (MLM) 领域适应
5. 命名实体识别 (NER)

关键洞见:
- 冻结底层可以加速训练，同时保持大部分性能
- SetFit 只需少量标注数据就能达到不错效果
- 领域特定 MLM 预训练可以提升下游任务性能
- NER 需要特殊的标签对齐处理
"""

import gc
import warnings
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Part 1: 数据准备
# ============================================================
def load_rotten_tomatoes():
    """
    加载 Rotten Tomatoes 电影评论数据集
    二分类任务: 正面/负面评价
    """
    from datasets import load_dataset
    
    print("=" * 60)
    print("Part 1: 加载数据")
    print("=" * 60)
    
    tomatoes = load_dataset("rotten_tomatoes")
    train_data, test_data = tomatoes["train"], tomatoes["test"]
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"\n示例: {train_data[0]}")
    
    return train_data, test_data, tomatoes


# ============================================================
# Part 2: 监督分类 - HuggingFace Trainer
# ============================================================
def supervised_classification(train_data, test_data):
    """
    使用 HuggingFace Trainer 进行全参数微调
    
    流程:
    1. 加载预训练 BERT + 分类头
    2. Tokenize 数据
    3. 使用 Trainer 训练
    4. 评估
    """
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer
    )
    import evaluate
    
    print("\n" + "=" * 60)
    print("Part 2: 监督分类 - 全参数微调")
    print("=" * 60)
    
    # 加载模型和分词器
    model_id = "bert-base-cased"
    print(f"\n加载模型: {model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tokenize
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    print("Tokenizing 数据...")
    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)
    
    # Data collator - 动态 padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        load_f1 = evaluate.load("f1")
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"f1": f1}
    
    # 训练参数
    training_args = TrainingArguments(
        "model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="none",
        fp16=False,  # MPS 不支持 fp16
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\n开始训练 (全参数微调)...")
    trainer.train()
    
    print("\n评估结果:")
    results = trainer.evaluate()
    print(f"  F1 Score: {results['eval_f1']:.4f}")
    
    clear_memory()
    return results


# ============================================================
# Part 3: 冻结层策略
# ============================================================
def freeze_layers_experiment(train_data, test_data, tokenized_train, tokenized_test):
    """
    探索冻结不同层的效果
    
    策略:
    1. 只训练分类头 (冻结所有 BERT 层) - 最快但效果一般
    2. 冻结底层，训练顶层 - 平衡方案
    3. 全参数微调 - 最慢但效果最好
    
    BERT 层结构:
    - embeddings (word, position, token_type, LayerNorm)
    - encoder.layer.0 ~ encoder.layer.11 (12个 Transformer 块)
    - pooler
    - classifier (分类头)
    """
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer
    )
    import evaluate
    
    print("\n" + "=" * 60)
    print("Part 3: 冻结层策略实验")
    print("=" * 60)
    
    model_id = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        load_f1 = evaluate.load("f1")
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"f1": f1}
    
    training_args = TrainingArguments(
        "model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
    )
    
    # 实验1: 只训练分类头
    print("\n[实验1] 只训练分类头 (冻结所有 BERT 层)")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
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
    results1 = trainer.evaluate()
    print(f"  F1 Score: {results1['eval_f1']:.4f}")
    
    clear_memory()
    
    # 实验2: 冻结前10层，训练后2层 + 分类头
    print("\n[实验2] 冻结 layer 0-9，训练 layer 10-11 + 分类头")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    
    # 参数索引: layer 10 从 index 165 开始
    for index, (name, param) in enumerate(model.named_parameters()):
        if index < 165:  # 冻结前面的层
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
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
    results2 = trainer.evaluate()
    print(f"  F1 Score: {results2['eval_f1']:.4f}")
    
    clear_memory()
    
    print("\n冻结层实验总结:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  策略                        │  训练速度  │  效果           │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  只训练分类头                │  最快      │  一般           │")
    print("│  冻结底层，训练顶层          │  较快      │  接近全微调     │")
    print("│  全参数微调                  │  最慢      │  最好           │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    return results1, results2


# ============================================================
# Part 4: Few-shot 分类 - SetFit
# ============================================================
def few_shot_classification(tomatoes):
    """
    使用 SetFit 进行 few-shot 分类
    
    SetFit 原理:
    1. 从少量样本生成句子对 (正/负样本对)
    2. 使用对比学习微调 Sentence Transformer
    3. 训练一个分类头 (如 LogisticRegression)
    
    优势:
    - 只需每类 8-16 个样本
    - 无需 prompt 工程
    - 训练快速
    """
    from setfit import SetFitModel, sample_dataset
    from setfit import TrainingArguments as SetFitTrainingArguments
    from setfit import Trainer as SetFitTrainer
    
    print("\n" + "=" * 60)
    print("Part 4: Few-shot 分类 (SetFit)")
    print("=" * 60)
    
    # 模拟 few-shot: 每类采样 16 个样本
    sampled_train_data = sample_dataset(tomatoes["train"], num_samples=16)
    print(f"Few-shot 训练样本: {len(sampled_train_data)} (每类 16 个)")
    
    # 加载预训练 Sentence Transformer
    print("\n加载模型: sentence-transformers/all-mpnet-base-v2")
    model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    
    # 训练参数
    args = SetFitTrainingArguments(
        num_epochs=3,       # 对比学习 epochs
        num_iterations=20,  # 每类生成的句子对数量
    )
    args.eval_strategy = args.evaluation_strategy
    
    # Trainer
    trainer = SetFitTrainer(
        model=model,
        args=args,
        train_dataset=sampled_train_data,
        eval_dataset=tomatoes["test"],
        metric="f1"
    )
    
    print("\n开始训练...")
    trainer.train()
    
    print("\n评估结果:")
    results = trainer.evaluate()
    print(f"  F1 Score: {results['f1']:.4f}")
    
    # SetFit 使用 LogisticRegression 作为分类头
    print(f"\n分类头类型: {type(model.model_head).__name__}")
    
    clear_memory()
    return results


# ============================================================
# Part 5: 掩码语言模型 (MLM) 领域适应
# ============================================================
def domain_adaptation_mlm(train_data, test_data):
    """
    使用 MLM 进行领域适应
    
    原理:
    1. 在领域数据上继续 MLM 预训练
    2. BERT 学习领域特定的语言模式
    3. 再用于下游任务效果更好
    
    这里用电影评论数据做 MLM，让模型学习电影领域词汇
    """
    from transformers import (
        AutoTokenizer, 
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
        pipeline
    )
    
    print("\n" + "=" * 60)
    print("Part 5: 掩码语言模型 (MLM) 领域适应")
    print("=" * 60)
    
    model_id = "bert-base-cased"
    print(f"\n加载 MLM 模型: {model_id}")
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tokenize
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    print("Tokenizing 数据...")
    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_train = tokenized_train.remove_columns("label")
    tokenized_test = test_data.map(preprocess_function, batched=True)
    tokenized_test = tokenized_test.remove_columns("label")
    
    # MLM Data Collator - 随机 mask 15% 的 token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # 训练参数 (MLM 通常需要更多 epochs)
    training_args = TrainingArguments(
        "mlm_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,  # MLM 需要更多 epochs
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
        data_collator=data_collator
    )
    
    print("\n开始 MLM 预训练...")
    trainer.train()
    
    # 保存模型
    tokenizer.save_pretrained("mlm_movie")
    model.save_pretrained("mlm_movie")
    
    # 对比: 原始 BERT vs 领域适应后
    print("\n对比 [MASK] 预测:")
    print("输入: 'What a horrible [MASK]!'")
    
    print("\n原始 BERT:")
    mask_filler = pipeline("fill-mask", model="bert-base-cased")
    preds = mask_filler("What a horrible [MASK]!")
    for pred in preds[:3]:
        print(f"  {pred['sequence']}")
    
    print("\n领域适应后:")
    mask_filler = pipeline("fill-mask", model="mlm_movie")
    preds = mask_filler("What a horrible [MASK]!")
    for pred in preds[:3]:
        print(f"  {pred['sequence']}")
    
    clear_memory()


# ============================================================
# Part 6: 命名实体识别 (NER)
# ============================================================
def named_entity_recognition():
    """
    使用 BERT 进行 NER (Token Classification)
    
    数据集: CoNLL-2003
    标签体系: BIO 格式
    - B-XXX: 实体开始
    - I-XXX: 实体内部
    - O: 非实体
    
    实体类型: PER (人名), ORG (组织), LOC (地点), MISC (其他)
    
    关键挑战: 子词标签对齐
    - "Maarten" -> ["Ma", "##arte", "##n"]
    - 原始标签: B-PER
    - 对齐后: [B-PER, I-PER, I-PER]
    """
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
    
    print("\n" + "=" * 60)
    print("Part 6: 命名实体识别 (NER)")
    print("=" * 60)
    
    # 加载 CoNLL-2003 数据集
    # 原始数据源已失效，使用 HuggingFace 镜像或替代数据集
    print("\n加载 CoNLL-2003 数据集...")
    
    use_wnut = False
    try:
        # 方案1: 尝试使用 eriktks/conll2003 (社区维护版本)
        dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
        print("使用 eriktks/conll2003 数据集")
    except Exception as e1:
        try:
            # 方案2: 使用 HuggingFace 镜像
            import os
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            dataset = load_dataset("conll2003", trust_remote_code=True)
            print("使用 HuggingFace 镜像加载")
        except Exception as e2:
            # 方案3: 使用 wnut_17 作为替代 NER 数据集
            print(f"CoNLL-2003 加载失败: {e2}")
            print("使用替代数据集: wnut_17")
            dataset = load_dataset("wnut_17", trust_remote_code=True)
            use_wnut = True
            print("注意: wnut_17 使用不同的实体类型")
    
    # 根据数据集设置标签映射
    if use_wnut:
        # wnut_17 标签体系: 13 个标签
        # 实体类型: corporation, creative-work, group, location, person, product
        label_names = dataset["train"].features["ner_tags"].feature.names
        label2id = {name: i for i, name in enumerate(label_names)}
        id2label = {i: name for i, name in enumerate(label_names)}
        print(f"WNUT-17 标签: {label_names}")
    else:
        # CoNLL-2003 标签体系: 9 个标签
        label2id = {
            'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
            'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
        }
        id2label = {v: k for k, v in label2id.items()}
        print(f"CoNLL-2003 标签: {list(label2id.keys())}")
    
    # 示例
    example = dataset["train"][min(848, len(dataset["train"])-1)]
    print(f"\n示例:")
    print(f"  Tokens: {example['tokens']}")
    print(f"  NER Tags: {[id2label[t] for t in example['ner_tags']]}")
    
    # 加载模型
    model_id = "bert-base-cased"
    print(f"\n加载模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    # 标签对齐函数
    def align_labels(examples):
        """
        对齐子词和标签
        
        问题: BERT 使用 WordPiece，一个词可能被拆成多个子词
        解决: 第一个子词用原标签，后续子词用 I-XXX 或 -100 (忽略)
        """
        token_ids = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True
        )
        labels = examples["ner_tags"]
        
        updated_labels = []
        for index, label in enumerate(labels):
            word_ids = token_ids.word_ids(batch_index=index)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx != previous_word_idx:
                    previous_word_idx = word_idx
                    updated_label = -100 if word_idx is None else label[word_idx]
                    label_ids.append(updated_label)
                elif word_idx is None:
                    label_ids.append(-100)
                else:
                    # 子词: B-XXX -> I-XXX
                    updated_label = label[word_idx]
                    if updated_label % 2 == 1:  # B-XXX
                        updated_label += 1      # -> I-XXX
                    label_ids.append(updated_label)
            
            updated_labels.append(label_ids)
        
        token_ids["labels"] = updated_labels
        return token_ids
    
    print("\n对齐标签...")
    tokenized = dataset.map(align_labels, batched=True)
    
    # 展示对齐效果
    print(f"\n对齐示例 (index=848):")
    print(f"  原始: {example['ner_tags']}")
    print(f"  对齐后: {tokenized['train'][848]['labels']}")
    
    # 评估指标
    seqeval = evaluate.load("seqeval")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)
        
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for token_pred, token_label in zip(prediction, label):
                if token_label != -100:
                    true_predictions.append([id2label[token_pred]])
                    true_labels.append([id2label[token_label]])
        
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"f1": results["overall_f1"]}
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # 训练
    training_args = TrainingArguments(
        "ner_model",
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
    
    print("\n开始训练...")
    trainer.train()
    
    print("\n评估结果:")
    results = trainer.evaluate()
    print(f"  F1 Score: {results['eval_f1']:.4f}")
    
    # 推理示例
    trainer.save_model("ner_model_final")
    token_classifier = pipeline(
        "token-classification",
        model="ner_model_final",
    )
    
    print("\n推理示例:")
    text = "My name is Maarten and I work at Hugging Face in New York."
    print(f"输入: {text}")
    preds = token_classifier(text)
    print("识别的实体:")
    for p in preds:
        print(f"  {p['word']}: {p['entity']} (score: {p['score']:.3f})")
    
    clear_memory()
    return results


# ============================================================
# 总结
# ============================================================
def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("Chapter 11 总结")
    print("=" * 60)
    
    summary = """
┌─────────────────────────────────────────────────────────────┐
│            微调表征模型用于分类的方法                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 全参数微调 (Full Fine-tuning)                           │
│     - 所有参数都更新                                         │
│     - 效果最好，但训练慢                                     │
│     - 需要足够的数据                                         │
│                                                             │
│  2. 冻结层策略 (Layer Freezing)                             │
│     - 冻结底层，只训练顶层                                   │
│     - 底层学通用特征，顶层学任务特征                          │
│     - 平衡效率与性能                                         │
│                                                             │
│  3. Few-shot 学习 (SetFit)                                  │
│     - 只需每类 8-16 个样本                                   │
│     - 使用对比学习微调嵌入                                   │
│     - 无需 prompt 工程                                       │
│                                                             │
│  4. 领域适应 (Domain Adaptation)                            │
│     - 在领域数据上继续 MLM 预训练                            │
│     - 学习领域特定的语言模式                                 │
│     - 再进行下游任务微调                                     │
│                                                             │
│  5. Token Classification (NER)                              │
│     - 需要处理子词标签对齐                                   │
│     - BIO 标注体系                                           │
│     - 特殊 token 用 -100 忽略                                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  关键洞见:                                                   │
│  • BERT 底层学通用语法，顶层学语义 - 可分层微调               │
│  • 少样本场景用 SetFit，效果接近全监督                        │
│  • 领域 MLM 预训练是有效的迁移学习策略                        │
│  • NER 的子词对齐是常见坑，要特别注意                         │
└─────────────────────────────────────────────────────────────┘
"""
    print(summary)


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    print("=" * 60)
    print("Chapter 11 - Fine-Tuning Representation Models")
    print("=" * 60)
    
    device = get_device()
    print(f"\n使用设备: {device}")
    
    # Part 1: 加载数据
    train_data, test_data, tomatoes = load_rotten_tomatoes()
    
    # Part 2: 监督分类
    print("\n运行监督分类...")
    results_supervised = supervised_classification(train_data, test_data)
    
    # Part 4: Few-shot (SetFit)
    # 注意: SetFit 需要安装 setfit 包
    try:
        print("\n运行 Few-shot 分类 (SetFit)...")
        results_setfit = few_shot_classification(tomatoes)
    except ImportError:
        print("\nSetFit 未安装，跳过 few-shot 实验")
        print("安装: pip install setfit")
    
    # Part 6: NER
    print("\n运行命名实体识别...")
    results_ner = named_entity_recognition()
    
    # 总结
    print_summary()
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
