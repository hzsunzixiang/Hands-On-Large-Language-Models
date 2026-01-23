#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 12 - 微调生成模型 (Fine-Tuning Generation Models)

本章探索微调生成式大语言模型的两步法：
1. SFT (Supervised Fine-Tuning) - 监督微调
2. Preference Tuning (DPO/PPO) - 偏好对齐

核心技术:
- QLoRA: 4-bit 量化 + LoRA，大幅降低显存需求
- SFTTrainer: HuggingFace TRL 库的监督微调工具
- DPOTrainer: 直接偏好优化，无需训练奖励模型

使用的模型: TinyLlama-1.1B (适合演示和学习)

依赖安装:
    pip install accelerate peft bitsandbytes transformers trl sentencepiece

运行方式:
    python fine_tuning_generation_models.py --mode sft    # 只运行 SFT
    python fine_tuning_generation_models.py --mode dpo    # 只运行 DPO
    python fine_tuning_generation_models.py --mode all    # 完整流程
"""

import argparse
import gc
import warnings
import torch

warnings.filterwarnings("ignore")


# ============================================================
# 配置
# ============================================================
# 基础模型 (未经指令微调的预训练模型)
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Chat 模板模型 (用于获取对话格式)
TEMPLATE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# SFT 数据集
SFT_DATASET = "HuggingFaceH4/ultrachat_200k"

# DPO 数据集 (包含 chosen/rejected 对)
DPO_DATASET = "argilla/distilabel-intel-orca-dpo-pairs"

# 输出目录
SFT_OUTPUT = "./sft_results"
DPO_OUTPUT = "./dpo_results"


# ============================================================
# 工具函数
# ============================================================
def get_device():
    """检测可用设备"""
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


def check_gpu():
    """检查 GPU 是否可用"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ 检测到 CUDA GPU: {gpu_name}")
        print(f"  显存: {gpu_memory:.1f} GB")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("✓ 检测到 MPS (Apple Silicon)")
        print("  注意: MPS 不支持 bitsandbytes 量化，将使用标准 LoRA")
        return "mps"
    else:
        print("⚠️  警告: 未检测到 GPU")
        print("   建议在 Google Colab (T4 GPU) 上运行")
        return None


# ============================================================
# Part 1: 数据预处理
# ============================================================
def prepare_sft_dataset(num_samples: int = 3000):
    """
    准备 SFT 数据集
    
    使用 UltraChat 数据集，包含多轮对话
    需要将对话格式转换为模型的 Chat Template
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print("\n" + "=" * 60)
    print("  Part 1: 准备 SFT 数据集")
    print("=" * 60)
    
    # 加载模板分词器 (用于获取 Chat Template)
    print(f"\n加载 Chat Template: {TEMPLATE_MODEL}")
    template_tokenizer = AutoTokenizer.from_pretrained(TEMPLATE_MODEL)
    
    def format_prompt(example):
        """
        使用 Chat Template 格式化对话
        
        TinyLlama 格式:
        <|user|>
        用户消息</s>
        <|assistant|>
        助手回复</s>
        """
        chat = example["messages"]
        prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": prompt}
    
    # 加载并处理数据集
    print(f"\n加载数据集: {SFT_DATASET}")
    dataset = (
        load_dataset(SFT_DATASET, split="test_sft")
        .shuffle(seed=42)
        .select(range(num_samples))
    )
    
    print("格式化数据...")
    # 同时保留 text 字段和 messages 字段，兼容不同版本 trl
    dataset = dataset.map(format_prompt)
    
    print(f"✓ 数据集准备完成: {len(dataset)} 条样本")
    
    # 展示示例
    print("\n示例对话:")
    print("-" * 40)
    print(dataset["text"][0][:500] + "...")
    
    # 移除 messages 字段，只保留 text 字段（兼容 trl 0.12+）
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    
    return dataset


def prepare_dpo_dataset():
    """
    准备 DPO 数据集
    
    DPO 需要三元组: (prompt, chosen, rejected)
    - prompt: 用户输入
    - chosen: 人类偏好的回答
    - rejected: 被拒绝的回答
    """
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("  准备 DPO 数据集")
    print("=" * 60)
    
    def format_prompt(example):
        """格式化为 DPO 格式"""
        system = "<|system|>\n" + example['system'] + "</s>\n"
        prompt = "<|user|>\n" + example['input'] + "</s>\n<|assistant|>\n"
        chosen = example['chosen'] + "</s>\n"
        rejected = example['rejected'] + "</s>\n"
        
        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    
    # 加载并过滤数据集
    print(f"\n加载数据集: {DPO_DATASET}")
    dpo_dataset = load_dataset(DPO_DATASET, split="train")
    
    # 过滤高质量样本
    dpo_dataset = dpo_dataset.filter(
        lambda r:
            r["status"] != "tie" and
            r["chosen_score"] >= 8 and
            not r["in_gsm8k_train"]
    )
    
    dpo_dataset = dpo_dataset.map(format_prompt, remove_columns=dpo_dataset.column_names)
    
    print(f"✓ DPO 数据集: {len(dpo_dataset)} 条样本")
    
    return dpo_dataset


# ============================================================
# Part 2: 模型加载与量化
# ============================================================
def load_model_for_training(device_type="cuda"):
    """
    加载模型并应用量化（如果支持）
    
    QLoRA 配置 (CUDA only):
    - load_in_4bit: 4-bit 量化
    - bnb_4bit_quant_type: nf4 (normalized float 4-bit)
    - bnb_4bit_compute_dtype: float16 (计算时使用的精度)
    - bnb_4bit_use_double_quant: 二次量化，进一步压缩
    
    MPS: 不使用量化，直接 float16 加载
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "=" * 60)
    print("  Part 2: 加载模型")
    print("=" * 60)
    
    if device_type == "cuda":
        # CUDA: 使用 4-bit 量化
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"\n加载模型: {BASE_MODEL}")
        print("  量化配置: 4-bit NF4 + 二次量化")
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            quantization_config=bnb_config,
        )
    else:
        # MPS/CPU: 不使用量化，使用 float32
        print(f"\n加载模型: {BASE_MODEL}")
        print("  模式: 标准 LoRA (无量化)")
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,  # MPS 需要 float32
            device_map={"": device_type} if device_type == "mps" else "auto",
        )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"
    
    # 设置 chat template (TinyLlama 使用 Llama 2 格式)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>\n' + message['content'] + '</s>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>\n' + message['content'] + '</s>\n' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<|system|>\n' + message['content'] + '</s>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>\n' }}"
            "{% endif %}"
        )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ 模型加载完成")
    print(f"  总参数量: {total_params / 1e9:.2f}B")
    
    return model, tokenizer


def setup_lora(model, device_type="cuda"):
    """
    配置 LoRA (Low-Rank Adaptation)
    
    LoRA 原理:
    - 冻结原始权重 W
    - 添加低秩分解 ΔW = BA (B: d×r, A: r×k)
    - 只训练 A 和 B，大幅减少可训练参数
    
    关键参数:
    - r: 秩，越大表达能力越强，但参数越多
    - lora_alpha: 缩放因子，通常设为 2*r
    - target_modules: 要添加 LoRA 的层
    """
    from peft import LoraConfig, get_peft_model
    
    print("\n" + "=" * 60)
    print("  配置 LoRA")
    print("=" * 60)
    
    # LoRA 配置
    peft_config = LoraConfig(
        r=64,                    # 秩 (rank)
        lora_alpha=32,           # 缩放因子
        lora_dropout=0.1,        # Dropout
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[         # 目标层 (Transformer 中的投影层)
            'k_proj', 'v_proj', 'q_proj', 'o_proj',  # Attention
            'gate_proj', 'up_proj', 'down_proj'       # FFN
        ]
    )
    
    # 准备模型进行训练
    if device_type == "cuda":
        # CUDA: 使用 k-bit 训练准备
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        # MPS/CPU: 直接冻结原始参数
        for param in model.parameters():
            param.requires_grad = False
    
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n✓ LoRA 配置完成")
    print(f"  秩 (r): {peft_config.r}")
    print(f"  缩放因子 (alpha): {peft_config.lora_alpha}")
    print(f"  目标层: {len(peft_config.target_modules)} 个")
    print(f"  可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, peft_config


# ============================================================
# Part 3: SFT 训练
# ============================================================
def train_sft(model, tokenizer, dataset, peft_config, device_type="cuda"):
    """
    监督微调 (Supervised Fine-Tuning)
    
    使用 TRL 库的 SFTTrainer:
    - 自动处理 Chat Template
    - 支持 packing (多个短样本打包)
    - 与 PEFT/LoRA 无缝集成
    """
    from trl import SFTTrainer, SFTConfig
    
    print("\n" + "=" * 60)
    print("  Part 3: SFT 训练")
    print("=" * 60)
    
    # 根据设备类型设置训练参数
    if device_type == "cuda":
        sft_config = SFTConfig(
            output_dir="./sft_results",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            logging_steps=10,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",
            # SFT 特有参数
            dataset_text_field="text",
            max_length=512,
        )
    else:
        # MPS: 不使用 fp16，使用更保守的配置
        sft_config = SFTConfig(
            output_dir="./sft_results",
            per_device_train_batch_size=1,      # MPS 内存有限
            gradient_accumulation_steps=8,
            optim="adamw_torch",                # 标准优化器
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            logging_steps=10,
            fp16=False,                         # MPS 不支持 fp16
            bf16=False,
            gradient_checkpointing=True,
            report_to="none",
            # SFT 特有参数
            dataset_text_field="text",
            max_length=512,
        )
    
    print("\n训练配置:")
    print(f"  设备: {device_type}")
    print(f"  Batch size: {sft_config.per_device_train_batch_size}")
    print(f"  梯度累积: {sft_config.gradient_accumulation_steps}")
    print(f"  有效 batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"  学习率: {sft_config.learning_rate}")
    print(f"  Epochs: {sft_config.num_train_epochs}")
    
    # 创建 SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
        peft_config=peft_config,
    )
    
    print("\n开始 SFT 训练...")
    trainer.train()
    
    # 保存 LoRA 权重
    trainer.model.save_pretrained(SFT_OUTPUT)
    print(f"\n✓ SFT 完成，权重保存到: {SFT_OUTPUT}")
    
    return trainer


# ============================================================
# Part 4: DPO 训练
# ============================================================
def train_dpo(sft_model_path: str, tokenizer, dpo_dataset, peft_config):
    """
    直接偏好优化 (Direct Preference Optimization)
    
    DPO vs PPO:
    - PPO: 需要训练奖励模型，然后用 RL 优化
    - DPO: 直接从偏好数据学习，无需奖励模型
    
    DPO 损失函数:
    L = -log σ(β * (log π(chosen) - log π(rejected) 
                 - log π_ref(chosen) + log π_ref(rejected)))
    
    其中 β 控制 KL 散度惩罚强度
    """
    from peft import AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
    from transformers import BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer
    
    print("\n" + "=" * 60)
    print("  Part 4: DPO 训练")
    print("=" * 60)
    
    # 加载 SFT 后的模型
    print(f"\n加载 SFT 模型: {sft_model_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        sft_model_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    # 合并 SFT LoRA 权重
    merged_model = model.merge_and_unload()
    
    # 为 DPO 添加新的 LoRA
    model = prepare_model_for_kbit_training(merged_model)
    model = get_peft_model(model, peft_config)
    
    # DPO 训练配置
    training_args = DPOConfig(
        output_dir="./dpo_results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=1e-5,              # DPO 学习率通常比 SFT 小
        lr_scheduler_type="cosine",
        max_steps=200,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        report_to="none",
    )
    
    print("\nDPO 配置:")
    print(f"  β (KL 惩罚): 0.1")
    print(f"  学习率: {training_args.learning_rate}")
    print(f"  最大步数: {training_args.max_steps}")
    
    # 创建 DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,                    # KL 散度惩罚系数
        max_prompt_length=512,
        max_length=512,
    )
    
    print("\n开始 DPO 训练...")
    dpo_trainer.train()
    
    # 保存 DPO LoRA 权重
    dpo_trainer.model.save_pretrained(DPO_OUTPUT)
    print(f"\n✓ DPO 完成，权重保存到: {DPO_OUTPUT}")
    
    return dpo_trainer


# ============================================================
# Part 5: 推理与测试
# ============================================================
def inference(model_path: str = None, merged_model=None):
    """
    使用微调后的模型进行推理
    """
    import os
    import glob
    from transformers import pipeline, AutoTokenizer
    
    print("\n" + "=" * 60)
    print("  Part 5: 推理测试")
    print("=" * 60)
    
    if merged_model is None:
        # 查找最新的 checkpoint
        if os.path.isdir(model_path):
            checkpoints = glob.glob(os.path.join(model_path, "checkpoint-*"))
            if checkpoints:
                # 按步数排序，取最新的
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                model_path = checkpoints[-1]
        
        print(f"\n加载模型: {model_path}")
        
        # 加载基础模型
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        
        # 加载 LoRA 适配器
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
        merged_model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = "<PAD>"
    
    # 创建 pipeline
    pipe = pipeline(
        task="text-generation",
        model=merged_model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )
    
    # 测试提示
    prompts = [
        "Tell me something about Large Language Models.",
        "What is the difference between SFT and DPO?",
        "写一首关于人工智能的诗。",
    ]
    
    for prompt in prompts:
        formatted_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        print(f"\n[用户] {prompt}")
        output = pipe(formatted_prompt)[0]["generated_text"]
        
        # 提取助手回复
        response = output.split("<|assistant|>\n")[-1].replace("</s>", "").strip()
        print(f"[助手] {response[:500]}...")
    
    return merged_model


# ============================================================
# 教学总结
# ============================================================
def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("  Chapter 12 总结")
    print("=" * 60)
    
    summary = """
    ┌─────────────────────────────────────────────────────────┐
    │           微调生成模型的两步法                           │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  Step 1: SFT (监督微调)                                 │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  预训练模型 → 指令遵循能力                       │    │
    │  │                                                 │    │
    │  │  • 数据: (instruction, response) 对            │    │
    │  │  • 目标: 学会遵循指令格式                       │    │
    │  │  • 工具: SFTTrainer (TRL 库)                   │    │
    │  └─────────────────────────────────────────────────┘    │
    │                          ↓                              │
    │  Step 2: Preference Tuning (偏好对齐)                   │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  SFT 模型 → 符合人类偏好                        │    │
    │  │                                                 │    │
    │  │  • 数据: (prompt, chosen, rejected) 三元组     │    │
    │  │  • 方法: DPO (直接) 或 PPO (强化学习)          │    │
    │  │  • 目标: 生成人类更喜欢的回答                   │    │
    │  └─────────────────────────────────────────────────┘    │
    │                                                         │
    ├─────────────────────────────────────────────────────────┤
    │  QLoRA: 高效微调技术                                    │
    │                                                         │
    │  • Q (Quantization): 4-bit 量化，大幅降低显存          │
    │  • LoRA: 低秩适配，只训练少量参数                      │
    │  • 效果: 7B 模型只需 ~6GB 显存即可微调                 │
    │                                                         │
    │  LoRA 原理:                                             │
    │    W' = W + ΔW = W + BA                                │
    │    其中 B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)        │
    │                                                         │
    ├─────────────────────────────────────────────────────────┤
    │  DPO vs PPO                                             │
    │                                                         │
    │  PPO (强化学习):                                        │
    │    1. 训练奖励模型                                      │
    │    2. 用 RL 优化策略                                    │
    │    优点: 灵活，理论完善                                 │
    │    缺点: 复杂，需要额外模型                             │
    │                                                         │
    │  DPO (直接优化):                                        │
    │    直接从偏好数据学习，无需奖励模型                     │
    │    优点: 简单，稳定                                     │
    │    缺点: 依赖高质量偏好数据                             │
    │                                                         │
    ├─────────────────────────────────────────────────────────┤
    │  关键洞见:                                              │
    │  • SFT 教会模型「怎么说」，DPO 教会模型「说什么好」     │
    │  • QLoRA 让个人也能微调大模型                          │
    │  • 两步法是 ChatGPT/Claude 等模型的标准训练流程         │
    └─────────────────────────────────────────────────────────┘
    """
    print(summary)


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Chapter 12 - 微调生成模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["sft", "dpo", "all", "summary"],
        default="summary",
        help="运行模式: sft/dpo/all/summary"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="SFT 训练样本数 (默认: 1000)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Chapter 12 - Fine-Tuning Generation Models")
    print("=" * 60)
    
    device = get_device()
    print(f"\n设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    if args.mode == "summary":
        print_summary()
        return
    
    # 检查 GPU
    device_type = check_gpu()
    if device_type is None:
        print("\n由于没有 GPU，只显示教学总结")
        print_summary()
        return
    
    if args.mode in ["sft", "all"]:
        # Step 1: 准备数据
        sft_dataset = prepare_sft_dataset(args.num_samples)
        
        # Step 2: 加载模型
        model, tokenizer = load_model_for_training(device_type)
        
        # Step 3: 配置 LoRA
        model, peft_config = setup_lora(model, device_type)
        
        # Step 4: SFT 训练
        train_sft(model, tokenizer, sft_dataset, peft_config, device_type)
        
        clear_memory()
    
    if args.mode in ["dpo", "all"]:
        # 准备 DPO 数据
        dpo_dataset = prepare_dpo_dataset()
        
        # 重新配置 LoRA
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=64, lora_alpha=32, lora_dropout=0.1,
            bias="none", task_type="CAUSAL_LM",
            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj']
        )
        
        tokenizer_for_dpo = None
        from transformers import AutoTokenizer
        tokenizer_for_dpo = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer_for_dpo.pad_token = "<PAD>"
        tokenizer_for_dpo.padding_side = "left"
        
        # DPO 训练
        train_dpo(SFT_OUTPUT, tokenizer_for_dpo, dpo_dataset, peft_config)
        
        clear_memory()
    
    # 推理测试
    if args.mode == "all":
        inference(DPO_OUTPUT)
    elif args.mode == "sft":
        inference(SFT_OUTPUT)
    
    # 总结
    print_summary()
    
    print("\n" + "=" * 60)
    print("  训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
