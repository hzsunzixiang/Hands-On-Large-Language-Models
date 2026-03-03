"""
Chapter 12 - Part 2: SFT (Supervised Fine-Tuning) 训练
自动检测设备：CUDA → QLoRA (4-bit)，MPS/CPU → LoRA (float16/float32)

依赖: 先运行 ch12_01_sft_data_prep.py 中的数据准备逻辑（此处内联）。
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig

# ============================================================
# 0. 设备检测
# ============================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

USE_CUDA = (DEVICE == "cuda")
USE_MPS = (DEVICE == "mps")

print(f"[INFO] 检测到设备: {DEVICE}")
if USE_MPS:
    print("[INFO] MPS 模式: 使用 LoRA (非量化) + bf16 + adamw_torch")
elif USE_CUDA:
    print("[INFO] CUDA 模式: 使用 QLoRA (4-bit NF4) + fp16 + paged_adamw_32bit")
else:
    print("[INFO] CPU 模式: 使用 LoRA (非量化) + float32")

# ============================================================
# 1. 数据准备（内联 ch12_01 的逻辑）
# ============================================================
template_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def format_prompt(example):
    """Format the prompt to using the <|user|> template TinyLLama is using"""
    chat = example["messages"]
    prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}


dataset = (
    load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    .shuffle(seed=42)
    .select(range(3_000))
)
dataset = dataset.map(format_prompt)

# ============================================================
# 2. 模型加载（CUDA: 4-bit 量化 / MPS: float16 / CPU: float32）
# ============================================================
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

if USE_CUDA:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
elif USE_MPS:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(DEVICE)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"

# ============================================================
# 3. LoRA 配置
# ============================================================
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
)

if USE_CUDA:
    model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# ============================================================
# 4. 训练配置 + SFT 训练
# ============================================================
output_dir = "./results"

if USE_CUDA:
    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_seq_length=512,
    )
elif USE_MPS:
    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataset_text_field="text",
        max_seq_length=512,
    )
else:
    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        logging_steps=10,
        no_cuda=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_seq_length=512,
    )

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_arguments,
    peft_config=peft_config,
)

trainer.train()

# Save LoRA/QLoRA weights
trainer.model.save_pretrained("TinyLlama-1.1B-qlora")

# ============================================================
# 5. 合并 Adapter + 推理
# ============================================================
if USE_CUDA:
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        "TinyLlama-1.1B-qlora",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
elif USE_MPS:
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        "TinyLlama-1.1B-qlora",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(DEVICE)
else:
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        "TinyLlama-1.1B-qlora",
        low_cpu_mem_usage=True,
    )

merged_model = merged_model.merge_and_unload()

prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

pipe = pipeline(
    task="text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    device=DEVICE if DEVICE != "cuda" else None,  # CUDA 用 device_map，MPS/CPU 手动指定
)
print("\n" + "=" * 60)
print("SFT 模型推理结果:")
print("=" * 60)
print(pipe(prompt)[0]["generated_text"])
