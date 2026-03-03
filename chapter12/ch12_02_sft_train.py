"""
Chapter 12 - Part 2: SFT (Supervised Fine-Tuning) 训练
自动检测设备：CUDA → QLoRA (4-bit)，MPS/CPU → LoRA (float16/float32)

依赖: 先运行 ch12_01_sft_data_prep.py 中的数据准备逻辑（此处内联）。
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# ============================================================
# 配置 HuggingFace 镜像和超时设置
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 加速下载
# 增加超时时间
import huggingface_hub
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 300  # 5分钟

# ============================================================
# 0. 设备检测 + 按设备构建差异化参数
# ============================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

USE_CUDA = (DEVICE == "cuda")
USE_MPS = (DEVICE == "mps")

DEVICE_PROFILE = {
    "cuda": {
        "model_load_kwargs": {},
        "model_dtype": None,
        "training_overrides": dict(optim="paged_adamw_32bit", fp16=True),
    },
    "mps": {
        "model_load_kwargs": dict(dtype=torch.float16),
        "model_dtype": torch.float16,
        "training_overrides": dict(optim="adamw_torch", bf16=True, dataloader_pin_memory=False),
    },
    "cpu": {
        "model_load_kwargs": {},
        "model_dtype": None,
        "training_overrides": dict(optim="adamw_torch", no_cuda=True),
    },
}[DEVICE]

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
print("[INFO] 正在加载 tokenizer...")
try:
    template_tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        trust_remote_code=True,
        resume_download=True,
    )
    print("[INFO] Tokenizer 加载成功")
except Exception as e:
    print(f"[ERROR] Tokenizer 加载失败: {e}")
    print("[INFO] 尝试使用本地缓存...")
    template_tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        local_files_only=True,
    )


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
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# ============================================================
# 2. 模型加载（CUDA: 4-bit 量化 / MPS: float16 / CPU: float32）
# ============================================================
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

print(f"[INFO] 正在加载模型: {model_name}")
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
else:
    load_kwargs = DEVICE_PROFILE["model_load_kwargs"]
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model = model.to(DEVICE)

model.config.use_cache = False
model.config.pretraining_tp = 1

print("[INFO] 正在加载主 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, resume_download=True)
print("[INFO] Tokenizer 加载完成")
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

# ============================================================
# 4. 训练配置 + SFT 训练（统一入口，差异化参数）
# ============================================================
output_dir = "./results"

sft_base_config = dict(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    gradient_checkpointing=True,
    dataset_text_field="text",
    max_length=512,
    report_to="none",
)
sft_base_config.update(DEVICE_PROFILE["training_overrides"])
training_arguments = SFTConfig(**sft_base_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_arguments,
    peft_config=peft_config,
)

trainer.train()
trainer.model.save_pretrained("TinyLlama-1.1B-qlora")

# ============================================================
# 5. 合并 Adapter + 推理
# ============================================================
# 手动加载 base model + adapter（避免 AutoPeftModelForCausalLM 依赖 adapter_config 中的 base_model_name_or_path）
base_load_kwargs = dict(low_cpu_mem_usage=True)
if USE_CUDA:
    base_load_kwargs["device_map"] = "auto"
else:
    base_load_kwargs.update(DEVICE_PROFILE["model_load_kwargs"])

base_model = AutoModelForCausalLM.from_pretrained(model_name, **base_load_kwargs)
if not USE_CUDA:
    base_model = base_model.to(DEVICE)

merged_model = PeftModel.from_pretrained(base_model, "TinyLlama-1.1B-qlora")
merged_model = merged_model.merge_and_unload()

prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

pipe = pipeline(
    task="text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    device=DEVICE if DEVICE != "cuda" else None,
)
print("\n" + "=" * 60)
print("SFT 模型推理结果:")
print("=" * 60)
print(pipe(prompt)[0]["generated_text"])
