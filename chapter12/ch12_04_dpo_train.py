"""
Chapter 12 - Part 4: DPO (Direct Preference Optimization) 训练
自动检测设备：CUDA → QLoRA (4-bit) / MPS → LoRA (float16) / CPU → LoRA (float32)

前置条件: 需要先运行 ch12_02_sft_train.py 生成 TinyLlama-1.1B-qlora 目录。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import DPOConfig, DPOTrainer

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

# 按设备差异化的参数（只记录不同的部分）
DEVICE_PROFILE = {
    "cuda": {
        "model_load_kwargs": {},          # CUDA 用 device_map="auto" + quantization_config
        "model_dtype": None,              # 由 quantization_config 控制
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
# 辅助函数：统一的模型加载 / 设备放置
# ============================================================
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def load_base_model():
    """加载 base model，CUDA 走量化+device_map，MPS/CPU 走 dtype+手动 to"""
    load_kwargs = dict(low_cpu_mem_usage=True)
    if USE_CUDA:
        from transformers import BitsAndBytesConfig
        load_kwargs["device_map"] = "auto"
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs.update(DEVICE_PROFILE["model_load_kwargs"])

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **load_kwargs)
    if not USE_CUDA:
        model = model.to(DEVICE)
    return model


def load_adapter(base_model, adapter_path):
    """在 base_model 上加载 LoRA adapter"""
    model = PeftModel.from_pretrained(base_model, adapter_path)
    if not USE_CUDA:
        model = model.to(DEVICE)
    return model


# ============================================================
# 1. DPO 数据准备（内联 ch12_03 的逻辑）
# ============================================================
def format_prompt(example):
    """Format the prompt to using the <|user|> template TinyLLama is using"""
    system = "<|system|>\n" + example['system'] + "</s>\n"
    prompt = "<|user|>\n" + example['input'] + "</s>\n<|assistant|>\n"
    chosen = example['chosen'] + "</s>\n"
    rejected = example['rejected'] + "</s>\n"
    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


dpo_dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
dpo_dataset = dpo_dataset.filter(
    lambda r:
        r["status"] != "tie" and
        r["chosen_score"] >= 8 and
        not r["in_gsm8k_train"]
)
dpo_dataset = dpo_dataset.map(format_prompt, remove_columns=dpo_dataset.column_names)

# ============================================================
# 2. 加载 SFT Adapter 权重 + 合并为基础模型
# ============================================================
base_model = load_base_model()
sft_model = load_adapter(base_model, "TinyLlama-1.1B-qlora")
merged_model = sft_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=False)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"

# ============================================================
# 3. LoRA 配置（用于 DPO）
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
    merged_model = prepare_model_for_kbit_training(merged_model)

# ============================================================
# 4. DPO 训练配置 + 训练（统一入口，差异化参数）
# ============================================================
output_dir = "./results_dpo"

dpo_base_config = dict(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    logging_steps=10,
    gradient_checkpointing=True,
    warmup_ratio=0.1,
    report_to="none",
    max_length=512,
)
dpo_base_config.update(DEVICE_PROFILE["training_overrides"])
training_arguments = DPOConfig(**dpo_base_config)

dpo_trainer = DPOTrainer(
    merged_model,
    args=training_arguments,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    beta=0.1,
)

dpo_trainer.train()
dpo_trainer.model.save_pretrained("TinyLlama-1.1B-dpo-qlora")

# ============================================================
# 5. 合并两层 Adapter (SFT + DPO) + 推理
# ============================================================
base_model2 = load_base_model()
sft_merged = load_adapter(base_model2, "TinyLlama-1.1B-qlora").merge_and_unload()
dpo_model = load_adapter(sft_merged, "TinyLlama-1.1B-dpo-qlora").merge_and_unload()

prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

pipe = pipeline(
    task="text-generation",
    model=dpo_model,
    tokenizer=tokenizer,
    device=DEVICE if DEVICE != "cuda" else None,
)
print("\n" + "=" * 60)
print("DPO 模型推理结果:")
print("=" * 60)
print(pipe(prompt)[0]["generated_text"])
