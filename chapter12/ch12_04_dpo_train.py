"""
Chapter 12 - Part 4: DPO (Direct Preference Optimization) 训练
加载 SFT 阶段的 QLoRA 权重 → 4-bit 量化 → LoRA 配置 → DPO 训练 → 合并两层 Adapter → 推理验证

前置条件: 需要先运行 ch12_02_sft_train.py 生成 TinyLlama-1.1B-qlora 目录。
需要 GPU 环境。
"""

from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    AutoPeftModelForCausalLM,
    PeftModel,
)
from trl import DPOConfig, DPOTrainer


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
# 2. 加载 SFT QLoRA 权重 + 4-bit 量化
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=bnb_config,
)
merged_model = model.merge_and_unload()

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
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

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# ============================================================
# 4. DPO 训练配置 + 训练
# ============================================================
output_dir = "./results"

training_arguments = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    warmup_ratio=0.1,
)

dpo_trainer = DPOTrainer(
    model,
    args=training_arguments,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=512,
    max_length=512,
)

dpo_trainer.train()

# Save DPO adapter
dpo_trainer.model.save_pretrained("TinyLlama-1.1B-dpo-qlora")

# ============================================================
# 5. 合并两层 Adapter (SFT + DPO) + 推理
# ============================================================
# 先合并 SFT LoRA
model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    low_cpu_mem_usage=True,
    device_map="auto",
)
sft_model = model.merge_and_unload()

# 再合并 DPO LoRA
dpo_model = PeftModel.from_pretrained(
    sft_model,
    "TinyLlama-1.1B-dpo-qlora",
    device_map="auto",
)
dpo_model = dpo_model.merge_and_unload()

# 推理
prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

pipe = pipeline(task="text-generation", model=dpo_model, tokenizer=tokenizer)
print("\n" + "=" * 60)
print("DPO 模型推理结果:")
print("=" * 60)
print(pipe(prompt)[0]["generated_text"])
