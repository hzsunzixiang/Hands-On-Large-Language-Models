# ch12_04_dpo_train.py 调试运行总结

## 环境信息

| 项目 | 版本 |
|---|---|
| 系统 | macOS (Apple Silicon, MPS) |
| Python | 3.13 (conda env: d2l_3.13) |
| transformers | 4.57.6 |
| trl | 0.29.0 |
| peft | 最新版 |
| PyTorch | 支持 MPS 后端 |

## 原始 Notebook 到 Mac MPS 的适配改动

与 `ch12_02_sft_train.py` 一致的三路设备适配：

| 维度 | CUDA (原始) | MPS (Mac 适配) | CPU (兜底) |
|---|---|---|---|
| **量化** | 4-bit NF4 (bitsandbytes) | 不量化，float16 直接加载 | 不量化，float32 |
| **精度** | `fp16=True` | `bf16=True` | float32 |
| **优化器** | `paged_adamw_32bit` | `adamw_torch` | `adamw_torch` |
| **device_map** | `"auto"` | 手动 `.to("mps")` | 默认 CPU |
| **其他** | — | `dataloader_pin_memory=False` | — |

## 遇到的问题及解决方案（按时间顺序）

### 问题 1：`AutoPeftModelForCausalLM` 加载失败 — `base_model_name_or_path` 为 null

```
RepositoryNotFoundError: 404 Client Error.
Repository Not Found for url: https://huggingface.co/None/resolve/main/config.json
```

**原因**：`ch12_02` 中 `SFTTrainer` 自动包装 LoRA 并保存 adapter 时，`adapter_config.json` 中的 `base_model_name_or_path` 被写为 `null`。`AutoPeftModelForCausalLM.from_pretrained()` 读取该字段去 HuggingFace 下载 base model，拼出了 `https://huggingface.co/None/...`。

**验证**：
```bash
$ cat TinyLlama-1.1B-qlora/adapter_config.json | grep base_model
  "base_model_name_or_path": null,
```

**解决**：放弃 `AutoPeftModelForCausalLM`（它依赖 adapter_config 中的 base model 路径），改为手动两步加载：

```python
# 1. 显式加载 base model（路径由代码控制，不依赖 adapter_config）
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, ...)
# 2. 在 base model 上挂载 adapter
peft_model = PeftModel.from_pretrained(base_model, "TinyLlama-1.1B-qlora")
# 3. 合并
merged = peft_model.merge_and_unload()
```

> 此修复同步应用到了 `ch12_02_sft_train.py` 的合并推理阶段。

---

### 问题 2：`DPOConfig` 不接受 `max_prompt_length`

```
TypeError: DPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'
```

**原因**：trl 0.29 的 `DPOConfig` 移除了 `max_prompt_length` 参数（DPO 内部自动处理 prompt 截断），只保留 `max_length`。

**验证**：
```python
>>> [p for p in DPOConfig.__init__.__signature__.parameters if 'max' in p]
['max_grad_norm', 'max_steps', 'max_length']  # 没有 max_prompt_length
```

**解决**：删除 `max_prompt_length=512`，只保留 `max_length=512`。

---

### 问题 3：`DPOTrainer` 不接受 `beta`

```
TypeError: DPOTrainer.__init__() got an unexpected keyword argument 'beta'
```

**原因**：trl 0.29 将 `beta` 从 `DPOTrainer()` 构造参数移到了 `DPOConfig` 中统一管理。

**解决**：将 `beta=0.1` 从 `DPOTrainer(...)` 移到 `dpo_base_config` 字典中：

```python
dpo_base_config = dict(
    ...,
    beta=0.1,  # 从 DPOTrainer 移到 DPOConfig
)
training_arguments = DPOConfig(**dpo_base_config)
```

---

## 代码重构：统一入口 + 差异化参数

原始代码对每个设备写一整块 if/elif/else 的 `DPOConfig(...)`，参数大量重复且难以维护。重构为：

### 1. `DEVICE_PROFILE` 字典集中管理设备差异

```python
DEVICE_PROFILE = {
    "cuda": { "training_overrides": dict(optim="paged_adamw_32bit", fp16=True), ... },
    "mps":  { "training_overrides": dict(optim="adamw_torch", bf16=True, dataloader_pin_memory=False), ... },
    "cpu":  { "training_overrides": dict(optim="adamw_torch", no_cuda=True), ... },
}[DEVICE]
```

### 2. base_config + update 模式

```python
dpo_base_config = dict(output_dir=..., learning_rate=..., ...)  # 所有设备共享
dpo_base_config.update(DEVICE_PROFILE["training_overrides"])     # 覆盖设备差异项
training_arguments = DPOConfig(**dpo_base_config)                # 统一一行构造
```

### 3. 辅助函数封装模型加载

```python
def load_base_model():
    """CUDA: 量化+device_map / MPS: float16+to / CPU: 默认"""
    ...

def load_adapter(base_model, adapter_path):
    """在 base_model 上挂载 LoRA adapter"""
    ...
```

调用方简洁：
```python
sft_merged = load_adapter(load_base_model(), "TinyLlama-1.1B-qlora").merge_and_unload()
dpo_model = load_adapter(sft_merged, "TinyLlama-1.1B-dpo-qlora").merge_and_unload()
```

---

## trl 0.29 DPO 相关 API 变更速查表

| 旧 API | 新 API | 说明 |
|---|---|---|
| `DPOTrainer(beta=...)` | `DPOConfig(beta=...)` | 移到 config |
| `DPOTrainer(max_prompt_length=...)` | 已移除 | DPO 内部自动处理 |
| `DPOTrainer(max_length=...)` | `DPOConfig(max_length=...)` | 移到 config |
| `DPOTrainer(tokenizer=...)` | `DPOTrainer(processing_class=...)` | 参数改名 |
| `AutoPeftModelForCausalLM.from_pretrained(adapter_path)` | `AutoModelForCausalLM` + `PeftModel.from_pretrained` | 避免 `base_model_name_or_path: null` |
| 手动 `get_peft_model()` + `DPOTrainer(peft_config=...)` | 只传 `peft_config`，不手动包装 | 避免冲突 |

## 最终可运行代码的关键结构

```
设备检测 (CUDA/MPS/CPU) → DEVICE_PROFILE
    ↓
数据准备 (intel-orca-dpo-pairs → 过滤 → prompt/chosen/rejected 三元组)
    ↓
加载 base model → 挂载 SFT adapter → merge_and_unload → 得到 SFT merged model
    ↓
LoRA 配置 (r=64, alpha=32, 7个target modules)
    ↓
dpo_base_config + DEVICE_PROFILE overrides → DPOConfig → DPOTrainer
    ↓
训练 → 保存 DPO adapter
    ↓
合并两层 Adapter: base → +SFT adapter → merge → +DPO adapter → merge → 推理
```
