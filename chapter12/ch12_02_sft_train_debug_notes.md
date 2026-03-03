# ch12_02_sft_train.py 调试运行总结

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

原始 Notebook 基于 CUDA + bitsandbytes 实现 QLoRA（4-bit 量化 + LoRA），Mac 上需要做以下适配：

| 维度 | CUDA (原始) | MPS (Mac 适配) | CPU (兜底) |
|---|---|---|---|
| **量化** | 4-bit NF4 (bitsandbytes) | 不量化，float16 直接加载 | 不量化，float32 |
| **精度** | `fp16=True` | `bf16=True`（MPS 上 fp16 不稳定） | float32 |
| **优化器** | `paged_adamw_32bit` (bnb) | `adamw_torch`（标准 PyTorch） | `adamw_torch` |
| **device_map** | `"auto"` | 手动 `.to("mps")` | 默认 CPU |
| **其他** | — | `dataloader_pin_memory=False` | — |

> **注意**：MPS 分支丢掉了 QLoRA 中的 "Q"（量化），只保留 LoRA。因为 bitsandbytes 不支持 MPS 后端。如需完整体验 QLoRA，可改用 bitsandbytes CPU 模式（>=0.43）。

## 遇到的问题及解决方案（按时间顺序）

### 问题 1：缺少依赖模块

```
ModuleNotFoundError: No module named 'peft'
```

**原因**：conda 环境中未安装 peft、trl 等库。

**解决**：
```bash
pip install peft trl bitsandbytes accelerate sentencepiece
```

---

### 问题 2：`torch_dtype` 参数废弃

```
`torch_dtype` is deprecated! Use `dtype` instead!
```

**原因**：transformers 4.57 将 `from_pretrained()` 的 `torch_dtype` 参数改名为 `dtype`。

**解决**：将所有 `torch_dtype=...` 替换为 `dtype=...`。

---

### 问题 3：`use_mps_device` 参数废弃

```
UserWarning: `use_mps_device` is deprecated and will be removed in version 5.0 of Transformers.
```

**原因**：新版 transformers 自动检测 MPS，不再需要手动设置。

**解决**：删除 `use_mps_device=True`。

---

### 问题 4：`SFTTrainer` 不接受 `dataset_text_field` 和 `max_seq_length`

```
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field'
```

**原因**：trl 0.29 将 `dataset_text_field` 和 `max_seq_length` 从 `SFTTrainer` 构造参数移到了 `SFTConfig` 中。

**解决**：
- 用 `SFTConfig` 替代 `TrainingArguments`
- 将 `dataset_text_field` 和 `max_seq_length` 放入 `SFTConfig`

---

### 问题 5：`max_seq_length` 参数改名

```
TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'. Did you mean 'max_length'?
```

**原因**：trl 0.29 将 `max_seq_length` 改名为 `max_length`。

**解决**：`max_seq_length=512` → `max_length=512`。

---

### 问题 6：`SFTTrainer` 不接受 `tokenizer` 参数

```
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'
```

**原因**：trl 0.29 将 `tokenizer` 参数改名为 `processing_class`。

**解决**：`tokenizer=tokenizer` → `processing_class=tokenizer`。

---

### 问题 7：传入 PeftModel 与 peft_config 冲突

```
ValueError: You passed a `PeftModel` instance together with a `peft_config` to the trainer.
```

**原因**：代码先手动调用 `get_peft_model(model, peft_config)` 将模型包装成 `PeftModel`，又在 `SFTTrainer` 中传了 `peft_config`，导致双重包装冲突。

**解决**：去掉手动的 `get_peft_model()` 调用，让 `SFTTrainer` 内部通过 `peft_config` 自动应用 LoRA。

---

### 问题 8：数据集字段冲突 (`prompt` vs `completion`)

```
KeyError: 'completion'
```

**原因**：`ultrachat_200k` 数据集自带 `prompt` 字段，trl 0.29 检测到 `prompt` 存在后自动寻找 `completion` 字段（prompt-completion 模式），忽略了 `dataset_text_field="text"` 配置。

**解决**：在 `dataset.map()` 时加 `remove_columns=dataset.column_names`，只保留新建的 `text` 字段，避免字段名冲突。

```python
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
```

---

### 问题 9：TensorBoard / TensorFlow 版本冲突

```
AttributeError: module 'tensorflow' has no attribute 'io'
```

**原因**：环境中安装的 TensorFlow 与 TensorBoard 版本不兼容，导致 `SFTTrainer` 初始化 TensorBoard callback 时报错。

**解决**：在 `SFTConfig` 中禁用 TensorBoard 报告：

```python
training_arguments = SFTConfig(
    ...,
    report_to="none",
)
```

---

## trl 0.29 + transformers 4.57 API 变更速查表

| 旧 API | 新 API | 说明 |
|---|---|---|
| `TrainingArguments` | `SFTConfig` | SFT 训练配置统一到 SFTConfig |
| `SFTTrainer(dataset_text_field=...)` | `SFTConfig(dataset_text_field=...)` | 移到 config |
| `SFTTrainer(max_seq_length=...)` | `SFTConfig(max_length=...)` | 移到 config + 改名 |
| `SFTTrainer(tokenizer=...)` | `SFTTrainer(processing_class=...)` | 参数改名 |
| `from_pretrained(torch_dtype=...)` | `from_pretrained(dtype=...)` | 参数改名 |
| `TrainingArguments(use_mps_device=True)` | 自动检测，无需设置 | 已废弃 |
| 手动 `get_peft_model()` + `SFTTrainer(peft_config=...)` | 只传 `peft_config`，不手动包装 | 避免冲突 |

## 最终可运行代码的关键结构

```
设备检测 (CUDA/MPS/CPU)
    ↓
数据准备 (ultrachat_200k → chat template → 只保留 text 字段)
    ↓
模型加载 (CUDA: 4-bit量化 / MPS: float16 / CPU: float32)
    ↓
LoRA 配置 (r=64, alpha=32, 7个target modules)
    ↓
SFTConfig + SFTTrainer (自动应用 LoRA, report_to="none")
    ↓
训练 → 保存 Adapter → 合并 → 推理
```
