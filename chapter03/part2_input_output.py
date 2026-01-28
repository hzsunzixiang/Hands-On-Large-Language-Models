"""
Part 2: LLM 的输入和输出
深入理解 Transformer LLM 的前向传播过程：
- 输入 token IDs
- 模型内部的隐藏状态
- lm_head 输出 logits
- 选择下一个 token
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("Part 2: LLM 的输入和输出")
print("=" * 60)

# 检测设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\n使用设备: {device}")

# 加载模型
MODEL_NAME = "gpt2"
print(f"加载模型: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# ============================================================
# Step 1: Token 化输入
# ============================================================
print("\n" + "-" * 60)
print("Step 1: Token 化输入")
print("-" * 60)

prompt = "The capital of France is"
print(f"\n输入文本: '{prompt}'")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(f"Token IDs: {input_ids}")
print(f"Token IDs 形状: {input_ids.shape}  # [batch_size, seq_len]")

# 展示每个 token
print("\n各个 Token:")
for i, token_id in enumerate(input_ids[0]):
    token = tokenizer.decode(token_id)
    print(f"  位置 {i}: ID={token_id.item():5d} -> '{token}'")

# ============================================================
# Step 2: 通过模型主体 (不包含 lm_head)
# ============================================================
print("\n" + "-" * 60)
print("Step 2: 通过模型主体 (Transformer layers)")
print("-" * 60)

with torch.no_grad():
    # GPT-2 使用 model.transformer 而不是 model.model
    if hasattr(model, 'transformer'):
        model_output = model.transformer(input_ids)
        hidden_states = model_output.last_hidden_state
    else:
        model_output = model.model(input_ids)
        hidden_states = model_output[0]

print(f"\n隐藏状态形状: {hidden_states.shape}")
print(f"  - batch_size: {hidden_states.shape[0]}")
print(f"  - seq_len: {hidden_states.shape[1]} (输入 token 数量)")
print(f"  - hidden_dim: {hidden_states.shape[2]} (模型隐藏层维度)")

print(f"\n最后一个位置的隐藏状态 (用于预测下一个 token):")
last_hidden = hidden_states[0, -1, :]
print(f"  形状: {last_hidden.shape}")
print(f"  前5个值: {last_hidden[:5].cpu().numpy()}")

# ============================================================
# Step 3: 通过 lm_head 得到 logits
# ============================================================
print("\n" + "-" * 60)
print("Step 3: 通过 lm_head 得到 logits")
print("-" * 60)

with torch.no_grad():
    lm_head_output = model.lm_head(hidden_states)

print(f"\nlm_head 输出形状: {lm_head_output.shape}")
print(f"  - batch_size: {lm_head_output.shape[0]}")
print(f"  - seq_len: {lm_head_output.shape[1]}")
print(f"  - vocab_size: {lm_head_output.shape[2]} (词表大小，每个位置预测所有词的分数)")

# ============================================================
# Step 4: 选择下一个 token
# ============================================================
print("\n" + "-" * 60)
print("Step 4: 选择下一个 token (Greedy Decoding)")
print("-" * 60)

# 取最后一个位置的 logits
last_position_logits = lm_head_output[0, -1, :]
print(f"\n最后位置的 logits 形状: {last_position_logits.shape}")

# 找到分数最高的 token (argmax = greedy decoding)
predicted_token_id = last_position_logits.argmax(-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"\n预测的 token ID: {predicted_token_id.item()}")
print(f"预测的 token: '{predicted_token}'")
print(f"\n完整句子: '{prompt}{predicted_token}'")

# ============================================================
# Step 5: Top-K 候选 tokens
# ============================================================
print("\n" + "-" * 60)
print("Step 5: Top-K 候选 tokens")
print("-" * 60)

# 转换为概率
probs = torch.softmax(last_position_logits, dim=-1)

# 获取 top 10 候选
top_k = 10
top_probs, top_indices = torch.topk(probs, top_k)

print(f"\nTop {top_k} 候选 tokens:")
print("-" * 50)
for i in range(top_k):
    token_id = top_indices[i].item()
    prob = top_probs[i].item()
    token = tokenizer.decode(token_id)
    bar = "█" * int(prob * 50)
    print(f"  {token:15s} (ID={token_id:5d}): {prob:.4f} {bar}")

# ============================================================
# 总结
# ============================================================
print("\n" + "-" * 60)
print("LLM 前向传播流程总结")
print("-" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│  输入: "The capital of France is"                           │
│                    ↓                                        │
│  Tokenizer: [464, 3139, 286, 4881, 318]                     │
│                    ↓                                        │
│  Embedding: [5, 768] (5个token, 768维向量)                   │
│                    ↓                                        │
│  Transformer Layers (12层 self-attention + FFN)             │
│                    ↓                                        │
│  Hidden States: [5, 768]                                    │
│                    ↓                                        │
│  lm_head: 768 -> 50257 (词表大小)                           │
│                    ↓                                        │
│  Logits: [5, 50257] (每个位置对所有词的分数)                 │
│                    ↓                                        │
│  取最后位置 + argmax                                        │
│                    ↓                                        │
│  预测: "Paris"                                              │
└─────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 60)
print("Part 2 完成!")
print("=" * 60)
