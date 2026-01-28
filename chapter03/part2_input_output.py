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
# Step 2: 获取 Embedding (Transformer 之前的词向量)
# ============================================================
print("\n" + "-" * 60)
print("Step 2: 获取 Embedding (Transformer 之前的词向量)")
print("-" * 60)

with torch.no_grad():
    # 获取 token embedding (只是查表，还没有经过 Transformer)
    if hasattr(model, 'transformer'):
        # GPT-2 的 embedding 层
        wte = model.transformer.wte  # word token embedding
        wpe = model.transformer.wpe  # word position embedding
        token_embeddings = wte(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        position_embeddings = wpe(position_ids)
        input_embeddings = token_embeddings + position_embeddings
    else:
        embed_tokens = model.model.embed_tokens
        input_embeddings = embed_tokens(input_ids)

print(f"\n输入 Embedding 形状: {input_embeddings.shape}")
print(f"  - batch_size: {input_embeddings.shape[0]}")
print(f"  - seq_len: {input_embeddings.shape[1]} (输入 token 数量)")
print(f"  - embed_dim: {input_embeddings.shape[2]} (embedding 维度)")

# 打印每个词的 embedding 前几个维度
print("\n【Transformer 之前】每个词的 Embedding (前8维):")
print("-" * 70)
for i, token_id in enumerate(input_ids[0]):
    token = tokenizer.decode(token_id)
    embed = input_embeddings[0, i, :8].cpu().numpy()
    embed_str = ", ".join([f"{v:+.4f}" for v in embed])
    print(f"  位置 {i} '{token:10s}': [{embed_str}, ...]")

# ============================================================
# Step 3: 通过模型主体 (Transformer layers)
# ============================================================
print("\n" + "-" * 60)
print("Step 3: 通过模型主体 (Transformer layers)")
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

# 打印每个词经过 Transformer 后的向量
print("\n【Transformer 之后】每个词的隐藏状态 (前8维):")
print("-" * 70)
for i, token_id in enumerate(input_ids[0]):
    token = tokenizer.decode(token_id)
    hidden = hidden_states[0, i, :8].cpu().numpy()
    hidden_str = ", ".join([f"{v:+.4f}" for v in hidden])
    print(f"  位置 {i} '{token:10s}': [{hidden_str}, ...]")

# 对比 Transformer 前后的变化
print("\n" + "-" * 60)
print("【对比】Transformer 前后向量的变化")
print("-" * 60)
print("\n每个位置向量的 L2 变化量 (欧氏距离):")
for i, token_id in enumerate(input_ids[0]):
    token = tokenizer.decode(token_id)
    before = input_embeddings[0, i, :].cpu()
    after = hidden_states[0, i, :].cpu()
    diff = torch.norm(after - before).item()
    bar = "█" * int(diff * 2)
    print(f"  位置 {i} '{token:10s}': L2 距离 = {diff:.4f} {bar}")

print(f"\n最后一个位置的隐藏状态 (用于预测下一个 token):")
last_hidden = hidden_states[0, -1, :]
print(f"  形状: {last_hidden.shape}")
print(f"  前5个值: {last_hidden[:5].cpu().numpy()}")

# ============================================================
# Step 4: 通过 lm_head 得到 logits
# ============================================================
print("\n" + "-" * 60)
print("Step 4: 通过 lm_head 得到 logits")
print("-" * 60)

with torch.no_grad():
    lm_head_output = model.lm_head(hidden_states)

print(f"\nlm_head 输出形状: {lm_head_output.shape}")
print(f"  - batch_size: {lm_head_output.shape[0]}")
print(f"  - seq_len: {lm_head_output.shape[1]}")
print(f"  - vocab_size: {lm_head_output.shape[2]} (词表大小，每个位置预测所有词的分数)")

# ============================================================
# Step 5: 选择下一个 token
# ============================================================
print("\n" + "-" * 60)
print("Step 5: 选择下一个 token (Greedy Decoding)")
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
# Step 6: Top-K 候选 tokens
# ============================================================
print("\n" + "-" * 60)
print("Step 6: Top-K 候选 tokens")
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
