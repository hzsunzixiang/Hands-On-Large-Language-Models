"""
Part 3: KV Cache 加速生成
演示 use_cache=True 如何加速文本生成

核心原理:
- 不使用 cache: 每生成一个新 token，都要重新计算所有之前 token 的 K, V
- 使用 cache: 缓存之前 token 的 K, V，只计算新 token 的 K, V, Q
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("Part 3: KV Cache 加速生成")
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
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# 准备输入
prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print(f"\n输入提示: '{prompt[:50]}...'")
print(f"输入 token 数: {input_ids.shape[1]}")

# ============================================================
# 对比测试: 有无 KV Cache
# ============================================================
print("\n" + "-" * 60)
print("对比测试: KV Cache 的效果")
print("-" * 60)

max_new_tokens = 50
n_runs = 3

# 定义清理缓存函数（支持多种设备）
def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

# 测试 use_cache=True
print(f"\n测试 use_cache=True ({n_runs} 次运行)...")
times_with_cache = []
for i in range(n_runs):
    clear_cache()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - start
    times_with_cache.append(elapsed)
    print(f"  运行 {i+1}: {elapsed:.3f}s")

avg_with_cache = sum(times_with_cache) / len(times_with_cache)
print(f"  平均: {avg_with_cache:.3f}s")

# 测试 use_cache=False
print(f"\n测试 use_cache=False ({n_runs} 次运行)...")
times_without_cache = []
for i in range(n_runs):
    clear_cache()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - start
    times_without_cache.append(elapsed)
    print(f"  运行 {i+1}: {elapsed:.3f}s")

avg_without_cache = sum(times_without_cache) / len(times_without_cache)
print(f"  平均: {avg_without_cache:.3f}s")

# 结果对比
print("\n" + "-" * 60)
print("结果对比")
print("-" * 60)
speedup = avg_without_cache / avg_with_cache
print(f"""
  use_cache=True:  {avg_with_cache:.3f}s
  use_cache=False: {avg_without_cache:.3f}s
  
  加速比: {speedup:.2f}x
""")

# 生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n生成的文本:")
print("-" * 60)
print(generated_text)

# ============================================================
# 原理解释
# ============================================================
print("\n" + "-" * 60)
print("KV Cache 原理解释")
print("-" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│  不使用 KV Cache (use_cache=False)                          │
├─────────────────────────────────────────────────────────────┤
│  生成 token 1: 计算 [t0, t1, t2, t3, t4] 的 K, V, Q         │
│  生成 token 2: 重新计算 [t0, t1, t2, t3, t4, t5] 的 K, V, Q │
│  生成 token 3: 重新计算 [t0, ..., t6] 的 K, V, Q            │
│  ...                                                        │
│  复杂度: O(n²) - 每次都重新计算所有位置                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  使用 KV Cache (use_cache=True)                             │
├─────────────────────────────────────────────────────────────┤
│  生成 token 1: 计算 [t0, t1, t2, t3, t4] 的 K, V, Q         │
│                缓存 K, V                                    │
│  生成 token 2: 只计算 t5 的 K, V, Q，复用缓存的 K, V        │
│  生成 token 3: 只计算 t6 的 K, V, Q，复用缓存的 K, V        │
│  ...                                                        │
│  复杂度: O(n) - 每次只计算新 token                          │
└─────────────────────────────────────────────────────────────┘

Self-Attention 计算:
  Q @ K^T -> Attention Scores -> Softmax -> @ V

KV Cache 的关键:
  - Q 只需要当前 token 的
  - K, V 需要所有历史 token 的 (可以缓存!)
""")

print("\n" + "=" * 60)
print("Part 3 完成!")
print("=" * 60)
