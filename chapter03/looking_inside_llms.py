"""
Chapter 3 - Looking Inside Transformer LLMs
深入了解生成式 LLM 的 Transformer 架构

主要内容：
1. 加载 LLM 模型
2. Transformer LLM 的输入和输出
3. 从概率分布中选择 token (采样/解码)
4. 通过缓存 Key/Value 加速生成
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_device():
    """检测可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# Part 1: 加载 LLM 模型
# ============================================================
def load_model_and_tokenizer(model_name="microsoft/Phi-3-mini-4k-instruct"):
    """加载模型和分词器"""
    print("=" * 60)
    print("Part 1: 加载 LLM 模型")
    print("=" * 60)
    
    device = get_device()
    print(f"\n使用设备: {device}")
    print(f"加载模型: {model_name}")
    print("首次运行需要下载模型 (约 7GB)，请稍候...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 根据设备选择数据类型
    # MPS 和 CPU 使用 float32 更稳定
    if device == "cuda":
        dtype = torch.float16
        device_map = "cuda"
    else:
        dtype = torch.float32
        device_map = "cpu"  # MPS 有兼容性问题，退回 CPU
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=dtype,  # 新版 transformers 用 dtype 替代 torch_dtype
        trust_remote_code=True,
        attn_implementation="eager",  # 避免 flash_attn 兼容问题
    )
    
    print(f"模型已加载到: {device_map}")
    
    return model, tokenizer, device_map


def create_generator(model, tokenizer):
    """创建文本生成 pipeline"""
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=50,
        do_sample=False,
        use_cache=False,  # 避免 DynamicCache 兼容问题
    )
    return generator


# ============================================================
# Part 2: Transformer LLM 的输入和输出
# ============================================================
def demo_inputs_outputs(model, tokenizer, generator, device):
    """演示 Transformer LLM 的输入和输出"""
    print("\n" + "=" * 60)
    print("Part 2: Transformer LLM 的输入和输出")
    print("=" * 60)
    
    # 使用 pipeline 生成文本
    prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
    
    print(f"\n输入 Prompt:\n{prompt}")
    print("\n生成文本中...")
    
    output = generator(prompt)
    
    print(f"\n生成结果:\n{output[0]['generated_text']}")
    
    # 打印模型架构
    print("\n" + "-" * 40)
    print("模型架构:")
    print("-" * 40)
    print(model)


# ============================================================
# Part 3: 从概率分布中选择 token (采样/解码)
# ============================================================
def demo_token_selection(model, tokenizer, device):
    """演示如何从概率分布中选择下一个 token"""
    print("\n" + "=" * 60)
    print("Part 3: 从概率分布中选择 token (采样/解码)")
    print("=" * 60)
    
    prompt = "The capital of France is"
    
    print(f"\n输入 Prompt: '{prompt}'")
    
    # 将输入转换为 token IDs
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Token IDs shape: {input_ids.shape}")
    
    # 获取模型主体的输出 (lm_head 之前)
    print("\n获取模型输出...")
    with torch.no_grad():
        # 使用完整的 forward 获取 logits
        outputs = model(input_ids, use_cache=False)
        lm_head_output = outputs.logits  # [batch, seq_len, vocab_size]
    
    # 从 hidden states 推断维度
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    
    print(f"模型隐藏层维度 (hidden_size): {hidden_size}")
    print(f"  - 序列长度: {input_ids.shape[1]}")
    
    print(f"\nlm_head 输出 shape: {lm_head_output.shape}")
    print(f"  - batch_size: {lm_head_output.shape[0]}")
    print(f"  - sequence_length: {lm_head_output.shape[1]}")
    print(f"  - vocab_size: {lm_head_output.shape[2]}")
    
    # 从最后一个位置的 logits 中选择概率最高的 token
    # lm_head_output[0, -1] 是最后一个位置的 logits
    last_token_logits = lm_head_output[0, -1]
    
    # argmax 获取概率最高的 token ID
    predicted_token_id = last_token_logits.argmax(-1)
    
    # 解码 token ID 为文本
    predicted_token = tokenizer.decode(predicted_token_id)
    
    print(f"\n预测的下一个 token:")
    print(f"  Token ID: {predicted_token_id.item()}")
    print(f"  Token: '{predicted_token}'")
    
    # 显示 top-5 预测
    print("\nTop-5 预测:")
    top5_values, top5_indices = torch.topk(last_token_logits, 5)
    for i, (value, idx) in enumerate(zip(top5_values, top5_indices)):
        token = tokenizer.decode(idx)
        print(f"  {i+1}. '{token}' (logit: {value.item():.4f})")


# ============================================================
# Part 4: 通过缓存 Key/Value 加速生成
# ============================================================
def demo_kv_cache(model, tokenizer, device):
    """演示 KV Cache 对生成速度的影响"""
    print("\n" + "=" * 60)
    print("Part 4: 通过缓存 Key/Value 加速生成")
    print("=" * 60)
    
    prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
    
    print(f"\n输入 Prompt:\n{prompt}")
    
    # 将输入转换为 token IDs
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    
    print(f"\n输入 Token 数量: {input_ids.shape[1]}")
    print(f"生成 Token 数量: 100")
    
    # 测试 use_cache=True
    print("\n测试 use_cache=True (使用 KV Cache)...")
    start_time = time.time()
    try:
        with torch.no_grad():
            generation_output_cached = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                use_cache=True,
                do_sample=False,
            )
        time_cached = time.time() - start_time
        print(f"耗时: {time_cached:.2f} 秒")
    except Exception as e:
        print(f"KV Cache 模式出错 (可能是版本兼容问题): {type(e).__name__}")
        time_cached = None
        generation_output_cached = None
    
    # 测试 use_cache=False
    print("\n测试 use_cache=False (不使用 KV Cache)...")
    start_time = time.time()
    with torch.no_grad():
        generation_output_no_cache = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            use_cache=False,
            do_sample=False,
        )
    time_no_cache = time.time() - start_time
    print(f"耗时: {time_no_cache:.2f} 秒")
    
    # 比较结果
    print("\n" + "-" * 40)
    print("性能比较:")
    print("-" * 40)
    if time_cached:
        print(f"使用 KV Cache:    {time_cached:.2f} 秒")
        print(f"不使用 KV Cache:  {time_no_cache:.2f} 秒")
        print(f"加速比:           {time_no_cache/time_cached:.2f}x")
        generated_text = tokenizer.decode(generation_output_cached[0], skip_special_tokens=True)
    else:
        print(f"不使用 KV Cache:  {time_no_cache:.2f} 秒")
        print("(KV Cache 因版本兼容问题未能测试)")
        generated_text = tokenizer.decode(generation_output_no_cache[0], skip_special_tokens=True)
    
    # 显示生成的文本
    print("\n" + "-" * 40)
    print("生成的文本:")
    print("-" * 40)
    print(generated_text)


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("Chapter 3: Looking Inside Transformer LLMs")
    print("深入了解生成式 LLM 的 Transformer 架构")
    print("=" * 60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # Part 1: 加载模型
    model, tokenizer, device = load_model_and_tokenizer()
    generator = create_generator(model, tokenizer)
    
    # Part 2: 输入输出演示
    demo_inputs_outputs(model, tokenizer, generator, device)
    
    # Part 3: Token 选择演示
    demo_token_selection(model, tokenizer, device)
    
    # Part 4: KV Cache 演示
    demo_kv_cache(model, tokenizer, device)
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
