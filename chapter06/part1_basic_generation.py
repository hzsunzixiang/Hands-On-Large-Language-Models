"""
Chapter 6 - Part 1: 基本生成与采样参数

本节内容:
1. 加载模型并进行基本文本生成
2. 了解 chat template 的作用
3. 探索采样参数: temperature 和 top_p

关键概念:
- Chat Template: 将用户消息格式化为模型期望的输入格式
- Temperature: 控制生成的随机性 (高=更创造性, 低=更确定性)
- Top-p (Nucleus Sampling): 从累积概率前p%的token中采样
"""

from common import load_model, cleanup, print_section


def basic_generation(pipe, tokenizer):
    """
    基本生成示例
    展示如何使用 chat template 和 pipeline 进行文本生成
    """
    print_section("1.1 基本生成", level=2)
    
    # 简单提示
    messages = [
        {"role": "user", "content": "Create a funny joke about chickens."}
    ]
    
    # 查看实际发送给模型的提示
    # Chat template 会将消息列表转换为模型期望的格式
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\n实际提示模板 (Chat Template 转换后):")
    print(prompt)
    
    print("\n说明:")
    print("- Chat Template 将 {role, content} 转换为模型特定的格式")
    print("- 不同模型有不同的 template (如 <|user|>...<|assistant|>)")
    print("- 使用 pipeline 时会自动应用 template")
    
    # 生成输出
    print("\n生成输出:")
    output = pipe(messages)
    print(output[0]["generated_text"])
    
    return output


def sampling_parameters(pipe):
    """
    采样参数演示: temperature 和 top_p
    
    采样策略说明:
    - do_sample=False: 贪婪解码，总是选择概率最高的token
    - do_sample=True: 从概率分布中采样
    
    温度 (Temperature):
    - 低温度 (0.1-0.5): 更确定性，适合事实性任务
    - 中温度 (0.7-0.9): 平衡创造性和连贯性
    - 高温度 (1.0+): 更随机，适合创意写作
    
    Top-p (Nucleus Sampling):
    - 只从累积概率达到 p 的最小 token 集合中采样
    - top_p=0.9: 从概率累积90%的token中采样
    - top_p=1.0: 考虑所有token
    """
    print_section("1.2 采样参数", level=2)
    
    messages = [
        {"role": "user", "content": "Create a funny joke about chickens."}
    ]
    
    print("\n采样参数说明:")
    print("┌─────────────────┬────────────────────────────────────┐")
    print("│ do_sample=False │ 贪婪解码 - 总是选最高概率的token    │")
    print("│ temperature=0.1 │ 低随机性 - 更确定、更保守           │")
    print("│ temperature=1.0 │ 高随机性 - 更创造性、更多样         │")
    print("│ top_p=0.9       │ 从概率累积90%的token中采样          │")
    print("└─────────────────┴────────────────────────────────────┘")
    
    # 高 temperature (更随机)
    print("\n高 temperature (=1.0) - 更多创造性:")
    output = pipe(messages, do_sample=True, temperature=1.0)
    print(output[0]["generated_text"])
    
    # 高 top_p (nucleus sampling)
    print("\n高 top_p (=1.0) - nucleus sampling:")
    output = pipe(messages, do_sample=True, top_p=1.0)
    print(output[0]["generated_text"])
    
    # 对比：低温度
    print("\n低 temperature (=0.3) - 更确定性:")
    output = pipe(messages, do_sample=True, temperature=0.3)
    print(output[0]["generated_text"])


def main():
    """主函数"""
    print_section("Part 1: 基本生成与采样参数")
    
    # 加载模型
    pipe, tokenizer, model, device = load_model()
    
    try:
        # 基本生成
        basic_generation(pipe, tokenizer)
        
        # 采样参数
        sampling_parameters(pipe)
        
    finally:
        # 清理资源
        cleanup(model, tokenizer, pipe)


if __name__ == "__main__":
    main()
