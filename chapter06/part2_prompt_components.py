"""
Chapter 6 - Part 2: 提示的组成部分

本节内容:
构建一个完整的提示，包含以下7个组件:
1. Persona (角色) - 定义 AI 的身份和专业领域
2. Instruction (指令) - 明确告诉模型要做什么
3. Context (上下文) - 提供背景信息和约束
4. Format (格式) - 指定输出格式
5. Audience (受众) - 说明输出的目标读者
6. Tone (语气) - 指定语言风格
7. Data (数据) - 需要处理的输入内容

关键技巧:
- 越清晰的指令 → 越好的输出
- 不一定每次都需要所有组件，根据任务选择
- 顺序可以调整，但逻辑要清晰
"""

from common import load_model, cleanup, print_section


def demonstrate_prompt_components(pipe, tokenizer):
    """
    演示提示的7个组成部分
    任务: 总结 Transformer 论文的关键发现
    """
    print_section("提示组件详解", level=2)
    
    # 待总结的文本 (Transformer 论文介绍)
    text = """The Transformer model utilizes attention mechanisms to improve the speed 
and performance of deep learning models. It outperforms previous models in 
machine translation tasks and enables parallelization for faster training."""
    
    # ========== 7个提示组件 ==========
    
    # 1. Persona (角色) - 让模型扮演特定身份
    persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries.\n"
    
    # 2. Instruction (指令) - 核心任务描述
    instruction = "Summarize the key findings of the paper provided.\n"
    
    # 3. Context (上下文) - 背景信息和约束
    context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper.\n"
    
    # 4. Format (格式) - 输出结构要求
    data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results.\n"
    
    # 5. Audience (受众) - 目标读者
    audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models.\n"
    
    # 6. Tone (语气) - 语言风格
    tone = "The tone should be professional and clear.\n"
    
    # 7. Data (数据) - 输入内容
    data = f"Text to summarize: {text}"
    
    # ========== 组合完整提示 ==========
    query = persona + instruction + context + data_format + audience + tone + data
    
    # 打印各组件
    print("\n提示的7个组件:")
    print("=" * 50)
    
    components = [
        ("1. Persona (角色)", persona, "定义 AI 的身份，使其回答更专业"),
        ("2. Instruction (指令)", instruction, "核心任务，必须清晰明确"),
        ("3. Context (上下文)", context, "补充背景，约束输出范围"),
        ("4. Format (格式)", data_format, "指定输出结构"),
        ("5. Audience (受众)", audience, "调整语言复杂度"),
        ("6. Tone (语气)", tone, "控制语言风格"),
        ("7. Data (数据)", data[:50] + "...", "需要处理的输入"),
    ]
    
    for name, content, description in components:
        print(f"\n{name}")
        print(f"  作用: {description}")
        print(f"  内容: {content.strip()[:60]}...")
    
    # 发送给模型
    messages = [{"role": "user", "content": query}]
    
    print("\n" + "=" * 50)
    print("完整提示模板 (前500字符):")
    print("=" * 50)
    template = tokenizer.apply_chat_template(messages, tokenize=False)
    print(template[:500] + "...")
    
    print("\n" + "=" * 50)
    print("模型生成的摘要:")
    print("=" * 50)
    outputs = pipe(messages)
    print(outputs[0]["generated_text"])


def compare_with_without_components(pipe):
    """
    对比: 有组件 vs 无组件的提示效果
    """
    print_section("对比: 简单提示 vs 完整提示", level=2)
    
    text = """The Transformer model utilizes attention mechanisms to improve the speed 
and performance of deep learning models."""
    
    # 简单提示 (无组件)
    simple_prompt = [
        {"role": "user", "content": f"Summarize: {text}"}
    ]
    
    # 完整提示 (有组件)
    full_prompt = [
        {"role": "user", "content": f"""You are an expert in Large Language models.
Summarize the key findings for busy researchers.
Use bullet points for methods, then a paragraph for results.
Be professional and clear.
Text: {text}"""}
    ]
    
    print("\n简单提示:")
    print(f"  'Summarize: {text[:50]}...'")
    print("\n输出:")
    outputs = pipe(simple_prompt)
    print(outputs[0]["generated_text"])
    
    print("\n" + "-" * 40)
    
    print("\n完整提示 (含角色、格式、语气等):")
    print("  [见上方组件]")
    print("\n输出:")
    outputs = pipe(full_prompt)
    print(outputs[0]["generated_text"])
    
    print("\n结论:")
    print("  - 完整提示通常产生更结构化、更符合预期的输出")
    print("  - 简单任务可以用简单提示，复杂任务建议添加更多组件")


def main():
    """主函数"""
    print_section("Part 2: 提示的组成部分")
    
    # 加载模型
    pipe, tokenizer, model, device = load_model()
    
    try:
        # 演示提示组件
        demonstrate_prompt_components(pipe, tokenizer)
        
        # 对比实验
        compare_with_without_components(pipe)
        
    finally:
        # 清理资源
        cleanup(model, tokenizer, pipe)


if __name__ == "__main__":
    main()
