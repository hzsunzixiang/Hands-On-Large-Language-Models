"""
Chapter 6 - Part 3: 上下文学习 (In-Context Learning)

本节内容:
1. Zero-shot Learning - 无示例，直接提问
2. One-shot Learning - 提供1个示例
3. Few-shot Learning - 提供多个示例

关键概念:
- In-Context Learning: 模型通过提示中的示例学习任务模式
- 不需要微调模型参数，只需要构造好的提示
- 示例质量比数量更重要

适用场景:
- 教模型使用新词汇或概念
- 约束输出格式
- 引导特定的推理模式
"""

from common import load_model, cleanup, print_section


def zero_shot_demo(pipe):
    """
    Zero-shot Learning: 无示例直接提问
    模型完全依赖预训练知识
    """
    print_section("3.1 Zero-shot Learning", level=2)
    
    print("特点: 不提供任何示例，直接提问")
    print("适用: 模型预训练时已学过的常见任务")
    
    prompt = [
        {"role": "user", "content": "What is the sentiment of this sentence? 'I love this product!' Answer with positive, negative, or neutral."}
    ]
    
    print("\n提示:")
    print("  'What is the sentiment of this sentence? \"I love this product!\"'")
    
    print("\n输出:")
    outputs = pipe(prompt)
    print(outputs[0]["generated_text"])


def one_shot_demo(pipe, tokenizer):
    """
    One-shot Learning: 提供1个示例
    用于教模型新任务或新概念
    """
    print_section("3.2 One-shot Learning", level=2)
    
    print("特点: 提供1个示例，教模型理解任务模式")
    print("适用: 教模型使用虚构词汇或特定格式")
    
    # 教模型使用虚构词汇
    one_shot_prompt = [
        {
            "role": "user",
            "content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
        },
        {
            "role": "assistant",
            "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."
        },
        {
            "role": "user",
            "content": "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
        }
    ]
    
    print("\n示例结构:")
    print("┌──────────────────────────────────────────────────────┐")
    print("│ User: 定义 'Gigamuru' = 日本乐器                      │")
    print("│ Assistant: 用 'Gigamuru' 造句的示例                   │")
    print("│ User: 定义 'screeg' = 挥剑，请造句                    │")
    print("└──────────────────────────────────────────────────────┘")
    
    print("\n完整提示模板:")
    print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
    
    print("\n模型输出:")
    outputs = pipe(one_shot_prompt)
    print(outputs[0]["generated_text"])
    
    print("\n分析:")
    print("  - 模型通过示例学会了如何使用虚构词汇造句")
    print("  - 'Gigamuru' 和 'screeg' 是虚构的词，模型预训练时未见过")
    print("  - 但通过 one-shot 示例，模型理解了任务模式")


def few_shot_demo(pipe):
    """
    Few-shot Learning: 提供多个示例
    更复杂的任务可能需要多个示例
    """
    print_section("3.3 Few-shot Learning", level=2)
    
    print("特点: 提供多个示例，强化模型对任务的理解")
    print("适用: 复杂任务、需要特定输出格式")
    
    # 情感分析 Few-shot
    few_shot_prompt = [
        {"role": "user", "content": "Review: 'This movie was fantastic!' Sentiment:"},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "Review: 'Terrible waste of time.' Sentiment:"},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "Review: 'It was okay, nothing special.' Sentiment:"},
        {"role": "assistant", "content": "neutral"},
        {"role": "user", "content": "Review: 'Absolutely loved every minute of it!' Sentiment:"}
    ]
    
    print("\n示例结构 (3个示例):")
    print("┌──────────────────────────────────────────────────────┐")
    print("│ 示例1: 'fantastic!' → positive                       │")
    print("│ 示例2: 'Terrible waste' → negative                   │")
    print("│ 示例3: 'okay, nothing special' → neutral             │")
    print("│ 任务: 'loved every minute' → ?                       │")
    print("└──────────────────────────────────────────────────────┘")
    
    print("\n模型输出:")
    outputs = pipe(few_shot_prompt)
    print(outputs[0]["generated_text"])
    
    print("\n分析:")
    print("  - 通过3个示例，模型学会了只输出 positive/negative/neutral")
    print("  - 示例展示了期望的输出格式 (单词而非句子)")


def comparison_demo(pipe):
    """
    对比: Zero-shot vs One-shot vs Few-shot
    """
    print_section("3.4 对比实验", level=2)
    
    task = "将以下英文翻译成 emoji: 'I love sunny days'"
    
    # Zero-shot
    zero_shot = [
        {"role": "user", "content": f"{task}"}
    ]
    
    # One-shot
    one_shot = [
        {"role": "user", "content": "将以下英文翻译成 emoji: 'Hello world'"},
        {"role": "assistant", "content": "👋🌍"},
        {"role": "user", "content": task}
    ]
    
    # Few-shot
    few_shot = [
        {"role": "user", "content": "将以下英文翻译成 emoji: 'Hello world'"},
        {"role": "assistant", "content": "👋🌍"},
        {"role": "user", "content": "将以下英文翻译成 emoji: 'I am happy'"},
        {"role": "assistant", "content": "😊"},
        {"role": "user", "content": "将以下英文翻译成 emoji: 'Good night'"},
        {"role": "assistant", "content": "🌙😴"},
        {"role": "user", "content": task}
    ]
    
    print(f"\n任务: {task}")
    print("\n" + "-" * 40)
    
    print("\nZero-shot (无示例):")
    outputs = pipe(zero_shot)
    print(f"  输出: {outputs[0]['generated_text']}")
    
    print("\nOne-shot (1个示例):")
    outputs = pipe(one_shot)
    print(f"  输出: {outputs[0]['generated_text']}")
    
    print("\nFew-shot (3个示例):")
    outputs = pipe(few_shot)
    print(f"  输出: {outputs[0]['generated_text']}")
    
    print("\n结论:")
    print("  - 更多示例通常带来更稳定、更符合预期的输出")
    print("  - 但示例太多会占用 context window，需要权衡")
    print("  - 高质量示例比大量低质量示例更有效")


def main():
    """主函数"""
    print_section("Part 3: 上下文学习 (In-Context Learning)")
    
    # 加载模型
    pipe, tokenizer, model, device = load_model()
    
    try:
        # Zero-shot
        zero_shot_demo(pipe)
        
        # One-shot
        one_shot_demo(pipe, tokenizer)
        
        # Few-shot
        few_shot_demo(pipe)
        
        # 对比实验
        comparison_demo(pipe)
        
    finally:
        # 清理资源
        cleanup(model, tokenizer, pipe)


if __name__ == "__main__":
    main()
