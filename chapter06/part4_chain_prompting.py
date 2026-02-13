"""
Chapter 6 - Part 4: 链式提示 (Chain Prompting)

本节内容:
将复杂任务分解为多个步骤，前一步的输出作为下一步的输入

关键概念:
- 复杂任务往往难以用单个提示完成
- 分解任务可以提高每一步的准确性
- 中间结果可以被检查和修正

应用场景:
- 多步骤写作 (大纲 → 段落 → 润色)
- 代码生成 (需求 → 设计 → 实现 → 测试)
- 内容创作 (构思 → 初稿 → 审核 → 发布)
"""

from common import load_model, cleanup, print_section


def basic_chain_demo(pipe):
    """
    基础链式提示演示
    任务: 为一个聊天机器人创建名称、口号，然后生成销售文案
    """
    print_section("4.1 基础链式提示", level=2)
    
    print("任务分解:")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ Step 1: 创建产品名称和口号                           │")
    print("│    ↓                                                │")
    print("│ Step 2: 基于名称和口号生成销售文案                    │")
    print("└─────────────────────────────────────────────────────┘")
    
    # Step 1: 创建产品名称和口号
    print("\n步骤 1: 创建产品名称和口号")
    print("-" * 40)
    
    product_prompt = [
        {"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}
    ]
    outputs = pipe(product_prompt)
    product_description = outputs[0]["generated_text"]
    print(f"输出:\n{product_description}")
    
    # Step 2: 基于第一步的输出生成销售文案
    print("\n步骤 2: 基于名称生成销售文案")
    print("-" * 40)
    print(f"输入 (来自步骤1): '{product_description[:50]}...'")
    
    sales_prompt = [
        {"role": "user", "content": f"Generate a very short sales pitch for the following product: '{product_description}'"}
    ]
    outputs = pipe(sales_prompt)
    sales_pitch = outputs[0]["generated_text"]
    print(f"\n输出:\n{sales_pitch}")
    
    return product_description, sales_pitch


def article_writing_chain(pipe):
    """
    文章写作链式提示
    任务: 写一篇关于 AI 的短文
    """
    print_section("4.2 文章写作链", level=2)
    
    print("任务分解:")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ Step 1: 生成大纲                                     │")
    print("│    ↓                                                │")
    print("│ Step 2: 基于大纲写开头段落                           │")
    print("│    ↓                                                │")
    print("│ Step 3: 润色和优化                                   │")
    print("└─────────────────────────────────────────────────────┘")
    
    topic = "The future of AI in healthcare"
    
    # Step 1: 生成大纲
    print(f"\n步骤 1: 为 '{topic}' 生成大纲")
    print("-" * 40)
    
    outline_prompt = [
        {"role": "user", "content": f"Create a brief outline (3 main points) for an article about: {topic}"}
    ]
    outputs = pipe(outline_prompt)
    outline = outputs[0]["generated_text"]
    print(f"输出:\n{outline}")
    
    # Step 2: 基于大纲写开头
    print("\n步骤 2: 基于大纲写开头段落")
    print("-" * 40)
    
    intro_prompt = [
        {"role": "user", "content": f"Based on this outline:\n{outline}\n\nWrite an engaging introduction paragraph (2-3 sentences)."}
    ]
    outputs = pipe(intro_prompt)
    intro = outputs[0]["generated_text"]
    print(f"输出:\n{intro}")
    
    # Step 3: 润色
    print("\n步骤 3: 润色和优化")
    print("-" * 40)
    
    polish_prompt = [
        {"role": "user", "content": f"Polish this introduction to make it more professional and engaging:\n{intro}"}
    ]
    outputs = pipe(polish_prompt)
    polished = outputs[0]["generated_text"]
    print(f"输出:\n{polished}")
    
    return outline, intro, polished


def iterative_refinement_chain(pipe):
    """
    迭代优化链式提示
    任务: 生成并优化一个概念解释
    """
    print_section("4.3 迭代优化链", level=2)
    
    print("任务分解:")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ Step 1: 初始解释                                     │")
    print("│    ↓                                                │")
    print("│ Step 2: 简化语言                                     │")
    print("│    ↓                                                │")
    print("│ Step 3: 添加类比                                     │")
    print("└─────────────────────────────────────────────────────┘")
    
    concept = "Transformer attention mechanism"
    
    # Step 1: 初始解释
    print(f"\n步骤 1: 解释 '{concept}'")
    print("-" * 40)
    
    explain_prompt = [
        {"role": "user", "content": f"Explain {concept} in 2-3 sentences."}
    ]
    outputs = pipe(explain_prompt)
    initial = outputs[0]["generated_text"]
    print(f"输出:\n{initial}")
    
    # Step 2: 简化
    print("\n步骤 2: 简化语言 (面向初学者)")
    print("-" * 40)
    
    simplify_prompt = [
        {"role": "user", "content": f"Simplify this explanation for a beginner:\n{initial}"}
    ]
    outputs = pipe(simplify_prompt)
    simplified = outputs[0]["generated_text"]
    print(f"输出:\n{simplified}")
    
    # Step 3: 添加类比
    print("\n步骤 3: 添加日常生活类比")
    print("-" * 40)
    
    analogy_prompt = [
        {"role": "user", "content": f"Add a simple real-world analogy to this explanation:\n{simplified}"}
    ]
    outputs = pipe(analogy_prompt)
    with_analogy = outputs[0]["generated_text"]
    print(f"输出:\n{with_analogy}")
    
    return initial, simplified, with_analogy


def main():
    """主函数"""
    print_section("Part 4: 链式提示 (Chain Prompting)")
    
    print("""
链式提示的优势:
1. 将复杂任务分解为简单步骤
2. 每一步都可以检查和修正
3. 中间结果可以作为参考
4. 更容易调试和优化

注意事项:
- 每一步的提示要清晰
- 确保步骤之间的输入输出匹配
- 控制每一步的输出长度，避免上下文溢出
""")
    
    # 加载模型
    pipe, tokenizer, model, device = load_model()
    
    try:
        # 基础链式提示
        basic_chain_demo(pipe)
        
        # 文章写作链
        article_writing_chain(pipe)
        
        # 迭代优化链
        iterative_refinement_chain(pipe)
        
    finally:
        # 清理资源
        cleanup(model, tokenizer, pipe)


if __name__ == "__main__":
    main()
