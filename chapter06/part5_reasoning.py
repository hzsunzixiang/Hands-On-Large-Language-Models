"""
Chapter 6 - Part 5: 推理技术 (Reasoning Techniques)

本节内容:
1. Chain-of-Thought (CoT) - 思维链推理
   - Few-shot CoT: 提供带推理过程的示例
   - Zero-shot CoT: "Let's think step-by-step"
2. Tree-of-Thought (ToT) - 思维树推理
   - 模拟多专家协作

关键概念:
- CoT 让模型"展示工作过程"，而不是直接给出答案
- 研究表明 CoT 显著提升数学推理和逻辑推理能力
- "Let's think step-by-step" 是一个简单但有效的技巧
"""

from common import load_model, cleanup, print_section


def few_shot_cot_demo(pipe):
    """
    Few-shot Chain-of-Thought 演示
    通过示例展示推理过程
    """
    print_section("5.1 Few-shot Chain-of-Thought", level=2)
    
    print("核心思想: 在示例中展示推理步骤，模型会模仿这种推理模式")
    print()
    
    # 提供带推理过程的示例
    cot_prompt = [
        {
            "role": "user", 
            "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
        },
        {
            "role": "assistant", 
            "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."
        },
        {
            "role": "user", 
            "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"
        }
    ]
    
    print("示例结构:")
    print("┌──────────────────────────────────────────────────────┐")
    print("│ 示例问题: Roger 有 5 个网球，买了 2 罐，每罐 3 个     │")
    print("│ 示例答案: 5 开始 → 2×3=6 新买 → 5+6=11              │")
    print("│ 实际问题: 食堂有 23 个苹果，用了 20，又买 6          │")
    print("└──────────────────────────────────────────────────────┘")
    
    print("\n关键: 示例答案展示了完整的推理步骤!")
    
    outputs = pipe(cot_prompt)
    print(f"\n模型输出:\n{outputs[0]['generated_text']}")
    
    print("\n分析:")
    print("  - 模型学会了分步骤解题")
    print("  - 先陈述初始状态，再计算变化，最后得出答案")


def zero_shot_cot_demo(pipe):
    """
    Zero-shot Chain-of-Thought 演示
    只需在提示末尾添加 "Let's think step-by-step"
    """
    print_section("5.2 Zero-shot Chain-of-Thought", level=2)
    
    print("核心技巧: 在问题末尾添加 'Let's think step-by-step.'")
    print("来源: Kojima et al. (2022) 'Large Language Models are Zero-Shot Reasoners'")
    print()
    
    # 对比: 不加 CoT 触发词
    without_cot = [
        {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
    ]
    
    # 加 CoT 触发词
    with_cot = [
        {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step."}
    ]
    
    print("不加 'Let's think step-by-step':")
    print("-" * 40)
    outputs = pipe(without_cot)
    print(outputs[0]["generated_text"])
    
    print("\n加 'Let's think step-by-step':")
    print("-" * 40)
    outputs = pipe(with_cot)
    print(outputs[0]["generated_text"])
    
    print("\n其他有效的 CoT 触发词:")
    print("  - 'Let's think step-by-step.'")
    print("  - 'Let's work this out step by step.'")
    print("  - 'Take a deep breath and think step by step.'")
    print("  - '让我们一步一步思考。' (中文)")


def complex_reasoning_demo(pipe):
    """
    复杂推理问题演示
    """
    print_section("5.3 复杂推理问题", level=2)
    
    # 逻辑推理问题
    logic_problem = """
    A farmer needs to cross a river with a wolf, a goat, and a cabbage. 
    The boat can only carry the farmer and one item at a time.
    If left alone, the wolf will eat the goat, and the goat will eat the cabbage.
    How can the farmer get everything across safely?
    Let's think step-by-step.
    """
    
    print("问题: 农夫过河 (狼、羊、白菜)")
    print("-" * 40)
    
    prompt = [{"role": "user", "content": logic_problem.strip()}]
    outputs = pipe(prompt, max_new_tokens=800)
    print(outputs[0]["generated_text"])


def tree_of_thought_demo(pipe):
    """
    Tree-of-Thought 演示
    模拟多专家协作解决问题
    """
    print_section("5.4 Tree-of-Thought (思维树)", level=2)
    
    print("核心思想:")
    print("  - Chain-of-Thought 是单条推理路径")
    print("  - Tree-of-Thought 是多条推理路径并行探索")
    print("  - 模拟多专家讨论，错误的专家会被淘汰")
    print()
    
    tot_prompt = [
        {"role": "user", "content": """Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. 
If any expert realises they're wrong at any point then they leave.

The question is 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?'

Make sure to discuss the results."""}
    ]
    
    print("提示结构:")
    print("┌──────────────────────────────────────────────────────┐")
    print("│ 1. 假设有3位专家                                      │")
    print("│ 2. 每位专家写下一步思考                               │")
    print("│ 3. 分享并讨论                                         │")
    print("│ 4. 发现错误的专家退出                                 │")
    print("│ 5. 重复直到得出答案                                   │")
    print("└──────────────────────────────────────────────────────┘")
    
    print("\n模型输出:")
    print("-" * 40)
    outputs = pipe(tot_prompt, max_new_tokens=800)
    print(outputs[0]["generated_text"])
    
    print("\n分析:")
    print("  - ToT 通过多角度验证减少错误")
    print("  - 适合复杂问题，但消耗更多 tokens")
    print("  - 可以结合自洽性 (Self-Consistency) 进一步提升准确性")


def main():
    """主函数"""
    print_section("Part 5: 推理技术 (Reasoning)")
    
    print("""
推理技术总结:
┌───────────────────┬────────────────────────────────────┐
│ 技术              │ 特点                               │
├───────────────────┼────────────────────────────────────┤
│ Few-shot CoT      │ 提供带推理步骤的示例               │
│ Zero-shot CoT     │ 添加 "Let's think step-by-step"    │
│ Tree-of-Thought   │ 多专家/多路径并行推理              │
│ Self-Consistency  │ 多次采样取多数投票 (未演示)        │
└───────────────────┴────────────────────────────────────┘

研究发现:
- CoT 在数学推理任务上提升 10-30% 准确率
- 模型越大，CoT 效果越明显
- 简单问题不需要 CoT，复杂问题强烈推荐
""")
    
    # 加载模型
    pipe, tokenizer, model, device = load_model()
    
    try:
        # Few-shot CoT
        few_shot_cot_demo(pipe)
        
        # Zero-shot CoT
        zero_shot_cot_demo(pipe)
        
        # 复杂推理
        complex_reasoning_demo(pipe)
        
        # Tree-of-Thought
        tree_of_thought_demo(pipe)
        
    finally:
        # 清理资源
        cleanup(model, tokenizer, pipe)


if __name__ == "__main__":
    main()
