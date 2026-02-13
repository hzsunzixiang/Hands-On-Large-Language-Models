"""
Chapter 6 - Prompt Engineering 完整运行

按顺序运行所有部分，展示完整的提示工程技术

使用方法:
    python run_all.py           # 运行所有部分
    python run_all.py --part 1  # 只运行 Part 1
    python run_all.py --part 3 5  # 运行 Part 3 和 Part 5
"""

import argparse
from common import load_model, cleanup, print_section


def print_chapter_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("Chapter 6 总结: 提示工程技术")
    print("=" * 60)
    
    print("""
┌──────────────────────────────────────────────────────────┐
│  提示的基本组成部分                                        │
├──────────────────────────────────────────────────────────┤
│  1. Persona (角色)    - 定义 AI 的身份和专业领域           │
│  2. Instruction (指令) - 明确告诉模型要做什么              │
│  3. Context (上下文)   - 提供背景信息                      │
│  4. Format (格式)      - 指定输出格式                      │
│  5. Audience (受众)    - 说明输出的目标读者                │
│  6. Tone (语气)        - 指定语言风格                      │
│  7. Data (数据)        - 需要处理的输入内容                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  高级提示技术                                              │
├──────────────────────────────────────────────────────────┤
│  In-Context Learning   - 通过示例教模型                    │
│    - Zero-shot: 无示例                                    │
│    - One-shot: 1个示例                                    │
│    - Few-shot: 多个示例                                   │
│                                                          │
│  Chain Prompting       - 将任务分解为多个步骤              │
│    - 前一步的输出作为下一步的输入                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  推理技术                                                  │
├──────────────────────────────────────────────────────────┤
│  Chain-of-Thought (CoT) - 思维链                          │
│    - Few-shot CoT: 提供带推理的示例                       │
│    - Zero-shot CoT: "Let's think step-by-step"           │
│                                                          │
│  Tree-of-Thought (ToT)  - 思维树                          │
│    - 模拟多专家协作                                       │
│    - 探索多条推理路径                                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  输出验证                                                  │
├──────────────────────────────────────────────────────────┤
│  示例约束       - 通过示例展示期望的输出格式               │
│  语法约束       - 使用 GBNF 语法强制输出格式               │
│  JSON Mode      - response_format={"type": "json_object"} │
└──────────────────────────────────────────────────────────┘

关键技巧:
- 越清晰的指令 → 越好的输出
- 提供示例可以大幅提高输出质量
- "Let's think step-by-step" 显著提升推理能力
- 语法约束可以保证输出格式正确
""")


def run_part1(pipe, tokenizer):
    """Part 1: 基本生成与采样参数"""
    from part1_basic_generation import basic_generation, sampling_parameters
    
    print_section("Part 1: 基本生成与采样参数")
    basic_generation(pipe, tokenizer)
    sampling_parameters(pipe)


def run_part2(pipe, tokenizer):
    """Part 2: 提示的组成部分"""
    from part2_prompt_components import demonstrate_prompt_components, compare_with_without_components
    
    print_section("Part 2: 提示的组成部分")
    demonstrate_prompt_components(pipe, tokenizer)
    compare_with_without_components(pipe)


def run_part3(pipe, tokenizer):
    """Part 3: 上下文学习"""
    from part3_in_context_learning import zero_shot_demo, one_shot_demo, few_shot_demo, comparison_demo
    
    print_section("Part 3: 上下文学习 (In-Context Learning)")
    zero_shot_demo(pipe)
    one_shot_demo(pipe, tokenizer)
    few_shot_demo(pipe)
    comparison_demo(pipe)


def run_part4(pipe):
    """Part 4: 链式提示"""
    from part4_chain_prompting import basic_chain_demo, article_writing_chain, iterative_refinement_chain
    
    print_section("Part 4: 链式提示 (Chain Prompting)")
    basic_chain_demo(pipe)
    article_writing_chain(pipe)
    iterative_refinement_chain(pipe)


def run_part5(pipe):
    """Part 5: 推理技术"""
    from part5_reasoning import few_shot_cot_demo, zero_shot_cot_demo, complex_reasoning_demo, tree_of_thought_demo
    
    print_section("Part 5: 推理技术 (Reasoning)")
    few_shot_cot_demo(pipe)
    zero_shot_cot_demo(pipe)
    complex_reasoning_demo(pipe)
    tree_of_thought_demo(pipe)


def run_part6(pipe):
    """Part 6: 输出格式控制"""
    from part6_output_format import example_based_formatting, instruction_based_formatting, json_validation_demo
    
    print_section("Part 6: 输出格式控制")
    example_based_formatting(pipe)
    instruction_based_formatting(pipe)
    json_validation_demo(pipe)


def run_grammar_demo():
    """运行语法约束演示 (独立于主模型)"""
    from part6_output_format import grammar_constrained_demo
    grammar_constrained_demo()


def main():
    parser = argparse.ArgumentParser(description='Chapter 6 - Prompt Engineering')
    parser.add_argument('--part', nargs='*', type=int, 
                       help='指定要运行的部分 (1-6)，不指定则运行所有')
    parser.add_argument('--summary', action='store_true',
                       help='只打印总结')
    args = parser.parse_args()
    
    if args.summary:
        print_chapter_summary()
        return
    
    # 确定要运行的部分
    if args.part:
        parts_to_run = args.part
    else:
        parts_to_run = [1, 2, 3, 4, 5, 6]
    
    print("=" * 60)
    print("Chapter 6 - Prompt Engineering")
    print("=" * 60)
    print(f"将运行: Part {', '.join(map(str, parts_to_run))}")
    
    # 加载模型
    pipe, tokenizer, model, device = load_model()
    
    try:
        # 运行各部分
        if 1 in parts_to_run:
            run_part1(pipe, tokenizer)
        
        if 2 in parts_to_run:
            run_part2(pipe, tokenizer)
        
        if 3 in parts_to_run:
            run_part3(pipe, tokenizer)
        
        if 4 in parts_to_run:
            run_part4(pipe)
        
        if 5 in parts_to_run:
            run_part5(pipe)
        
        if 6 in parts_to_run:
            run_part6(pipe)
            
    finally:
        # 清理 transformers 模型
        cleanup(model, tokenizer, pipe)
    
    # 语法约束演示 (需要独立运行，因为使用不同的库)
    if 6 in parts_to_run:
        run_grammar_demo()
    
    # 总结
    print_chapter_summary()


if __name__ == "__main__":
    main()
