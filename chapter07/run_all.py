"""
Chapter 7 - Advanced Text Generation 完整运行

按顺序运行所有部分，展示完整的高级文本生成技术

使用方法:
    python run_all.py              # 运行所有部分
    python run_all.py --part 1     # 只运行 Part 1
    python run_all.py --part 1 3   # 运行 Part 1 和 Part 3
    python run_all.py --summary    # 只打印总结
"""

import argparse
from common import get_device, load_llm, print_section


def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("Chapter 7 总结: 高级文本生成技术")
    print("=" * 60)
    
    print("""
┌──────────────────────────────────────────────────────────┐
│  LangChain 核心概念                                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. 链 (Chains)                                          │
│     ┌─────────┐   ┌─────────┐   ┌─────────┐            │
│     │ Prompt  │ → │   LLM   │ → │ Output  │            │
│     └─────────┘   └─────────┘   └─────────┘            │
│                                                          │
│  2. 多链组合                                              │
│     ┌────────┐   ┌────────┐   ┌────────┐              │
│     │ Chain1 │ → │ Chain2 │ → │ Chain3 │              │
│     │ (标题) │   │ (角色) │   │ (故事) │              │
│     └────────┘   └────────┘   └────────┘              │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  记忆 (Memory) 类型                                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  类型                  │  特点                 │ 适用场景 │
│  ─────────────────────┼─────────────────────┼────────── │
│  ConversationBuffer   │  保存全部历史         │ 短对话   │
│  ConversationWindow   │  只保留最近 k 轮      │ 长对话   │
│  ConversationSummary  │  压缩为摘要          │ 超长对话  │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  代理 (Agent) - ReAct 模式                                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  思考 (Thought) → 行动 (Action) → 观察 (Observation)      │
│       ↑                                    │            │
│       └────────────────────────────────────┘            │
│                   (循环直到得出答案)                      │
│                                                          │
│  工具示例:                                                │
│    - Calculator: 数学计算                                │
│    - DuckDuckGo: 网络搜索                                │
│    - 自定义工具: API 调用、数据库查询等                   │
│                                                          │
└──────────────────────────────────────────────────────────┘

关键收获:
- Chain 让 LLM 调用更模块化、可组合
- Memory 解决 LLM 无状态问题，实现多轮对话
- Agent 让 LLM 能调用外部工具，扩展能力边界
- ReAct = Reasoning + Acting，让 LLM 边思考边行动
""")


def run_part1(llm):
    """Part 1: 加载模型与基本链"""
    from part1_load_and_chain import test_basic_invoke, basic_chain_demo
    
    print_section("Part 1: 加载模型与基本链")
    test_basic_invoke(llm)
    basic_chain_demo(llm)


def run_part2(llm):
    """Part 2: 多链组合"""
    from part2_chains import multiple_chains_demo
    
    print_section("Part 2: 多链组合")
    multiple_chains_demo(llm)


def run_part3(llm):
    """Part 3: 对话记忆"""
    from part3_memory import memory_buffer_demo, memory_window_demo, memory_summary_demo
    
    print_section("Part 3: 对话记忆")
    memory_buffer_demo(llm)
    memory_window_demo(llm)
    # 摘要记忆比较耗时（每轮额外调用 LLM 生成摘要），按需开启
    memory_summary_demo(llm)


def run_part4():
    """Part 4: 代理 (独立于本地 LLM，使用 OpenAI)"""
    from part4_agent import agent_demo
    
    print_section("Part 4: 代理 (Agent)")
    agent_demo()


def main():
    parser = argparse.ArgumentParser(description='Chapter 7 - Advanced Text Generation')
    parser.add_argument('--part', nargs='*', type=int,
                       help='指定要运行的部分 (1-4)，不指定则运行所有')
    parser.add_argument('--summary', action='store_true',
                       help='只打印总结')
    args = parser.parse_args()
    
    if args.summary:
        print_summary()
        return
    
    # 确定要运行的部分
    if args.part:
        parts_to_run = args.part
    else:
        parts_to_run = [1, 2, 3, 4]
    
    print("=" * 60)
    print("Chapter 7 - Advanced Text Generation")
    print("=" * 60)
    print(f"将运行: Part {', '.join(map(str, parts_to_run))}")
    
    # Part 1-3 共享同一个本地 LLM
    needs_local_llm = any(p in parts_to_run for p in [1, 2, 3])
    
    if needs_local_llm:
        device, n_gpu_layers = get_device()
        llm = load_llm(n_gpu_layers=n_gpu_layers)
        
        if 1 in parts_to_run:
            run_part1(llm)
        if 2 in parts_to_run:
            run_part2(llm)
        if 3 in parts_to_run:
            run_part3(llm)
    
    # Part 4 使用 OpenAI API，独立运行
    if 4 in parts_to_run:
        run_part4()
    
    # 总结
    print_summary()


if __name__ == "__main__":
    main()
