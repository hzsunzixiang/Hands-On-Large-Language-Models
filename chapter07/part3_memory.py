"""
Chapter 7 - Part 3: 对话记忆 (Memory)

本节内容:
1. 手动管理对话历史 (最直接的方式)
2. 滑动窗口记忆 - 只保留最近 k 轮
3. 摘要记忆 - 将对话历史压缩为摘要

关键概念:
- LLM 本身是无状态的，每次调用都独立
- 通过在 prompt 中注入对话历史实现记忆
- 不同策略在 token 消耗和信息保留之间权衡

对比:
┌───────────────────────┬──────────────┬──────────────┬──────────┐
│ 类型                   │ Token 消耗   │ 信息保留     │ 适用场景 │
├───────────────────────┼──────────────┼──────────────┼──────────┤
│ 完整历史 (Buffer)      │ 线性增长     │ 完整保留     │ 短对话   │
│ 滑动窗口 (Window)      │ 固定 (k轮)   │ 近期保留     │ 长对话   │
│ 摘要 (Summary)         │ 缓慢增长     │ 摘要保留     │ 超长对话 │
└───────────────────────┴──────────────┴──────────────┴──────────┘
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import get_device, load_llm, PHI3_TEMPLATE_WITH_HISTORY, print_section


def memory_buffer_demo(llm):
    """
    3.1 对话缓冲记忆 (完整历史)
    手动维护对话历史列表
    """
    print_section("3.1 对话缓冲记忆 (完整历史)", level=2)
    
    prompt = PromptTemplate(
        template=PHI3_TEMPLATE_WITH_HISTORY,
        input_variables=["input_prompt", "chat_history"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # 手动维护对话历史
    chat_history = ""
    
    print("特点: 保存所有对话历史，一字不漏")
    print("缺点: 对话越长，token 消耗越大")
    print()
    
    def chat(user_input):
        nonlocal chat_history
        response = chain.invoke({
            "input_prompt": user_input,
            "chat_history": chat_history
        })
        # 更新历史
        chat_history += f"\nHuman: {user_input}\nAI: {response}"
        return response
    
    # 第一轮对话
    print("[轮次 1] 输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    result1 = chat("Hi! My name is Maarten. What is 1 + 1?")
    print(f"输出: {result1}")
    
    # 第二轮对话 - 询问名字
    print("\n[轮次 2] 输入: 'What is my name?'")
    result2 = chat("What is my name?")
    print(f"输出: {result2}")
    
    print("\n✓ LLM 成功记住了用户的名字!")
    print("  原理: 完整对话历史被注入到 {chat_history} 占位符中")


def memory_window_demo(llm):
    """
    3.2 滑动窗口记忆
    只保留最近 k 轮对话
    """
    print_section("3.2 滑动窗口记忆 (k=2)", level=2)
    
    prompt = PromptTemplate(
        template=PHI3_TEMPLATE_WITH_HISTORY,
        input_variables=["input_prompt", "chat_history"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # 用列表维护对话历史，只保留最近 k 轮
    k = 2
    history_turns = []  # [(user, ai), ...]
    
    print("特点: 只保留最近 k 轮对话，更早的被丢弃")
    print("优点: token 消耗固定，不会随对话增长")
    print(f"设置: k={k} (只保留最近 {k} 轮)")
    print()
    
    def chat(user_input):
        # 构建历史字符串 (只用最近 k 轮)
        chat_history = ""
        for human, ai in history_turns[-k:]:
            chat_history += f"\nHuman: {human}\nAI: {ai}"
        
        response = chain.invoke({
            "input_prompt": user_input,
            "chat_history": chat_history
        })
        
        # 记录本轮
        history_turns.append((user_input, response))
        return response
    
    # 对话 1: 提供名字和年龄
    print("[轮次 1] 输入: 'Hi! My name is Maarten and I am 33 years old. What is 1 + 1?'")
    chat("Hi! My name is Maarten and I am 33 years old. What is 1 + 1?")
    
    # 对话 2
    print("[轮次 2] 输入: 'What is 3 + 3?'")
    chat("What is 3 + 3?")
    
    # 对话 3: 询问名字 (窗口: 轮次2+3, 轮次1还在窗口边缘)
    print("\n[轮次 3] 输入: 'What is my name?'")
    result3 = chat("What is my name?")
    print(f"输出: {result3}")
    
    # 对话 4: 询问年龄 (窗口: 轮次3+4, 轮次1已出窗口)
    print("\n[轮次 4] 输入: 'What is my age?'")
    result4 = chat("What is my age?")
    print(f"输出: {result4}")
    
    print("\n分析:")
    print("  - 轮次1 包含名字和年龄信息")
    print(f"  - 当对话超过 k={k} 轮，轮次1 被遗忘")
    print("  - 年龄信息丢失!")


def memory_summary_demo(llm):
    """
    3.3 摘要记忆
    将对话历史压缩为摘要
    """
    print_section("3.3 摘要记忆 (Summary)", level=2)
    
    # 对话链
    chat_prompt = PromptTemplate(
        template=PHI3_TEMPLATE_WITH_HISTORY,
        input_variables=["input_prompt", "chat_history"]
    )
    chat_chain = chat_prompt | llm | StrOutputParser()
    
    # 摘要链 - 用于压缩对话历史
    summary_template = """<s><|user|>Summarize the conversations and update with the new lines.

    Current summary:
    {summary}

    new lines of conversation:
    {new_lines}

    New summary:<|end|>
    <|assistant|>"""
    
    summary_prompt = PromptTemplate(
        input_variables=["new_lines", "summary"],
        template=summary_template
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # 维护摘要状态
    current_summary = ""
    
    print("特点: 用 LLM 将对话历史压缩为摘要")
    print("优点: 能保留关键信息，token 消耗增长缓慢")
    print("缺点: 压缩过程本身需要额外的 LLM 调用")
    print()
    
    def chat(user_input):
        nonlocal current_summary
        
        # 用当前摘要作为历史
        response = chat_chain.invoke({
            "input_prompt": user_input,
            "chat_history": current_summary
        })
        
        # 更新摘要
        new_lines = f"Human: {user_input}\nAI: {response}"
        current_summary = summary_chain.invoke({
            "summary": current_summary,
            "new_lines": new_lines
        })
        
        return response
    
    # 多轮对话
    print("[轮次 1] 输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    chat("Hi! My name is Maarten. What is 1 + 1?")
    
    print("[轮次 2] 输入: 'What is my name?'")
    chat("What is my name?")
    
    print("\n[轮次 3] 输入: 'What was the first question I asked?'")
    result = chat("What was the first question I asked?")
    print(f"输出: {result}")
    
    # 查看摘要
    print("\n当前对话摘要:")
    print("-" * 40)
    print(current_summary)
    
    print("\n说明:")
    print("  - 不管对话多长，记忆大小只是一段摘要")
    print("  - 适合需要长期记忆但不需要逐字回忆的场景")


def main():
    """主函数"""
    print_section("Part 3: 对话记忆 (Memory)")
    
    print("""
LLM 本身是无状态的:
  - 每次调用都是独立的
  - 模型不知道之前说过什么
  - 需要在 prompt 中注入对话历史来实现记忆

本节演示三种记忆策略:
  1. Buffer - 保存全部历史
  2. Window - 只保留最近 k 轮
  3. Summary - 压缩为摘要
""")
    
    # 检测设备
    device, n_gpu_layers = get_device()
    
    # 加载模型
    llm = load_llm(n_gpu_layers=n_gpu_layers)
    
    # 缓冲记忆
    memory_buffer_demo(llm)
    
    # 滑动窗口记忆
    memory_window_demo(llm)
    
    # 摘要记忆
    memory_summary_demo(llm)


if __name__ == "__main__":
    main()
