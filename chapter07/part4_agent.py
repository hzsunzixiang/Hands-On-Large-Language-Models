"""
Chapter 7 - Part 4: 代理 (Agent) - ReAct 模式

本节内容:
让 LLM 使用外部工具 (计算器、搜索引擎) 解决问题

关键概念:
- Agent: 让 LLM 决定何时调用什么工具
- ReAct: Reasoning + Acting，边思考边行动
- Tool: 代理可以调用的外部能力 (计算、搜索等)

ReAct 循环:
  思考 (Thought) → 行动 (Action) → 观察 (Observation)
       ↑                                    │
       └────────────────────────────────────┘
                  (循环直到得出答案)

注意: 本节需要 DeepSeek API Key
"""

import os
from common import print_section

# DeepSeek API 配置
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def _load_api_key():
    """从环境变量或 ~/.deepseek 文件加载 DeepSeek API Key"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        key_file = os.path.expanduser("~/.deepseek")
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                api_key = f.read().strip()
    return api_key


def agent_demo():
    """
    代理演示 - 使用 ReAct 模式 (langgraph 实现)
    工具: 计算器 + DuckDuckGo 搜索
    """
    print_section("4.1 ReAct 代理", level=2)
    
    # 加载 DeepSeek API Key
    api_key = _load_api_key()
    if not api_key:
        print("[跳过] 需要设置 DeepSeek API Key")
        print("方式1: export DEEPSEEK_API_KEY='your-key'")
        print("方式2: 将 key 写入 ~/.deepseek 文件")
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain_community.tools import DuckDuckGoSearchResults
        from langgraph.prebuilt import create_react_agent
    except ImportError as e:
        print(f"[跳过] 缺少依赖: {e}")
        print("安装: pip install langchain-openai langgraph duckduckgo-search")
        return None
    
    # 使用 DeepSeek API (兼容 OpenAI 接口)
    print(f"使用模型: {DEEPSEEK_MODEL} (DeepSeek API)")
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_key=api_key,
        openai_api_base=DEEPSEEK_API_BASE,
        temperature=0,
    )
    
    # ========== 创建工具 ==========
    # 工具1: 计算器 (用 @tool 装饰器定义)
    @tool
    def calculator(expression: str) -> str:
        """Calculate a math expression. Input should be a valid math expression like '2 + 3 * 4'."""
        try:
            # 安全地计算数学表达式
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    # 工具2: DuckDuckGo 搜索
    search = DuckDuckGoSearchResults()
    
    tools = [calculator, search]
    
    print("\n可用工具:")
    print("  1. calculator - 数学计算")
    print("  2. DuckDuckGo - 网络搜索")
    
    print("\nReAct 工作流程:")
    print("┌──────────────────────────────────────────────────────┐")
    print("│ 用户提问                                              │")
    print("│   → LLM 思考: 需要什么信息?                           │")
    print("│   → 调用工具: 搜索/计算                               │")
    print("│   → 观察结果                                          │")
    print("│   → 继续思考或给出最终答案                             │")
    print("└──────────────────────────────────────────────────────┘")
    
    # ========== 创建 ReAct 代理 (langgraph) ==========
    agent = create_react_agent(llm, tools)
    
    # 运行代理
    print("\n" + "-" * 40)
    print("问题: MacBook Pro 的价格是多少？按 0.85 汇率换算成欧元是多少？")
    print("-" * 40 + "\n")
    
    result = agent.invoke({
        "messages": [
            ("user", "What is the current price of a MacBook Pro in USD? How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?")
        ]
    })
    
    # 打印完整的推理过程
    print("\n===== 推理过程 =====")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        if role == "HumanMessage":
            print(f"\n[用户] {msg.content}")
        elif role == "AIMessage":
            if msg.content:
                print(f"\n[AI 思考] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[调用工具] {tc['name']}({tc['args']})")
        elif role == "ToolMessage":
            print(f"[工具结果] {msg.content[:200]}")
    
    # 最终答案
    final_msg = result["messages"][-1]
    print(f"\n{'=' * 40}")
    print(f"最终答案: {final_msg.content}")
    
    print("\n说明:")
    print("  - Agent 自动决定先搜索价格，再计算汇率转换")
    print("  - 整个过程是自主的，LLM 自己决定使用什么工具")
    print("  - langgraph 的 create_react_agent 自动处理 ReAct 循环")
    
    return agent


def main():
    """主函数"""
    print_section("Part 4: 代理 (Agent)")
    
    print("""
代理 (Agent) 是什么:
  - 让 LLM 不仅能"说"，还能"做"
  - LLM 自主决定调用什么工具、何时调用
  - ReAct = Reasoning (推理) + Acting (行动)

前提条件:
  - DeepSeek API Key: 从 ~/.deepseek 文件自动读取，或 export DEEPSEEK_API_KEY='your-key'
  - 安装依赖: pip install langchain-openai langgraph duckduckgo-search
""")
    
    agent_demo()


if __name__ == "__main__":
    main()
