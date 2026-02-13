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

注意: 本节需要 OpenAI API Key
"""

import os
from common import print_section


def agent_demo():
    """
    代理演示 - 使用 ReAct 模式
    工具: 计算器 + DuckDuckGo 搜索
    """
    print_section("4.1 ReAct 代理", level=2)
    
    # 检查 OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "MY_KEY":
        print("[跳过] 需要设置 OPENAI_API_KEY 环境变量")
        print("示例: export OPENAI_API_KEY='your-api-key'")
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain.agents import load_tools, Tool, AgentExecutor, create_react_agent
        from langchain_community.tools import DuckDuckGoSearchResults
    except ImportError as e:
        print(f"[跳过] 缺少依赖: {e}")
        print("安装: pip install langchain-openai duckduckgo-search")
        return None
    
    # 加载 OpenAI LLM
    openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # ========== ReAct 提示模板 ==========
    react_template = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
    
    prompt = PromptTemplate(
        template=react_template,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )
    
    print("ReAct 提示模板结构:")
    print("┌──────────────────────────────────────────────────────┐")
    print("│ Question: 用户问题                                    │")
    print("│ Thought: LLM 思考应该做什么                           │")
    print("│ Action: 选择使用哪个工具                              │")
    print("│ Action Input: 工具的输入参数                          │")
    print("│ Observation: 工具返回的结果                           │")
    print("│ ... (重复直到找到答案)                                │")
    print("│ Final Answer: 最终答案                                │")
    print("└──────────────────────────────────────────────────────┘")
    
    # ========== 创建工具 ==========
    search = DuckDuckGoSearchResults()
    search_tool = Tool(
        name="duckduck",
        description="A web search engine. Use this for general queries.",
        func=search.run,
    )
    
    tools = load_tools(["llm-math"], llm=openai_llm)
    tools.append(search_tool)
    
    print("\n可用工具:")
    print("  1. Calculator (llm-math) - 数学计算")
    print("  2. DuckDuckGo - 网络搜索")
    
    # ========== 创建 ReAct 代理 ==========
    agent = create_react_agent(openai_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )
    
    # 运行代理
    print("\n" + "-" * 40)
    print("问题: MacBook Pro 的价格是多少？按 0.85 汇率换算成欧元是多少？")
    print("-" * 40)
    
    result = agent_executor.invoke({
        "input": "What is the current price of a MacBook Pro in USD? How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
    })
    
    print(f"\n最终答案: {result['output']}")
    
    print("\n说明:")
    print("  - Agent 自动决定先搜索价格，再计算汇率转换")
    print("  - 整个过程是自主的，LLM 自己决定使用什么工具")
    print("  - verbose=True 可以看到完整的推理过程")
    
    return agent_executor


def main():
    """主函数"""
    print_section("Part 4: 代理 (Agent)")
    
    print("""
代理 (Agent) 是什么:
  - 让 LLM 不仅能"说"，还能"做"
  - LLM 自主决定调用什么工具、何时调用
  - ReAct = Reasoning (推理) + Acting (行动)

前提条件:
  - 需要 OpenAI API Key: export OPENAI_API_KEY='your-key'
  - 安装依赖: pip install langchain-openai duckduckgo-search
""")
    
    agent_demo()


if __name__ == "__main__":
    main()
