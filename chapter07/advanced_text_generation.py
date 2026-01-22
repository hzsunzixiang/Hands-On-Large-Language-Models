"""
Chapter 7 - Advanced Text Generation Techniques and Tools
超越基础提示工程的高级技术

本章内容:
1. 使用 LangChain 框架
2. 链 (Chains) - 组合多个 LLM 调用
3. 记忆 (Memory) - 让 LLM 记住对话历史
4. 代理 (Agents) - 让 LLM 使用工具
"""

import warnings
warnings.filterwarnings("ignore")

import os
import torch


def get_device():
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        device = "cuda"
        n_gpu_layers = -1
        print(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        n_gpu_layers = -1  # llama.cpp 支持 Metal
        print("使用设备: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        n_gpu_layers = 0
        print("使用设备: CPU")
    return device, n_gpu_layers


def download_model():
    """下载 Phi-3 GGUF 模型"""
    import urllib.request
    
    model_path = "Phi-3-mini-4k-instruct-q4_k_m.gguf"
    
    if os.path.exists(model_path):
        print(f"模型已存在: {model_path}")
        return model_path
    
    print("下载 Phi-3 GGUF 模型...")
    url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4_k_m.gguf"
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"下载完成: {model_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载模型或使用 HuggingFace 自动下载")
        model_path = None
    
    return model_path


def load_llm_langchain(model_path=None, n_gpu_layers=-1):
    """
    Part 1: 使用 LangChain 加载 LLM
    """
    print("\n" + "=" * 60)
    print("Part 1: 加载 LLM (LangChain + llama.cpp)")
    print("=" * 60)
    
    try:
        from langchain_community.llms import LlamaCpp
    except ImportError:
        from langchain.llms import LlamaCpp
    
    if model_path is None:
        # 使用 HuggingFace 自动下载
        from llama_cpp import Llama
        llm_raw = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",  # 实际可用的文件名
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False
        )
        model_path = llm_raw.model_path
        del llm_raw
    
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        max_tokens=500,
        n_ctx=2048,
        seed=42,
        verbose=False
    )
    
    print("模型加载完成!")
    
    # 简单测试
    print("\n测试生成:")
    response = llm.invoke("Hi! My name is Maarten. What is 1 + 1?")
    print(f"输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    print(f"输出: {response}")
    
    return llm


def basic_chain_demo(llm):
    """
    Part 2: 基本链 (Basic Chain)
    使用 PromptTemplate 和 LLM 组合
    """
    print("\n" + "=" * 60)
    print("Part 2: 基本链 (Chain)")
    print("=" * 60)
    
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        from langchain import PromptTemplate
    
    # 创建提示模板 (Phi-3 格式)
    template = """<s><|user|>
    {input_prompt}<|end|>
    <|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["input_prompt"]
    )
    
    # 使用 LCEL (LangChain Expression Language) 创建链
    basic_chain = prompt | llm
    
    print("\n链的结构: PromptTemplate -> LLM")
    print("-" * 40)
    
    # 使用链
    response = basic_chain.invoke(
        {"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"}
    )
    print(f"输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    print(f"输出: {response}")
    
    return basic_chain


def multiple_chains_demo(llm):
    """
    Part 3: 多链组合 (Multiple Chains)
    将多个链串联起来完成复杂任务
    """
    print("\n" + "=" * 60)
    print("Part 3: 多链组合 - 故事生成器")
    print("=" * 60)
    
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        from langchain import PromptTemplate, LLMChain
    
    # 链 1: 生成标题
    title_template = """<s><|user|>
    Create a title for a story about {summary}. Only return the title.<|end|>
    <|assistant|>"""
    title_prompt = PromptTemplate(template=title_template, input_variables=["summary"])
    title_chain = LLMChain(llm=llm, prompt=title_prompt, output_key="title")
    
    # 链 2: 生成角色描述
    character_template = """<s><|user|>
    Describe the main character of a story about {summary} with the title {title}. Use only two sentences.<|end|>
    <|assistant|>"""
    character_prompt = PromptTemplate(
        template=character_template, input_variables=["summary", "title"]
    )
    character_chain = LLMChain(llm=llm, prompt=character_prompt, output_key="character")
    
    # 链 3: 生成故事
    story_template = """<s><|user|>
    Create a story about {summary} with the title {title}. The main character is: {character}. Only return the story and it cannot be longer than one paragraph.<|end|>
    <|assistant|>"""
    story_prompt = PromptTemplate(
        template=story_template, input_variables=["summary", "title", "character"]
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt, output_key="story")
    
    # 组合所有链
    full_chain = title_chain | character_chain | story_chain
    
    print("\n链的结构:")
    print("  输入(summary) -> 标题链 -> 角色链 -> 故事链 -> 完整故事")
    print("-" * 40)
    
    # 运行完整链
    print("\n生成关于 'a girl that lost her mother' 的故事:")
    result = full_chain.invoke("a girl that lost her mother")
    
    print(f"\n标题: {result.get('title', 'N/A')}")
    print(f"\n角色: {result.get('character', 'N/A')}")
    print(f"\n故事: {result.get('story', 'N/A')}")
    
    return full_chain


def memory_buffer_demo(llm):
    """
    Part 4.1: 对话缓冲记忆 (ConversationBufferMemory)
    保存完整的对话历史
    """
    print("\n" + "=" * 60)
    print("Part 4.1: 对话缓冲记忆 (ConversationBufferMemory)")
    print("=" * 60)
    
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        from langchain import PromptTemplate, LLMChain
        from langchain.memory import ConversationBufferMemory
    
    # 包含对话历史的提示模板
    template = """<s><|user|>Current conversation:{chat_history}

    {input_prompt}<|end|>
    <|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["input_prompt", "chat_history"]
    )
    
    # 创建记忆
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # 创建链
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    
    print("\n演示: LLM 能记住之前的对话")
    print("-" * 40)
    
    # 第一轮对话
    print("\n[轮次 1] 输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    result1 = chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
    print(f"输出: {result1['text']}")
    
    # 第二轮对话 - 询问名字
    print("\n[轮次 2] 输入: 'What is my name?'")
    result2 = chain.invoke({"input_prompt": "What is my name?"})
    print(f"输出: {result2['text']}")
    
    print("\n✓ LLM 成功记住了用户的名字!")
    
    return chain


def memory_window_demo(llm):
    """
    Part 4.2: 滑动窗口记忆 (ConversationBufferWindowMemory)
    只保留最近 k 轮对话
    """
    print("\n" + "=" * 60)
    print("Part 4.2: 滑动窗口记忆 (k=2)")
    print("=" * 60)
    
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.memory import ConversationBufferWindowMemory
    except ImportError:
        from langchain import PromptTemplate, LLMChain
        from langchain.memory import ConversationBufferWindowMemory
    
    template = """<s><|user|>Current conversation:{chat_history}

    {input_prompt}<|end|>
    <|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["input_prompt", "chat_history"]
    )
    
    # 只保留最近 2 轮对话
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    
    print("\n演示: 只保留最近 2 轮对话，更早的会被遗忘")
    print("-" * 40)
    
    # 对话 1: 提供名字和年龄
    print("\n[轮次 1] 输入: 'Hi! My name is Maarten and I am 33 years old. What is 1 + 1?'")
    chain.invoke({"input_prompt": "Hi! My name is Maarten and I am 33 years old. What is 1 + 1?"})
    
    # 对话 2
    print("[轮次 2] 输入: 'What is 3 + 3?'")
    chain.invoke({"input_prompt": "What is 3 + 3?"})
    
    # 对话 3: 询问名字 (应该还记得)
    print("\n[轮次 3] 输入: 'What is my name?'")
    result3 = chain.invoke({"input_prompt": "What is my name?"})
    print(f"输出: {result3['text']}")
    
    # 对话 4: 询问年龄 (可能忘记了，因为超出窗口)
    print("\n[轮次 4] 输入: 'What is my age?'")
    result4 = chain.invoke({"input_prompt": "What is my age?"})
    print(f"输出: {result4['text']}")
    
    print("\n✓ 年龄信息已超出窗口(k=2)，被遗忘了!")
    
    return chain


def memory_summary_demo(llm):
    """
    Part 4.3: 摘要记忆 (ConversationSummaryMemory)
    将对话历史压缩为摘要
    """
    print("\n" + "=" * 60)
    print("Part 4.3: 摘要记忆 (ConversationSummaryMemory)")
    print("=" * 60)
    
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.memory import ConversationSummaryMemory
    except ImportError:
        from langchain import PromptTemplate, LLMChain
        from langchain.memory import ConversationSummaryMemory
    
    # 对话提示模板
    template = """<s><|user|>Current conversation:{chat_history}

    {input_prompt}<|end|>
    <|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["input_prompt", "chat_history"]
    )
    
    # 摘要提示模板
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
    
    # 创建摘要记忆
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        prompt=summary_prompt
    )
    
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    
    print("\n演示: 对话被压缩为摘要，节省 token")
    print("-" * 40)
    
    # 多轮对话
    print("\n[轮次 1] 输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
    
    print("[轮次 2] 输入: 'What is my name?'")
    chain.invoke({"input_prompt": "What is my name?"})
    
    print("\n[轮次 3] 输入: 'What was the first question I asked?'")
    result = chain.invoke({"input_prompt": "What was the first question I asked?"})
    print(f"输出: {result['text']}")
    
    # 查看摘要
    print("\n当前对话摘要:")
    print("-" * 40)
    summary = memory.load_memory_variables({})
    print(summary.get('chat_history', 'N/A'))
    
    return chain


def agent_demo():
    """
    Part 5: 代理 (Agent) - 让 LLM 使用工具
    使用 ReAct 模式
    """
    print("\n" + "=" * 60)
    print("Part 5: 代理 (Agent) - ReAct 模式")
    print("=" * 60)
    
    # 检查 OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "MY_KEY":
        print("\n[跳过] 需要设置 OPENAI_API_KEY 环境变量")
        print("示例: export OPENAI_API_KEY='your-api-key'")
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain.agents import load_tools, Tool, AgentExecutor, create_react_agent
        from langchain_community.tools import DuckDuckGoSearchResults
    except ImportError as e:
        print(f"\n[跳过] 缺少依赖: {e}")
        return None
    
    # 加载 OpenAI LLM
    openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # ReAct 提示模板
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
    
    # 创建工具
    search = DuckDuckGoSearchResults()
    search_tool = Tool(
        name="duckduck",
        description="A web search engine. Use this for general queries.",
        func=search.run,
    )
    
    tools = load_tools(["llm-math"], llm=openai_llm)
    tools.append(search_tool)
    
    # 创建 ReAct 代理
    agent = create_react_agent(openai_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )
    
    print("\n可用工具:")
    print("  1. Calculator - 数学计算")
    print("  2. DuckDuckGo - 网络搜索")
    print("-" * 40)
    
    # 运行代理
    print("\n问题: MacBook Pro 的价格是多少？按 0.85 汇率换算成欧元是多少？")
    result = agent_executor.invoke({
        "input": "What is the current price of a MacBook Pro in USD? How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
    })
    
    print(f"\n最终答案: {result['output']}")
    
    return agent_executor


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


def main():
    """主函数"""
    # 检测设备
    device, n_gpu_layers = get_device()
    
    # Part 1: 加载模型
    llm = load_llm_langchain(n_gpu_layers=n_gpu_layers)
    
    # Part 2: 基本链
    basic_chain_demo(llm)
    
    # Part 3: 多链组合
    multiple_chains_demo(llm)
    
    # Part 4.1: 对话缓冲记忆
    memory_buffer_demo(llm)
    
    # Part 4.2: 滑动窗口记忆
    memory_window_demo(llm)
    
    # Part 4.3: 摘要记忆
    memory_summary_demo(llm)
    
    # Part 5: 代理 (需要 OpenAI API Key)
    agent_demo()
    
    # 总结
    print_summary()


if __name__ == "__main__":
    main()
