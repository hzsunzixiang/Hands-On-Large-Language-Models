"""
Chapter 7 - Part 1: 加载模型与基本链 (Chain)

本节内容:
1. 使用 LangChain + llama.cpp 加载本地 LLM
2. 直接调用 LLM 生成文本
3. 使用 PromptTemplate + LLM 构建基本链
4. 理解 LCEL (LangChain Expression Language)

关键概念:
- LangChain: LLM 应用开发框架，提供链、记忆、代理等抽象
- llama.cpp: 高效的本地 LLM 推理引擎，支持 GGUF 量化模型
- Chain: 将 Prompt 和 LLM 组合成可复用的调用单元
- LCEL: 使用 | 运算符组合组件 (prompt | llm)
"""

from common import get_device, load_llm, PHI3_TEMPLATE, print_section


def test_basic_invoke(llm):
    """
    直接调用 LLM 生成文本
    """
    print_section("1.1 直接调用 LLM", level=2)
    
    response = llm.invoke("Hi! My name is Maarten. What is 1 + 1?")
    print(f"输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    print(f"输出: {response}")
    
    print("\n说明:")
    print("  - llm.invoke() 是最基本的调用方式")
    print("  - 直接传入字符串，返回生成的文本")
    print("  - 没有使用 chat template，模型可能表现不佳")
    
    return response


def basic_chain_demo(llm):
    """
    使用 PromptTemplate + LLM 构建基本链
    """
    print_section("1.2 基本链 (PromptTemplate | LLM)", level=2)
    
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        from langchain import PromptTemplate
    
    # 创建提示模板 (Phi-3 格式)
    prompt = PromptTemplate(
        template=PHI3_TEMPLATE,
        input_variables=["input_prompt"]
    )
    
    # 使用 LCEL (LangChain Expression Language) 创建链
    # | 运算符将组件串联: 先格式化 prompt，再传给 llm
    basic_chain = prompt | llm
    
    print("链的结构: PromptTemplate -> LLM")
    print()
    print("LCEL 语法:")
    print("  basic_chain = prompt | llm")
    print("  等价于: llm.invoke(prompt.format(...))")
    print()
    
    # 使用链
    response = basic_chain.invoke(
        {"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"}
    )
    print(f"输入: 'Hi! My name is Maarten. What is 1 + 1?'")
    print(f"输出: {response}")
    
    print("\n对比直接调用:")
    print("  - 直接调用: llm.invoke('原始文本') → 无 chat template")
    print("  - 使用链: chain.invoke({变量}) → 自动应用 Phi-3 template")
    
    return basic_chain


def main():
    """主函数"""
    print_section("Part 1: 加载模型与基本链")
    
    # 检测设备
    device, n_gpu_layers = get_device()
    
    # 加载模型
    llm = load_llm(n_gpu_layers=n_gpu_layers)
    
    # 直接调用
    test_basic_invoke(llm)
    
    # 基本链
    basic_chain_demo(llm)


if __name__ == "__main__":
    main()
