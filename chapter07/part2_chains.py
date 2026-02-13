"""
Chapter 7 - Part 2: 多链组合 (Multiple Chains)

本节内容:
将多个链串联起来完成复杂任务 - 故事生成器

链的结构:
  输入(summary) → 标题链 → 角色链 → 故事链 → 完整故事

关键概念:
- LCEL (LangChain Expression Language): 使用 | 运算符组合组件
- RunnablePassthrough: 将上游变量透传给下游
- PromptTemplate | LLM | StrOutputParser: 标准链结构
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from common import get_device, load_llm, print_section


def multiple_chains_demo(llm):
    """
    多链组合 - 故事生成器
    将复杂任务拆成3个步骤，逐步生成
    """
    print_section("2.1 多链组合 - 故事生成器", level=2)
    
    parser = StrOutputParser()
    
    # ========== 链 1: 生成标题 ==========
    title_template = """<s><|user|>
    Create a title for a story about {summary}. Only return the title.<|end|>
    <|assistant|>"""
    title_prompt = PromptTemplate(template=title_template, input_variables=["summary"])
    title_chain = title_prompt | llm | parser
    
    # ========== 链 2: 生成角色描述 ==========
    character_template = """<s><|user|>
    Describe the main character of a story about {summary} with the title {title}. Use only two sentences.<|end|>
    <|assistant|>"""
    character_prompt = PromptTemplate(
        template=character_template, input_variables=["summary", "title"]
    )
    character_chain = character_prompt | llm | parser
    
    # ========== 链 3: 生成故事 ==========
    story_template = """<s><|user|>
    Create a story about {summary} with the title {title}. The main character is: {character}. Only return the story and it cannot be longer than one paragraph.<|end|>
    <|assistant|>"""
    story_prompt = PromptTemplate(
        template=story_template, input_variables=["summary", "title", "character"]
    )
    story_chain = story_prompt | llm | parser
    
    # ========== 用 LCEL 组合所有链 ==========
    # 每一步: 保留上游变量 + 新增本步输出
    full_chain = (
        # Step 1: summary → title
        RunnablePassthrough.assign(title=title_chain)
        # Step 2: summary + title → character
        | RunnablePassthrough.assign(character=character_chain)
        # Step 3: summary + title + character → story
        | RunnablePassthrough.assign(story=story_chain)
    )
    
    print("链的结构 (LCEL):")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 输入: {summary: '...'}                              │")
    print("│   ↓                                                │")
    print("│ Step 1: assign(title=title_chain)                  │")
    print("│   → {summary, title}                               │")
    print("│   ↓                                                │")
    print("│ Step 2: assign(character=character_chain)           │")
    print("│   → {summary, title, character}                    │")
    print("│   ↓                                                │")
    print("│ Step 3: assign(story=story_chain)                  │")
    print("│   → {summary, title, character, story}             │")
    print("└─────────────────────────────────────────────────────┘")
    
    # 运行完整链
    print("\n生成关于 'a girl that lost her mother' 的故事:")
    print("-" * 40)
    result = full_chain.invoke({"summary": "a girl that lost her mother"})
    
    print(f"\n标题: {result.get('title', 'N/A')}")
    print(f"\n角色: {result.get('character', 'N/A')}")
    print(f"\n故事: {result.get('story', 'N/A')}")
    
    print("\n说明 (LCEL 方式):")
    print("  - RunnablePassthrough.assign(key=chain) 保留上游变量并新增输出")
    print("  - | 运算符将步骤串联")
    print("  - 每一步都能访问之前所有步骤的输出")
    
    return full_chain


def main():
    """主函数"""
    print_section("Part 2: 多链组合 (Multiple Chains)")
    
    # 检测设备
    device, n_gpu_layers = get_device()
    
    # 加载模型
    llm = load_llm(n_gpu_layers=n_gpu_layers)
    
    # 多链组合
    multiple_chains_demo(llm)


if __name__ == "__main__":
    main()
