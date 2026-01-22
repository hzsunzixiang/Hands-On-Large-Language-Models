"""
Chapter 6 - Prompt Engineering
通过提示工程改进大语言模型的输出质量

本章内容:
1. 加载模型和基本使用
2. 提示的基本组成部分 (角色、指令、上下文、格式、受众、语气)
3. 高级提示技术 (In-Context Learning, Chain Prompting)
4. 推理技术 (Chain-of-Thought, Tree-of-Thought)
5. 输出验证 (示例约束、Grammar约束)
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import json
import gc


def get_device():
    """
    自动检测最佳可用设备
    优先级: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"使用设备: CUDA ({device_name})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("使用设备: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("使用设备: CPU")
    return device


def load_model(device=None):
    """
    加载 Phi-3-mini-4k-instruct 模型
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    print("\n" + "=" * 60)
    print("Part 1: 加载模型")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    print("\n加载 Phi-3-mini-4k-instruct 模型...")
    
    # 根据设备选择配置
    if device == "cuda":
        device_map = "cuda"
    elif device == "mps":
        device_map = "mps"
    else:
        device_map = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    # 创建 pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False,
    )
    
    print("模型加载完成!")
    return pipe, tokenizer, model, device


def basic_generation(pipe, tokenizer):
    """
    基本生成示例
    """
    print("\n" + "=" * 60)
    print("Part 1.1: 基本生成")
    print("=" * 60)
    
    # 简单提示
    messages = [
        {"role": "user", "content": "Create a funny joke about chickens."}
    ]
    
    # 查看实际发送给模型的提示
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\n实际提示模板:")
    print(prompt)
    
    # 生成输出
    print("\n生成输出:")
    output = pipe(messages)
    print(output[0]["generated_text"])
    
    return output


def sampling_parameters(pipe):
    """
    采样参数演示: temperature 和 top_p
    """
    print("\n" + "=" * 60)
    print("Part 1.2: 采样参数")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Create a funny joke about chickens."}
    ]
    
    # 高 temperature (更随机)
    print("\n高 temperature (=1.0) - 更多创造性:")
    output = pipe(messages, do_sample=True, temperature=1.0)
    print(output[0]["generated_text"])
    
    # 高 top_p (nucleus sampling)
    print("\n高 top_p (=1.0) - nucleus sampling:")
    output = pipe(messages, do_sample=True, top_p=1.0)
    print(output[0]["generated_text"])


def complex_prompt_demo(pipe, tokenizer):
    """
    复杂提示组成演示
    展示如何构建一个完整的提示，包含多个组件
    """
    print("\n" + "=" * 60)
    print("Part 2: 复杂提示的组成部分")
    print("=" * 60)
    
    # 待总结的文本 (Transformer 论文介绍)
    text = """The Transformer model utilizes attention mechanisms to improve the speed 
and performance of deep learning models. It outperforms previous models in 
machine translation tasks and enables parallelization for faster training."""
    
    # 提示组件
    persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries.\n"
    instruction = "Summarize the key findings of the paper provided.\n"
    context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper.\n"
    data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results.\n"
    audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models.\n"
    tone = "The tone should be professional and clear.\n"
    data = f"Text to summarize: {text}"
    
    # 完整提示
    query = persona + instruction + context + data_format + audience + tone + data
    
    print("\n提示组件:")
    print("-" * 40)
    print(f"1. Persona (角色): {persona[:50]}...")
    print(f"2. Instruction (指令): {instruction[:50]}...")
    print(f"3. Context (上下文): {context[:50]}...")
    print(f"4. Format (格式): {data_format[:50]}...")
    print(f"5. Audience (受众): {audience[:50]}...")
    print(f"6. Tone (语气): {tone[:30]}...")
    print(f"7. Data (数据): [用户提供的文本]")
    
    messages = [{"role": "user", "content": query}]
    
    print("\n完整提示模板:")
    print(tokenizer.apply_chat_template(messages, tokenize=False)[:500] + "...")
    
    print("\n生成摘要:")
    outputs = pipe(messages)
    print(outputs[0]["generated_text"])


def in_context_learning_demo(pipe, tokenizer):
    """
    上下文学习 (In-Context Learning) 演示
    通过提供示例来教模型完成任务
    """
    print("\n" + "=" * 60)
    print("Part 3: 上下文学习 (Few-shot Learning)")
    print("=" * 60)
    
    # One-shot 学习: 提供一个示例
    one_shot_prompt = [
        {
            "role": "user",
            "content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
        },
        {
            "role": "assistant",
            "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."
        },
        {
            "role": "user",
            "content": "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
        }
    ]
    
    print("\nOne-shot 学习 - 教模型使用虚构词汇:")
    print("-" * 40)
    print("示例: 'Gigamuru' = 日本乐器")
    print("任务: 使用 'screeg' (挥剑) 造句")
    
    print("\n提示模板:")
    print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
    
    print("\n模型输出:")
    outputs = pipe(one_shot_prompt)
    print(outputs[0]["generated_text"])


def chain_prompting_demo(pipe):
    """
    链式提示 (Chain Prompting) 演示
    将复杂任务分解为多个步骤
    """
    print("\n" + "=" * 60)
    print("Part 4: 链式提示 (Chain Prompting)")
    print("=" * 60)
    
    print("\n步骤 1: 创建产品名称和口号")
    print("-" * 40)
    
    # 第一步: 创建产品名称和口号
    product_prompt = [
        {"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}
    ]
    outputs = pipe(product_prompt)
    product_description = outputs[0]["generated_text"]
    print(product_description)
    
    print("\n步骤 2: 基于名称生成销售文案")
    print("-" * 40)
    
    # 第二步: 基于第一步的输出生成销售文案
    sales_prompt = [
        {"role": "user", "content": f"Generate a very short sales pitch for the following product: '{product_description}'"}
    ]
    outputs = pipe(sales_prompt)
    sales_pitch = outputs[0]["generated_text"]
    print(sales_pitch)


def chain_of_thought_demo(pipe):
    """
    思维链 (Chain-of-Thought) 演示
    让模型展示推理过程
    """
    print("\n" + "=" * 60)
    print("Part 5: 思维链推理 (Chain-of-Thought)")
    print("=" * 60)
    
    # Few-shot CoT: 提供带推理过程的示例
    print("\n5.1 Few-shot Chain-of-Thought:")
    print("-" * 40)
    
    cot_prompt = [
        {"role": "user", "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
        {"role": "assistant", "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."},
        {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
    ]
    
    outputs = pipe(cot_prompt)
    print("问题: 食堂有23个苹果，用了20个做午餐，又买了6个，现在有多少个？")
    print(f"模型回答: {outputs[0]['generated_text']}")
    
    # Zero-shot CoT: 只需添加 "Let's think step-by-step"
    print("\n5.2 Zero-shot Chain-of-Thought:")
    print("-" * 40)
    print("技巧: 在问题末尾添加 'Let's think step-by-step.'")
    
    zeroshot_cot_prompt = [
        {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step."}
    ]
    
    outputs = pipe(zeroshot_cot_prompt)
    print(outputs[0]["generated_text"])


def tree_of_thought_demo(pipe):
    """
    思维树 (Tree-of-Thought) 演示
    让多个"专家"协作解决问题
    """
    print("\n" + "=" * 60)
    print("Part 6: 思维树 (Tree-of-Thought)")
    print("=" * 60)
    
    print("\n模拟三位专家协作解决问题:")
    print("-" * 40)
    
    zeroshot_tot_prompt = [
        {"role": "user", "content": """Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?' Make sure to discuss the results."""}
    ]
    
    outputs = pipe(zeroshot_tot_prompt)
    print(outputs[0]["generated_text"])


def output_format_demo(pipe):
    """
    输出格式控制演示
    通过示例约束输出格式
    """
    print("\n" + "=" * 60)
    print("Part 7: 输出格式控制")
    print("=" * 60)
    
    # Zero-shot: 不提供格式示例
    print("\n7.1 Zero-shot (无格式约束):")
    print("-" * 40)
    
    zeroshot_prompt = [
        {"role": "user", "content": "Create a character profile for an RPG game in JSON format."}
    ]
    
    outputs = pipe(zeroshot_prompt)
    print(outputs[0]["generated_text"][:500] + "...")
    
    # One-shot: 提供格式模板
    print("\n7.2 One-shot (提供格式模板):")
    print("-" * 40)
    
    one_shot_template = """Create a short character profile for an RPG game. Make sure to only use this format:

{
  "description": "A SHORT DESCRIPTION",
  "name": "THE CHARACTER'S NAME",
  "armor": "ONE PIECE OF ARMOR",
  "weapon": "ONE OR MORE WEAPONS"
}
"""
    one_shot_prompt = [
        {"role": "user", "content": one_shot_template}
    ]
    
    outputs = pipe(one_shot_prompt)
    print(outputs[0]["generated_text"])


def grammar_constrained_demo():
    """
    语法约束采样演示
    使用 llama-cpp-python 强制输出 JSON 格式
    """
    print("\n" + "=" * 60)
    print("Part 8: 语法约束采样 (Grammar-Constrained Sampling)")
    print("=" * 60)
    
    try:
        from llama_cpp.llama import Llama
        
        print("\n加载 Phi-3 GGUF 模型...")
        
        # 检测 GPU 层数
        n_gpu = -1 if torch.cuda.is_available() else 0
        
        llm = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="*q4_k_m.gguf",  # 使用量化版本节省内存
            n_gpu_layers=n_gpu,
            n_ctx=2048,
            verbose=False
        )
        
        print("\n使用 JSON 模式生成:")
        print("-" * 40)
        
        # 使用 response_format 强制 JSON 输出
        output = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": "Create a warrior for an RPG in JSON format."},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )['choices'][0]['message']["content"]
        
        # 格式化 JSON 输出
        json_output = json.dumps(json.loads(output), indent=4)
        print(json_output)
        
        return llm
        
    except ImportError:
        print("\n[跳过] llama-cpp-python 未安装")
        print("安装命令: pip install llama-cpp-python")
        return None
    except Exception as e:
        print(f"\n[错误] {e}")
        return None


def cleanup(model=None, tokenizer=None, pipe=None):
    """清理 GPU 内存"""
    print("\n" + "=" * 60)
    print("清理资源")
    print("=" * 60)
    
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if pipe is not None:
        del pipe
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA 缓存已清理")
    elif torch.backends.mps.is_available():
        # MPS 没有显式的缓存清理 API
        print("MPS 资源已释放")
    else:
        print("CPU 资源已释放")


def print_summary():
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


def main():
    """主函数"""
    # Part 1: 加载模型
    pipe, tokenizer, model, device = load_model()
    
    # Part 1.1: 基本生成
    basic_generation(pipe, tokenizer)
    
    # Part 1.2: 采样参数
    sampling_parameters(pipe)
    
    # Part 2: 复杂提示
    complex_prompt_demo(pipe, tokenizer)
    
    # Part 3: 上下文学习
    in_context_learning_demo(pipe, tokenizer)
    
    # Part 4: 链式提示
    chain_prompting_demo(pipe)
    
    # Part 5: 思维链推理
    chain_of_thought_demo(pipe)
    
    # Part 6: 思维树
    tree_of_thought_demo(pipe)
    
    # Part 7: 输出格式控制
    output_format_demo(pipe)
    
    # 清理 transformers 模型
    cleanup(model, tokenizer, pipe)
    
    # Part 8: 语法约束采样 (可选)
    grammar_constrained_demo()
    
    # 总结
    print_summary()


if __name__ == "__main__":
    main()
