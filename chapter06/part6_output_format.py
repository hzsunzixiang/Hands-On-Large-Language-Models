"""
Chapter 6 - Part 6: 输出格式控制

本节内容:
1. 通过示例约束输出格式 (Few-shot formatting)
2. 通过指令约束输出格式
3. 语法约束采样 (Grammar-Constrained Sampling)
   - 使用 llama-cpp-python 的 JSON Mode

关键概念:
- LLM 输出默认是自由文本
- 实际应用中往往需要结构化输出 (JSON, XML, Markdown 等)
- 语法约束可以保证 100% 格式正确
"""

import json
import torch
from common import load_model, cleanup, print_section


def example_based_formatting(pipe):
    """
    通过示例约束输出格式
    """
    print_section("6.1 示例约束格式", level=2)
    
    print("方法: 在提示中提供期望的输出格式示例")
    print()
    
    # Zero-shot: 不提供格式示例
    print("Zero-shot (无格式约束):")
    print("-" * 40)
    
    zeroshot_prompt = [
        {"role": "user", "content": "Create a character profile for an RPG game in JSON format."}
    ]
    
    outputs = pipe(zeroshot_prompt)
    print(outputs[0]["generated_text"][:600])
    
    # One-shot: 提供格式模板
    print("\n\nOne-shot (提供格式模板):")
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
    
    print("\n分析:")
    print("  - 提供模板大大提高了格式一致性")
    print("  - 但仍不能 100% 保证格式正确")


def instruction_based_formatting(pipe):
    """
    通过指令约束输出格式
    """
    print_section("6.2 指令约束格式", level=2)
    
    print("方法: 在指令中明确说明格式要求")
    print()
    
    # 结构化指令
    structured_prompt = [
        {"role": "user", "content": """Generate a product review with the following structure:
        
## Product Name
[name here]

## Rating
[1-5 stars]

## Pros
- [bullet point 1]
- [bullet point 2]

## Cons
- [bullet point 1]

## Summary
[one sentence summary]

The product is a wireless mouse."""}
    ]
    
    print("提示中指定 Markdown 结构:")
    print("-" * 40)
    outputs = pipe(structured_prompt)
    print(outputs[0]["generated_text"])


def validate_json(text):
    """尝试解析 JSON，返回是否有效"""
    try:
        json.loads(text)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)


def json_validation_demo(pipe):
    """
    JSON 输出验证演示
    """
    print_section("6.3 JSON 验证", level=2)
    
    print("问题: LLM 生成的 JSON 可能无效")
    print("解决: 验证 + 重试 或 使用语法约束")
    print()
    
    prompt = [
        {"role": "user", "content": "Generate a JSON object with name, age, and city fields for a random person. Only output valid JSON, nothing else."}
    ]
    
    outputs = pipe(prompt)
    result = outputs[0]["generated_text"].strip()
    
    print(f"模型输出:\n{result}")
    
    is_valid, error = validate_json(result)
    if is_valid:
        print("\n✓ JSON 有效!")
        parsed = json.loads(result)
        print(f"解析结果: {json.dumps(parsed, indent=2)}")
    else:
        print(f"\n✗ JSON 无效: {error}")
        print("  建议: 使用语法约束采样确保格式正确")


def grammar_constrained_demo():
    """
    语法约束采样演示
    使用 llama-cpp-python 强制输出 JSON 格式
    """
    print_section("6.4 语法约束采样 (Grammar-Constrained)", level=2)
    
    print("核心思想: 在采样时强制遵循特定语法")
    print("实现: llama-cpp-python 支持 JSON Mode 和 GBNF 语法")
    print()
    
    try:
        from llama_cpp.llama import Llama
        
        print("加载 Phi-3 GGUF 模型...")
        print("(GGUF 是量化格式，适合 CPU/低显存运行)")
        print()
        
        # 检测 GPU 层数
        n_gpu = -1 if torch.cuda.is_available() else 0
        
        llm = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="*q4_k_m.gguf",  # 使用量化版本节省内存
            n_gpu_layers=n_gpu,
            n_ctx=2048,
            verbose=False
        )
        
        print("JSON Mode 生成:")
        print("-" * 40)
        
        # 使用 response_format 强制 JSON 输出
        output = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": "Create a warrior character for an RPG game. Include name, class, health, attack, and skills array."},
            ],
            response_format={"type": "json_object"},  # 强制 JSON 输出!
            temperature=0,
        )['choices'][0]['message']["content"]
        
        # 格式化 JSON 输出
        json_output = json.dumps(json.loads(output), indent=2)
        print(json_output)
        
        print("\n✓ JSON Mode 保证输出 100% 有效!")
        
        # 清理
        del llm
        
    except ImportError:
        print("[跳过] llama-cpp-python 未安装")
        print()
        print("安装命令:")
        print("  pip install llama-cpp-python")
        print()
        print("如需 GPU 加速 (CUDA):")
        print("  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python")
        
    except Exception as e:
        print(f"[错误] {e}")


def output_format_summary():
    """输出格式控制总结"""
    print_section("6.5 总结", level=2)
    
    print("""
输出格式控制方法对比:
┌───────────────────┬─────────────────┬─────────────────┬───────────────┐
│ 方法              │ 格式正确率      │ 实现复杂度      │ 推荐场景      │
├───────────────────┼─────────────────┼─────────────────┼───────────────┤
│ Zero-shot         │ 低 (~50%)       │ 简单            │ 简单任务      │
│ 示例约束          │ 中等 (~80%)     │ 简单            │ 一般任务      │
│ 指令 + 验证重试   │ 高 (~95%)       │ 中等            │ 重要任务      │
│ 语法约束采样      │ 100%            │ 需要特定库      │ 生产环境      │
└───────────────────┴─────────────────┴─────────────────┴───────────────┘

语法约束采样支持:
- llama-cpp-python: JSON Mode, GBNF 语法
- Outlines (dottxt-ai/outlines): 正则表达式约束
- Guidance (microsoft/guidance): 模板约束
- vLLM: 支持 JSON Schema 约束
""")


def main():
    """主函数"""
    print_section("Part 6: 输出格式控制")
    
    # 加载 transformers 模型
    pipe, tokenizer, model, device = load_model()
    
    try:
        # 示例约束
        example_based_formatting(pipe)
        
        # 指令约束
        instruction_based_formatting(pipe)
        
        # JSON 验证
        json_validation_demo(pipe)
        
    finally:
        # 清理 transformers 模型
        cleanup(model, tokenizer, pipe)
    
    # 语法约束 (使用 llama-cpp-python)
    grammar_constrained_demo()
    
    # 总结
    output_format_summary()


if __name__ == "__main__":
    main()
