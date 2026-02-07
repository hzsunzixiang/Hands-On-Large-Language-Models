"""
Chapter 4 - Part 6: 使用 ChatGPT 进行分类
需要 OpenAI API key
"""

import os
from tqdm import tqdm

from common import load_data, evaluate_performance


def chatgpt_generation(client, prompt, document, model="gpt-3.5-turbo-0125"):
    """
    使用 ChatGPT 生成回复
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    return chat_completion.choices[0].message.content


def chatgpt_classification(data, api_key=None, model="gpt-3.5-turbo-0125", max_samples=None):
    """
    使用 ChatGPT 进行文本分类
    
    参数:
        data: 数据集
        api_key: OpenAI API key (如果为 None，从环境变量读取)
        model: 使用的模型名称
        max_samples: 最大样本数 (用于测试，None 表示全部)
    """
    print("\n" + "=" * 60)
    print("Part 6: ChatGPT 分类")
    print("=" * 60)
    
    # 检查 API key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("\n警告: 未设置 OPENAI_API_KEY 环境变量")
        print("请设置环境变量或传入 api_key 参数")
        print("\n显示演示代码:")
        show_demo_code()
        return None
    
    # 创建客户端
    import openai
    client = openai.OpenAI(api_key=api_key)
    
    # 定义提示模板
    prompt = '''Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
'''
    
    print(f"\n使用模型: {model}")
    print(f"提示模板:\n{prompt}")
    
    # 准备测试数据
    test_texts = data["test"]["text"]
    test_labels = data["test"]["label"]
    
    if max_samples:
        test_texts = test_texts[:max_samples]
        test_labels = test_labels[:max_samples]
        print(f"\n仅测试前 {max_samples} 个样本")
    
    # 运行推理
    print(f"\n在 {len(test_texts)} 个样本上运行推理...")
    y_pred = []
    for text in tqdm(test_texts):
        try:
            result = chatgpt_generation(client, prompt, text, model=model)
            pred = int(result.strip())
            y_pred.append(pred)
        except Exception as e:
            print(f"\n错误: {e}")
            y_pred.append(1)  # 默认预测为正面
    
    print("\nChatGPT 分类结果:")
    evaluate_performance(test_labels, y_pred)
    
    return y_pred


def show_demo_code():
    """显示 ChatGPT 分类的演示代码"""
    print("""
ChatGPT 分类代码示例:

```python
import openai

# 创建客户端
client = openai.OpenAI(api_key="YOUR_KEY_HERE")

def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    return chat_completion.choices[0].message.content

# 定义提示模板
prompt = '''Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
'''

# 预测
document = "unpretentious, charming, quirky, original"
result = chatgpt_generation(prompt, document)  # 返回 "1"
```

注意: 运行完整测试集需要 API 调用 1066 次，请确保有足够的 API 额度。
预期准确率约 91%。
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ChatGPT 文本分类")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125", help="模型名称")
    parser.add_argument("--max-samples", type=int, help="最大测试样本数")
    parser.add_argument("--demo", action="store_true", help="只显示演示代码")
    
    args = parser.parse_args()
    
    if args.demo:
        show_demo_code()
        return
    
    # 加载数据
    data = load_data()
    
    # 运行分类
    y_pred = chatgpt_classification(
        data, 
        api_key=args.api_key, 
        model=args.model,
        max_samples=args.max_samples
    )
    
    return y_pred


if __name__ == "__main__":
    main()
