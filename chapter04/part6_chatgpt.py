"""
Chapter 4 - Part 6: 使用大模型 API 进行分类
支持 OpenAI / DeepSeek / 其他兼容 OpenAI 格式的 API

API Key 读取优先级:
1. 命令行参数 --api-key
2. 环境变量 (DEEPSEEK_API_KEY 或 OPENAI_API_KEY)
3. .env 文件
"""

import os
from pathlib import Path
from tqdm import tqdm

from common import load_data, evaluate_performance


def load_env_file():
    """从 .env 文件加载环境变量"""
    # 查找 .env 文件 (当前目录或上级目录)
    env_paths = [
        Path(__file__).parent / ".env",           # chapter04/.env
        Path(__file__).parent.parent / ".env",    # Hands-On-Large-Language-Models/.env
        Path.home() / ".env",                      # ~/.env
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            print(f"从 {env_path} 加载配置...")
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:  # 不覆盖已有环境变量
                            os.environ[key] = value
            return True
    return False


def load_api_key(provider):
    """
    从多个位置加载 API Key
    
    查找顺序:
    1. 环境变量 (DEEPSEEK_API_KEY / OPENAI_API_KEY)
    2. ~/.deepseek 或 ~/.openai (单行 key 文件)
    3. ~/.config/deepseek/api_key 或 ~/.config/openai/api_key
    4. .env 文件 (已在 main 中加载)
    """
    config = API_PROVIDERS.get(provider, {})
    env_key = config.get("env_key", "")
    
    # 1. 先检查环境变量 (包括从 .env 加载的)
    if env_key and os.environ.get(env_key):
        return os.environ.get(env_key)
    
    # 2. 检查 home 目录下的专用文件 (~/.deepseek, ~/.openai)
    home_key_file = Path.home() / f".{provider}"
    if home_key_file.exists():
        print(f"从 {home_key_file} 加载 API key...")
        with open(home_key_file, "r") as f:
            key = f.read().strip()
            if key:
                return key
    
    # 3. 检查 XDG 配置目录 (~/.config/deepseek/api_key)
    xdg_paths = [
        Path.home() / ".config" / provider / "api_key",
        Path.home() / ".config" / provider / "config",
    ]
    for xdg_path in xdg_paths:
        if xdg_path.exists():
            print(f"从 {xdg_path} 加载 API key...")
            with open(xdg_path, "r") as f:
                content = f.read().strip()
                # 如果是 key=value 格式，提取 value
                if "=" in content:
                    for line in content.split("\n"):
                        if line.startswith("api_key=") or line.startswith("API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
                else:
                    return content
    
    return None

# 支持的 API 提供商配置
API_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-3.5-turbo-0125",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
}


def llm_generation(client, prompt, document, model):
    """
    使用大模型 API 生成回复
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


def llm_classification(data, provider="deepseek", api_key=None, model=None, max_samples=None):
    """
    使用大模型 API 进行文本分类
    
    参数:
        data: 数据集
        provider: API 提供商 (openai, deepseek)
        api_key: API key (如果为 None，从环境变量读取)
        model: 使用的模型名称 (如果为 None，使用默认模型)
        max_samples: 最大样本数 (用于测试，None 表示全部)
    """
    print("\n" + "=" * 60)
    print(f"Part 6: 大模型 API 分类 ({provider.upper()})")
    print("=" * 60)
    
    # 获取提供商配置
    if provider not in API_PROVIDERS:
        print(f"不支持的提供商: {provider}")
        print(f"支持的提供商: {list(API_PROVIDERS.keys())}")
        return None
    
    config = API_PROVIDERS[provider]
    
    # 检查 API key
    if api_key is None:
        api_key = load_api_key(provider)
    
    if not api_key:
        print(f"\n警告: 未设置 {config['env_key']} 环境变量")
        print("请设置环境变量或传入 --api-key 参数")
        show_demo_code()
        return None
    
    # 使用默认模型
    if model is None:
        model = config["default_model"]
    
    # 创建客户端
    import openai
    client = openai.OpenAI(
        api_key=api_key,
        base_url=config["base_url"]
    )
    
    # 定义提示模板
    prompt = '''Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
'''
    
    print(f"\n使用模型: {model}")
    print(f"API 地址: {config['base_url']}")
    
    # 准备测试数据
    test_texts = data["test"]["text"]
    test_labels = data["test"]["label"]
    
    if max_samples:
        test_texts = test_texts[:max_samples]
        test_labels = test_labels[:max_samples]
        print(f"\n仅测试前 {max_samples} 个样本")
    else:
        print(f"\n测试全部 {len(test_texts)} 个样本 (可能需要较长时间和 API 费用)")
    
    # 运行推理
    print(f"\n开始推理...")
    y_pred = []
    errors = 0
    for text in tqdm(test_texts):
        try:
            result = llm_generation(client, prompt, text, model=model)
            # 尝试提取数字
            result_clean = result.strip()
            if result_clean in ["0", "1"]:
                pred = int(result_clean)
            elif "0" in result_clean and "1" not in result_clean:
                pred = 0
            elif "1" in result_clean and "0" not in result_clean:
                pred = 1
            else:
                pred = 1  # 默认正面
                errors += 1
            y_pred.append(pred)
        except Exception as e:
            print(f"\n错误: {e}")
            y_pred.append(1)
            errors += 1
    
    if errors > 0:
        print(f"\n警告: {errors} 个样本解析失败，使用默认值")
    
    print(f"\n{provider.upper()} 分类结果:")
    evaluate_performance(test_labels, y_pred)
    
    return y_pred


def show_demo_code():
    """显示演示代码"""
    print("""
大模型 API 分类代码示例:

```python
import openai

# DeepSeek
client = openai.OpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com"
)

# 或 OpenAI
# client = openai.OpenAI(api_key="sk-xxx")

def llm_generation(prompt, document, model="deepseek-chat"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    return response.choices[0].message.content

# 预测
document = "unpretentious, charming, quirky, original"
result = llm_generation(prompt, document)  # 返回 "1"
```
""")


def main():
    import argparse
    
    # 加载 .env 文件
    load_env_file()
    
    parser = argparse.ArgumentParser(
        description="大模型 API 文本分类",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python part6_chatgpt.py --provider deepseek --max-samples 10
  python part6_chatgpt.py --provider openai --api-key sk-xxx
  python part6_chatgpt.py --demo
        """
    )
    parser.add_argument("--provider", "-p", type=str, default="deepseek",
                        choices=["openai", "deepseek"],
                        help="API 提供商 (默认: deepseek)")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--model", type=str, help="模型名称")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="最大测试样本数 (默认: 50，设为 0 表示全部)")
    parser.add_argument("--demo", action="store_true", help="只显示演示代码")
    
    args = parser.parse_args()
    
    if args.demo:
        show_demo_code()
        return
    
    # 加载数据
    data = load_data()
    
    # 处理 max_samples
    max_samples = args.max_samples if args.max_samples > 0 else None
    
    # 运行分类
    y_pred = llm_classification(
        data, 
        provider=args.provider,
        api_key=args.api_key, 
        model=args.model,
        max_samples=max_samples
    )
    
    return y_pred


if __name__ == "__main__":
    main()
