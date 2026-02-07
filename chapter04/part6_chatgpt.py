"""
Chapter 4 - Part 6: 使用 DeepSeek 进行分类
API Key 从 ~/.deepseek 读取
"""

from pathlib import Path
from tqdm import tqdm
import openai

from common import load_data, evaluate_performance

# DeepSeek 配置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def load_api_key():
    """从 ~/.deepseek 读取 API key"""
    key_file = Path.home() / ".deepseek"
    if not key_file.exists():
        raise FileNotFoundError(f"请创建 {key_file} 文件并写入 API key")
    return key_file.read_text().strip()


def classify_with_deepseek(data, max_samples=50):
    """使用 DeepSeek 进行文本分类"""
    print("\n" + "=" * 60)
    print("Part 6: DeepSeek 分类")
    print("=" * 60)
    
    # 加载 API key
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    
    # 提示模板
    prompt = '''Predict whether the following document is a positive or negative movie review:

{text}

If it is positive return 1 and if it is negative return 0. Do not give any other answers.'''
    
    # 准备数据
    test_texts = data["test"]["text"][:max_samples] if max_samples else data["test"]["text"]
    test_labels = data["test"]["label"][:max_samples] if max_samples else data["test"]["label"]
    
    print(f"\n模型: {DEEPSEEK_MODEL}")
    print(f"测试样本数: {len(test_texts)}")
    
    # 推理
    y_pred = []
    for text in tqdm(test_texts):
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt.format(text=text)}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        y_pred.append(0 if "0" in result else 1)
    
    print("\nDeepSeek 分类结果:")
    evaluate_performance(test_labels, y_pred)
    return y_pred


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek 文本分类")
    parser.add_argument("--max-samples", "-n", type=int, default=50,
                        help="测试样本数 (默认: 50, 0=全部)")
    args = parser.parse_args()
    
    data = load_data()
    max_samples = args.max_samples if args.max_samples > 0 else None
    classify_with_deepseek(data, max_samples)


if __name__ == "__main__":
    main()
