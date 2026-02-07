"""
Chapter 4 - Text Classification
使用表示模型和生成模型进行文本分类

本章内容:
1. 加载数据 (Rotten Tomatoes 电影评论数据集)
2. 使用任务特定模型进行情感分类
3. 使用嵌入 + 监督学习分类
4. 零样本分类 (Zero-shot Classification)
5. 使用生成模型 (Flan-T5) 进行分类
6. (可选) 使用 ChatGPT 进行分类
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


def evaluate_performance(y_true, y_pred):
    """创建并打印分类报告"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)


def load_data():
    """加载 Rotten Tomatoes 数据集"""
    from datasets import load_dataset
    
    print("=" * 60)
    print("Part 1: 加载数据")
    print("=" * 60)
    
    data = load_dataset("rotten_tomatoes")
    print(f"\n数据集结构:")
    print(data)
    
    print(f"\n样本示例:")
    print(f"训练集第一条: {data['train'][0]}")
    print(f"训练集最后一条: {data['train'][-1]}")
    
    return data


def task_specific_classification(data, device="cpu"):
    """
    Part 2: 使用任务特定模型 (Twitter RoBERTa) 进行情感分类
    这个模型专门为 Twitter 情感分析训练
    """
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    
    print("\n" + "=" * 60)
    print("Part 2: 使用任务特定模型 (Twitter-RoBERTa)")
    print("=" * 60)
    
    # 加载预训练的情感分析模型
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"\n加载模型: {model_path}")
    
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        top_k=None,  # 返回所有类别的分数 (替代废弃的 return_all_scores)
        device=device
    )
    
    # 运行推理
    print("\n在测试集上运行推理...")
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        # 模型输出: negative, neutral, positive
        # 我们只关心 negative (index 0) 和 positive (index 2)
        scores = {item['label']: item['score'] for item in output}
        negative_score = scores.get('negative', 0)
        positive_score = scores.get('positive', 0)
        assignment = np.argmax([negative_score, positive_score])
        y_pred.append(assignment)
    
    print("\n分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return y_pred


def embedding_supervised_classification(data):
    """
    Part 3: 使用嵌入 + 监督学习分类
    先用 Sentence Transformer 生成嵌入，再用逻辑回归分类
    """
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "=" * 60)
    print("Part 3: 嵌入 + 监督学习分类")
    print("=" * 60)
    
    # 加载 Sentence Transformer 模型
    print("\n加载 Sentence Transformer 模型...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # 生成嵌入
    print("生成训练集嵌入...")
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    print("生成测试集嵌入...")
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
    
    print(f"\n训练集嵌入维度: {train_embeddings.shape}")
    print(f"测试集嵌入维度: {test_embeddings.shape}")
    
    # 训练逻辑回归分类器
    print("\n训练逻辑回归分类器...")
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(train_embeddings, data["train"]["label"])
    
    # 预测
    y_pred = clf.predict(test_embeddings)
    
    print("\n监督分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    # 额外实验: 不使用分类器，直接用平均嵌入 + 余弦相似度
    print("\n" + "-" * 40)
    print("额外实验: 平均嵌入 + 余弦相似度 (无分类器)")
    print("-" * 40)
    
    # 计算每个类别的平均嵌入
    df = pd.DataFrame(np.hstack([
        train_embeddings, 
        np.array(data["train"]["label"]).reshape(-1, 1)
    ]))
    averaged_target_embeddings = df.groupby(768).mean().values
    
    # 计算相似度并预测
    sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
    y_pred_avg = np.argmax(sim_matrix, axis=1)
    
    print("\n平均嵌入分类结果:")
    evaluate_performance(data["test"]["label"], y_pred_avg)
    
    return model, test_embeddings


def zero_shot_classification(data, model, test_embeddings):
    """
    Part 4: 零样本分类
    使用标签描述的嵌入与文档嵌入比较
    """
    print("\n" + "=" * 60)
    print("Part 4: 零样本分类 (Zero-shot)")
    print("=" * 60)
    
    # 为标签创建嵌入
    print("\n创建标签嵌入...")
    label_embeddings = model.encode(["A negative review", "A positive review"])
    
    # 计算相似度
    sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)
    
    print("\n零样本分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    # 尝试不同的标签描述
    print("\n" + "-" * 40)
    print("尝试更详细的标签描述:")
    print("-" * 40)
    
    label_embeddings_v2 = model.encode([
        "A very negative movie review",
        "A very positive movie review"
    ])
    
    sim_matrix_v2 = cosine_similarity(test_embeddings, label_embeddings_v2)
    y_pred_v2 = np.argmax(sim_matrix_v2, axis=1)
    
    print("\n使用详细标签描述的结果:")
    evaluate_performance(data["test"]["label"], y_pred_v2)
    
    return y_pred


def generative_classification(data, device="cpu"):
    """
    Part 5: 使用生成模型 (Flan-T5) 进行分类
    """
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    
    print("\n" + "=" * 60)
    print("Part 5: 生成模型分类 (Flan-T5)")
    print("=" * 60)
    
    # 加载 Flan-T5 模型
    print("\n加载 Flan-T5-small 模型...")
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-small",
        device=device,
        max_new_tokens=10,
    )
    
    # 准备数据: 添加提示词
    prompt = "Is the following sentence positive or negative? "
    data_with_prompt = data.map(lambda example: {"t5": prompt + example['text']})
    
    print(f"\n提示词模板: '{prompt}'")
    print(f"示例输入: {data_with_prompt['test'][0]['t5'][:100]}...")
    
    # 运行推理
    print("\n在测试集上运行推理...")
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data_with_prompt["test"], "t5")), total=len(data_with_prompt["test"])):
        text = output[0]["generated_text"].lower()
        y_pred.append(0 if "negative" in text else 1)
    
    print("\nFlan-T5 分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return y_pred


def chatgpt_classification_demo(data):
    """
    Part 6: ChatGPT 分类演示 (需要 API key)
    这里只展示代码结构，不实际运行
    """
    print("\n" + "=" * 60)
    print("Part 6: ChatGPT 分类 (演示代码)")
    print("=" * 60)
    
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


def print_summary():
    """打印各方法的性能对比总结"""
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    
    print("""
| 方法                                | 准确率 | 说明                           |
|-------------------------------------|--------|--------------------------------|
| Twitter-RoBERTa (任务特定模型)      | ~80%   | 预训练于 Twitter 数据          |
| 嵌入 + 逻辑回归 (监督学习)          | ~85%   | 使用训练数据                   |
| 嵌入 + 平均类别嵌入                 | ~84%   | 无需额外分类器                 |
| 零样本分类 (简单标签描述)           | ~78%   | 无需训练数据                   |
| Flan-T5 (生成模型)                  | ~84%   | 小型模型，零样本               |
| ChatGPT (GPT-3.5)                   | ~91%   | 最高准确率，需要 API 费用      |

关键发现:
1. 监督学习方法 (嵌入+逻辑回归) 在有标注数据时表现很好
2. 零样本方法无需训练数据，适合快速原型
3. 大型生成模型 (如 ChatGPT) 准确率最高，但成本也最高
4. 标签描述的选择会影响零样本分类的性能
""")


def main():
    """主函数"""
    import torch
    
    # 检测设备
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"使用设备: {device}")
    
    # Part 1: 加载数据
    data = load_data()
    
    # Part 2: 任务特定模型分类
    # 注意: 在 CPU 上运行较慢
    task_specific_classification(data, device=device if device != "mps" else "cpu")
    
    # Part 3: 嵌入 + 监督学习
    model, test_embeddings = embedding_supervised_classification(data)
    
    # Part 4: 零样本分类
    zero_shot_classification(data, model, test_embeddings)
    
    # Part 5: 生成模型分类
    generative_classification(data, device=device if device != "mps" else "cpu")
    
    # Part 6: ChatGPT 演示 (仅显示代码)
    chatgpt_classification_demo(data)
    
    # 总结
    print_summary()


if __name__ == "__main__":
    main()
