"""
Chapter 4 - 运行全部文本分类实验
这是一个便捷脚本，用于运行所有部分

使用方法:
    python run_all.py              # 运行全部 (除了 ChatGPT)
    python run_all.py --parts 1,2  # 只运行指定部分
    python run_all.py --summary    # 只显示性能对比总结
"""

import argparse

from common import load_data, get_device


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
    parser = argparse.ArgumentParser(
        description="Chapter 4 - 文本分类 (运行全部)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
各部分说明:
  Part 1: 加载数据 (Rotten Tomatoes)
  Part 2: 任务特定模型 (Twitter-RoBERTa)
  Part 3: 嵌入 + 监督学习 (Sentence Transformer + LogisticRegression)
  Part 4: 零样本分类 (Zero-shot)
  Part 5: 生成模型 (Flan-T5)
  Part 6: ChatGPT (需要 API key)

示例:
  python run_all.py              # 运行 Part 1-5
  python run_all.py --parts 1,2  # 只运行 Part 1 和 2
  python run_all.py --parts all  # 运行全部包括 ChatGPT
  python run_all.py --summary    # 只显示总结
        """
    )
    parser.add_argument(
        "--parts", "-p",
        type=str,
        default="1,2,3,4,5",
        help="运行哪些部分: 1,2,3,4,5,6 或 all (默认: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="只显示性能对比总结"
    )
    
    args = parser.parse_args()
    
    if args.summary:
        print_summary()
        return
    
    # 解析要运行的部分
    if args.parts == "all":
        parts = [1, 2, 3, 4, 5, 6]
    else:
        parts = [int(p.strip()) for p in args.parts.split(",")]
    
    print(f"将运行部分: {parts}")
    
    # 获取设备
    device = get_device()
    device_for_pipeline = device if device != "mps" else "cpu"
    
    # Part 1: 加载数据
    data = None
    if any(p in parts for p in [1, 2, 3, 4, 5, 6]):
        data = load_data()
        if parts == [1]:
            return
    
    # Part 2: 任务特定模型
    if 2 in parts:
        from part2_task_specific import task_specific_classification
        task_specific_classification(data, device=device_for_pipeline)
    
    # Part 3 & 4: 嵌入相关
    model, test_embeddings = None, None
    if 3 in parts:
        from part3_embedding_supervised import embedding_supervised_classification, average_embedding_classification
        model, train_embeddings, test_embeddings, _ = embedding_supervised_classification(data)
        average_embedding_classification(data, train_embeddings, test_embeddings)
    
    if 4 in parts:
        from part4_zero_shot import zero_shot_classification, zero_shot_detailed_labels
        model, test_embeddings, _ = zero_shot_classification(data, model, test_embeddings)
        zero_shot_detailed_labels(data, model, test_embeddings)
    
    # Part 5: 生成模型
    if 5 in parts:
        from part5_generative import generative_classification
        generative_classification(data, device=device_for_pipeline)
    
    # Part 6: ChatGPT
    if 6 in parts:
        from part6_chatgpt import chatgpt_classification
        chatgpt_classification(data)
    
    # 总结
    print_summary()


if __name__ == "__main__":
    main()
