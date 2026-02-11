"""
Chapter 4 - Part 1: 加载数据
加载 Rotten Tomatoes 电影评论数据集
"""

from common import load_data


def main():
    print("=" * 60)
    print("Part 1: 加载数据")
    print("=" * 60)
    
    data = load_data()
    
    # 显示更多数据集信息
    print("\n" + "-" * 40)
    print("数据集详细信息:")
    print("-" * 40)
    print(f"训练集大小: {len(data['train'])}")
    print(f"验证集大小: {len(data['validation'])}")
    print(f"测试集大小: {len(data['test'])}")
    
    # 显示标签分布
    import numpy as np
    train_labels = np.array(data['train']['label'])
    print(f"\n训练集标签分布:")
    print(f"  负面 (0): {(train_labels == 0).sum()}")
    print(f"  正面 (1): {(train_labels == 1).sum()}")
    
    # 显示训练集示例
    print("\n" + "-" * 40)
    print("训练集正面评论示例 (label=1):")
    print("-" * 40)
    count = 0
    for i, item in enumerate(data['train']):
        if item['label'] == 1:  # 正面
            print(f"[{i}] 👍 {item['text']}")
            print()
            count += 1
            if count >= 3:
                break
    
    print("-" * 40)
    print("训练集负面评论示例 (label=0):")
    print("-" * 40)
    count = 0
    for i, item in enumerate(data['train']):
        if item['label'] == 0:  # 负面
            print(f"[{i}] 👎 {item['text']}")
            print()
            count += 1
            if count >= 3:
                break
    
    return data


if __name__ == "__main__":
    main()
