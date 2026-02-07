"""
Chapter 5 - Part 1: 加载数据
加载 ArXiv NLP 论文数据集
"""

from common import load_data


def main():
    print("=" * 60)
    print("Part 1: 加载数据")
    print("=" * 60)
    
    abstracts, titles = load_data(sample_size=None)  # 加载全部数据
    
    # 显示更多信息
    print("\n" + "-" * 40)
    print("数据集统计:")
    print("-" * 40)
    print(f"论文总数: {len(abstracts)}")
    
    # 摘要长度统计
    lengths = [len(a) for a in abstracts]
    print(f"摘要平均长度: {sum(lengths)/len(lengths):.0f} 字符")
    print(f"摘要最短: {min(lengths)} 字符")
    print(f"摘要最长: {max(lengths)} 字符")
    
    return abstracts, titles


if __name__ == "__main__":
    main()
