"""
Part 5: 传统词嵌入 (Word2Vec/GloVe)
演示传统词嵌入的使用和经典的词向量运算
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60)
print("Part 5: 传统词嵌入 (GloVe)")
print("=" * 60)

try:
    import gensim.downloader as api
except ImportError:
    print("\n请安装: pip install gensim")
    exit(1)

# 下载 GloVe 词嵌入（约 66MB）
print("\n下载 GloVe 词嵌入 (glove-wiki-gigaword-50)...")
print("首次下载约 66MB，请稍候...")
model = api.load("glove-wiki-gigaword-50")

print(f"\n词表大小: {len(model)}")
print(f"嵌入维度: {model.vector_size}")

# 查找相似词
print("\n" + "-" * 60)
print("相似词查找:")
print("-" * 60)

words_to_test = ["king", "computer", "happy"]
for word in words_to_test:
    print(f"\n与 '{word}' 最相似的词:")
    similar_words = model.most_similar([model[word]], topn=5)
    for w, score in similar_words:
        bar = "█" * int(score * 20)
        print(f"  {score:.4f} {bar} {w}")

# 经典词向量运算
print("\n" + "-" * 60)
print("经典词向量运算:")
print("-" * 60)

print("\n1. king - man + woman = ?")
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
for word, score in result:
    print(f"   {word}: {score:.4f}")

print("\n2. paris - france + germany = ?")
result = model.most_similar(positive=['paris', 'germany'], negative=['france'], topn=3)
for word, score in result:
    print(f"   {word}: {score:.4f}")

print("\n3. bigger - big + small = ?")
result = model.most_similar(positive=['bigger', 'small'], negative=['big'], topn=3)
for word, score in result:
    print(f"   {word}: {score:.4f}")

# 词向量可视化（简单版）
print("\n" + "-" * 60)
print("词向量示例:")
print("-" * 60)

word = "king"
vector = model[word]
print(f"\n'{word}' 的词向量 (前10维):")
print(f"  [{', '.join(f'{v:.4f}' for v in vector[:10])}, ...]")

print("\n" + "=" * 60)
print("Part 5 完成!")
print("=" * 60)
