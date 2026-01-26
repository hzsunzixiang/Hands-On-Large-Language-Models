"""
Part 4: 句子嵌入 (Sentence Embeddings)
演示使用 Sentence Transformers 生成句子嵌入并计算相似度
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"

print("=" * 60)
print("Part 4: 句子嵌入 (Sentence Embeddings)")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("\n请安装依赖:")
    print("  pip install sentence-transformers scikit-learn")
    exit(1)

# 加载句子嵌入模型
print("\n加载 Sentence Transformer 模型...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 测试句子
sentences = [
    "Best movie ever!",
    "This film is amazing!",
    "I love programming in Python.",
    "The weather is nice today."
]

print("\n测试句子:")
for i, s in enumerate(sentences):
    print(f"  {i}: {s}")

# 生成嵌入
embeddings = model.encode(sentences)

print(f"\n句子数量: {len(sentences)}")
print(f"嵌入维度: {embeddings.shape[1]}")

# 计算相似度
print("\n" + "-" * 60)
print("句子相似度矩阵:")
print("-" * 60)
similarity_matrix = cosine_similarity(embeddings)

# 打印相似度
for i, s1 in enumerate(sentences):
    print(f"\n'{s1}' 与其他句子的相似度:")
    for j, s2 in enumerate(sentences):
        if i != j:
            sim = similarity_matrix[i][j]
            bar = "█" * int(sim * 20)
            print(f"  {sim:.4f} {bar} '{s2}'")

# 语义搜索示例
print("\n" + "-" * 60)
print("语义搜索示例:")
print("-" * 60)

query = "I really enjoyed watching that movie"
query_embedding = model.encode([query])

print(f"\n查询: '{query}'")
print("\n搜索结果 (按相似度排序):")

similarities = cosine_similarity(query_embedding, embeddings)[0]
ranked_indices = similarities.argsort()[::-1]

for rank, idx in enumerate(ranked_indices):
    sim = similarities[idx]
    print(f"  {rank+1}. [{sim:.4f}] {sentences[idx]}")

print("\n" + "=" * 60)
print("Part 4 完成!")
print("=" * 60)
