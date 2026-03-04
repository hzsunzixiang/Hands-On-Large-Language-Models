"""
Chapter 8 - Part 1: Dense Retrieval
使用 Cohere Embedding + FAISS 进行语义搜索

流程:
  1. 准备文本并分句 (chunking)
  2. 使用 Cohere API 生成 embedding
  3. 构建 FAISS 向量索引
  4. 对查询进行语义搜索
"""

import cohere
import numpy as np
import faiss
import pandas as pd

# ============================================================
# 1. 初始化 Cohere 客户端
# ============================================================
api_key = ''  # 在此填入你的 Cohere API key
co = cohere.Client(api_key)

# ============================================================
# 2. 准备文本数据并分句 (Chunking)
# ============================================================
text = """
Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.

Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock, expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014.
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight.
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""

# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = [t.strip(' \n') for t in texts]

# ============================================================
# 3. 使用 Cohere 生成 Embedding
# ============================================================
response = co.embed(
    texts=texts,
    input_type="search_document",
).embeddings

embeds = np.array(response)
print(f"Embedding shape: {embeds.shape}")

# ============================================================
# 4. 构建 FAISS 索引
# ============================================================
dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.float32(embeds))

# ============================================================
# 5. 定义语义搜索函数
# ============================================================
def search(query, number_of_results=3):
    """使用 Cohere embedding + FAISS 进行语义搜索"""
    # 1. Get the query's embedding
    query_embed = co.embed(
        texts=[query],
        input_type="search_query",
    ).embeddings[0]

    # 2. Retrieve the nearest neighbors
    distances, similar_item_ids = index.search(
        np.float32([query_embed]), number_of_results
    )

    # 3. Format the results
    texts_np = np.array(texts)
    results = pd.DataFrame(data={
        'texts': texts_np[similar_item_ids[0]],
        'distance': distances[0]
    })

    # 4. Print and return the results
    print(f"Query: '{query}'\nNearest neighbors:")
    return results


# ============================================================
# 6. 测试语义搜索
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: 语义相关查询")
    print("=" * 60)
    query = "how precise was the science"
    results = search(query)
    print(results)

    print("\n" + "=" * 60)
    print("测试 2: 无关查询 (Dense Retrieval 的缺陷)")
    print("=" * 60)
    query = "What is the mass of the moon?"
    results = search(query)
    print(results)
