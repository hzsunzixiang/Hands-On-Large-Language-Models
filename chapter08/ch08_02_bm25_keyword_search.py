"""
Chapter 8 - Part 2: BM25 Keyword Search (词法搜索)
使用 BM25 算法进行关键词匹配搜索, 与语义搜索形成对比

流程:
  1. 复用 Part 1 的文本数据
  2. 自定义 BM25 tokenizer (去停用词、去标点)
  3. 构建 BM25 索引
  4. 执行关键词搜索
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm import tqdm

# ============================================================
# 1. 准备文本数据 (与 Part 1 相同)
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

texts = text.split('.')
texts = [t.strip(' \n') for t in texts]


# ============================================================
# 2. BM25 Tokenizer
# ============================================================
def bm25_tokenizer(text):
    """自定义 tokenizer: 小写化、去标点、去停用词"""
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


# ============================================================
# 3. 构建 BM25 索引
# ============================================================
tokenized_corpus = []
for passage in tqdm(texts, desc="Tokenizing corpus"):
    tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)


# ============================================================
# 4. 关键词搜索函数
# ============================================================
def keyword_search(query, top_k=3, num_candidates=15):
    """使用 BM25 进行词法搜索"""
    print("Input question:", query)

    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    print(f"Top-{top_k} lexical search (BM25) hits")
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(
            hit['score'],
            texts[hit['corpus_id']].replace("\n", " ")
        ))

    return bm25_hits[:top_k]


# ============================================================
# 5. 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BM25 关键词搜索测试")
    print("=" * 60)
    keyword_search(query="how precise was the science")
