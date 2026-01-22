"""
Chapter 8 - Semantic Search and Retrieval-Augmented Generation
语义搜索与检索增强生成 (RAG)

本章内容:
1. Dense Retrieval (密集检索) - 使用嵌入向量进行语义搜索
2. Sparse Retrieval (稀疏检索) - BM25 关键词搜索
3. Reranking (重排序) - 使用 Cross-Encoder 优化搜索结果
4. RAG - 检索增强生成，结合搜索与 LLM 生成

注意: 本章原始代码使用 Cohere API，此版本改为使用本地模型
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch


def get_device():
    """
    自动检测最佳可用设备
    优先级: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"使用设备: CUDA ({device_name})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("使用设备: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("使用设备: CPU")
    return device


# ============================================================
# 示例数据: 电影《星际穿越》的介绍文本
# ============================================================
INTERSTELLAR_TEXT = """
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


def prepare_text_chunks(text):
    """
    Part 1: 文本分块
    将长文本按句子分割成小块
    """
    print("\n" + "=" * 60)
    print("Part 1: 文本分块 (Text Chunking)")
    print("=" * 60)
    
    # 按句号分割
    texts = text.split('.')
    # 清理空白字符
    texts = [t.strip(' \n') for t in texts if t.strip()]
    
    print(f"原文长度: {len(text)} 字符")
    print(f"分块数量: {len(texts)} 个句子")
    print(f"\n前 3 个文本块:")
    for i, t in enumerate(texts[:3]):
        print(f"  [{i}] {t[:80]}...")
    
    return texts


def create_embeddings_local(texts, device=None):
    """
    Part 2: 使用本地 Sentence Transformer 生成嵌入
    替代原版的 Cohere API
    """
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "=" * 60)
    print("Part 2: 生成文档嵌入 (Document Embeddings)")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 使用 BGE 嵌入模型 (与原版 Cohere 嵌入维度不同但效果类似)
    print("\n加载嵌入模型: BAAI/bge-small-en-v1.5")
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    
    # 生成嵌入
    print("生成嵌入中...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    print(f"\n嵌入维度: {embeddings.shape}")
    print(f"每个文档被转换为 {embeddings.shape[1]} 维向量")
    
    return embeddings, embedding_model


def build_faiss_index(embeddings):
    """
    Part 3: 构建 FAISS 向量索引
    用于高效的相似性搜索
    """
    import faiss
    
    print("\n" + "=" * 60)
    print("Part 3: 构建向量索引 (FAISS Index)")
    print("=" * 60)
    
    # 获取嵌入维度
    dim = embeddings.shape[1]
    
    # 创建 L2 距离索引 (欧几里得距离)
    index = faiss.IndexFlatL2(dim)
    
    # 添加向量到索引
    index.add(np.float32(embeddings))
    
    print(f"索引类型: IndexFlatL2 (L2 距离)")
    print(f"向量维度: {dim}")
    print(f"索引中向量数量: {index.ntotal}")
    
    return index


def dense_search(query, index, embedding_model, texts, top_k=3):
    """
    Part 4.1: 密集检索 (Dense Retrieval)
    使用嵌入向量进行语义搜索
    """
    print(f"\n--- 密集检索: '{query}' ---")
    
    # 1. 将查询转换为嵌入向量
    query_embed = embedding_model.encode([query])
    
    # 2. 在索引中搜索最近邻
    distances, similar_ids = index.search(np.float32(query_embed), top_k)
    
    # 3. 格式化结果
    texts_np = np.array(texts)
    results = pd.DataFrame({
        'text': texts_np[similar_ids[0]],
        'distance': distances[0]
    })
    
    print(f"最近邻结果 (距离越小越相似):")
    for i, row in results.iterrows():
        print(f"  [{i}] 距离={row['distance']:.2f}: {row['text'][:60]}...")
    
    return results


def bm25_search(query, texts, top_k=3):
    """
    Part 4.2: 稀疏检索 (Sparse Retrieval) - BM25
    基于关键词的传统搜索方法
    """
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
    import string
    
    print(f"\n--- BM25 关键词搜索: '{query}' ---")
    
    # 分词器
    def bm25_tokenizer(text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)
            if len(token) > 0 and token not in ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc
    
    # 构建 BM25 索引
    tokenized_corpus = [bm25_tokenizer(doc) for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 搜索
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argsort(bm25_scores)[::-1][:top_k]
    
    print(f"Top-{top_k} BM25 结果 (分数越高越相关):")
    for idx in top_n:
        print(f"  分数={bm25_scores[idx]:.3f}: {texts[idx][:60]}...")
    
    return bm25, bm25_scores


def compare_search_methods(texts, index, embedding_model):
    """
    Part 4.3: 对比密集检索和稀疏检索
    展示两种方法的优缺点
    """
    print("\n" + "=" * 60)
    print("Part 4: 对比搜索方法")
    print("=" * 60)
    
    # 测试查询 1: 语义理解测试
    query1 = "how precise was the science"
    print(f"\n[测试 1] 查询: '{query1}'")
    print("=" * 40)
    dense_search(query1, index, embedding_model, texts)
    bm25_search(query1, texts)
    
    # 测试查询 2: 域外问题测试 (数据中不存在的信息)
    query2 = "What is the mass of the moon?"
    print(f"\n[测试 2] 查询: '{query2}'")
    print("=" * 40)
    print("注意: 这个问题的答案不在文档中")
    dense_search(query2, index, embedding_model, texts)


def reranking_demo(texts):
    """
    Part 5: 重排序 (Reranking)
    使用 Cross-Encoder 对初步检索结果进行精细排序
    """
    from sentence_transformers import CrossEncoder
    
    print("\n" + "=" * 60)
    print("Part 5: 重排序 (Reranking with Cross-Encoder)")
    print("=" * 60)
    
    print("\n加载 Cross-Encoder 重排序模型...")
    # 使用 MS MARCO 训练的 Cross-Encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    query = "how precise was the science"
    print(f"\n查询: '{query}'")
    
    # 对所有文档进行重排序
    pairs = [[query, text] for text in texts]
    scores = reranker.predict(pairs)
    
    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\nCross-Encoder 重排序结果 (分数越高越相关):")
    for i, idx in enumerate(sorted_indices[:3]):
        print(f"  [{i}] 分数={scores[idx]:.4f}: {texts[idx][:60]}...")
    
    return reranker


def rag_with_local_model(texts, device=None):
    """
    Part 6: 检索增强生成 (RAG) - 使用本地模型
    结合向量检索和 LLM 生成
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    
    print("\n" + "=" * 60)
    print("Part 6: RAG - 检索增强生成")
    print("=" * 60)
    
    if device is None:
        device = get_device()
    
    # 6.1 加载嵌入模型
    print("\n[6.1] 加载嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        model_kwargs={'device': device}
    )
    
    # 6.2 创建向量数据库
    print("[6.2] 创建向量数据库...")
    db = FAISS.from_texts(texts, embedding_model)
    
    # 6.3 加载 LLM
    print("[6.3] 加载 LLM (llama.cpp)...")
    try:
        from langchain_community.llms import LlamaCpp
        from llama_cpp import Llama
        
        # 下载模型
        llm_raw = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
            n_gpu_layers=-1 if device != "cpu" else 0,
            n_ctx=2048,
            verbose=False
        )
        model_path = llm_raw.model_path
        del llm_raw
        
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1 if device != "cpu" else 0,
            max_tokens=500,
            n_ctx=2048,
            seed=42,
            verbose=False
        )
    except Exception as e:
        print(f"加载 LlamaCpp 失败: {e}")
        print("使用 HuggingFace 模型替代...")
        
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_new_tokens=200,
            device=0 if device == "cuda" else -1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    
    # 6.4 创建 RAG 提示模板
    print("[6.4] 创建 RAG 流水线...")
    template = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 6.5 创建 RAG 链
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )
    
    # 6.6 测试 RAG
    print("\n[6.6] 测试 RAG 查询...")
    queries = [
        "Income generated by the film",
        "Who directed the film?",
        "What awards did it win?"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        try:
            result = rag.invoke(query)
            print(f"回答: {result['result']}")
        except Exception as e:
            print(f"错误: {e}")
    
    return rag


def print_summary():
    """打印章节总结"""
    print("\n" + "=" * 60)
    print("Chapter 8 总结")
    print("=" * 60)
    
    summary = """
┌─────────────────────────────────────────────────────────────┐
│                    语义搜索与 RAG                            │
├─────────────────────────────────────────────────────────────┤
│  1. Dense Retrieval (密集检索)                              │
│     - 使用嵌入向量表示文档和查询                              │
│     - 通过向量相似度进行语义匹配                              │
│     - 优点: 理解语义，处理同义词                              │
│     - 缺点: 可能返回不相关但语义接近的结果                    │
│                                                             │
│  2. Sparse Retrieval (稀疏检索 / BM25)                      │
│     - 基于词频的传统关键词匹配                                │
│     - 优点: 精确匹配关键词                                   │
│     - 缺点: 无法处理同义词和语义变化                          │
│                                                             │
│  3. Hybrid Search (混合搜索)                                │
│     - 结合 Dense + Sparse 的优点                            │
│     - 先用 BM25 召回候选，再用向量排序                        │
│                                                             │
│  4. Reranking (重排序)                                      │
│     - 使用 Cross-Encoder 对初步结果精细排序                  │
│     - 比 Bi-Encoder 更精确但更慢                            │
│                                                             │
│  5. RAG (检索增强生成)                                      │
│     - Query → 检索相关文档 → 作为上下文 → LLM 生成回答        │
│     - 解决 LLM 知识过时和幻觉问题                            │
│     - 组件: Embedding + VectorDB + Retriever + LLM          │
└─────────────────────────────────────────────────────────────┘

RAG 流水线:
  ┌────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
  │ 用户   │ -> │ 检索器   │ -> │ 相关    │ -> │ LLM     │
  │ 查询   │    │ (向量库) │    │ 文档    │    │ 生成    │
  └────────┘    └──────────┘    └─────────┘    └─────────┘
                                     ↓
                              ┌─────────────┐
                              │ 最终回答    │
                              │ (带引用)    │
                              └─────────────┘
"""
    print(summary)


def main():
    """主函数"""
    device = get_device()
    
    # Part 1: 准备文本数据
    texts = prepare_text_chunks(INTERSTELLAR_TEXT)
    
    # Part 2: 生成嵌入
    embeddings, embedding_model = create_embeddings_local(texts, device)
    
    # Part 3: 构建索引
    index = build_faiss_index(embeddings)
    
    # Part 4: 对比搜索方法
    compare_search_methods(texts, index, embedding_model)
    
    # Part 5: 重排序演示
    try:
        reranking_demo(texts)
    except Exception as e:
        print(f"\n重排序演示跳过: {e}")
    
    # Part 6: RAG 演示 (可选，需要更多资源)
    print("\n" + "=" * 60)
    print("是否运行 RAG 演示? (需要下载 LLM)")
    print("=" * 60)
    
    try:
        run_rag = input("输入 'y' 运行 RAG 演示，其他跳过: ").strip().lower()
        if run_rag == 'y':
            rag_with_local_model(texts, device)
    except EOFError:
        print("非交互模式，跳过 RAG 演示")
    
    # 打印总结
    print_summary()


if __name__ == "__main__":
    main()
