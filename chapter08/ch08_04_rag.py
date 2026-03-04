"""
Chapter 8 - Part 4: Retrieval-Augmented Generation (RAG)
两种 RAG 实现方式:
  A) 使用 Cohere API (云端 LLM) 进行 Grounded Generation
  B) 使用本地模型 (Phi-3 GGUF + HuggingFace Embedding + FAISS + LangChain)

流程:
  A: Cohere embedding search -> Cohere chat (grounded generation with citations)
  B: 下载 Phi-3 GGUF -> 加载本地 LLM -> HuggingFace embedding -> FAISS 向量库 -> LangChain RAG 链
"""

import numpy as np
import faiss
import pandas as pd

# ============================================================
# 共用的文本数据
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
# Part A: Cohere API 的 Grounded Generation (RAG)
# ============================================================
def run_cohere_rag():
    """使用 Cohere API 进行 RAG: embedding search + grounded chat"""
    import cohere

    api_key = ''  # 在此填入你的 Cohere API key
    co = cohere.Client(api_key)

    # 1. 构建 embedding 索引
    response = co.embed(
        texts=texts,
        input_type="search_document",
    ).embeddings
    embeds = np.array(response)

    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.float32(embeds))

    # 2. 搜索函数
    def search(query, number_of_results=3):
        query_embed = co.embed(
            texts=[query],
            input_type="search_query",
        ).embeddings[0]
        distances, similar_item_ids = index.search(
            np.float32([query_embed]), number_of_results
        )
        texts_np = np.array(texts)
        results = pd.DataFrame(data={
            'texts': texts_np[similar_item_ids[0]],
            'distance': distances[0]
        })
        print(f"Query: '{query}'\nNearest neighbors:")
        return results

    # 3. RAG: 检索 + 生成
    query = "income generated"
    results = search(query)

    docs_dict = [{'text': t} for t in results['texts']]
    response = co.chat(
        message=query,
        documents=docs_dict,
    )

    print(f"\nRAG Response: {response.text}")
    print(f"Citations: {response.citations}")

    return response


# ============================================================
# Part B: 本地模型的 RAG (Phi-3 + HuggingFace + LangChain)
# ============================================================
def run_local_rag():
    """使用本地模型进行 RAG: Phi-3 GGUF + BGE embedding + FAISS + LangChain"""
    import os
    import subprocess
    from langchain import LlamaCpp, PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS as LangchainFAISS

    # 1. 下载 Phi-3 模型 (如果不存在)
    model_filename = "Phi-3-mini-4k-instruct-q4.gguf"
    if not os.path.exists(model_filename):
        print(f"Downloading {model_filename}...")
        subprocess.run([
            "wget",
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
        ], check=True)

    # 2. 加载本地 LLM
    llm = LlamaCpp(
        model_path=model_filename,
        n_gpu_layers=-1,
        max_tokens=500,
        n_ctx=2048,
        seed=42,
        verbose=False,
    )

    # 3. 加载 Embedding 模型
    embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-small-en-v1.5'
    )

    # 4. 创建 FAISS 向量库
    db = LangchainFAISS.from_texts(texts, embedding_model)

    # 5. 定义 RAG prompt
    template = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    # 6. 构建 RAG 链
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
    )

    # 7. 执行查询
    result = rag.invoke('Income generated')
    print(f"\nRAG Result: {result}")

    return result


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "local":
        print("=" * 60)
        print("Part B: 本地模型 RAG (Phi-3 + BGE + LangChain)")
        print("=" * 60)
        run_local_rag()
    else:
        print("=" * 60)
        print("Part A: Cohere API RAG (Grounded Generation)")
        print("=" * 60)
        run_cohere_rag()

    print("\n提示: 运行 'python ch08_04_rag.py local' 使用本地模型")
