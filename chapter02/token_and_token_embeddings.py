"""
Chapter 2 - Tokens 和 Token Embeddings
探索 tokens 和 embeddings 作为构建 LLM 的重要组成部分

主要内容：
1. Tokenizer 基础 - 文本如何被切分和编码
2. 比较不同模型的 Tokenizer
3. 上下文词嵌入 (Contextualized Word Embeddings)
4. 句子/文档嵌入 (Sentence Embeddings)
5. 传统词嵌入 (Word2Vec/GloVe)
6. 实战：基于嵌入的歌曲推荐系统
"""

import torch
import numpy as np


def get_device():
    """检测可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# Part 1: Tokenizer 基础
# ============================================================
def demo_tokenizer_basics():
    """演示 Tokenizer 的基本使用"""
    print("\n" + "=" * 60)
    print("Part 1: Tokenizer 基础")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    
    # 加载 Phi-3 的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    # 测试文本
    prompt = "Write an email apologizing to Sarah for the tragic gardening mishap."
    
    # 将文本转换为 token IDs
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"\n原始文本: {prompt}")
    print(f"\nToken IDs shape: {input_ids.shape}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # 逐个解码 token 查看分词结果
    print("\n分词结果:")
    for i, token_id in enumerate(input_ids[0]):
        token = tokenizer.decode(token_id)
        print(f"  {i}: ID={token_id.item():5d} -> '{token}'")
    
    # 演示子词组合
    print("\n子词组合示例:")
    print(f"  tokenizer.decode([3323, 622]) = '{tokenizer.decode([3323, 622])}'")
    
    return tokenizer


# ============================================================
# Part 2: 比较不同模型的 Tokenizer
# ============================================================
def demo_compare_tokenizers():
    """比较不同 LLM 的分词方式"""
    print("\n" + "=" * 60)
    print("Part 2: 比较不同模型的 Tokenizer")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    
    # ANSI 颜色代码用于可视化
    colors_list = [
        '102;194;165', '252;141;98', '141;160;203',
        '231;138;195', '166;216;84', '255;217;47'
    ]
    
    def show_tokens(sentence, tokenizer_name):
        """可视化展示分词结果"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            token_ids = tokenizer(sentence).input_ids
            print(f"\n{tokenizer_name} ({len(token_ids)} tokens):")
            tokens = []
            for idx, t in enumerate(token_ids):
                color = colors_list[idx % len(colors_list)]
                token = tokenizer.decode(t)
                tokens.append(f'\x1b[0;30;48;2;{color}m{token}\x1b[0m')
            print(' '.join(tokens))
        except Exception as e:
            print(f"\n{tokenizer_name}: 加载失败 - {e}")
    
    # 测试文本（包含各种特殊情况）
    text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else:
12.0*50=600
"""
    
    print(f"测试文本: {text}")
    
    # 比较不同的 tokenizer
    tokenizers_to_compare = [
        "bert-base-uncased",      # BERT (小写)
        "bert-base-cased",        # BERT (保留大小写)
        "gpt2",                   # GPT-2
        "google/flan-t5-small",   # T5
    ]
    
    for tokenizer_name in tokenizers_to_compare:
        show_tokens(text, tokenizer_name)


# ============================================================
# Part 3: 上下文词嵌入 (Contextualized Embeddings)
# ============================================================
def demo_contextualized_embeddings():
    """演示从语言模型获取上下文感知的词嵌入"""
    print("\n" + "=" * 60)
    print("Part 3: 上下文词嵌入 (Contextualized Embeddings)")
    print("=" * 60)
    
    from transformers import AutoModel, AutoTokenizer
    
    # 加载 DeBERTa 模型和 tokenizer
    print("\n加载 DeBERTa 模型...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
    
    # 处理文本
    text = "Hello world"
    tokens = tokenizer(text, return_tensors='pt')
    
    # 获取上下文嵌入
    with torch.no_grad():
        output = model(**tokens)[0]
    
    print(f"\n输入文本: '{text}'")
    print(f"Token 数量: {output.shape[1]}")
    print(f"嵌入维度: {output.shape[2]}")
    print(f"输出形状: {output.shape}")  # [batch, tokens, embedding_dim]
    
    # 展示每个 token 的信息
    print("\n每个 Token 的嵌入:")
    for i, token_id in enumerate(tokens['input_ids'][0]):
        token = tokenizer.decode(token_id)
        embedding = output[0, i, :5].tolist()  # 只显示前5维
        print(f"  {i}: '{token}' -> [{', '.join(f'{x:.4f}' for x in embedding)}, ...]")


# ============================================================
# Part 4: 句子嵌入 (Sentence Embeddings)
# ============================================================
def demo_sentence_embeddings():
    """演示使用 Sentence Transformers 生成句子嵌入"""
    print("\n" + "=" * 60)
    print("Part 4: 句子嵌入 (Sentence Embeddings)")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("请安装: pip install sentence-transformers")
        return
    
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
    
    # 生成嵌入
    embeddings = model.encode(sentences)
    
    print(f"\n句子数量: {len(sentences)}")
    print(f"嵌入维度: {embeddings.shape[1]}")
    
    # 计算相似度
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n句子相似度矩阵:")
    similarity_matrix = cosine_similarity(embeddings)
    
    # 打印相似度
    for i, s1 in enumerate(sentences):
        print(f"\n'{s1[:30]}...' 与其他句子的相似度:")
        for j, s2 in enumerate(sentences):
            if i != j:
                print(f"  -> '{s2[:30]}...': {similarity_matrix[i][j]:.4f}")


# ============================================================
# Part 5: 传统词嵌入 (Word2Vec/GloVe)
# ============================================================
def demo_word_embeddings():
    """演示传统词嵌入的使用"""
    print("\n" + "=" * 60)
    print("Part 5: 传统词嵌入 (GloVe)")
    print("=" * 60)
    
    try:
        import gensim.downloader as api
    except ImportError:
        print("请安装: pip install gensim")
        return
    
    # 下载 GloVe 词嵌入（约 66MB）
    print("\n下载 GloVe 词嵌入 (glove-wiki-gigaword-50)...")
    print("首次下载约 66MB，请稍候...")
    model = api.load("glove-wiki-gigaword-50")
    
    # 查找相似词
    word = "king"
    print(f"\n与 '{word}' 最相似的词:")
    similar_words = model.most_similar([model[word]], topn=10)
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")
    
    # 词向量运算: king - man + woman ≈ queen
    print("\n词向量运算: king - man + woman = ?")
    result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
    for word, score in result:
        print(f"  {word}: {score:.4f}")


# ============================================================
# Part 6: 实战 - 基于嵌入的歌曲推荐系统
# ============================================================
def demo_song_recommendation():
    """使用 Word2Vec 构建歌曲推荐系统"""
    print("\n" + "=" * 60)
    print("Part 6: 基于嵌入的歌曲推荐系统")
    print("=" * 60)
    
    import pandas as pd
    from urllib import request
    from gensim.models import Word2Vec
    
    print("\n下载播放列表数据...")
    
    # 获取播放列表数据
    data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')
    lines = data.read().decode("utf-8").split('\n')[2:]
    playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]
    
    # 获取歌曲元数据
    songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
    songs_file = songs_file.read().decode("utf-8").split('\n')
    songs = [s.rstrip().split('\t') for s in songs_file]
    songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
    songs_df = songs_df.set_index('id')
    
    print(f"播放列表数量: {len(playlists)}")
    print(f"歌曲数量: {len(songs_df)}")
    
    # 训练 Word2Vec 模型
    # 把歌曲 ID 当作 "词"，播放列表当作 "句子"
    print("\n训练 Word2Vec 模型...")
    model = Word2Vec(
        playlists,
        vector_size=32,   # 嵌入维度
        window=20,        # 上下文窗口
        negative=50,      # 负采样
        min_count=1,
        workers=4
    )
    
    def get_recommendations(song_id, topn=5):
        """获取歌曲推荐"""
        similar_songs = np.array(
            model.wv.most_similar(positive=str(song_id), topn=topn)
        )[:, 0]
        return songs_df.loc[similar_songs]
    
    # 测试推荐
    test_song_id = 2172
    print(f"\n测试歌曲 (ID={test_song_id}):")
    print(songs_df.loc[str(test_song_id)])
    
    print(f"\n推荐歌曲:")
    recommendations = get_recommendations(test_song_id)
    print(recommendations)
    
    # 另一个测试
    test_song_id2 = 842
    print(f"\n\n测试歌曲 (ID={test_song_id2}):")
    print(songs_df.loc[str(test_song_id2)])
    
    print(f"\n推荐歌曲:")
    recommendations2 = get_recommendations(test_song_id2)
    print(recommendations2)


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("Chapter 2: Tokens 和 Token Embeddings")
    print("=" * 60)
    
    device = get_device()
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"运行设备: {device}")
    
    # 运行各个演示
    # 可以注释掉不想运行的部分
    
    # Part 1: Tokenizer 基础
    demo_tokenizer_basics()
    
    # Part 2: 比较不同 Tokenizer
    demo_compare_tokenizers()
    
    # Part 3: 上下文词嵌入
    demo_contextualized_embeddings()
    
    # Part 4: 句子嵌入 (需要 sentence-transformers)
    demo_sentence_embeddings()
    
    # Part 5: 传统词嵌入 (需要下载数据，约 66MB)
    # demo_word_embeddings()
    
    # Part 6: 歌曲推荐 (需要下载数据和训练)
    # demo_song_recommendation()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
