"""
Part 6: 基于嵌入的歌曲推荐系统
使用 Word2Vec 把歌曲当作"词"，播放列表当作"句子"来训练推荐模型

完全按照 notebook 的实现方式
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60)
print("Part 6: 基于嵌入的歌曲推荐系统")
print("=" * 60)

import numpy as np
import pandas as pd
from urllib import request
from gensim.models import Word2Vec

# 数据缓存目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

PLAYLIST_CACHE = os.path.join(CACHE_DIR, 'playlists_train.txt')
SONGS_CACHE = os.path.join(CACHE_DIR, 'song_hash.txt')

def download_if_not_cached(url, cache_path):
    """检查本地缓存，没有则下载"""
    if os.path.exists(cache_path):
        print(f"  使用本地缓存: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"  下载: {url}")
        data = request.urlopen(url).read().decode("utf-8")
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"  已缓存到: {cache_path}")
        return data

# 下载/加载数据
print("\n加载播放列表数据...")

# 获取播放列表数据
playlist_data = download_if_not_cached(
    'https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt',
    PLAYLIST_CACHE
)
lines = playlist_data.split('\n')[2:]
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# 获取歌曲元数据 (注意：不设置 index，使用默认整数索引)
songs_data = download_if_not_cached(
    'https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt',
    SONGS_CACHE
)
songs = [s.rstrip().split('\t') for s in songs_data.split('\n')]
songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
# 不使用 set_index，保持默认整数索引，这样 iloc[2172] 就是第 2172 行

print(f"\n数据统计:")
print(f"  播放列表数量: {len(playlists)}")
print(f"  歌曲数量: {len(songs_df)}")
print(f"  平均每个列表歌曲数: {np.mean([len(p) for p in playlists]):.1f}")

# 查看一个播放列表示例
print("\n" + "-" * 60)
print("播放列表示例 (第一个列表的前5首歌):")
print("-" * 60)
print(f"  歌曲ID列表: {playlists[0][:5]}")

# 训练 Word2Vec 模型
print("\n" + "-" * 60)
print("训练 Word2Vec 模型...")
print("-" * 60)
print("核心思想: 把歌曲ID当作'词'，播放列表当作'句子'")
print("经常出现在同一播放列表中的歌曲会有相似的嵌入向量")

model = Word2Vec(
    playlists,
    vector_size=32,   # 嵌入维度
    window=20,        # 上下文窗口（考虑前后20首歌）
    negative=50,      # 负采样
    min_count=1,
    workers=4,
    epochs=10
)

print(f"\n模型训练完成!")
print(f"  嵌入维度: {model.wv.vector_size}")
print(f"  词表大小: {len(model.wv)}")

# 按照 notebook 的方式：使用 iloc (位置索引)
def print_recommendations(song_id, topn=5):
    """获取歌曲推荐 - 使用 iloc 按位置索引"""
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id), topn=topn)
    )[:, 0]
    # 将字符串 ID 转为整数作为位置索引
    indices = [int(s) for s in similar_songs]
    return songs_df.iloc[indices]

# 测试推荐 1: Metallica
print("\n" + "-" * 60)
print("推荐测试 1: Metallica - Fade To Black")
print("-" * 60)

song_id = 2172
print(f"\n输入歌曲 (位置索引={song_id}):")
print(songs_df.iloc[song_id])

print(f"\n推荐的相似歌曲:")
recommendations = print_recommendations(song_id)
for idx, row in recommendations.iterrows():
    print(f"  {row['title']} - {row['artist']}")

# 测试推荐 2: 2Pac
print("\n" + "-" * 60)
print("推荐测试 2: 2Pac")
print("-" * 60)

song_id2 = 842
print(f"\n输入歌曲 (位置索引={song_id2}):")
print(songs_df.iloc[song_id2])

print(f"\n推荐的相似歌曲:")
recommendations2 = print_recommendations(song_id2)
for idx, row in recommendations2.iterrows():
    print(f"  {row['title']} - {row['artist']}")

# 查看歌曲嵌入向量
print("\n" + "-" * 60)
print("歌曲嵌入向量示例:")
print("-" * 60)
song_vector = model.wv[str(song_id)]
print(f"\n'Fade To Black' 的嵌入向量 (前10维):")
print(f"  [{', '.join(f'{v:.4f}' for v in song_vector[:10])}, ...]")

# 解释原理
print("\n" + "-" * 60)
print("原理解释:")
print("-" * 60)
print("""
这个推荐系统的核心思想:

1. 把歌曲ID当作"词"，播放列表当作"句子"
   - 播放列表: ["2172", "2849", "2640", ...]  ← 相当于一个句子
   - 每首歌: "2172" ← 相当于一个词

2. 使用 Word2Vec 训练
   - 经常在同一播放列表中出现的歌曲 → 向量相似
   - 就像经常在同一上下文出现的词 → 向量相似

3. 推荐逻辑
   - 给定一首歌的嵌入向量
   - 找到向量空间中最近的其他歌曲
   - 这些歌曲就是推荐结果

这就是 Word2Vec 在推荐系统中的应用！
""")

print("\n" + "=" * 60)
print("Part 6 完成!")
print("=" * 60)
