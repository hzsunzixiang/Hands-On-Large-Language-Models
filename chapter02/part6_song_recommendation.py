"""
Part 6: 基于嵌入的歌曲推荐系统
使用 Word2Vec 把歌曲当作"词"，播放列表当作"句子"来训练推荐模型
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

# 下载数据
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

print(f"\n数据统计:")
print(f"  播放列表数量: {len(playlists)}")
print(f"  歌曲数量: {len(songs_df)}")
print(f"  平均每个列表歌曲数: {np.mean([len(p) for p in playlists]):.1f}")

# 查看一个播放列表示例
print("\n" + "-" * 60)
print("播放列表示例 (第一个列表的前5首歌):")
print("-" * 60)
for song_id in playlists[0][:5]:
    try:
        song_info = songs_df.loc[song_id]
        print(f"  ID={song_id}: {song_info['title']} - {song_info['artist']}")
    except:
        print(f"  ID={song_id}: (未找到)")

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

def print_recommendations(song_id, topn=5):
    """获取歌曲推荐"""
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id), topn=topn)
    )[:, 0]
    return songs_df.loc[[s for s in similar_songs]]

# 测试推荐
print("\n" + "-" * 60)
print("推荐测试 1: Metallica")
print("-" * 60)

test_song_id = "2172"
print(f"\n输入歌曲 (ID={test_song_id}):")
print(f"  {songs_df.loc[test_song_id]['title']} - {songs_df.loc[test_song_id]['artist']}")

print(f"\n推荐歌曲:")
recommendations = print_recommendations(test_song_id)
for idx, row in recommendations.iterrows():
    print(f"  {row['title']} - {row['artist']}")

# 另一个测试
print("\n" + "-" * 60)
print("推荐测试 2: 2Pac")
print("-" * 60)

test_song_id2 = "842"
print(f"\n输入歌曲 (ID={test_song_id2}):")
print(f"  {songs_df.loc[test_song_id2]['title']} - {songs_df.loc[test_song_id2]['artist']}")

print(f"\n推荐歌曲:")
recommendations2 = print_recommendations(test_song_id2)
for idx, row in recommendations2.iterrows():
    print(f"  {row['title']} - {row['artist']}")

# 查看歌曲嵌入
print("\n" + "-" * 60)
print("歌曲嵌入向量示例:")
print("-" * 60)
song_vector = model.wv[test_song_id]
print(f"\n'{songs_df.loc[test_song_id]['title']}' 的嵌入向量 (前10维):")
print(f"  [{', '.join(f'{v:.4f}' for v in song_vector[:10])}, ...]")

print("\n" + "=" * 60)
print("Part 6 完成!")
print("=" * 60)
