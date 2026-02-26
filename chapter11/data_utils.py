"""
数据集缓存工具
首次从 HuggingFace 下载后保存到本地 data_cache/ 目录，
之后直接从本地加载，无需联网。
"""
import os
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict

# 缓存目录: chapter11/data_cache/
CACHE_DIR = Path(__file__).parent / "data_cache"


def load_cached_dataset(name, revision=None, **kwargs):
    """
    带本地缓存的数据集加载

    参数:
        name: 数据集名称, 如 "rotten_tomatoes", "conll2003"
        revision: HuggingFace revision, 如 "refs/convert/parquet"
        **kwargs: 传给 load_dataset 的其他参数

    流程:
        1. 检查 data_cache/{safe_name}/ 是否存在
        2. 存在 → load_from_disk (完全离线，秒级加载)
        3. 不存在 → load_dataset 下载 → save_to_disk 缓存
    """
    # 将 name 和 revision 组合成安全的目录名
    safe_name = name.replace("/", "__")
    if revision:
        safe_name += f"__{revision.replace('/', '_')}"
    cache_path = CACHE_DIR / safe_name

    if cache_path.exists():
        print(f"  [缓存命中] 从本地加载: {cache_path}")
        dataset = load_from_disk(str(cache_path))
        return dataset

    print(f"  [首次下载] {name}" + (f" (revision={revision})" if revision else ""))
    load_kwargs = {**kwargs}
    if revision:
        load_kwargs["revision"] = revision
    dataset = load_dataset(name, **load_kwargs)

    # 保存到本地
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    print(f"  [已缓存] 保存到: {cache_path}")

    return dataset


def load_rotten_tomatoes():
    """加载 Rotten Tomatoes 数据集 (带缓存)"""
    return load_cached_dataset("rotten_tomatoes")


def load_conll2003():
    """
    加载 CoNLL-2003 NER 数据集 (带缓存)
    datasets >= 4.x 需要使用 Parquet 版本
    """
    try:
        return load_cached_dataset("conll2003", revision="refs/convert/parquet")
    except Exception as e1:
        print(f"  conll2003 Parquet 加载失败: {e1}")
        try:
            return load_cached_dataset("conll2003")
        except Exception as e2:
            print(f"  conll2003 加载失败: {e2}, 使用 wnut_17 替代")
            return load_cached_dataset("wnut_17", revision="refs/convert/parquet")
