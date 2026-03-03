#!/usr/bin/env python3
"""
FP4 vs FP8 量化模拟：分块量化的"存储 + 还原"两阶段机制（NumPy 版）

核心逻辑：
  - FP4 只能存 0~15 的等级编号（4-bit = 16 个离散值）
  - 精细浮点值通过 "等级数 * scale + offset" 还原
  - 全局量化 vs 分块量化的精度对比

依赖：numpy
运行：conda activate d2l_3.13 && python fp4_fp8_quantization_simulation_numpy.py
"""

import numpy as np

# ─── FP4 硬件常量 ───────────────────────────────────────────
FP4_LEVELS = 16       # 2^4 = 16 个等级（0~15）
FP4_MAX_LEVEL = 15    # 最大等级编号


# ══════════════════════════════════════════════════════════════
# 第一部分：核心量化函数（存储 + 还原）—— 向量化实现
# ══════════════════════════════════════════════════════════════

def fp4_quantize_block(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    阶段 1（存储）：将一组 FP8 真值量化为 FP4 等级数

    返回：(等级数数组 int, scale, offset)
    """
    values = np.asarray(values, dtype=np.float64)
    offset = values.min()
    value_range = values.max() - offset

    if value_range == 0:
        return np.zeros(len(values), dtype=np.int32), 0.0, float(offset)

    scale = value_range / FP4_MAX_LEVEL
    levels = np.clip(np.round((values - offset) / scale), 0, FP4_MAX_LEVEL).astype(np.int32)
    return levels, float(scale), float(offset)


def fp4_dequantize_block(levels: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """
    阶段 2（还原）：将 FP4 等级数还原为近似精细值

    还原公式：精细值 = offset + 等级数 * scale
    """
    return offset + np.asarray(levels, dtype=np.float64) * scale


def quantize_errors(original: np.ndarray, restored: np.ndarray) -> np.ndarray:
    """计算逐元素绝对误差"""
    return np.abs(original - restored)


# ══════════════════════════════════════════════════════════════
# 第二部分：全局量化 vs 分块量化 —— 批量向量化
# ══════════════════════════════════════════════════════════════

def global_quantize(values: np.ndarray) -> dict:
    """全局量化：所有值共用一本翻译册"""
    values = np.asarray(values, dtype=np.float64)
    levels, scale, offset = fp4_quantize_block(values)
    restored = fp4_dequantize_block(levels, scale, offset)
    return {
        "method": "全局量化",
        "levels": levels,
        "scale": scale,
        "offset": offset,
        "restored": restored,
        "errors": quantize_errors(values, restored),
    }


def blockwise_quantize(values: np.ndarray, block_size: int) -> dict:
    """
    分块量化：按 block_size 自动切块，每块独立量化

    返回的 levels/restored/errors 与原始 values 等长（尾部不足一块的部分单独处理）
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    n_full_blocks = n // block_size

    all_levels = np.empty(n, dtype=np.int32)
    all_restored = np.empty(n, dtype=np.float64)
    block_info = []

    for i in range(n_full_blocks):
        s, e = i * block_size, (i + 1) * block_size
        blk = values[s:e]
        levels, scale, offset = fp4_quantize_block(blk)
        all_levels[s:e] = levels
        all_restored[s:e] = fp4_dequantize_block(levels, scale, offset)
        block_info.append({"block_id": i + 1, "range": f"[{blk.min():.4f}, {blk.max():.4f}]",
                           "scale": scale, "offset": offset})

    # 尾部不足一块
    tail = n - n_full_blocks * block_size
    if tail > 0:
        s = n_full_blocks * block_size
        blk = values[s:]
        levels, scale, offset = fp4_quantize_block(blk)
        all_levels[s:] = levels
        all_restored[s:] = fp4_dequantize_block(levels, scale, offset)
        block_info.append({"block_id": n_full_blocks + 1,
                           "range": f"[{blk.min():.4f}, {blk.max():.4f}]",
                           "scale": scale, "offset": offset})

    return {
        "method": f"分块量化 (block_size={block_size})",
        "levels": all_levels,
        "restored": all_restored,
        "errors": quantize_errors(values, all_restored),
        "blocks": block_info,
        "block_size": block_size,
    }


def blockwise_quantize_manual(values: np.ndarray, blocks: list[np.ndarray]) -> dict:
    """手动指定分块的量化（用于文档示例中不等长的块）"""
    values = np.asarray(values, dtype=np.float64)
    all_levels = []
    all_restored = []
    block_info = []

    for i, blk in enumerate(blocks):
        blk = np.asarray(blk, dtype=np.float64)
        levels, scale, offset = fp4_quantize_block(blk)
        restored = fp4_dequantize_block(levels, scale, offset)
        all_levels.append(levels)
        all_restored.append(restored)
        block_info.append({"block_id": i + 1, "range": f"[{blk.min():.2f}, {blk.max():.2f}]",
                           "scale": scale, "offset": offset})

    all_levels = np.concatenate(all_levels)
    all_restored = np.concatenate(all_restored)
    return {
        "method": "分块量化 (QLoRA)",
        "levels": all_levels,
        "restored": all_restored,
        "errors": quantize_errors(values, all_restored),
        "blocks": block_info,
    }


# ══════════════════════════════════════════════════════════════
# 第三部分：可视化打印
# ══════════════════════════════════════════════════════════════

def print_separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_level_table(block_name: str, values: np.ndarray, levels: np.ndarray,
                      restored: np.ndarray, scale: float, offset: float):
    """打印某个块的完整等级映射表"""
    print(f"\n  [{block_name}] offset={offset:.4f}, scale={scale:.6f}")
    print(f"  {'FP8真值':>10}  ->  {'FP4等级':>7}  ->  {'还原公式':>30}  ->  {'还原值':>10}  {'误差':>10}")
    print(f"  {'-'*10}      {'-'*7}      {'-'*30}      {'-'*10}  {'-'*10}")

    for v, l, r in zip(values, levels, restored):
        formula = f"{offset:.2f} + {l} * {scale:.6f}"
        error = abs(v - r)
        mark = "OK" if error < 0.001 else ("~" if error < 0.01 else "!!")
        print(f"  {v:10.4f}  ->  {l:7d}  ->  {formula:>30}  ->  {r:10.4f}  {error:10.4f} {mark}")


def print_full_level_grid(offset: float, scale: float, block_name: str):
    """打印某个块所有 16 个等级的完整映射网格"""
    levels = np.arange(FP4_LEVELS)
    vals = offset + levels * scale

    print(f"\n  [{block_name}] 完整 16 等级网格：")
    print(f"  {'等级':>4}  {'还原公式':>30}  {'还原值':>10}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*10}")
    for level, val in zip(levels, vals):
        formula = f"{offset:.2f} + {level} * {scale:.6f}"
        print(f"  {level:4d}  {formula:>30}  {val:10.4f}")


def print_comparison(original: np.ndarray, global_result: dict, block_result: dict):
    """打印全局 vs 分块对比表"""
    print(f"\n  {'FP8原值':>10}  {'全局等级':>8}  {'全局还原':>10}  {'全局误差':>10}  "
          f"{'分块等级':>8}  {'分块还原':>10}  {'分块误差':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    for i in range(len(original)):
        v = original[i]
        gl, gr, ge = global_result["levels"][i], global_result["restored"][i], global_result["errors"][i]
        bl, br, be = block_result["levels"][i], block_result["restored"][i], block_result["errors"][i]
        print(f"  {v:10.4f}  {gl:8d}  {gr:10.4f}  {ge:10.4f}  "
              f"{bl:8d}  {br:10.4f}  {be:10.4f}")

    g_total = global_result["errors"].sum()
    b_total = block_result["errors"].sum()
    print(f"\n  总误差：全局 = {g_total:.4f}，分块 = {b_total:.4f}，"
          f"分块精度提升 = {g_total / max(b_total, 1e-10):.1f}x")


def print_same_level_different_blocks(blocks_data: list):
    """展示同一等级数在不同块中还原出完全不同的值"""
    print('\n  同一个等级数 3 在不同"翻译册"下的还原结果：')
    print(f"  {'块':>6}  {'范围':>16}  {'scale':>12}  {'等级3还原公式':>35}  {'还原值':>10}")
    print(f"  {'-'*6}  {'-'*16}  {'-'*12}  {'-'*35}  {'-'*10}")

    for b in blocks_data:
        val = b["offset"] + 3 * b["scale"]
        formula = f"{b['offset']:.2f} + 3 * {b['scale']:.6f}"
        print(f"  {'块'+str(b['block_id']):>6}  {b['range']:>16}  {b['scale']:12.6f}  "
              f"{formula:>35}  {val:10.4f}")


# ══════════════════════════════════════════════════════════════
# 第四部分：主实验
# ══════════════════════════════════════════════════════════════

def run_store_restore_demo():
    """逐步演示存储 + 还原两阶段"""
    print_separator('实验 1：逐步演示"存储 + 还原"两阶段（以 1.12 为例）')

    fp8_value = 1.12
    block_min, block_max = 1.10, 1.20
    scale = (block_max - block_min) / FP4_MAX_LEVEL

    print(f"\n  == 阶段 1：存储（FP8 真值 -> FP4 等级数）==")
    print(f"  FP8 真值：{fp8_value}")
    print(f"  块范围：[{block_min}, {block_max}]")
    print(f"  scale = ({block_max} - {block_min}) / {FP4_MAX_LEVEL} = {scale:.6f}")

    relative_pos = fp8_value - block_min
    print(f"\n  第 1 步：块内相对位置 = {fp8_value} - {block_min} = {relative_pos:.4f}")

    raw_level = relative_pos / scale
    print(f"  第 2 步：原始等级数 = {relative_pos:.4f} / {scale:.6f} = {raw_level:.4f}")

    level = int(np.clip(np.round(raw_level), 0, FP4_MAX_LEVEL))
    print(f"  第 3 步：取整并裁剪 -> FP4 存储等级数 = {level}")
    print(f"\n  [存储] FP4 芯片上实际存储的值：{level}（不是 {fp8_value}！）")

    print(f"\n  == 阶段 2：还原（FP4 等级数 -> 近似精细值）==")
    print(f"  读取 FP4 等级数：{level}")

    restored = block_min + level * scale
    print(f"  还原公式：{block_min} + {level} * {scale:.6f} = {restored:.4f}")

    error = abs(fp8_value - restored)
    print(f"\n  [对比]")
    print(f"     FP8 原始值：{fp8_value}")
    print(f"     FP4 还原值：{restored:.4f}")
    print(f"     量化误差：  {error:.4f}")
    print(f"     相对误差：  {error/fp8_value*100:.2f}%")


def run_basic_experiment():
    """基础实验：文档中的 7 个值"""
    print_separator("实验 2：基础场景（文档示例 7 个值）")

    original = np.array([1.10, 1.12, 1.15, 1.20, 9.10, 9.15, 9.20])
    print(f"\n  FP8 原始值：{original}")

    # 全局量化
    print_separator("2.1 全局量化（灾难性信息丢失）")
    g = global_quantize(original)
    print_level_table("全局（一本翻译册）", original, g["levels"], g["restored"], g["scale"], g["offset"])
    unique_levels = len(np.unique(g["levels"]))
    print(f"\n  !! 7 个不同 FP8 值 -> 只用了 {unique_levels} 个 FP4 等级 -> 信息严重丢失")

    # 分块量化（手动分块，与文档一致）
    print_separator("2.2 分块量化（QLoRA 方式）")
    block1 = np.array([1.10, 1.12, 1.15, 1.20])
    block2 = np.array([9.10, 9.15, 9.20])
    b = blockwise_quantize_manual(original, [block1, block2])

    levels1, scale1, offset1 = fp4_quantize_block(block1)
    levels2, scale2, offset2 = fp4_quantize_block(block2)
    restored1 = fp4_dequantize_block(levels1, scale1, offset1)
    restored2 = fp4_dequantize_block(levels2, scale2, offset2)

    print_level_table("块1", block1, levels1, restored1, scale1, offset1)
    print_level_table("块2", block2, levels2, restored2, scale2, offset2)

    # 完整 16 等级网格
    print_separator("2.3 完整 16 等级网格（块1 的翻译册）")
    print_full_level_grid(offset1, scale1, "块1: 1.10~1.20")

    # 对比
    print_separator("2.4 全局 vs 分块对比")
    print_comparison(original, g, b)

    # 同一等级不同翻译册
    print_separator("2.5 核心思想：同一等级数，不同翻译册")
    print_same_level_different_blocks(b["blocks"])


def run_outlier_experiment():
    """异常值实验：展示块内 outlier 对精度的影响"""
    print_separator("实验 3：异常值（Outlier）对分块量化的影响")

    normal = np.array([1.10, 1.12, 1.15, 1.18])
    levels_n, scale_n, offset_n = fp4_quantize_block(normal)
    restored_n = fp4_dequantize_block(levels_n, scale_n, offset_n)

    with_outlier = np.array([1.10, 1.12, 1.15, 100.0])
    levels_o, scale_o, offset_o = fp4_quantize_block(with_outlier)
    restored_o = fp4_dequantize_block(levels_o, scale_o, offset_o)

    print(f"\n  场景对比：同样的前 3 个值，最后一个值不同")
    print(f"  正常块：   {normal}")
    print(f"  异常块：   {with_outlier}")

    print_level_table("正常块", normal, levels_n, restored_n, scale_n, offset_n)
    print_level_table("异常块（含 outlier）", with_outlier, levels_o, restored_o, scale_o, offset_o)

    print(f"\n  !! 异常值 100.0 拉大了 scale（{scale_n:.6f} -> {scale_o:.6f}），")
    print(f"     导致 1.10/1.12/1.15 全部被映射到等级 0，无法区分")
    print(f"     退化为全局量化的困境 -> 这就是 LLM.int8() 要对 outlier 做 mixed-precision 处理的原因")


def run_large_scale_experiment():
    """大规模实验：模拟真实 LLM 权重分布（NumPy 向量化）"""
    print_separator("实验 4：模拟真实 LLM 权重（1024 个值，不同 block size）")

    rng = np.random.default_rng(42)

    # 模拟真实权重：多个高斯分布混合
    weights = np.concatenate([
        rng.normal(0.5, 0.1, 256),
        rng.normal(2.0, 0.3, 256),
        rng.normal(-1.0, 0.2, 256),
        rng.normal(5.0, 0.5, 256),
    ])

    print(f"\n  权重数量：{len(weights)}")
    print(f"  权重范围：[{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  权重均值：{weights.mean():.4f}，标准差：{weights.std():.4f}")

    # 全局量化
    g = global_quantize(weights)
    g_mse = np.mean(g["errors"] ** 2)
    g_max = g["errors"].max()

    print(f"\n  全局量化：")
    print(f"    scale = {g['scale']:.6f}")
    print(f"    使用的等级数种类 = {len(np.unique(g['levels']))}/{FP4_LEVELS}")
    print(f"    MSE = {g_mse:.6f}，最大误差 = {g_max:.4f}")

    # 不同 block size 的分块量化
    for block_size in [16, 32, 64, 128, 256]:
        b = blockwise_quantize(weights, block_size)
        b_mse = np.mean(b["errors"] ** 2)
        b_max = b["errors"].max()

        n_blocks = len(b["blocks"])
        # scale 存储开销：每块一个 FP8 scale (1 byte) + 一个 FP32 offset (4 bytes)
        scale_overhead_bytes = n_blocks * (1 + 4)
        data_bytes = len(weights) * 0.5  # 4-bit = 0.5 byte per value
        overhead_pct = scale_overhead_bytes / (data_bytes + scale_overhead_bytes) * 100

        print(f"\n  分块量化 (block_size={block_size:3d})：")
        print(f"    块数 = {n_blocks}，MSE = {b_mse:.6f}，最大误差 = {b_max:.4f}")
        print(f"    MSE 提升 = {g_mse / max(b_mse, 1e-10):.1f}x")
        print(f"    scale 存储开销 = {overhead_pct:.1f}%")


def run_vectorized_demo():
    """NumPy 独有实验：向量化批量量化 + 误差分布直方图（文本版）"""
    print_separator("实验 5（NumPy 独有）：批量向量化量化 + 误差分布")

    rng = np.random.default_rng(123)
    weights = rng.normal(0, 1, 4096)

    print(f"\n  权重数量：{len(weights)}，分布：N(0, 1)")

    for block_size in [16, 64, 256]:
        b = blockwise_quantize(weights, block_size)
        errors = b["errors"]

        # 文本直方图
        bins = np.linspace(0, errors.max(), 11)
        counts, _ = np.histogram(errors, bins=bins)
        max_count = counts.max()

        print(f"\n  block_size={block_size}  MSE={np.mean(errors**2):.6f}  "
              f"max_err={errors.max():.4f}  mean_err={errors.mean():.6f}")
        print(f"  误差分布直方图（{len(weights)} 个值）：")

        bar_width = 40
        for i in range(len(counts)):
            lo, hi = bins[i], bins[i+1]
            bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
            bar = '#' * bar_len
            print(f"    [{lo:7.4f}, {hi:7.4f}) {counts[i]:5d} |{bar}")

    # 向量化性能展示：一次性量化大数组
    print(f"\n  -- 向量化性能展示 --")
    big = rng.normal(0, 2, 100_000)
    import time
    t0 = time.perf_counter()
    result = blockwise_quantize(big, 64)
    t1 = time.perf_counter()
    mse = np.mean(result["errors"] ** 2)
    print(f"  100,000 个权重，block_size=64：")
    print(f"    耗时 = {(t1 - t0)*1000:.1f} ms")
    print(f"    MSE  = {mse:.6f}")
    print(f"    块数 = {len(result['blocks'])}")


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  FP4 vs FP8 量化模拟（NumPy 版）")
    print('  分块量化的"存储 + 还原"两阶段机制')
    print(f"  核心：FP4 只存 0~15 等级数，精细值靠 scale/offset 还原")
    print(f"  NumPy {np.__version__}")
    print("=" * 70)

    run_store_restore_demo()      # 单值两阶段拆解
    run_basic_experiment()        # 文档中的 7 个值
    run_outlier_experiment()      # outlier 的破坏性
    run_large_scale_experiment()  # 1024 权重 + 不同 block size
    run_vectorized_demo()         # NumPy 独有：批量 + 误差直方图 + 性能

    print(f"\n{'=' * 70}")
    print("  总结：")
    print("  1. FP4 原生只存 0~15 等级编号，不认识精细浮点数")
    print("  2. 精细值 = offset + 等级数 * scale（翻译册还原）")
    print("  3. 全局量化：scale 太大 -> 等级间隔粗 -> 信息丢失严重")
    print("  4. 分块量化：scale 极小 -> 等级间隔细 -> 精度提升数十倍")
    print("  5. 块内 outlier 会拉大 scale，破坏精度 -> 需要特殊处理")
    print("  6. block size 是精度和存储开销的权衡点")
    print("  7. NumPy 向量化可高效处理 10 万级权重的批量量化")
    print(f"{'=' * 70}")
