#!/usr/bin/env python3
"""
FP4 vs FP8 量化模拟：分块量化的"存储 + 还原"两阶段机制

核心逻辑：
  - FP4 只能存 0~15 的等级编号（4-bit = 16 个离散值）
  - 精细浮点值通过 "等级数 * scale + offset" 还原
  - 全局量化 vs 分块量化的精度对比

零依赖：仅使用 Python 标准库
"""

import math
import random

# ─── FP4 硬件常量 ───────────────────────────────────────────
FP4_LEVELS = 16       # 2^4 = 16 个等级（0~15）
FP4_MAX_LEVEL = 15    # 最大等级编号


# ══════════════════════════════════════════════════════════════
# 第一部分：核心量化函数（存储 + 还原）
# ══════════════════════════════════════════════════════════════

def fp4_quantize_block(values):
    """
    阶段 1（存储）：将一组 FP8 真值量化为 FP4 等级数

    返回：(等级数列表, scale, offset)
    """
    offset = min(values)
    value_range = max(values) - min(values)

    if value_range == 0:
        return [0] * len(values), 0.0, offset

    scale = value_range / FP4_MAX_LEVEL

    levels = [max(0, min(FP4_MAX_LEVEL, round((v - offset) / scale))) for v in values]
    return levels, scale, offset


def fp4_dequantize_block(levels, scale, offset):
    """
    阶段 2（还原）：将 FP4 等级数还原为近似精细值

    还原公式：精细值 = offset + 等级数 * scale
    """
    return [offset + l * scale for l in levels]


# ══════════════════════════════════════════════════════════════
# 第二部分：全局量化 vs 分块量化
# ══════════════════════════════════════════════════════════════

def global_quantize(values):
    """全局量化：所有值共用一本翻译册"""
    levels, scale, offset = fp4_quantize_block(values)
    restored = fp4_dequantize_block(levels, scale, offset)
    errors = [abs(v - r) for v, r in zip(values, restored)]
    return {
        "method": "全局量化",
        "levels": levels,
        "scale": scale,
        "offset": offset,
        "restored": restored,
        "errors": errors,
    }


def blockwise_quantize(values, blocks):
    """
    分块量化：每个块独立配一本翻译册

    参数：
      values: 原始完整列表
      blocks: 分块后的子列表
    """
    all_levels = []
    all_restored = []
    all_errors = []
    block_info = []

    for i, block in enumerate(blocks):
        levels, scale, offset = fp4_quantize_block(block)
        restored = fp4_dequantize_block(levels, scale, offset)
        errors = [abs(v - r) for v, r in zip(block, restored)]

        all_levels.extend(levels)
        all_restored.extend(restored)
        all_errors.extend(errors)
        block_info.append({
            "block_id": i + 1,
            "range": f"[{min(block):.2f}, {max(block):.2f}]",
            "scale": scale,
            "offset": offset,
        })

    return {
        "method": "分块量化 (QLoRA)",
        "levels": all_levels,
        "restored": all_restored,
        "errors": all_errors,
        "blocks": block_info,
    }


# ══════════════════════════════════════════════════════════════
# 第三部分：可视化打印
# ══════════════════════════════════════════════════════════════

def print_separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_level_table(block_name, values, levels, restored, scale, offset):
    """打印某个块的完整等级映射表"""
    print(f"\n  [{block_name}] offset={offset:.4f}, scale={scale:.6f}")
    print(f"  {'FP8真值':>10}  ->  {'FP4等级':>7}  ->  {'还原公式':>30}  ->  {'还原值':>10}  {'误差':>10}")
    print(f"  {'-'*10}      {'-'*7}      {'-'*30}      {'-'*10}  {'-'*10}")

    for v, l, r in zip(values, levels, restored):
        formula = f"{offset:.2f} + {l} * {scale:.6f}"
        error = abs(v - r)
        if error < 0.001:
            mark = "OK"
        elif error < 0.01:
            mark = "~"
        else:
            mark = "!!"
        print(f"  {v:10.4f}  ->  {l:7d}  ->  {formula:>30}  ->  {r:10.4f}  {error:10.4f} {mark}")


def print_full_level_grid(offset, scale, block_name):
    """打印某个块所有 16 个等级的完整映射网格"""
    print(f"\n  [{block_name}] 完整 16 等级网格：")
    print(f"  {'等级':>4}  {'还原公式':>30}  {'还原值':>10}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*10}")
    for level in range(FP4_LEVELS):
        val = offset + level * scale
        formula = f"{offset:.2f} + {level} * {scale:.6f}"
        print(f"  {level:4d}  {formula:>30}  {val:10.4f}")


def print_comparison(original, global_result, block_result):
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

    g_total = sum(global_result["errors"])
    b_total = sum(block_result["errors"])
    ratio = g_total / max(b_total, 1e-10)
    print(f"\n  总误差：全局 = {g_total:.4f}，分块 = {b_total:.4f}，"
          f"分块精度提升 = {ratio:.1f}x")


def print_same_level_different_blocks(blocks_data):
    """展示同一等级数在不同块中还原出完全不同的值"""
    print('\n  同一个等级数 3 在不同"翻译册"下的还原结果：')
    print(f"  {'块':>6}  {'范围':>16}  {'scale':>12}  {'等级3还原公式':>35}  {'还原值':>10}")
    print(f"  {'-'*6}  {'-'*16}  {'-'*12}  {'-'*35}  {'-'*10}")

    for b in blocks_data:
        val = b["offset"] + 3 * b["scale"]
        formula = f"{b['offset']:.2f} + 3 * {b['scale']:.6f}"
        bname = f"块{b['block_id']}"
        print(f"  {bname:>6}  {b['range']:>16}  {b['scale']:12.6f}  "
              f"{formula:>35}  {val:10.4f}")


# ══════════════════════════════════════════════════════════════
# 第四部分：主实验
# ══════════════════════════════════════════════════════════════

def run_store_restore_demo():
    """逐步演示存储 + 还原两阶段（最直观的教学示例）"""
    print_separator('实验 1：逐步演示"存储 + 还原"两阶段（以 1.12 为例）')

    fp8_value = 1.12
    block_min = 1.10
    block_max = 1.20
    scale = (block_max - block_min) / FP4_MAX_LEVEL

    print(f"\n  == 阶段 1：存储（FP8 真值 -> FP4 等级数）==")
    print(f"  FP8 真值：{fp8_value}")
    print(f"  块范围：[{block_min}, {block_max}]")
    print(f"  scale = ({block_max} - {block_min}) / {FP4_MAX_LEVEL} = {scale:.6f}")

    relative_pos = fp8_value - block_min
    print(f"\n  第 1 步：块内相对位置 = {fp8_value} - {block_min} = {relative_pos:.4f}")

    raw_level = relative_pos / scale
    print(f"  第 2 步：原始等级数 = {relative_pos:.4f} / {scale:.6f} = {raw_level:.4f}")

    level = max(0, min(FP4_MAX_LEVEL, round(raw_level)))
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

    original = [1.10, 1.12, 1.15, 1.20, 9.10, 9.15, 9.20]
    print(f"\n  FP8 原始值：{original}")

    # 全局量化
    print_separator("2.1 全局量化（灾难性信息丢失）")
    g = global_quantize(original)
    print_level_table("全局（一本翻译册）", original, g["levels"], g["restored"], g["scale"], g["offset"])
    unique_levels = len(set(g["levels"]))
    print(f"\n  !! 7 个不同 FP8 值 -> 只用了 {unique_levels} 个 FP4 等级 -> 信息严重丢失")

    # 分块量化
    print_separator("2.2 分块量化（QLoRA 方式）")
    block1 = [1.10, 1.12, 1.15, 1.20]
    block2 = [9.10, 9.15, 9.20]
    b = blockwise_quantize(original, [block1, block2])

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

    normal = [1.10, 1.12, 1.15, 1.18]
    levels_n, scale_n, offset_n = fp4_quantize_block(normal)
    restored_n = fp4_dequantize_block(levels_n, scale_n, offset_n)

    with_outlier = [1.10, 1.12, 1.15, 100.0]
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
    """大规模实验：模拟真实 LLM 权重分布"""
    print_separator("实验 4：模拟真实 LLM 权重（1024 个值，不同 block size）")

    random.seed(42)

    def rand_normal(mu, sigma, n):
        """Box-Muller 变换生成正态分布"""
        result = []
        for _ in range(n // 2 + 1):
            u1 = random.random()
            u2 = random.random()
            while u1 == 0:
                u1 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
            result.append(mu + sigma * z0)
            result.append(mu + sigma * z1)
        return result[:n]

    weights = (rand_normal(0.5, 0.1, 256) +
               rand_normal(2.0, 0.3, 256) +
               rand_normal(-1.0, 0.2, 256) +
               rand_normal(5.0, 0.5, 256))

    w_min = min(weights)
    w_max = max(weights)
    w_mean = sum(weights) / len(weights)
    w_std = math.sqrt(sum((x - w_mean)**2 for x in weights) / len(weights))

    print(f"\n  权重数量：{len(weights)}")
    print(f"  权重范围：[{w_min:.4f}, {w_max:.4f}]")
    print(f"  权重均值：{w_mean:.4f}，标准差：{w_std:.4f}")

    # 全局量化
    g = global_quantize(weights)
    g_mse = sum(e**2 for e in g["errors"]) / len(g["errors"])
    g_max = max(g["errors"])

    print(f"\n  全局量化：")
    print(f"    scale = {g['scale']:.6f}")
    print(f"    使用的等级数种类 = {len(set(g['levels']))}/{FP4_LEVELS}")
    print(f"    MSE = {g_mse:.6f}，最大误差 = {g_max:.4f}")

    # 不同 block size 的分块量化
    for block_size in [16, 32, 64, 128, 256]:
        n_blocks = len(weights) // block_size
        blocks = [weights[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        all_values = []
        for bl in blocks:
            all_values.extend(bl)
        b = blockwise_quantize(all_values, blocks)
        b_mse = sum(e**2 for e in b["errors"]) / len(b["errors"])
        b_max = max(b["errors"])

        # scale 存储开销
        scale_overhead_bytes = n_blocks * (1 + 4)
        data_bytes = n_blocks * block_size * 0.5
        overhead_pct = scale_overhead_bytes / (data_bytes + scale_overhead_bytes) * 100

        ratio = g_mse / max(b_mse, 1e-10)
        print(f"\n  分块量化 (block_size={block_size:3d})：")
        print(f"    块数 = {n_blocks}，MSE = {b_mse:.6f}，最大误差 = {b_max:.4f}")
        print(f"    MSE 提升 = {ratio:.1f}x")
        print(f"    scale 存储开销 = {overhead_pct:.1f}%")


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  FP4 vs FP8 量化模拟")
    print('  分块量化的"存储 + 还原"两阶段机制')
    print("  核心：FP4 只存 0~15 等级数，精细值靠 scale/offset 还原")
    print("=" * 70)

    run_store_restore_demo()      # 先用最直观的单值示例建立认知
    run_basic_experiment()        # 文档中的 7 个值完整实验
    run_outlier_experiment()      # outlier 的破坏性
    run_large_scale_experiment()  # 真实 LLM 权重模拟

    print(f"\n{'=' * 70}")
    print("  总结：")
    print("  1. FP4 原生只存 0~15 等级编号，不认识精细浮点数")
    print("  2. 精细值 = offset + 等级数 * scale（翻译册还原）")
    print("  3. 全局量化：scale 太大 -> 等级间隔粗 -> 信息丢失严重")
    print("  4. 分块量化：scale 极小 -> 等级间隔细 -> 精度提升数十倍")
    print("  5. 块内 outlier 会拉大 scale，破坏精度 -> 需要特殊处理")
    print("  6. block size 是精度和存储开销的权衡点")
    print(f"{'=' * 70}")
