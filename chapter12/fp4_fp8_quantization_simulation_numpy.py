#!/usr/bin/env python3
"""
FP4 vs FP8 量化模拟：分块量化的"存储 + 还原"两阶段机制（NumPy 版）

核心：FP4 只存 0~15 等级编号，精细值靠 scale/offset 还原
运行：conda activate d2l_3.13 && python fp4_fp8_quantization_simulation_numpy.py
"""

import time
import numpy as np

FP4_MAX = 15  # FP4 最大等级编号，共 16 个等级（0~15）


# ── 核心：量化 & 反量化 ──────────────────────────────────────

def quantize(values):
    """存储：FP8 真值 -> FP4 等级数，返回 (levels, scale, offset)"""
    v = np.asarray(values, dtype=np.float64)
    offset, vrange = v.min(), v.max() - v.min()
    if vrange == 0:
        return np.zeros(len(v), dtype=np.int32), 0.0, float(offset)
    scale = vrange / FP4_MAX
    levels = np.clip(np.round((v - offset) / scale), 0, FP4_MAX).astype(np.int32)
    return levels, float(scale), float(offset)


def dequantize(levels, scale, offset):
    """还原：FP4 等级数 -> 近似精细值"""
    return offset + np.asarray(levels, dtype=np.float64) * scale


# ── 全局 & 分块量化 ─────────────────────────────────────────

def global_quantize(values):
    """全局量化：所有值共用一个 scale/offset"""
    v = np.asarray(values, dtype=np.float64)
    levels, scale, offset = quantize(v)
    restored = dequantize(levels, scale, offset)
    return {"levels": levels, "scale": scale, "offset": offset,
            "restored": restored, "errors": np.abs(v - restored)}


def blockwise_quantize(values, block_size_or_blocks):
    """
    分块量化。

    block_size_or_blocks:
      - int: 按固定大小自动切块
      - list[array]: 手动指定每个块
    """
    v = np.asarray(values, dtype=np.float64)

    if isinstance(block_size_or_blocks, int):
        bs = block_size_or_blocks
        blocks = [v[i:i+bs] for i in range(0, len(v), bs)]
    else:
        blocks = [np.asarray(b, dtype=np.float64) for b in block_size_or_blocks]

    all_levels, all_restored, block_info = [], [], []
    for i, blk in enumerate(blocks):
        lv, sc, off = quantize(blk)
        all_levels.append(lv)
        all_restored.append(dequantize(lv, sc, off))
        block_info.append({"block_id": i+1, "scale": sc, "offset": off,
                           "range": f"[{blk.min():.4f}, {blk.max():.4f}]"})

    restored = np.concatenate(all_restored)
    return {"levels": np.concatenate(all_levels), "restored": restored,
            "errors": np.abs(v - restored), "blocks": block_info}


# ── 打印工具 ────────────────────────────────────────────────

def sep(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def print_table(name, values, levels, restored, scale, offset):
    """打印块的等级映射表"""
    print(f"\n  [{name}] offset={offset:.4f}, scale={scale:.6f}")
    print(f"  {'FP8真值':>10}  ->  {'等级':>5}  ->  {'还原公式':>28}  ->  {'还原值':>8}  {'误差':>8}")
    print(f"  {'-'*10}     {'-'*5}     {'-'*28}     {'-'*8}  {'-'*8}")
    for v, l, r in zip(values, levels, restored):
        e = abs(v - r)
        mark = "OK" if e < 0.001 else ("~" if e < 0.01 else "!!")
        print(f"  {v:10.4f}  ->  {l:5d}  ->  {offset:.2f} + {l} * {scale:.6f}  ->  {r:8.4f}  {e:8.4f} {mark}")


# ── 实验 ────────────────────────────────────────────────────

def run_store_restore_demo():
    """实验 1：逐步拆解 1.12 的存储 + 还原"""
    sep('实验 1：逐步演示"存储 + 还原"（以 1.12 为例）')

    val, lo, hi = 1.12, 1.10, 1.20
    scale = (hi - lo) / FP4_MAX
    rel = val - lo
    raw = rel / scale
    level = int(np.clip(np.round(raw), 0, FP4_MAX))
    restored = lo + level * scale
    err = abs(val - restored)

    print(f"""
  == 阶段 1：存储 ==
  FP8 真值：{val}，块范围：[{lo}, {hi}]，scale = {scale:.6f}
  块内相对位置 = {rel:.4f}，原始等级 = {raw:.4f}，取整 -> 等级 {level}
  [存储] FP4 芯片存的是 {level}，不是 {val}！

  == 阶段 2：还原 ==
  {lo} + {level} * {scale:.6f} = {restored:.4f}
  误差 = {err:.4f}，相对误差 = {err/val*100:.2f}%""")


def run_basic_experiment():
    """实验 2：文档示例 7 个值"""
    sep("实验 2：基础场景（文档示例 7 个值）")

    orig = np.array([1.10, 1.12, 1.15, 1.20, 9.10, 9.15, 9.20])
    print(f"\n  FP8 原始值：{orig}")

    # 全局
    sep("2.1 全局量化")
    g = global_quantize(orig)
    print_table("全局", orig, g["levels"], g["restored"], g["scale"], g["offset"])
    print(f"\n  !! 7 值 -> 只用了 {len(np.unique(g['levels']))} 个等级 -> 信息严重丢失")

    # 分块
    sep("2.2 分块量化（QLoRA）")
    blk1, blk2 = np.array([1.10, 1.12, 1.15, 1.20]), np.array([9.10, 9.15, 9.20])
    b = blockwise_quantize(orig, [blk1, blk2])

    for blk, name in [(blk1, "块1"), (blk2, "块2")]:
        lv, sc, off = quantize(blk)
        print_table(name, blk, lv, dequantize(lv, sc, off), sc, off)

    # 16 等级网格
    sep("2.3 块1 完整 16 等级网格")
    _, sc1, off1 = quantize(blk1)
    levels = np.arange(16)
    vals = off1 + levels * sc1
    print(f"\n  [{' 等级':>4}]  {'还原值':>10}")
    for l, v in zip(levels, vals):
        print(f"   {l:4d}     {v:10.4f}")

    # 对比
    sep("2.4 全局 vs 分块对比")
    print(f"\n  {'FP8原值':>10}  {'全局等级':>6} {'全局还原':>8} {'全局误差':>8}  "
          f"{'分块等级':>6} {'分块还原':>8} {'分块误差':>8}")
    print(f"  {'-'*10}  {'-'*6} {'-'*8} {'-'*8}  {'-'*6} {'-'*8} {'-'*8}")
    for i in range(len(orig)):
        print(f"  {orig[i]:10.4f}  {g['levels'][i]:6d} {g['restored'][i]:8.4f} {g['errors'][i]:8.4f}  "
              f"{b['levels'][i]:6d} {b['restored'][i]:8.4f} {b['errors'][i]:8.4f}")
    print(f"\n  总误差：全局={g['errors'].sum():.4f}  分块={b['errors'].sum():.4f}  "
          f"提升={g['errors'].sum()/max(b['errors'].sum(),1e-10):.1f}x")

    # 同一等级不同翻译册
    sep("2.5 同一等级数 3，不同翻译册")
    for bi in b["blocks"]:
        v3 = bi["offset"] + 3 * bi["scale"]
        print(f"  块{bi['block_id']}  {bi['range']:>20}  scale={bi['scale']:.6f}  "
              f"等级3 -> {bi['offset']:.2f} + 3*{bi['scale']:.6f} = {v3:.4f}")


def run_outlier_experiment():
    """实验 3：Outlier 的破坏性"""
    sep("实验 3：Outlier 对分块量化的影响")

    normal = np.array([1.10, 1.12, 1.15, 1.18])
    outlier = np.array([1.10, 1.12, 1.15, 100.0])

    print(f"\n  正常块：{normal}\n  异常块：{outlier}")

    for arr, name in [(normal, "正常块"), (outlier, "异常块")]:
        lv, sc, off = quantize(arr)
        print_table(name, arr, lv, dequantize(lv, sc, off), sc, off)

    sc_n, sc_o = quantize(normal)[1], quantize(outlier)[1]
    print(f"\n  !! scale 从 {sc_n:.6f} 暴涨到 {sc_o:.6f}，1.10/1.12/1.15 全塌缩到等级 0")


def run_large_scale_experiment():
    """实验 4：1024 权重 + 不同 block size"""
    sep("实验 4：模拟 LLM 权重（1024 值，不同 block size）")

    rng = np.random.default_rng(42)
    w = np.concatenate([rng.normal(mu, sig, 256) for mu, sig in
                        [(0.5, 0.1), (2.0, 0.3), (-1.0, 0.2), (5.0, 0.5)]])

    print(f"\n  {len(w)} 个权重，范围 [{w.min():.4f}, {w.max():.4f}]，"
          f"均值 {w.mean():.4f}，std {w.std():.4f}")

    g = global_quantize(w)
    g_mse = np.mean(g["errors"]**2)
    print(f"\n  全局：scale={g['scale']:.6f}  等级种类={len(np.unique(g['levels']))}/16  "
          f"MSE={g_mse:.6f}  max_err={g['errors'].max():.4f}")

    for bs in [16, 32, 64, 128, 256]:
        b = blockwise_quantize(w, bs)
        b_mse = np.mean(b["errors"]**2)
        nb = len(b["blocks"])
        overhead = nb * 5 / (len(w) * 0.5 + nb * 5) * 100
        print(f"  block={bs:3d}：块数={nb:3d}  MSE={b_mse:.6f}  "
              f"提升={g_mse/max(b_mse,1e-10):.1f}x  开销={overhead:.1f}%")


def run_vectorized_demo():
    """实验 5：批量量化 + 误差直方图 + 性能"""
    sep("实验 5：批量向量化 + 误差分布")

    rng = np.random.default_rng(123)
    w = rng.normal(0, 1, 4096)
    print(f"\n  {len(w)} 个权重，N(0,1)")

    for bs in [16, 64, 256]:
        errs = blockwise_quantize(w, bs)["errors"]
        bins = np.linspace(0, errs.max(), 11)
        counts, _ = np.histogram(errs, bins)
        mx = counts.max()

        print(f"\n  block={bs}  MSE={np.mean(errs**2):.6f}  max={errs.max():.4f}  mean={errs.mean():.6f}")
        for i in range(len(counts)):
            bar = '#' * int(counts[i] / mx * 40) if mx else ''
            print(f"    [{bins[i]:7.4f},{bins[i+1]:7.4f}) {counts[i]:5d} |{bar}")

    # 性能
    big = rng.normal(0, 2, 100_000)
    t0 = time.perf_counter()
    r = blockwise_quantize(big, 64)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  100K 权重 block=64：{dt:.1f}ms  MSE={np.mean(r['errors']**2):.6f}  块数={len(r['blocks'])}")


# ── 主入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"  FP4 vs FP8 量化模拟（NumPy {np.__version__}）")
    print(f"  FP4 只存 0~15 等级数，精细值靠 scale/offset 还原")
    print(f"{'='*70}")

    run_store_restore_demo()
    run_basic_experiment()
    run_outlier_experiment()
    run_large_scale_experiment()
    run_vectorized_demo()

    print(f"\n{'='*70}")
    print("  总结：")
    for i, s in enumerate([
        "FP4 原生只存 0~15 等级编号", "精细值 = offset + 等级数 * scale",
        "全局量化 scale 太大 -> 信息丢失严重", "分块量化 scale 极小 -> 精度提升数十倍",
        "块内 outlier 拉大 scale -> 需特殊处理", "block size 是精度与存储开销的权衡",
        "NumPy 向量化可高效处理 10 万级权重",
    ], 1):
        print(f"  {i}. {s}")
    print(f"{'='*70}")
