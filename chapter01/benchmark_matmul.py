"""
CPU vs MPS 矩阵运算速度对比
这是最直接的 GPU 加速测试
"""
import torch
import time

def benchmark_matmul(device, size=4096, num_runs=10):
    print(f"\n{'='*40}")
    print(f"设备: {device.upper()}")
    print(f"{'='*40}")
    
    # 创建随机矩阵
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    for _ in range(3):
        _ = torch.mm(a, b)
    
    if device == "mps":
        torch.mps.synchronize()
    
    # 计时
    times = []
    for i in range(num_runs):
        if device == "mps":
            torch.mps.synchronize()
        
        start = time.time()
        c = torch.mm(a, b)
        
        if device == "mps":
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        # 计算 GFLOPS
        gflops = (2 * size**3) / elapsed / 1e9
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms, {gflops:.1f} GFLOPS")
    
    avg = sum(times) / len(times)
    avg_gflops = (2 * size**3) / avg / 1e9
    print(f"\n  平均: {avg*1000:.2f}ms, {avg_gflops:.1f} GFLOPS")
    return avg

if __name__ == "__main__":
    size = 4096
    print("=" * 40)
    print(f"CPU vs MPS 矩阵乘法对比")
    print(f"矩阵大小: {size} x {size}")
    print("=" * 40)
    
    cpu_time = benchmark_matmul("cpu", size)
    
    if torch.backends.mps.is_available():
        mps_time = benchmark_matmul("mps", size)
        
        print(f"\n{'='*40}")
        print("结果汇总")
        print(f"{'='*40}")
        print(f"  CPU: {cpu_time*1000:.2f}ms")
        print(f"  MPS: {mps_time*1000:.2f}ms")
        print(f"\n  MPS 比 CPU 快 {cpu_time/mps_time:.1f}x")
