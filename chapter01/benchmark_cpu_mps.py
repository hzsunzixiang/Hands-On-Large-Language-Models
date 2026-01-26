"""
CPU vs MPS 速度对比 - 展示 MPS 优势场景
"""
import torch
import time
import warnings
import gc
import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
warnings.filterwarnings("ignore")

def benchmark_matmul(device, size=4096, num_runs=10):
    """场景1: 大矩阵乘法 - MPS 优势最明显"""
    print(f"\n  设备: {device.upper()}")
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    for _ in range(3):
        _ = torch.mm(a, b)
    if device == "mps":
        torch.mps.synchronize()
    
    times = []
    for i in range(num_runs):
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()
        _ = torch.mm(a, b)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.time() - start)
    
    avg = sum(times) / len(times)
    gflops = (2 * size**3) / avg / 1e9
    print(f"    平均: {avg*1000:.2f}ms, {gflops:.1f} GFLOPS")
    return avg


def benchmark_batch_embedding(device, batch_size=64, seq_len=128, num_runs=10):
    """场景2: 批量文本嵌入 - MPS 优势明显"""
    from transformers import AutoModel, AutoTokenizer
    
    print(f"\n  设备: {device.upper()}")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # 构造批量输入
    texts = ["This is a sample text for embedding."] * batch_size
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 预热
    with torch.no_grad():
        _ = model(**inputs)
    if device == "mps":
        torch.mps.synchronize()
    
    times = []
    for i in range(num_runs):
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.time() - start)
    
    avg = sum(times) / len(times)
    throughput = batch_size / avg
    print(f"    平均: {avg*1000:.2f}ms, {throughput:.1f} samples/s")
    
    del model
    gc.collect()
    return avg


def benchmark_conv2d(device, batch_size=32, num_runs=10):
    """场景3: CNN 卷积运算 - MPS 优势明显"""
    print(f"\n  设备: {device.upper()}")
    
    # 模拟 ResNet 卷积层
    conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
    x = torch.randn(batch_size, 64, 224, 224, device=device)
    
    # 预热
    with torch.no_grad():
        _ = conv(x)
    if device == "mps":
        torch.mps.synchronize()
    
    times = []
    for i in range(num_runs):
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = conv(x)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.time() - start)
    
    avg = sum(times) / len(times)
    throughput = batch_size / avg
    print(f"    平均: {avg*1000:.2f}ms, {throughput:.1f} images/s")
    return avg


def benchmark_training_step(device, num_runs=10):
    """场景4: 训练步骤（前向+反向）- MPS 优势最大"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    print(f"\n  设备: {device.upper()}")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    model.train()
    
    # 批量输入
    batch_size = 16
    texts = ["This is a sample text for training."] * batch_size
    labels = torch.randint(0, 2, (batch_size,), device=device)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = labels
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 预热
    outputs = model(**inputs)
    outputs.loss.backward()
    optimizer.zero_grad()
    if device == "mps":
        torch.mps.synchronize()
    
    times = []
    for i in range(num_runs):
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()
        
        outputs = model(**inputs)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.time() - start)
    
    avg = sum(times) / len(times)
    throughput = batch_size / avg
    print(f"    平均: {avg*1000:.2f}ms, {throughput:.1f} samples/s")
    
    del model, optimizer
    gc.collect()
    return avg


if __name__ == "__main__":
    print("=" * 60)
    print("CPU vs MPS 性能对比 - MPS 优势场景")
    print("=" * 60)
    
    results = {}
    
    # 测试 1: 矩阵乘法
    print("\n" + "=" * 60)
    print("测试 1: 大矩阵乘法 (4096x4096)")
    print("=" * 60)
    cpu_matmul = benchmark_matmul("cpu")
    mps_matmul = benchmark_matmul("mps") if torch.backends.mps.is_available() else None
    results["矩阵乘法"] = (cpu_matmul, mps_matmul)
    
    # 测试 2: 批量嵌入
    print("\n" + "=" * 60)
    print("测试 2: 批量文本嵌入 (batch=64, DistilBERT)")
    print("=" * 60)
    cpu_embed = benchmark_batch_embedding("cpu")
    gc.collect()
    mps_embed = benchmark_batch_embedding("mps") if torch.backends.mps.is_available() else None
    results["批量嵌入"] = (cpu_embed, mps_embed)
    
    # 测试 3: CNN 卷积
    print("\n" + "=" * 60)
    print("测试 3: CNN 卷积运算 (batch=32, 224x224)")
    print("=" * 60)
    cpu_conv = benchmark_conv2d("cpu")
    gc.collect()
    mps_conv = benchmark_conv2d("mps") if torch.backends.mps.is_available() else None
    results["CNN卷积"] = (cpu_conv, mps_conv)
    
    # 测试 4: 训练步骤
    print("\n" + "=" * 60)
    print("测试 4: 训练步骤 (前向+反向+优化, batch=16)")
    print("=" * 60)
    cpu_train = benchmark_training_step("cpu")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    mps_train = benchmark_training_step("mps") if torch.backends.mps.is_available() else None
    results["训练步骤"] = (cpu_train, mps_train)
    
    # 汇总
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"{'场景':<12} {'CPU':<12} {'MPS':<12} {'加速比':<10}")
    print("-" * 46)
    for name, (cpu_t, mps_t) in results.items():
        if mps_t:
            speedup = cpu_t / mps_t
            print(f"{name:<12} {cpu_t*1000:>8.1f}ms   {mps_t*1000:>8.1f}ms   {speedup:>6.2f}x")
    
    print("\n结论: MPS 在批量计算和训练场景有明显优势!")
