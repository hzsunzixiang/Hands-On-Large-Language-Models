"""
Chapter 1 - 语言模型入门演示
本脚本演示如何使用 Hugging Face Transformers 加载和运行大语言模型
支持 CPU 和 GPU (CUDA/MPS) 运行
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ==================== 配置 ====================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# 也可以换成其他模型:
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# =============================================

def get_device():
    """检测可用设备: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main():
    print("=" * 50)
    print("Chapter 1: 语言模型入门")
    print("=" * 50)
    
    # 1. 环境信息
    device = get_device()
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"运行设备: {device}")
    print(f"模型: {MODEL_NAME}\n")
    
    # 2. 加载模型和分词器
    print("正在加载模型 (首次运行需要下载)...")
    
    # 根据设备选择数据类型
    if device == "cpu":
        dtype = torch.float32  # CPU 用 float32
        device_map = "cpu"
    else:
        dtype = torch.float16  # GPU 用 float16 节省显存
        device_map = device
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    print(f"模型加载完成!\n")
    
    # 3. 创建文本生成 Pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,  # 只返回生成的新内容
        max_new_tokens=200,      # 最大生成长度
        do_sample=False          # 贪婪解码，输出确定性结果
    )
    
    # 4. 构建对话消息
    messages = [
        {"role": "user", "content": "Create a funny joke about chickens."}
    ]
    
    print("用户输入:", messages[0]["content"])
    print("\n生成中...\n")
    
    # 5. 生成输出
    output = generator(messages)
    
    print("=" * 50)
    print("模型输出:")
    print("=" * 50)
    print(output[0]["generated_text"])

if __name__ == "__main__":
    main()
