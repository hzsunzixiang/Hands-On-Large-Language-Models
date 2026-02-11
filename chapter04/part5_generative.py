"""
Chapter 4 - Part 5: 使用生成模型 (Flan-T5) 进行分类
使用 pipeline 方式，与教材对齐
[ericksun@ERICKSUN-MC1:chapter04] (d2l_3.13) (main *%)$ conda install -c conda-forge llvm-openmp
"""
import os
# 解决 macOS 上 OpenMP 库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from common import load_data, get_device, evaluate_performance


def generative_classification(data, device="cpu"):
    """
    使用生成模型 (Flan-T5) 进行分类 - 使用 pipeline 方式
    """
    print("\n" + "=" * 60)
    print("Part 5: 生成模型分类 (Flan-T5)")
    print("=" * 60)
    
    # 加载 Flan-T5 模型 pipeline
    model_name = "google/flan-t5-small"
    print(f"\n加载模型: {model_name}")
    
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        device=device
    )
    
    # 准备数据: 添加提示词
    prompt = "Is the following sentence positive or negative? "
    print(f"\n提示词模板: '{prompt}'")
    
    # 使用 data.map 添加 t5 字段
    data_with_prompt = data.map(lambda example: {"t5": prompt + example['text']})
    
    print(f"示例输入: {data_with_prompt['test'][0]['t5'][:100]}...")
    
    # 运行推理 - 使用 KeyDataset 进行批量推理
    print("\n在测试集上运行推理...")
    y_pred = []
    
    for output in tqdm(pipe(KeyDataset(data_with_prompt["test"], "t5")), total=len(data["test"])):
        text = output[0]["generated_text"]
        y_pred.append(0 if text == "negative" else 1)
    
    print("\nFlan-T5 分类结果:")
    evaluate_performance(data["test"]["label"], y_pred)
    
    return y_pred


def main():
    # 加载数据
    data = load_data()
    
    # 获取设备 (PyTorch 2.x 已支持 MPS 运行 T5)
    device = get_device()
    
    # 运行分类
    y_pred = generative_classification(data, device=device)
    
    return y_pred


if __name__ == "__main__":
    main()
