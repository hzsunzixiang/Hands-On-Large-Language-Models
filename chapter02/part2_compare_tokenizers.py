"""
Part 2: 比较不同模型的 Tokenizer
展示不同 LLM 的分词方式差异
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

print("=" * 60)
print("Part 2: 比较不同模型的 Tokenizer")
print("=" * 60)

# ANSI 颜色代码用于可视化
colors_list = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence, tokenizer_name):
    """可视化展示分词结果"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        token_ids = tokenizer(sentence).input_ids
        print(f"\n{tokenizer_name} ({len(token_ids)} tokens):")
        tokens = []
        for idx, t in enumerate(token_ids):
            color = colors_list[idx % len(colors_list)]
            token = tokenizer.decode(t)
            tokens.append(f'\x1b[0;30;48;2;{color}m{token}\x1b[0m')
        print(' '.join(tokens))
        
        # 也打印纯文本版本
        print("  Tokens:", [tokenizer.decode(t) for t in token_ids])
    except Exception as e:
        print(f"\n{tokenizer_name}: 加载失败 - {e}")

# 测试文本（包含各种特殊情况）
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else:
12.0*50=600
"""

print(f"\n测试文本: {text}")
print("-" * 60)

# 比较不同的 tokenizer
tokenizers_to_compare = [
    "bert-base-uncased",      # BERT (小写)
    "bert-base-cased",        # BERT (保留大小写)
    "gpt2",                   # GPT-2
    "google/flan-t5-small",   # T5
]

for tokenizer_name in tokenizers_to_compare:
    show_tokens(text, tokenizer_name)

print("\n" + "=" * 60)
print("观察要点:")
print("=" * 60)
print("1. bert-base-uncased 会将所有字母转为小写")
print("2. 不同模型对 emoji、中文、代码的处理方式不同")
print("3. 数学表达式的分词方式差异很大")
print("4. Token 数量差异反映了词表设计的不同")

print("\n" + "=" * 60)
print("Part 2 完成!")
print("=" * 60)
