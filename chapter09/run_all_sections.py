"""
第9章 - 多模态大语言模型 完整学习流程
=====================================

这是第9章的主入口文件，可以按顺序运行所有章节，
或者选择性地运行特定章节。

使用方法:
python run_all_sections.py [section_number]

示例:
python run_all_sections.py        # 运行所有章节
python run_all_sections.py 1      # 只运行 9.1
python run_all_sections.py 1-3    # 运行 9.1 到 9.3
"""

import sys
import subprocess
import time
from pathlib import Path


def print_banner():
    """打印横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           第9章 - 多模态大语言模型 完整学习流程                    ║
║                                                                  ║
║  Chapter 9 - Multimodal Large Language Models                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def get_section_info():
    """获取章节信息"""
    sections = {
        1: {
            "file": "9.1_clip_basics.py",
            "title": "CLIP 基础 - 图文嵌入对齐",
            "description": "学习 CLIP 的核心原理和基础使用",
            "duration": "~5分钟",
            "difficulty": "⭐⭐☆☆☆"
        },
        2: {
            "file": "9.2_clip_similarity_matrix.py", 
            "title": "CLIP 相似度矩阵分析",
            "description": "深入理解多模态相似度计算和应用",
            "duration": "~8分钟",
            "difficulty": "⭐⭐⭐☆☆"
        },
        3: {
            "file": "9.3_sbert_clip.py",
            "title": "SBERT-CLIP 简化接口",
            "description": "掌握统一的多模态编程接口",
            "duration": "~6分钟", 
            "difficulty": "⭐⭐☆☆☆"
        },
        4: {
            "file": "9.4_blip2_vision_qa.py",
            "title": "BLIP-2 视觉问答系统",
            "description": "体验先进的视觉语言模型",
            "duration": "~15分钟",
            "difficulty": "⭐⭐⭐⭐☆"
        },
        5: {
            "file": "9.5_lightweight_vlm.py",
            "title": "轻量级视觉语言模型",
            "description": "学习资源友好的部署方案",
            "duration": "~10分钟",
            "difficulty": "⭐⭐⭐☆☆"
        },
        6: {
            "file": "9.6_multimodal_summary.py",
            "title": "多模态总结",
            "description": "整合知识，展望未来发展",
            "duration": "~5分钟",
            "difficulty": "⭐⭐☆☆☆"
        }
    }
    return sections


def show_menu():
    """显示菜单"""
    sections = get_section_info()
    
    print("\n📚 章节目录:")
    print("=" * 70)
    
    for num, info in sections.items():
        title = info["title"]
        desc = info["description"]
        duration = info["duration"]
        difficulty = info["difficulty"]
        
        print(f"9.{num} {title}")
        print(f"    📝 {desc}")
        print(f"    ⏱️  {duration} | 🎯 {difficulty}")
        print()
    
    print("🎮 运行选项:")
    print("  python run_all_sections.py        # 运行所有章节")
    print("  python run_all_sections.py 1      # 运行第1节")
    print("  python run_all_sections.py 1-3    # 运行第1-3节")
    print("  python run_all_sections.py menu   # 显示此菜单")


def run_section(section_num):
    """运行指定章节"""
    sections = get_section_info()
    
    if section_num not in sections:
        print(f"❌ 章节 {section_num} 不存在")
        return False
    
    section = sections[section_num]
    file_path = Path(__file__).parent / section["file"]
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {section['file']}")
        return False
    
    print(f"\n🚀 开始运行 9.{section_num}: {section['title']}")
    print("=" * 60)
    print(f"📝 {section['description']}")
    print(f"⏱️  预计时间: {section['duration']}")
    print(f"🎯 难度: {section['difficulty']}")
    print("=" * 60)
    
    try:
        # 运行 Python 文件
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=False,
            text=True,
            cwd=file_path.parent
        )
        
        if result.returncode == 0:
            print(f"\n✅ 9.{section_num} 运行完成")
            return True
        else:
            print(f"\n❌ 9.{section_num} 运行失败 (退出码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        return False


def run_range(start, end):
    """运行指定范围的章节"""
    sections = get_section_info()
    success_count = 0
    total_count = end - start + 1
    
    print(f"\n🎯 准备运行章节 9.{start} 到 9.{end} (共 {total_count} 个)")
    
    # 询问是否继续
    try:
        response = input("\n是否继续？(y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("👋 用户取消")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n👋 用户中断")
        return
    
    start_time = time.time()
    
    for section_num in range(start, end + 1):
        if section_num in sections:
            print(f"\n{'='*20} 第 {section_num}/{end} 节 {'='*20}")
            
            if run_section(section_num):
                success_count += 1
            
            # 章节间暂停
            if section_num < end:
                print(f"\n⏸️  章节间休息 3 秒...")
                time.sleep(3)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 运行总结")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{total_count} 个章节")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"📈 成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print(f"\n🎉 恭喜！所有章节都运行成功！")
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个章节运行失败")


def check_dependencies():
    """检查依赖"""
    print("\n🔍 检查依赖...")
    
    required_packages = [
        "torch",
        "transformers", 
        "sentence-transformers",
        "pillow",
        "matplotlib",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包:")
        for package in missing_packages:
            print(f"    pip install {package}")
        
        print(f"\n或者一次性安装:")
        print(f"    pip install {' '.join(missing_packages)}")
        return False
    
    print(f"\n✅ 所有依赖都已安装")
    return True


def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 请先安装缺少的依赖包")
        return
    
    # 解析命令行参数
    if len(sys.argv) == 1:
        # 无参数，运行所有章节
        run_range(1, 6)
    
    elif len(sys.argv) == 2:
        arg = sys.argv[1].lower()
        
        if arg in ['menu', 'help', '-h', '--help']:
            # 显示菜单
            show_menu()
        
        elif '-' in arg:
            # 范围运行，如 "1-3"
            try:
                start, end = map(int, arg.split('-'))
                if 1 <= start <= end <= 6:
                    run_range(start, end)
                else:
                    print("❌ 章节范围必须在 1-6 之间")
            except ValueError:
                print("❌ 无效的范围格式，请使用 '1-3' 格式")
        
        else:
            # 单个章节
            try:
                section_num = int(arg)
                if 1 <= section_num <= 6:
                    run_section(section_num)
                else:
                    print("❌ 章节号必须在 1-6 之间")
            except ValueError:
                print("❌ 无效的章节号")
                show_menu()
    
    else:
        print("❌ 参数过多")
        show_menu()


if __name__ == "__main__":
    main()