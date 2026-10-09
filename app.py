#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换系统 - 命令行演示
交互式命令行界面，支持 4 种风格转换
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import StyleTransferPipeline


def check_model_exists():
    """检查模型文件是否存在"""
    base_model = "models/Qwen2.5-1.5B-Instruct"
    lora_weights = "outputs/lora_checkpoint_v2/final_lora_weights"
    
    if not os.path.exists(base_model):
        print(f"✗ 基座模型不存在: {base_model}")
        print("请先运行 python scripts/download_models.py 下载模型")
        return False
    
    if not os.path.exists(lora_weights):
        print(f"✗ LoRA 权重不存在: {lora_weights}")
        print("请先训练模型或下载预训练权重")
        return False
    
    return True


def main():
    """主函数"""
    print("=" * 70)
    print("个性化语言风格转换系统")
    print("基于语义重构的 LoRA 微调 Qwen2.5 模型")
    print("=" * 70)
    print()
    
    if not check_model_exists():
        sys.exit(1)
    
    print("正在加载模型...")
    pipeline = StyleTransferPipeline(
        base_model_path="models/Qwen2.5-1.5B-Instruct",
        lora_weights_path="outputs/lora_checkpoint_v2/final_lora_weights",
        device="auto"
    )
    
    print("\n支持的转换风格:")
    styles = pipeline.style_generator.list_styles()
    for i, (style, desc) in enumerate(styles.items(), 1):
        print(f"  {i}. {style} - {desc}")
    print()
    
    # 交互式循环
    while True:
        try:
            # 获取输入文本
            text = input("\n请输入要转换的文本 (输入 q 退出): ").strip()
            if text.lower() in ['q', 'quit', 'exit']:
                print("再见！")
                break
            
            if not text:
                print("✗ 文本不能为空")
                continue
            
            # 获取目标风格
            style_choice = input("选择目标风格 (1-4): ").strip()
            style_map = {'1': '正式', '2': '口语化', '3': '简洁', '4': '丰富'}
            
            if style_choice not in style_map:
                print("✗ 无效选择")
                continue
            
            target_style = style_map[style_choice]
            
            # 执行转换
            print(f"\n正在转换为 {target_style} 风格...")
            result = pipeline.transfer(
                text=text,
                style=target_style,
                max_new_tokens=150
            )
            
            print("\n" + "-" * 70)
            print(f"【原文】{result['original_text']}")
            print(f"【风格】{result['target_style']}")
            print(f"【结果】{result['generated_text']}")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n✗ 错误: {e}")


if __name__ == "__main__":
    main()
