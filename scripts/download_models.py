#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载脚本
自动从 Hugging Face 下载 Qwen2.5-1.5B-Instruct 基座模型
"""
import os
import sys
from pathlib import Path


def download_model():
    """下载 Qwen2.5-1.5B-Instruct 模型"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("正在安装 huggingface-hub...")
        os.system(f"{sys.executable} -m pip install huggingface-hub")
        from huggingface_hub import snapshot_download
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    save_path = Path("models") / "Qwen2.5-1.5B-Instruct"
    
    print(f"开始下载模型: {model_name}")
    print(f"保存路径: {save_path.absolute()}")
    print("模型大小约 3GB，请确保有足够的磁盘空间和稳定的网络连接...")
    print()
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print()
        print(f"✓ 模型下载完成！保存在: {save_path}")
        print()
        print("下一步:")
        print("  1. 下载 LoRA 权重文件")
        print("  2. 运行 python app.py 启动演示")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n备选方案:")
        print("  1. 使用 Hugging Face 镜像站 (https://hf-mirror.com)")
        print("  2. 手动下载模型后放到 models/Qwen2.5-1.5B-Instruct/ 目录")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
