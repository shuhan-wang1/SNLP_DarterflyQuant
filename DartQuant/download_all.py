#!/usr/bin/env python3
"""
下载脚本 - 在无卡模式下运行，只下载模型和数据集，不加载模型
"""

import os
import sys

# 设置缓存目录（必须在导入 modelscope 之前）
MODEL_CACHE_DIR = '/root/autodl-tmp'
os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE_DIR

print(f"缓存目录: {MODEL_CACHE_DIR}")
print("=" * 60)

# 导入 modelscope
from modelscope import snapshot_download

def download_model(model_id):
    """下载模型（只下载文件，不加载到内存）"""
    print(f"\n>>> 下载模型: {model_id}")
    try:
        model_dir = snapshot_download(model_id, cache_dir=MODEL_CACHE_DIR)
        print(f"✓ 模型已下载到: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None

def download_dataset_via_git(dataset_name, repo_url):
    """通过 git clone 下载数据集"""
    import subprocess
    dataset_dir = os.path.join(MODEL_CACHE_DIR, 'datasets', dataset_name)
    
    if os.path.exists(dataset_dir):
        print(f"✓ 数据集已存在: {dataset_dir}")
        return dataset_dir
    
    print(f"\n>>> 下载数据集: {dataset_name}")
    os.makedirs(os.path.dirname(dataset_dir), exist_ok=True)
    
    try:
        # 使用 git lfs clone
        cmd = f"git clone {repo_url} {dataset_dir}"
        print(f"执行: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ 数据集已下载到: {dataset_dir}")
        return dataset_dir
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None

def main():
    print("开始下载所有需要的模型和数据集...")
    print("这个脚本可以在无卡模式下运行")
    print("=" * 60)
    
    # ============================================================
    # 1. 下载模型
    # ============================================================
    models_to_download = [
        'shakechen/Llama-2-7b-hf',
        # 如果需要其他模型，在这里添加:
        # 'shakechen/Llama-2-13b-hf',
        # 'LLM-Research/Meta-Llama-3-8B',
    ]
    
    print("\n" + "=" * 60)
    print("第一步: 下载模型")
    print("=" * 60)
    
    for model_id in models_to_download:
        download_model(model_id)
    
    # ============================================================
    # 2. 下载数据集 (通过 git clone)
    # ============================================================
    print("\n" + "=" * 60)
    print("第二步: 下载数据集")
    print("=" * 60)
    
    datasets_to_download = [
        ('wikitext-2-raw-v1', 'https://www.modelscope.cn/datasets/izhx/wikitext-2-raw-v1.git'),
        ('ptb_text_only', 'https://www.modelscope.cn/datasets/izhx/ptb_text_only.git'),
    ]
    
    for dataset_name, repo_url in datasets_to_download:
        download_dataset_via_git(dataset_name, repo_url)
    
    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 60)
    print("✓ 下载完成！")
    print("=" * 60)
    print(f"\n所有文件已下载到: {MODEL_CACHE_DIR}")
    print("\n你现在可以:")
    print("1. 关闭无卡模式")
    print("2. 启动GPU实例")
    print("3. 运行实际的训练/推理脚本")

if __name__ == '__main__':
    main()
