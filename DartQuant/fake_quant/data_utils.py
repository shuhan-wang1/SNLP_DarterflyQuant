import random
import os
import logging
import json

# Cache directory: read from HF_HOME (set by run_quantize.py), fallback to autodl default
MODEL_CACHE_DIR = os.environ.get('HF_HOME', '/root/autodl-tmp/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(MODEL_CACHE_DIR, 'datasets'))

from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset

# Legacy modelscope model paths (backward compat for old downloads)
MODEL_NAME_MAPPING = {
    'meta-llama/Llama-2-7b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-70b-hf',
    'meta-llama/Meta-Llama-3-8B': '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-70B': '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-70B',
    'shakechen/Llama-2-7b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-7b-hf',
    'shakechen/Llama-2-13b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-13b-hf',
    'shakechen/Llama-2-70b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-70b-hf',
}

# 本地数据集路径
LOCAL_DATASET_PATHS = {
    'wikitext2': '/root/autodl-tmp/datasets/wikitext/wikitext-2-v1/1.0.0/6280e5a53c82b20da4f99f484fa6f0ca9de738ff12f59efb0815fe7d8ae21478',
    'ptb': '/root/autodl-tmp/datasets/xeon09112___ptb_text_only/default-ce1f658bdfd12953/0.0.0/master',
}

def convert_model_name(model_name):
    """将模型名称转换为本地路径"""
    if model_name in MODEL_NAME_MAPPING:
        local_path = MODEL_NAME_MAPPING[model_name]
        if os.path.exists(local_path):
            logging.info(f'使用本地模型: {model_name} -> {local_path}')
            return local_path
    return model_name

def load_local_dataset(dataset_name, split='train'):
    """从本地加载数据集"""
    if dataset_name not in LOCAL_DATASET_PATHS:
        raise ValueError(f'未知数据集: {dataset_name}')

    base_path = LOCAL_DATASET_PATHS[dataset_name]

    if not os.path.exists(base_path):
        raise FileNotFoundError(f'数据集目录不存在: {base_path}')

    # 尝试使用 datasets 库加载
    try:
        ds = load_from_disk(base_path)
        if hasattr(ds, split):
            return ds[split]
        elif split in ds:
            return ds[split]
        else:
            return ds
    except Exception as e:
        print(f"load_from_disk 失败: {e}")

    # 列出目录中的所有文件
    files = os.listdir(base_path)
    print(f"目录 {base_path} 中的文件: {files}")

    # 使用 Dataset.from_file 直接加载 arrow 文件
    arrow_files = [f for f in files if f.endswith('.arrow') and split in f]
    if arrow_files:
        file_path = os.path.join(base_path, arrow_files[0])
        try:
            ds = Dataset.from_file(file_path)
            return ds
        except Exception as e:
            print(f"Dataset.from_file 失败: {e}")

    # 查找匹配 split 的 parquet 文件
    parquet_files = [f for f in files if f.endswith('.parquet') and split in f]
    if parquet_files:
        import pyarrow.parquet as pq
        all_data = {}
        for pq_file in sorted(parquet_files):
            file_path = os.path.join(base_path, pq_file)
            table = pq.read_table(file_path)
            data = table.to_pydict()
            if not all_data:
                all_data = data
            else:
                for key in data:
                    all_data[key].extend(data[key])
        return all_data

    # 查找 json 文件
    json_files = [f for f in files if f.endswith('.json')]
    if json_files:
        data_json = [f for f in json_files if 'dataset_info' not in f and split in f]
        if not data_json:
            data_json = [f for f in json_files if 'dataset_info' not in f]
        if data_json:
            with open(os.path.join(base_path, data_json[0]), 'r') as f:
                return json.load(f)

    raise FileNotFoundError(f'在 {base_path} 中找不到 {split} 数据集文件')


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):

    model = convert_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR, token=hf_token,
    )

    if eval_mode:
        # 使用本地数据集
        data = load_local_dataset('wikitext2', 'test')
        # 处理不同类型的返回值
        if hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            texts = [item['text'] if isinstance(item, dict) else item for item in data['text']] if 'text' in (data.column_names if hasattr(data, 'column_names') else data.keys()) else list(data)
        else:
            texts = data['text']
        testenc = tokenizer("\n\n".join(texts), return_tensors='pt')
        return testenc
    else:
        data = load_local_dataset('wikitext2', 'train')
        # 处理不同类型的返回值
        if hasattr(data, 'column_names'):
            texts = data['text']
        elif isinstance(data, dict):
            texts = data['text']
        else:
            texts = [item['text'] for item in data]
        trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    model = convert_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR, token=hf_token,
    )

    if eval_mode:
        # C4 数据集较大，如果本地没有则报错
        raise NotImplementedError("C4 数据集未下载，请使用 wikitext2 或 ptb")
    else:
        raise NotImplementedError("C4 数据集未下载，请使用 wikitext2 或 ptb")


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):

    model = convert_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR, token=hf_token,
    )

    if eval_mode:
        # 使用本地数据集
        data = load_local_dataset('ptb', 'test')
        if hasattr(data, 'column_names'):
            sentences = data['sentence']
        elif isinstance(data, dict):
            sentences = data['sentence']
        else:
            sentences = [item['sentence'] for item in data]
        testenc = tokenizer(" ".join(sentences), return_tensors='pt')
        return testenc
    else:
        data = load_local_dataset('ptb', 'train')
        if hasattr(data, 'column_names'):
            sentences = data['sentence']
        elif isinstance(data, dict):
            sentences = data['sentence']
        else:
            sentences = [item['sentence'] for item in data]
        trainenc = tokenizer(" ".join(sentences), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'ptb' in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
