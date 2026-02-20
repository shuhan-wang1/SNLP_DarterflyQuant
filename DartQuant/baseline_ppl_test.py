#!/usr/bin/env python3
"""
基准 PPL 测试脚本 - 验证 FP16 原始模型的 Perplexity

用于验证 PPL 评估方法是否正确。
标准 Llama-2-7B 在 Wikitext-2 上的 PPL 应约为 5.47

运行方式:
    python baseline_ppl_test.py
"""

import os
import sys
import torch
import torch.nn as nn
import logging
import argparse
from tqdm import tqdm

# 设置环境变量
MODEL_CACHE_DIR = '/root/autodl-tmp'
os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(MODEL_CACHE_DIR, 'datasets')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--seqlen', type=int, default=2048, help='序列长度')
    parser.add_argument('--stride', type=int, default=512, help='滑动窗口步长 (标准做法)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    return parser.parse_args()


def load_model_and_tokenizer(model_name, device, dtype):
    """加载模型和 tokenizer"""
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    
    # 本地路径映射
    model_mapping = {
        'meta-llama/Llama-2-7b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-7b-hf',
        'shakechen/Llama-2-7b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-7b-hf',
    }
    
    model_path = model_mapping.get(model_name, model_name)
    
    # 检查本地路径
    if not os.path.exists(model_path):
        # 尝试 modelscope 格式
        alt_path = os.path.join(MODEL_CACHE_DIR, 'hub', 'shakechen', 'Llama-2-7b-hf')
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            logging.info(f"使用 Modelscope 下载: {model_name}")
            model_path = 'shakechen/Llama-2-7b-hf'
    
    logging.info(f"加载模型: {model_path}")
    
    # 确定 dtype
    if dtype == 'float16':
        torch_dtype = torch.float16
    elif dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map='auto',
        trust_remote_code=True
    )
    
    model.eval()
    
    return model, tokenizer


def load_wikitext2_test():
    """加载 Wikitext-2 测试集"""
    from datasets import load_from_disk, load_dataset
    
    # 可能的本地路径列表
    local_paths = [
        '/root/autodl-tmp/datasets/wikitext-2-raw-v1',
        '/root/autodl-tmp/datasets/wikitext',
        '/root/autodl-tmp/datasets/wikitext/wikitext-2-v1',
        os.path.join(MODEL_CACHE_DIR, 'datasets', 'wikitext-2-raw-v1'),
        os.path.join(MODEL_CACHE_DIR, 'datasets', 'wikitext'),
    ]
    
    # 尝试从本地加载
    for local_path in local_paths:
        if os.path.exists(local_path):
            try:
                logging.info(f"尝试从本地加载: {local_path}")
                data = load_from_disk(local_path)
                if hasattr(data, 'keys') and 'test' in data.keys():
                    logging.info(f"成功加载本地数据集: {local_path}")
                    return data['test']
                elif hasattr(data, '__getitem__'):
                    logging.info(f"成功加载本地数据集: {local_path}")
                    return data
            except Exception as e:
                logging.debug(f"加载 {local_path} 失败: {e}")
                continue
    
    # 从 HuggingFace 下载
    logging.info("从 HuggingFace 下载 wikitext-2-raw-v1...")
    try:
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
        # 保存到本地以供后续使用
        save_path = '/root/autodl-tmp/datasets/wikitext-2-raw-v1'
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data.save_to_disk(save_path)
            logging.info(f"数据集已保存到: {save_path}")
        except Exception as e:
            logging.warning(f"保存数据集失败: {e}")
        
        return data
    except Exception as e:
        logging.error(f"下载数据集失败: {e}")
        raise


@torch.no_grad()
def evaluate_ppl_sliding_window(model, tokenizer, test_data, seqlen=2048, stride=512, device='cuda'):
    """
    使用滑动窗口方法计算 PPL（标准学术做法）
    
    原理：
    - 使用完整的测试文本
    - 滑动窗口保证每个 token 都有足够的上下文
    - 只计算窗口末尾新增 token 的 loss（避免重复计算）
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        test_data: 测试数据
        seqlen: 上下文窗口长度
        stride: 滑动步长
        device: 设备
    
    Returns:
        ppl: perplexity
    """
    model.eval()
    
    # 提取文本
    if hasattr(test_data, 'column_names'):
        texts = test_data['text']
    elif isinstance(test_data, dict):
        texts = test_data.get('text', [])
    else:
        texts = [item['text'] for item in test_data]
    
    # 拼接所有文本（过滤空行）
    text = '\n\n'.join([t for t in texts if t.strip()])
    
    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    
    seq_len = input_ids.size(1)
    logging.info(f"总 token 数: {seq_len}, 窗口大小: {seqlen}, 步长: {stride}")
    
    nlls = []
    prev_end_loc = 0
    total_tokens = 0
    
    pbar = tqdm(range(0, seq_len, stride), desc="计算 PPL")
    for begin_loc in pbar:
        end_loc = min(begin_loc + seqlen, seq_len)
        trg_len = end_loc - prev_end_loc  # 本次需要评估的新 token 数
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        
        # 只计算新增部分的 loss（前面的设为 -100 忽略）
        target_chunk[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            # outputs.loss 是 平均 loss（仅计算非 -100 的 token）
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        total_tokens += trg_len
        prev_end_loc = end_loc
        
        # 动态显示 PPL
        if len(nlls) > 0:
            current_ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
            pbar.set_description(f"PPL: {current_ppl.item():.2f}")
        
        if end_loc == seq_len:
            break
    
    # 计算最终 PPL
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_tokens)
    
    return ppl.item(), total_tokens


@torch.no_grad()
def evaluate_ppl_simple(model, tokenizer, test_data, seqlen=2048, device='cuda'):
    """
    简单的 PPL 评估（不使用滑动窗口）
    
    注意：这种方法会略微高估 PPL，因为每个片段的开头 token 没有上下文
    """
    model.eval()
    
    # 提取文本
    if hasattr(test_data, 'column_names'):
        texts = test_data['text']
    elif isinstance(test_data, dict):
        texts = test_data.get('text', [])
    else:
        texts = [item['text'] for item in test_data]
    
    text = '\n\n'.join([t for t in texts if t.strip()])
    
    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    
    # 按 seqlen 分割
    nsamples = input_ids.size(1) // seqlen
    logging.info(f"总 token 数: {input_ids.size(1)}, 样本数: {nsamples}, 序列长度: {seqlen}")
    
    nlls = []
    for i in tqdm(range(nsamples), desc="PPL 评估 (简单版)"):
        start = i * seqlen
        end = start + seqlen
        batch = input_ids[:, start:end]
        
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            nlls.append(outputs.loss)
    
    ppl = torch.exp(torch.stack(nlls).mean())
    
    return ppl.item()


@torch.no_grad()
def evaluate_ppl_manual(model, tokenizer, test_data, seqlen=2048, device='cuda'):
    """
    手动计算 PPL（不依赖 model(labels=...) 参数）
    用于调试或不支持 labels 参数的模型
    """
    model.eval()
    
    # 提取文本
    if hasattr(test_data, 'column_names'):
        texts = test_data['text']
    elif isinstance(test_data, dict):
        texts = test_data.get('text', [])
    else:
        texts = [item['text'] for item in test_data]
    
    text = '\n\n'.join([t for t in texts if t.strip()])
    
    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    
    # 按 seqlen 分割
    nsamples = input_ids.size(1) // seqlen
    logging.info(f"手动计算: 总 token 数: {input_ids.size(1)}, 样本数: {nsamples}")
    
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    nlls = []
    
    for i in tqdm(range(nsamples), desc="PPL 评估 (手动)"):
        start = i * seqlen
        end = start + seqlen
        batch = input_ids[:, start:end]
        
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Shift logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        nlls.append(loss)
    
    ppl = torch.exp(torch.stack(nlls).mean())
    
    return ppl.item()


def main():
    args = parse_args()
    
    logging.info("=" * 60)
    logging.info("基准 PPL 测试 - 验证评估方法正确性")
    logging.info("=" * 60)
    logging.info(f"模型: {args.model}")
    logging.info(f"数据类型: {args.dtype}")
    logging.info(f"序列长度: {args.seqlen}")
    logging.info(f"滑动步长: {args.stride}")
    logging.info("=" * 60)
    logging.info("")
    logging.info("期望值参考 (Llama-2-7B, Wikitext-2):")
    logging.info("  - FP16/BF16: ~5.47")
    logging.info("  - W4A16 (GPTQ): ~5.60-5.80")
    logging.info("  - W4A16 (QuaRot): ~5.55-5.70")
    logging.info("=" * 60)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)
    
    # 加载测试数据
    test_data = load_wikitext2_test()
    
    logging.info("\n" + "=" * 60)
    logging.info("方法 1: 滑动窗口 PPL (标准学术做法)")
    logging.info("=" * 60)
    ppl_sliding, tokens = evaluate_ppl_sliding_window(
        model, tokenizer, test_data, 
        seqlen=args.seqlen, 
        stride=args.stride,
        device=args.device
    )
    logging.info(f"★ 滑动窗口 PPL: {ppl_sliding:.4f} (评估 {tokens} 个 token)")
    
    logging.info("\n" + "=" * 60)
    logging.info("方法 2: 简单分割 PPL (使用 labels 参数)")
    logging.info("=" * 60)
    ppl_simple = evaluate_ppl_simple(
        model, tokenizer, test_data,
        seqlen=args.seqlen,
        device=args.device
    )
    logging.info(f"★ 简单分割 PPL: {ppl_simple:.4f}")
    
    logging.info("\n" + "=" * 60)
    logging.info("方法 3: 手动计算 PPL")
    logging.info("=" * 60)
    ppl_manual = evaluate_ppl_manual(
        model, tokenizer, test_data,
        seqlen=args.seqlen,
        device=args.device
    )
    logging.info(f"★ 手动计算 PPL: {ppl_manual:.4f}")
    
    # 汇总结果
    logging.info("\n" + "=" * 60)
    logging.info("结果汇总:")
    logging.info("=" * 60)
    logging.info(f"模型: {args.model}")
    logging.info(f"数据类型: {args.dtype}")
    logging.info(f"序列长度: {args.seqlen}")
    logging.info("")
    logging.info(f"滑动窗口 PPL (stride={args.stride}): {ppl_sliding:.4f}")
    logging.info(f"简单分割 PPL:                       {ppl_simple:.4f}")
    logging.info(f"手动计算 PPL:                       {ppl_manual:.4f}")
    logging.info("")
    
    if ppl_sliding < 7.0:
        logging.info("✓ PPL 在正常范围内 (< 7.0)")
    elif ppl_sliding < 10.0:
        logging.info("⚠ PPL 略高，可能存在轻微配置问题")
    else:
        logging.info("✗ PPL 过高！请检查模型加载和评估配置")
        logging.info("  可能原因:")
        logging.info("  1. tokenizer 配置问题 (BOS/EOS token)")
        logging.info("  2. 数据集版本问题")
        logging.info("  3. 模型权重问题")
    
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
