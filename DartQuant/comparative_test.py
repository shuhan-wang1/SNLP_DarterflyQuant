#!/usr/bin/env python3
"""
对比测试脚本：使用 SWD (Sliced Wasserstein Distance) Loss 替代 Whip Loss

与 quick_test.py 相比，唯一的区别是：
- R1 训练使用 SWD Loss 而非 Whip Loss
- R2 训练使用 SWD Loss 而非 Whip Loss

SWD Loss 的目标是将激活值分布匹配到均匀分布，而非简单地推离零点。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
from tqdm import tqdm
import gc
import functools
import transformers
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置环境变量（必须在使用前定义）
MODEL_CACHE_DIR = '/root/autodl-tmp'
os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(MODEL_CACHE_DIR, 'datasets')
os.environ['HF_DATASETS_OFFLINE'] = '1'

# ============================================================================
# 正确的 PPL 评估函数（使用滑动窗口）
# ============================================================================

@torch.no_grad()
def evaluate_ppl_simple(model, tokenizer, dataset_name='wikitext2', device='cuda'):
    """
    简化版 PPL 评估 - 适用于量化后的模型
    """
    from datasets import load_from_disk
    
    model.eval()
    seqlen = model.seqlen if hasattr(model, 'seqlen') else 2048
    
    # 加载测试数据
    test_data = None
    local_paths = [
        '/root/autodl-tmp/datasets/wikitext-2-raw-v1',
        '/root/autodl-tmp/datasets/wikitext',
        os.path.join(MODEL_CACHE_DIR, 'datasets', 'wikitext-2-raw-v1'),
    ]
    for local_path in local_paths:
        if os.path.exists(local_path):
            try:
                data = load_from_disk(local_path)
                if hasattr(data, 'keys') and 'test' in data.keys():
                    test_data = data['test']
                else:
                    test_data = data
                break
            except:
                continue
    if test_data is None:
        from datasets import load_dataset
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    texts = test_data['text'] if hasattr(test_data, 'column_names') else test_data.get('text', list(test_data))
    text = '\n\n'.join([t for t in texts if t.strip()])
    
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    
    # 按 seqlen 分割
    nsamples = input_ids.size(1) // seqlen
    logging.info(f"评估 {dataset_name}: {nsamples} 个样本, 序列长度 {seqlen}")
    
    nlls = []
    for i in tqdm(range(nsamples), desc="PPL 评估"):
        start = i * seqlen
        end = start + seqlen
        batch = input_ids[:, start:end].to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        nlls.append(loss)
    
    ppl = torch.exp(torch.stack(nlls).mean())
    logging.info(f"{dataset_name.upper()} PPL: {ppl.item():.4f}")
    
    return ppl.item()

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fake_quant'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'calibrater'))

from modelscope import AutoTokenizer
import data_utils
import model_utils
import rotation_utils
import quant_utils
import gptq_utils
import eval_utils
import hadamard_utils
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--nsamples', type=int, default=128, help='校准样本数')
    parser.add_argument('--seqlen', type=int, default=2048, help='序列长度')
    parser.add_argument('--seed', type=int, default=0)
    
    # 权重量化参数
    parser.add_argument('--w_bits', type=int, default=4, help='权重量化位数')
    parser.add_argument('--w_groupsize', type=int, default=-1, help='权重量化分组大小')
    parser.add_argument('--w_asym', action='store_true', default=False, help='非对称权重量化')
    parser.add_argument('--w_clip', action='store_true', default=False, help='权重裁剪')
    parser.add_argument('--w_rtn', action='store_true', default=False, help='使用RTN而非GPTQ')
    parser.add_argument('--percdamp', type=float, default=0.01, help='GPTQ dampening')
    parser.add_argument('--act_order', action='store_true', default=False, help='GPTQ act_order')
    parser.add_argument('--w_static_groups', action='store_true', default=False)
    
    # 激活量化参数
    parser.add_argument('--a_bits', type=int, default=16, help='激活量化位数')
    parser.add_argument('--a_groupsize', type=int, default=-1)
    parser.add_argument('--a_asym', action='store_true', default=False)
    parser.add_argument('--a_clip_ratio', type=float, default=1.0)
    
    # KV-cache 量化参数
    parser.add_argument('--k_bits', type=int, default=16)
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action='store_true', default=False)
    parser.add_argument('--k_clip_ratio', type=float, default=1.0)
    parser.add_argument('--v_bits', type=int, default=16)
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action='store_true', default=False)
    parser.add_argument('--v_clip_ratio', type=float, default=1.0)
    
    # 旋转参数
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--use_r1', action='store_true', default=True, help='使用R1旋转')
    parser.add_argument('--use_r2', type=str, default='offline', choices=['offline', 'online', 'none'])
    parser.add_argument('--use_r3', action='store_true', default=True, help='使用R3旋转(K-cache)')
    parser.add_argument('--use_r4', action='store_true', default=True, help='使用R4旋转(down_proj)')
    parser.add_argument('--fp32_had', action='store_true', default=False)
    
    # R1 训练参数
    parser.add_argument('--r1_epochs', type=int, default=10, help='R1训练epochs')
    parser.add_argument('--r1_lr', type=float, default=1e-3, help='R1学习率')
    parser.add_argument('--r1_bsz', type=int, default=64, help='R1训练batch size')
    parser.add_argument('--r1_train_subset', type=float, default=0.1, help='R1每epoch使用的数据比例')
    parser.add_argument('--r1_optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--r1_cos_lr', action='store_true', default=False)
    
    # R2 训练参数
    parser.add_argument('--r2_epochs', type=int, default=5, help='R2训练epochs')
    parser.add_argument('--r2_lr', type=float, default=1e-3, help='R2学习率')
    parser.add_argument('--r2_bsz', type=int, default=128, help='R2训练batch size')
    parser.add_argument('--r2_optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--r2_cos_lr', action='store_true', default=False)
    parser.add_argument('--r2_accumulation_steps', type=int, default=2)
    
    # 评估参数
    parser.add_argument('--eval_dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb'])
    parser.add_argument('--cal_dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--ppl_eval_batch_size', type=int, default=1)
    
    # 其他
    parser.add_argument('--train_r1', action='store_true', default=True, help='是否训练R1')
    parser.add_argument('--train_r2', action='store_true', default=True, help='是否训练R2')
    
    args = parser.parse_args()
    
    # 设置一些默认值
    args.r1_path = None  # 不从文件加载
    args.r2_path = None  # 不从文件加载
    args.smooth = None
    args.hf_token = None
    args.w_bits_down_proj = None
    args.a_bits_down_proj = None
    args.a_residual = False
    args.o_per_head = False
    
    return args


# ============================================================================
# SWD Loss（替代 Whip Loss）
# ============================================================================

def calc_swd_loss(outputs, device='cuda'):
    """
    SWD (Sliced Wasserstein Distance) Loss - 替代 Whip Loss
    
    将激活值分布匹配到均匀分布 Uniform[-b, b]，其中 b 由能量守恒约束确定。
    
    原理：
    - 旋转矩阵 R 不能改变 L2 范数（能量）
    - 均匀分布 Uniform[-b, b] 的期望能量 E[x^2] = b^2 / 3
    - 设置 b = sqrt(3) * RMS(x) 以匹配输入的能量
    
    Args:
        outputs: 旋转后的激活值 [batch, hidden_dim] 或 [batch, seqlen, hidden_dim]
        device: 设备
    
    Returns:
        loss: SWD loss 标量
    """
    # 1. 展平所有激活值
    x_flat = outputs.view(-1)
    N = x_flat.numel()
    
    # 2. 排序当前激活值（输入的分位数函数）
    x_sorted, _ = torch.sort(x_flat)
    
    # 3. 动态目标生成（能量守恒约束）
    with torch.no_grad():
        rms = torch.sqrt(torch.mean(x_flat ** 2))
        b = math.sqrt(3) * rms
        
        # 生成理想的均匀分布分位数
        target = torch.linspace(-b, b, steps=N, device=x_flat.device)
    
    # 4. 计算排序后输入与目标之间的 MSE
    loss = F.mse_loss(x_sorted, target)
    
    return loss


def calc_whip_loss(outputs):
    """
    原始 DartQuant Whip Loss（用于对比）
    L = sum( exp(-|x|) )
    促使值远离零点
    """
    return torch.sum(torch.exp(-outputs.abs()), dim=-1, keepdim=True).mean()


# ============================================================================
# R1 旋转矩阵训练（使用 SWD Loss）
# ============================================================================

class R1_QR(nn.Module):
    """R1 旋转矩阵（可训练）"""
    def __init__(self, hidden_size: int):
        super(R1_QR, self).__init__()
        self.hidden_size = hidden_size
        self.matrix = nn.Parameter(torch.eye(hidden_size))

    def forward(self, x):
        self.rotate, _ = torch.linalg.qr(self.matrix, mode='complete')
        o_x = torch.matmul(x, self.rotate)
        return o_x


def train_R1(r1_train_data, args, device='cuda'):
    """
    训练 R1 旋转矩阵 - 使用 SWD Loss
    
    Args:
        r1_train_data: 用于训练R1的激活值数据 [nsamples, seqlen, hidden_size]
        args: 训练参数
        device: 设备
    
    Returns:
        R1 旋转矩阵 [hidden_size, hidden_size]
    """
    logging.info("---> 开始训练 R1 旋转矩阵 (使用 SWD Loss)")
    
    hidden_size = r1_train_data.shape[-1]
    
    # 初始化 R1
    R1 = R1_QR(hidden_size=hidden_size).to(device)
    R1.matrix.data = rotation_utils.get_orthogonal_matrix(
        hidden_size, args.rotate_mode, device).float()
    
    # 优化器
    if args.r1_optim == 'sgd':
        optimizer = torch.optim.SGD(R1.parameters(), lr=args.r1_lr, momentum=0.9)
    elif args.r1_optim == 'adam':
        optimizer = torch.optim.Adam(R1.parameters(), lr=args.r1_lr)
    else:
        raise NotImplementedError
    
    # 学习率调度器
    scheduler = None
    if args.r1_cos_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.r1_epochs, eta_min=0)
    
    # 准备数据：将数据展平为 [total_tokens, hidden_size]
    flat_data = r1_train_data.reshape(-1, hidden_size)
    dataset = TensorDataset(flat_data)
    
    # 每个epoch训练的样本比例
    num_samples = int(len(dataset) * args.r1_train_subset)
    
    R1.train()
    for epoch in range(args.r1_epochs):
        loss_log = []
        
        # 随机采样
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        subset_data = flat_data[indices]
        
        # 创建数据加载器
        subset_dataset = TensorDataset(subset_data)
        dataloader = DataLoader(subset_dataset, batch_size=args.r1_bsz, shuffle=True)
        
        for batch_idx, (batch_samples,) in enumerate(dataloader):
            batch_samples = batch_samples.to(device).float()
            outputs = R1(batch_samples)
            
            # ★★★ 使用 SWD Loss 替代 Whip Loss ★★★
            loss = calc_swd_loss(outputs, device)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_log.append(loss.detach().cpu())
        
        if scheduler is not None:
            scheduler.step()
        
        # 打印日志
        mean_loss = torch.stack(loss_log).mean()
        log_message = f'R1 Epoch [{epoch+1}/{args.r1_epochs}], SWD Loss: {mean_loss.item():.6f}'
        if scheduler is not None:
            log_message += f', LR: {scheduler.get_last_lr()[0]:.6f}'
        logging.info(log_message)
    
    logging.info("---> R1 训练完成 (SWD Loss)")
    
    # 返回训练好的旋转矩阵
    return R1.rotate.data.detach()


# ============================================================================
# R2 旋转矩阵训练（使用 SWD Loss）
# ============================================================================

class R2_Per_Head(nn.Module):
    """R2 Per-Head 旋转矩阵（可训练）"""
    def __init__(self, hidden_size: int, head_num: int, kv_head: int):
        super(R2_Per_Head, self).__init__()
        assert hidden_size % head_num == 0, "hidden_size must be divisible by head_num"
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_dim = hidden_size // head_num
        self.kv_head = kv_head
        
        self.matrix = nn.Parameter(torch.eye(self.head_dim).repeat(self.kv_head, 1, 1))

    def forward(self, x):
        x_shape = x.shape  # [batch_size, seqlen, hidden_size]
        x = x.reshape(-1, x_shape[-1])
        x = x.reshape(-1, self.head_num, self.head_dim)  # 分头分块
        x = x.transpose(0, 1)
        self.rotate, _ = torch.linalg.qr(self.matrix)
        rotate_exp = self.rotate[:, None, :, :].expand(
            self.kv_head, self.head_num // self.kv_head,
            self.head_dim, self.head_dim)
        rotate_exp = rotate_exp.reshape(self.head_num, self.head_dim, self.head_dim)
        r_x = torch.matmul(x, rotate_exp)
        r_x = r_x.transpose(0, 1)
        r_x = r_x.reshape(x_shape)
        return r_x


def get_multi_head_init(hidden_size, head_num, kv_head, mode, device):
    """获取多头初始化矩阵"""
    org = rotation_utils.get_orthogonal_matrix(hidden_size // head_num, mode, device)
    return org.unsqueeze(0).repeat(kv_head, 1, 1)


def train_R2_single_layer(o_proj_data, layer_id, args, device='cuda'):
    """
    训练单层的 R2 旋转矩阵 - 使用 SWD Loss
    
    Args:
        o_proj_data: o_proj 的输入激活值 [nsamples, seqlen, hidden_size]
        layer_id: 层ID
        args: 训练参数
        device: 设备
    
    Returns:
        R2 旋转矩阵 [kv_head, head_dim, head_dim]
    """
    R2 = R2_Per_Head(
        hidden_size=args.hidden_size,
        head_num=args.head_num,
        kv_head=args.kv_head
    ).to(device)
    
    R2.matrix.data = get_multi_head_init(
        args.hidden_size, args.head_num, args.kv_head, 'hadamard', device
    ).float()
    
    # 优化器
    if args.r2_optim == 'sgd':
        optimizer = torch.optim.SGD(R2.parameters(), lr=args.r2_lr, momentum=0.9)
    elif args.r2_optim == 'adam':
        optimizer = torch.optim.Adam(R2.parameters(), lr=args.r2_lr)
    else:
        raise NotImplementedError
    
    # 学习率调度器
    scheduler = None
    if args.r2_cos_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.r2_epochs, eta_min=0)
    
    # 准备数据
    # o_proj_data: [nsamples, seqlen, hidden_size]
    # Shuffle
    o_proj_data = o_proj_data[torch.randperm(o_proj_data.size(0)), :, :]
    o_proj_data = o_proj_data[:, torch.randperm(o_proj_data.size(1)), :]
    
    dataset = TensorDataset(o_proj_data)
    dataloader = DataLoader(dataset, batch_size=args.r2_bsz, shuffle=True)
    
    R2.train()
    for epoch in range(args.r2_epochs):
        loss_log = []
        
        for batch_idx, (batch_samples,) in enumerate(dataloader):
            batch_samples = batch_samples.to(device).float()
            outputs = R2(batch_samples)
            
            # ★★★ 使用 SWD Loss 替代 Whip Loss ★★★
            loss = calc_swd_loss(outputs, device) / args.r2_accumulation_steps
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % args.r2_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            loss_log.append(loss.detach().cpu())
        
        if scheduler is not None:
            scheduler.step()
        
        # 打印日志
        mean_loss = torch.stack(loss_log).mean()
        log_message = f'  R2 Layer {layer_id} Epoch [{epoch+1}/{args.r2_epochs}], SWD Loss: {mean_loss.item():.6f}'
        if scheduler is not None:
            log_message += f', LR: {scheduler.get_last_lr()[0]:.6f}'
        logging.info(log_message)
    
    return R2.rotate.data.detach()


def train_all_R2(o_proj_data_dict, args, device='cuda'):
    """
    训练所有层的 R2 旋转矩阵
    
    Args:
        o_proj_data_dict: {layer_id: o_proj_data} 字典
        args: 训练参数
        device: 设备
    
    Returns:
        R2 字典 {f"model.layers.{layer_id}.self_attn.R2": R2_matrix}
    """
    logging.info("---> 开始训练 R2 旋转矩阵 (使用 SWD Loss)")
    
    r2_dict = {}
    for layer_id in sorted(o_proj_data_dict.keys()):
        logging.info(f"---> 训练 R2 layer {layer_id}")
        o_proj_data = o_proj_data_dict[layer_id]
        r2 = train_R2_single_layer(o_proj_data, layer_id, args, device)
        r2_dict[f"model.layers.{layer_id}.self_attn.R2"] = r2
        
        # 清理内存
        del o_proj_data
        torch.cuda.empty_cache()
    
    logging.info("---> R2 训练完成 (SWD Loss)")
    return r2_dict


# ============================================================================
# 收集激活值（与 quick_test.py 相同）
# ============================================================================

def collect_training_data(model, dataloader, args, device='cuda'):
    """
    收集用于训练 R1 和 R2 的激活值
    """
    logging.info("---> 收集训练数据（用于 R1 和 R2）")
    
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    config = model.config
    dtype = next(model.parameters()).dtype
    
    # 将 embedding 和第一层移到 GPU
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    
    layers = model.model.layers
    layers[0] = layers[0].to(device)
    
    # 收集第一层输入
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"i": 0}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(device))
            except ValueError:
                pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    
    torch.cuda.empty_cache()
    
    # R1 训练数据就是第一层的输入
    r1_train_data = inps.clone()
    
    # 准备收集 R2 训练数据
    fp_inps = inps
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    
    if attention_mask is not None:
        attention_mask = attention_mask.repeat(1, 1, 1, 1).to(dtype)
    
    o_proj_data_dict = {}
    
    # 逐层收集 o_proj 输入
    for idx, decoder_layer in tqdm(enumerate(layers), total=len(layers), desc="收集各层激活值"):
        decoder_layer = decoder_layer.to(device)
        
        o_proj_inp = {'idx': 0}
        
        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            if 'o_proj' in name:
                o_proj_inp[o_proj_inp['idx']] = x.detach().cpu()
                o_proj_inp['idx'] += 1
        
        # 插入钩子
        hooks = []
        for name, m in decoder_layer.named_modules():
            if isinstance(m, nn.Linear) and 'o_proj' in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)))
        
        # 前向传播
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_inps[j] = decoder_layer(
                    fp_inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]
        
        # 移除钩子
        for h in hooks:
            h.remove()
        
        # 收集 o_proj 数据
        if o_proj_inp['idx'] == args.nsamples:
            del o_proj_inp['idx']
            o_proj_data = [o_proj_inp[i] for i in range(args.nsamples)]
            o_proj_data = torch.cat(o_proj_data, dim=0)
            o_proj_data_dict[idx] = o_proj_data
        
        decoder_layer = decoder_layer.cpu()
        torch.cuda.empty_cache()
    
    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()
    
    model.config.use_cache = use_cache
    
    logging.info(f"---> 收集完成: R1 数据 shape={r1_train_data.shape}, R2 层数={len(o_proj_data_dict)}")
    
    return r1_train_data, o_proj_data_dict


# ============================================================================
# 应用旋转矩阵（从内存中）
# ============================================================================

def apply_rotation_from_memory(model, r1_matrix, r2_dict, args, device='cuda'):
    """
    从内存中应用旋转矩阵到模型
    """
    logging.info("应用旋转矩阵...")
    
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    kv_head = config.num_key_value_heads
    
    model_type = model_utils.model_type_extractor(model)
    
    # 获取 R1 矩阵
    if r1_matrix is not None:
        Q = r1_matrix.to(dtype=torch.float64)
    else:
        Q = rotation_utils.get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    
    # 应用 R1 旋转
    if args.use_r1:
        rotation_utils.rotate_embeddings(model, Q)
        rotation_utils.rotate_head(model, Q)
        utils.cleanup_memory()
        
        layers = model_utils.get_transformer_layers(model, model_type=model_type)
        for idx, layer in enumerate(tqdm(layers, unit="layer", desc="应用 R1 旋转")):
            rotation_utils.rotate_attention_inputs(layer, Q, model_type)
            rotation_utils.rotate_attention_output(layer, Q, model_type)
            rotation_utils.rotate_mlp_input(layer, Q, model_type, idx, None)
            rotation_utils.rotate_mlp_output(layer, Q, model_type, args.use_r4, idx, None)
    
    # 应用 R2 旋转
    if args.use_r2 != 'none' and args.use_r2 != 'online':
        layers = model_utils.get_transformer_layers(model, model_type=model_type)
        for idx, layer in enumerate(tqdm(layers, unit="layer", desc="应用 R2 旋转")):
            v_proj = layer.self_attn.v_proj
            o_proj = layer.self_attn.o_proj
            
            if r2_dict is not None and f"model.layers.{idx}.self_attn.R2" in r2_dict:
                r2 = r2_dict[f"model.layers.{idx}.self_attn.R2"].to(device=device, dtype=torch.float64)
            else:
                r2 = rotation_utils.get_orthogonal_matrix(head_dim, args.rotate_mode)
                if len(r2.shape) != 3:
                    r2 = r2.repeat(kv_head, 1, 1)
            
            rotation_utils.apply_multi_head_rotate(v_proj, r2, head_dim, idx, kv_head, output=True, smooth=None)
            rotation_utils.apply_multi_head_rotate(o_proj, r2, head_dim, idx, kv_head, output=False, smooth=None)
    
    logging.info("旋转矩阵应用完成")


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transformers.set_seed(args.seed)
    
    logging.info("=" * 60)
    logging.info("DartQuant 对比测试脚本 (SWD Loss)")
    logging.info("=" * 60)
    logging.info(f"模型: {args.model}")
    logging.info(f"校准样本数: {args.nsamples}, 序列长度: {args.seqlen}")
    logging.info(f"权重量化: W{args.w_bits}, 激活量化: A{args.a_bits}")
    logging.info(f"使用 R1: {args.use_r1}, R2: {args.use_r2}, R3: {args.use_r3}, R4: {args.use_r4}")
    logging.info(f"★ 使用 SWD Loss 替代 Whip Loss ★")
    logging.info("=" * 60)
    
    # ========================================================================
    # 1. 加载模型
    # ========================================================================
    logging.info("步骤 1: 加载模型...")
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    model.model_name = args.model.split('/')[-1]
    
    # 保存模型配置信息
    args.hidden_size = model.config.hidden_size
    args.head_num = model.config.num_attention_heads
    args.kv_head = model.config.num_key_value_heads
    args.num_layers = model.config.num_hidden_layers
    
    # 获取 tokenizer
    model_path = data_utils.convert_model_name(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    # ========================================================================
    # 2. 融合 LayerNorm（在收集数据前必须先融合）
    # ========================================================================
    logging.info("步骤 2: 融合 LayerNorm...")
    rotation_utils.fuse_layer_norms(model)
    utils.cleanup_memory(verbos=True)
    
    # ========================================================================
    # 3. 收集训练数据（用于 R1 和 R2）
    # ========================================================================
    logging.info("步骤 3: 收集训练数据...")
    
    # 加载校准数据
    trainloader = data_utils.get_loaders(
        args.cal_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
        eval_mode=False
    )
    
    r1_train_data, o_proj_data_dict = collect_training_data(model, trainloader, args, device)
    
    # ========================================================================
    # 4. 训练 R1 旋转矩阵 (使用 SWD Loss)
    # ========================================================================
    trained_r1 = None
    if args.train_r1 and args.use_r1:
        logging.info("步骤 4: 训练 R1 旋转矩阵 (SWD Loss)...")
        trained_r1 = train_R1(r1_train_data, args, device)
        logging.info(f"R1 训练完成, shape={trained_r1.shape}")
    
    # 释放 R1 训练数据
    del r1_train_data
    torch.cuda.empty_cache()
    gc.collect()
    
    # ========================================================================
    # 5. 训练 R2 旋转矩阵 (使用 SWD Loss)
    # ========================================================================
    trained_r2_dict = None
    if args.train_r2 and args.use_r2 != 'none':
        logging.info("步骤 5: 训练 R2 旋转矩阵 (SWD Loss)...")
        trained_r2_dict = train_all_R2(o_proj_data_dict, args, device)
        logging.info(f"R2 训练完成, 共 {len(trained_r2_dict)} 层")
    
    # 释放 R2 训练数据
    del o_proj_data_dict
    torch.cuda.empty_cache()
    gc.collect()
    
    # ========================================================================
    # 6. 重新加载模型并应用旋转
    # ========================================================================
    logging.info("步骤 6: 重新加载模型并应用旋转...")
    
    # 重新加载干净的模型
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    model.model_name = args.model.split('/')[-1]
    
    # 融合 LayerNorm
    rotation_utils.fuse_layer_norms(model)
    
    # 应用旋转（从内存中的矩阵）
    apply_rotation_from_memory(model, trained_r1, trained_r2_dict, args, device)
    
    utils.cleanup_memory(verbos=True)
    
    # ========================================================================
    # 7. 添加激活量化包装器
    # ========================================================================
    logging.info("步骤 7: 添加激活量化包装器...")
    quant_utils.add_actquant(model)
    
    # 配置 R4 (online Hadamard for down_proj)
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if args.use_r4 and 'down_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had
        if args.use_r2 == 'online' and 'o_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
            qlayers[name].fp32_had = args.fp32_had
    
    # ========================================================================
    # 8. GPTQ/RTN 权重量化
    # ========================================================================
    if args.w_bits < 16:
        logging.info(f"步骤 8: {'RTN' if args.w_rtn else 'GPTQ'} 权重量化 (W{args.w_bits})...")
        
        if args.w_rtn:
            # RTN 量化
            quantizers = gptq_utils.rtn_fwrd(model, device, args)
        else:
            # GPTQ 量化
            trainloader = data_utils.get_loaders(
                args.cal_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, device, args)
        
        logging.info("权重量化完成")
    
    # ========================================================================
    # 9. 配置激活量化
    # ========================================================================
    if args.a_bits < 16 or args.v_bits < 16:
        logging.info(f"步骤 9: 配置激活量化 (A{args.a_bits}, V{args.v_bits})...")
        
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not args.a_asym
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16:
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=args.v_groupsize,
                    sym=not args.v_asym,
                    clip_ratio=args.v_clip_ratio
                )
            
            if 'lm_head' in name:
                layer_input_bits = 16
            
            if args.o_per_head and 'o_proj' in name:
                layer_groupsize = model.config.hidden_size // model.config.num_attention_heads
            
            if 'down_proj' in name:
                if args.a_bits_down_proj is not None:
                    layer_input_bits = args.a_bits_down_proj
                layer_groupsize = down_proj_groupsize
            
            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
                residual=args.a_residual
            )
    
    # ========================================================================
    # 10. 配置 K-cache 量化
    # ========================================================================
    if args.k_bits < 16:
        logging.info(f"步骤 10: 配置 K-cache 量化 (K{args.k_bits})...")
        
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        k_quant_config = {
            'k_bits': args.k_bits,
            'k_groupsize': args.k_groupsize,
            'k_sym': not args.k_asym,
            'k_clip_ratio': args.k_clip_ratio,
            'use_r3': args.use_r3
        }
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn,
                rope_function_name,
                config=model.config,
                **k_quant_config
            )
    
    # ========================================================================
    # 11. 评估 PPL
    # ========================================================================
    logging.info("步骤 11: 评估 Perplexity...")
    
    model.to(device)
    
    # 使用正确的 PPL 评估方法
    try:
        ppl = evaluate_ppl_simple(model, tokenizer, args.eval_dataset, device)
    except Exception as e:
        logging.warning(f"PPL 评估失败: {e}")
        logging.info("回退到原始评估方法...")
        testenc = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )
        ppl = eval_utils.ppl_evaluator(model, testenc, device, args)
    
    # ========================================================================
    # 12. 输出结果
    # ========================================================================
    logging.info("=" * 60)
    logging.info("结果汇总 (SWD Loss):")
    logging.info(f"  模型: {args.model}")
    logging.info(f"  数据集: {args.eval_dataset}")
    logging.info(f"  权重量化: W{args.w_bits}, 激活量化: A{args.a_bits}")
    logging.info(f"  K-cache: K{args.k_bits}, V-cache: V{args.v_bits}")
    logging.info(f"  R1: {args.use_r1} (trained={args.train_r1})")
    logging.info(f"  R2: {args.use_r2} (trained={args.train_r2})")
    logging.info(f"  R3: {args.use_r3}, R4: {args.use_r4}")
    logging.info(f"  ★ Loss Type: SWD (Sliced Wasserstein Distance)")
    logging.info(f"  ★ Perplexity: {ppl:.2f}")
    logging.info("=" * 60)
    
    print(f"\n[SWD Loss] {args.eval_dataset.upper()} PPL: {ppl:.2f}")


if __name__ == '__main__':
    main()
