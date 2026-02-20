import sys
import wandb
from torch.utils.data import DataLoader, RandomSampler
from datetime import datetime
import torch.nn as nn
import torch
from pathlib import Path
import argparse
import os
import transformers
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import R1Dataset

sys.path.append('../fake_quant')

from rotation_utils import get_orthogonal_matrix
from utils import supported_models, supported_datasets
from model_utils import get_model

os.environ['WANDB_API_KEY'] = "XXXXXXXXXXXXXXX"
os.environ['WANDB_MODE'] = "offline"

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


default_path = '../data/'


class R1_QR(nn.Module):
    def __init__(self, hidden_size: int):
        super(R1_QR, self).__init__()
        self.hidden_size = hidden_size
        self.matrix = nn.Parameter(torch.eye(hidden_size))

    def forward(self, x):
        self.rotate, _ = torch.linalg.qr(self.matrix, mode='complete')
        o_x = torch.matmul(x, self.rotate)
        return o_x


def train_R1(dataset, args):
    # transformers.set_seed(args.seed)  # 初始化随机种子
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    R1 = R1_QR(hidden_size=args.hidden_size).to(device)
    R1.matrix.data = get_orthogonal_matrix(
        args.hidden_size, args.init_mode, device).float()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(R1.parameters(),
                                    lr=args.lr, momentum=args.mom)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(R1.parameters(),
                                     lr=args.lr,)
    else:
        raise NotImplementedError

    if args.cos_lr:  # 设置余弦退火学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ep, eta_min=0)

    if args.wandb:
        wandb.init(project="learnable_balance_rotate",
                   name=f"{args.run_time}_{args.st}",
                   config=vars(args)
                   )
        loss_table = wandb.Table(columns=['Losses'])
        for loss in args.loss_list:
            loss_table.add_data(loss)
        wandb.log({"loss list": loss_table})

        param_table = wandb.Table(columns=['Parameter', 'Value'])
        for key, value in vars(args).items():
            param_table.add_data(key, str(value))
        wandb.log({"parameter table": param_table})

    # 每个epoch训练的样本比例
    num_samples = int(len(dataset) * args.train_subset_size)
    R1.train()
    print("---> start training R1 ")
    for epoch in range(args.ep):
        loss_log = []
        # 创建 RandomSampler，每次从数据集中随机选择一部分样本
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        sampler = RandomSampler(indices)
        dataloader = DataLoader(dataset,        # 创建数据加载器
                                sampler=sampler,
                                batch_size=args.bsz,
                                num_workers=8,
                                prefetch_factor=3,
                                persistent_workers=True,
                                pin_memory=True)

        for batch_idx, batch_samples in enumerate(dataloader):
            batch_samples = batch_samples.to(device).float().reshape(-1, args.hidden_size)
            outputs = R1(batch_samples)
            loss = torch.sum(torch.exp((-outputs.abs())),
                             dim=-1, keepdim=True).mean() / args.accumulation_steps
            loss.backward()

            # 如果达到指定的累计步数，更新梯度并清零
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_log.append(loss.detach())

        if args.cos_lr:
            scheduler.step()

        # 计算平均损失 并打印日志
        mean_inv_loss = torch.stack(loss_log).mean()
        log_message = f'Epoch [{epoch+1}/{args.ep}], Whip Loss: {mean_inv_loss.item():.4f}'
        if args.cos_lr:
            log_message += f', LR: {scheduler.get_last_lr()[0]:.4f}'
        print(log_message)

        # 记录到 wandb
        if args.wandb:
            log_data = {
                "inv_loss": mean_inv_loss.item()
            }
            wandb.log(log_data)

        # if args.save_model:  # 保存训练好的旋转矩阵 每一epoch
        #     save_path = f"{args.save_path}/{args.save_folder}/"
        #     Path(save_path).mkdir(parents=True, exist_ok=True)
        #     save_path = Path(save_path)
        #     save_name = f"{save_path}/{args.save_folder}_{epoch}.pt"
        #     torch.save({'R1': R1.rotate.data}, save_name)

    print("---> R1 training done ")

    return R1.rotate.data


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, choices=supported_models,
                        help='model name.')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        choices=supported_datasets,
                        help='Dataset for Evaluation (default: wikitext2)',)
    parser.add_argument('--nsamples', type=int, default=128, help='Number of train sample.')
    parser.add_argument('--train_subset_size', type=float, default=0.1, help='')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of accumulation steps (default: 1)')
    parser.add_argument('--calib_sample', type=int, default=128,
                        help='Number of sample.')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--ep', type=int, default=10,
                        help='Number of epochs for training (default: 500)')
    parser.add_argument('--bsz', type=int, default=64,
                        help='Batch-size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for training (default: 1e-3)")
    parser.add_argument('--mom', type=float, default=0.9,
                        help='Momentum for training (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to use wandb for logging (default: False)')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help='Optimization method (default: sgdg)')
    parser.add_argument('--cos_lr', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to use cosine learning rate scheduler (default: True)')
    parser.add_argument('--init_mode', type=str, default='hadamard', choices=['hadamard', 'random'],
                        help='Optimization method (default: hadamard)')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether save learned r1 (default: True).')

    args = parser.parse_args()

    now = datetime.now()
    run_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    setattr(args, 'run_time', run_time)

    # {run_time}.
    save_folder = f'{args.optim}.{args.lr}.{args.mom}.{args.ep}.{args.bsz}.{args.train_subset_size}.{args.accumulation_steps}'
    save_folder = f'{save_folder}.cos' if args.cos_lr else save_folder
    setattr(args, 'save_folder', save_folder)

    data_path = os.path.join(
        default_path,
        f"train_data/{args.calib_dataset}_{args.calib_sample}samples/{args.model.split('/')[-1]}/r1_train")
    setattr(args, 'data_path', data_path)
    print(f'---> data path: {data_path}')
    save_path = os.path.join(
        default_path,
        f"trained_rotation/{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r1")
    setattr(args, 'save_path', save_path)

    return args, now


if __name__ == "__main__":
    args, run_time = parser_gen()

    model = get_model(args.model, args.hf_token)
    setattr(args, 'hidden_size', model.config.hidden_size)
    del model

    # 加载数据集
    dataset = R1Dataset(args.data_path, nsamples=args.nsamples)
    print(f'---> start time: {run_time}')
    r1 = train_R1(dataset, args)
    finish_time = datetime.now()
    train_time = finish_time - run_time
    print(f"---> Training time: {train_time}")

    os.makedirs(f'./logs/', exist_ok=True)
    with open(f'./logs/time_log.txt', 'a') as f:
        f.write(
            f'{args.model.split("/")[-1]} train r1 time: {train_time} {args.nsamples}sample {args.train_subset_size}% \n')

    if args.save_model:  # 保存训练好的旋转矩阵
        save_path = f"{args.save_path}/{args.save_folder}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_path)
        save_name = f"{save_path}/{args.save_folder}.pt"
        torch.save({'R1': r1}, save_name)

        print(f"---> R1 has been saved in: {save_path}")
