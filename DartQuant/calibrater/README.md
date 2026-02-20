# calibrater — 旋转矩阵校准模块

本目录负责 DartQuant 流程的**第一阶段**：从预训练模型中采集激活值数据，并训练用于分布校准的旋转矩阵 R1 和 R2。

---

## 目录结构

```
calibrater/
├── README.md            # 本文件
├── get_train_data.py    # 第一步：采集激活值数据
├── dataloader.py        # R1Dataset / R2Dataset 数据集类
├── r1_base_qr.py        # 第二步：训练 R1（全局旋转矩阵）
└── r2_base_qr.py        # 第三步：训练 R2（逐头旋转矩阵）
```

---

## 模块说明

### `get_train_data.py` — 激活值采集

通过 PyTorch Forward Hook 无侵入地捕获模型中间层的激活值，分别保存用于 R1 和 R2 的训练数据。

**采集目标：**

| 数据 | 来源层 | 用于训练 |
|------|--------|---------|
| `q_proj` 输入 | 各 Transformer 层注意力模块 | R1 |
| `up_proj` 输入 | 各 Transformer 层 MLP 模块 | R1 |
| `o_proj` 输入 | 各 Transformer 层注意力模块 | R2 |

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | — | 模型名称或本地路径 |
| `--calib_dataset` | `wikitext2` | 校准数据集（`wikitext2` / `ptb` / `c4`） |
| `--r_path` | — | 校准数据保存路径 |
| `--r_list` | `[r1, r2]` | 需要采集的旋转矩阵数据类型 |
| `--nsamples` | `128` | 校准样本数量 |
| `--seqlen` | `2048` | 序列长度 |

**示例：**

```bash
python get_train_data.py \
    --model meta-llama/Llama-2-7b-hf \
    --calib_dataset wikitext2 \
    --nsamples 128 --seqlen 2048 \
    --r_path /path/to/save/calib_data
```

---

### `dataloader.py` — 数据集类

提供两个 PyTorch Dataset 类，用于加载已保存的校准数据。

| 类名 | 用途 |
|------|------|
| `R1Dataset` | 加载 q_proj / up_proj 激活值，逐文件读取 |
| `R2Dataset` | 加载 o_proj 激活值，按层拼接并随机打乱 |

---

### `r1_base_qr.py` — 训练 R1

训练一个对全局隐藏维度（`hidden_size`）作用的正交旋转矩阵 R1，通过 QR 分解保证正交性。

**核心模块：** `R1_QR`
- 使用 Whip Loss：`L = sum(exp(-|x|))`，鼓励激活值远离零点
- 通过梯度下降 + QR 分解迭代更新旋转矩阵
- 支持 SGD / Adam，可配合余弦退火调度器

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | — | 模型名称（用于确定 `hidden_size`） |
| `--ep` | `10` | 训练轮次 |
| `--bsz` | `64` | 批大小 |
| `--lr` | `1.5e-3` | 学习率 |
| `--r_path` | — | 校准数据目录（`get_train_data.py` 的输出） |
| `--save_path` | — | R1 矩阵保存路径 |

**示例：**

```bash
python r1_base_qr.py \
    --model meta-llama/Llama-2-7b-hf \
    --ep 10 --bsz 64 --lr 1.5e-3 \
    --r_path /path/to/calib_data \
    --save_path /path/to/save/r1
```

---

### `r2_base_qr.py` — 训练 R2

逐层训练每个注意力头对应的旋转矩阵 R2，用于校准 o_proj 输入的逐头激活值分布。

**核心模块：** `R2_Per_Head`
- 每层独立维护 `num_attention_heads` 个旋转矩阵
- 同样使用 Whip Loss，同样通过 QR 分解保证正交性
- 逐层训练，所有层的 R2 矩阵统一保存到单个 `.pt` 文件

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | — | 模型名称（用于确定层数和头数） |
| `--ep` | `10` | 训练轮次 |
| `--bsz` | `64` | 批大小 |
| `--lr` | `1e-3` | 学习率 |
| `--r_path` | — | 校准数据目录 |
| `--save_path` | — | R2 矩阵保存路径 |

**示例：**

```bash
python r2_base_qr.py \
    --model meta-llama/Llama-2-7b-hf \
    --ep 10 --bsz 64 --lr 1e-3 \
    --r_path /path/to/calib_data \
    --save_path /path/to/save/r2
```

---

## 执行顺序

```
get_train_data.py  →  r1_base_qr.py  →  r2_base_qr.py
     ↓                     ↓                  ↓
  calib_data/           r1.pt              r2.pt
  (激活值数据)          (全局旋转矩阵)      (逐头旋转矩阵)
```

训练完成后，将 `r1.pt` 和 `r2.pt` 的路径传入 `fake_quant/main_for_test.py` 的 `--r1_path` 和 `--r2_path` 参数。
