# DartQuant

> **Efficient Rotational Distribution Calibration for LLM INT4 Quantization**

DartQuant 是一个基于 PyTorch 的大语言模型量化框架，通过可学习的旋转矩阵对激活值分布进行校准，显著降低 INT4 量化的精度损失，支持 LLaMA 系列模型。

---

## 目录

- [项目架构](#项目架构)
- [核心算法](#核心算法)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [完整工作流](#完整工作流)
- [NPU 支持](#npu-支持)

---

## 项目架构

```
DartQuant/
│
├── README.md                    # 项目总览（本文件）
├── requirement.txt              # Python 依赖
├── download_all.py              # 模型与数据集下载脚本
│
├── calibrater/                  # 第一阶段：旋转矩阵校准
│   ├── README.md
│   ├── get_train_data.py        # 采集 R1/R2 训练数据（激活值 Hook）
│   ├── dataloader.py            # R1Dataset / R2Dataset
│   ├── r1_base_qr.py            # 训练 R1（全局旋转矩阵）
│   └── r2_base_qr.py            # 训练 R2（逐头旋转矩阵）
│
├── fake_quant/                  # 第二阶段：量化推理与评测
│   ├── README.md
│   ├── main_for_test.py         # 主入口：量化流程 + 评测
│   ├── args_config_gen.py       # 参数解析与配置
│   ├── model_utils.py           # 模型加载与架构适配
│   ├── rotation_utils.py        # 旋转矩阵融合与应用
│   ├── quant_utils.py           # 量化包装器（激活/权重）
│   ├── gptq_utils.py            # GPTQ / RTN 权重量化
│   ├── hadamard_utils.py        # 快速 Hadamard 变换（R3/R4 在线旋转）
│   ├── eval_utils.py            # PPL 评测
│   ├── data_utils.py            # 数据集加载
│   ├── utils.py                 # 通用工具（日志/内存管理/模型分块 I/O）
│   ├── monkeypatch.py           # 推理兼容性补丁
│   └── Script/
│       ├── dart_gptq_wxaykvz_test.sh    # 测试脚本
│       ├── dart_gptq_wxaykvz_save.sh    # 量化模型保存脚本
│       └── dart_gptq_wxaykvz_load.sh    # 量化模型加载脚本
│
├── NPU_DartQuant/               # NPU（Ascend）运行时版本
│   ├── README.md
│   ├── requirement.txt
│   ├── calibrater/              # 与 GPU 版本结构相同
│   └── fake_quant/              # 与 GPU 版本结构相同
│       └── Script/
│           ├── dart_gptq_wxaykvz_test.sh
│           ├── dart_gptq_wxaykvz_save.sh
│           ├── dart_gptq_wxaykvz_load.sh
│           └── llama_3_70b.sh
│
└── tests/                       # 端到端测试脚本（根目录）
    ├── baseline_ppl_test.py     # FP16 基线 PPL 测评
    ├── quick_test.py            # 全流程集成测试（Whip Loss，无磁盘 I/O）
    └── comparative_test.py      # 对比测试（SWD Loss vs Whip Loss）
```

---

## 核心算法

### 旋转矩阵体系（R1 ~ R4）

DartQuant 使用四组旋转矩阵对模型的激活值分布进行校准，使其更适合低比特量化：

| 矩阵 | 作用对象 | 实现方式 | 是否需要训练 |
|------|---------|---------|------------|
| **R1** | Q/K/V 投影 + up-proj/gate-proj 输入 | 全隐藏维度正交矩阵 | 是（Whip Loss） |
| **R2** | O-proj（注意力输出）输入 | 逐头正交矩阵 | 是（Whip Loss） |
| **R3** | K-cache（在线） | 逐头 Hadamard 变换 | 否 |
| **R4** | down-proj 输入（在线） | 全中间维度 Hadamard 变换 | 否 |

### 训练损失函数

- **Whip Loss**（默认）：`L = sum(exp(-|x|))`，鼓励激活值远离零点，使分布更均匀
- **SWD Loss**（对比方案）：Sliced Wasserstein Distance，将激活值分布匹配到 Uniform[-b, b]

### 量化方案

| 量化目标 | 支持方案 |
|---------|---------|
| 权重量化 | GPTQ（基于 Hessian 矩阵）/ RTN（逐元素取整） |
| 激活量化 | 对称 / 非对称，逐 token / 逐通道 |
| KV-Cache 量化 | K-cache（配合 R3）/ V-cache（独立量化） |

---

## 环境配置

**基础依赖**

```bash
# Python 3.10+, PyTorch >= 2.0 (with CUDA)
pip install -r requirement.txt
```

**第三方依赖（须手动安装）**

```bash
# 快速 Hadamard 变换（R3/R4 在线旋转必需）
mkdir -p third-party && cd third-party
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform && pip install .

# LM 评测框架
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness && pip install -e .
```

**模型与数据集下载**

```bash
python download_all.py
```

---

## 快速开始

### 一键端到端测试（无磁盘 I/O）

```bash
# 在内存中完成：校准 -> 训练 R1/R2 -> GPTQ 量化 -> PPL 评测
python tests/quick_test.py \
    --model meta-llama/Llama-2-7b-hf \
    --w_bits 4 --a_bits 8 \
    --ppl_eval
```

### 基线 PPL 测评（FP16）

```bash
python tests/baseline_ppl_test.py \
    --model meta-llama/Llama-2-7b-hf
```

---

## 完整工作流

> 分步执行，适合正式实验。`calibrater/` 和 `fake_quant/` 的详细参数说明见各目录的 `README.md`。

### 第一步：采集校准数据

```bash
cd calibrater
python get_train_data.py \
    --model meta-llama/Llama-2-7b-hf \
    --calib_dataset wikitext2 \
    --nsamples 128 --seqlen 2048 \
    --r_path /path/to/save/calib_data
```

### 第二步：训练旋转矩阵

```bash
# 训练 R1（全局旋转矩阵）
python r1_base_qr.py \
    --model meta-llama/Llama-2-7b-hf \
    --ep 10 --bsz 64 --lr 1.5e-3

# 训练 R2（逐头旋转矩阵）
python r2_base_qr.py \
    --model meta-llama/Llama-2-7b-hf \
    --ep 10 --bsz 64 --lr 1e-3
```

### 第三步：量化推理与评测

```bash
cd ../fake_quant

# 使用预置脚本（推荐）
bash Script/dart_gptq_wxaykvz_test.sh \
    <GPU_ID> <MODEL> <W_BITS> <A_BITS> <KV_BITS> <R2_PATH> <R1_PATH>

# 示例：LLaMA-2-7B，W4A8，KV8
bash Script/dart_gptq_wxaykvz_test.sh \
    0 meta-llama/Llama-2-7b-hf 4 8 8 \
    /path/to/r2 /path/to/r1
```

---

## NPU 支持

`NPU_DartQuant/` 目录包含适配 Ascend NPU 的完整实现，算法与 GPU 版本完全一致，使用方式相同，详见 `NPU_DartQuant/README.md`。

---

## 支持模型

| 模型 | 参数量 | 状态 |
|------|-------|------|
| LLaMA-2-7B | 7B | 完整支持 |
| LLaMA-2-13B | 13B | 完整支持 |
| LLaMA-2-70B | 70B | 完整支持 |
| LLaMA-3-8B | 8B | 完整支持 |
| LLaMA-3-70B | 70B | 完整支持 |
| OPT-125M / 1.3B | — | 仅用于调试 |
