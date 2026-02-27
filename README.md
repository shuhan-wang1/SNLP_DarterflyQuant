# DartQuant v2：大语言模型统一量化框架

> UCL SNLP 项目 — INT4 (W4A4) 权重-激活联合量化研究

DartQuant v2 是一个**一键执行**的统一量化管道，基于 DartQuant 框架构建，集成了多项核心创新：支持 7 种损失函数、NF4 量化器、通用模型接口（含 MoE）、**逐层独立 R1 训练**，以及可学习的 Butterfly 旋转算法（含 Hadamard 预热、权重感知重建损失）。无需复杂配置，即可对任意 LLM 模型进行高效的 INT4 后训练量化。

---

## 快速对比：DartQuant v2 vs v1

| 特性 | DartQuant v1 | DartQuant v2 | 改进 |
|------|------------|------------|------|
| 一键执行 | 否 | 是 | 完整管道编排 |
| 损失函数 | Whip 损失 | Whip / SWD_Unif / SWD_Gauss / KL_Unif / KL_Gauss / Bin_KL_Unif / Bin_KL_NF4 | **7 种可选** |
| 量化器 | INT4 仅 | INT4 / NF4 | 支持 NF4 |
| R1 训练 | 全局单矩阵 | **逐层独立训练** | 更精确，避免梯度混叠 |
| Dense 模型 | Llama/OPT 硬编码 | 自动检测，可扩展 | 注册新架构只需新建文件 |
| MoE 模型 | 不支持 | 原生支持（Mixtral、Qwen2-MoE） | 专家共享 R4，无需子类化 |
| R3/R4 旋转 | 固定 Hadamard | 可学习 Butterfly（含 Hadamard 预热 + Eq17 权重感知重建） | 性能更优 |
| 使用门槛 | 高（需修改代码） | 低（命令行参数） | 易用性提升 |

---

## 新增功能详解

### 功能 1：一键统一量化脚本

**核心价值**：从模型加载到量化评估，12 步完整管道，单条命令执行。

**包含组件**：
- `run_quantize.py` — 命令行入口
- `pipeline.py` — 12 步流程编排
- `args.py` — 统一参数解析

**特点**：
- 完全参数化，无需修改代码
- 支持所有模型架构（自动检测）
- 集成 R1/R2/R3/R4 全流程
- 可复现的实验管道

**基础用法**：
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss whip \
    --quantizer_type int4 \
    --w_bits 4 --a_bits 4 \
    --nsamples 128
```

---

### 功能 2：NF4 量化器集成

**原理**：QLoRA 权重量化方法，基于高斯分布假设，仅对权重进行 4 位量化。

**NF4 vs INT4 对比**：

| 对比项 | INT4（均匀） | NF4（高斯） |
|--------|-----------|-----------|
| 量化目标 | 权重 + 激活 | 权重仅 |
| 分布假设 | 均匀分布 | 高斯分布 |
| 显存占用 | 较低 | 更低 |
| 推理速度 | 中等 | 较快 |
| 量化精度 | 一般 | 较好 |
| 推荐场景 | 通用训练 | 显存紧张、推理优先 |

**集成方式**：使用 `bitsandbytes` 库，自动替换 `nn.Linear` 为 `bnb.nn.Linear4bit(quant_type='nf4')`。

**使用示例**：
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_gauss \
    --quantizer_type nf4 \
    --nsamples 128
```

---

### 功能 3：UnifiedQuantModel 通用模型接口

**问题解决**：原始 DartQuant 仅硬编码支持 Llama 和 OPT，新增模型需修改代码。

**解决方案**：基于 HuggingFace config 自动检测架构，注册表模式支持无限扩展。

**工作机制**：
```python
# 自动检测（无需用户干预）
umodel = UnifiedQuantModel("meta-llama/Llama-3.2-1B")
# 自动识别 LlamaConfig，获取所有访问器

# 统一接口，架构无关
embeddings = umodel.get_embeddings()
layers     = umodel.get_layers()
attn_q, attn_k, attn_v, attn_o = umodel.get_attn_projs(layer)
```

**默认支持**：Llama（所有版本）、OPT（所有尺寸）、Mixtral、Qwen2-MoE。

**添加新架构示例**：
```python
from dartquant_v2.unified_model import register_arch, ModelArchConfig

config = ModelArchConfig(
    embed_tokens_path="model.embed_tokens",
    layers_path="model.layers",
    lm_head_path="lm_head",
    # ... 其他路径 ...
)
register_arch("YourModelConfig", config)
```

---

### 功能 4：逐层独立 R1 训练

**v2 的 R1 变化**：原始 DartQuant 将所有层的激活混合后训练单一全局旋转矩阵；v2 为每一个 Transformer 层独立训练 R1，使用该层自身的激活数据。

**优势**：
- 保留每层的激活分布特性，不被其他层的分布均值化
- 避免层间梯度相互抵消（不同层协方差结构混合导致的梯度取消）
- 同层多模块的激活（如 `up_proj` 和 `q_proj`）自动合并训练

**函数映射**：

| 行为 | v1（旧） | v2（新） |
|------|---------|---------|
| R1 训练 | `train_r1()` 全局单矩阵 | `train_r1_all_layers()` 每层独立 |
| R1 应用 | `apply_r1_rotation()` 单矩阵广播 | `apply_r1_rotation_per_layer()` 逐层应用 |
| 输出格式 | 单个 Tensor `(hidden_size, hidden_size)` | `dict {layer_idx → Tensor}` |

**向后兼容**：`--r1_path` 加载时同时兼容旧版单矩阵格式（自动广播到所有层）和新版逐层字典格式。

---

### 功能 5：扩展损失函数（7 种）

v2 新增 4 种损失函数，共支持 7 种：

#### 原有损失（v1 继承）

| 损失函数 | 原理 | 推荐配对 |
|---------|------|---------|
| `whip` | 指数排斥 `exp(-|x|)`，推离零点 | INT4 |
| `swd_unif` | Sliced Wasserstein Distance to Uniform | INT4 |
| `swd_gauss` | Sliced Wasserstein Distance to Gaussian | NF4 |

#### 新增损失（v2）

| 损失函数 | 原理 | 推荐配对 |
|---------|------|---------|
| `kl_unif` | KL 散度至均匀分布（Vasicek 间距熵估计，最大化熵）| INT4 |
| `kl_gauss` | KL 散度至高斯分布（Gram-Charlier 矩匹配，最小化偏度²+峰度²）| NF4 |
| `bin_kl_unif` | 离散 bin KL 散度至 INT4 量化级别均匀分布（论文 Eq 19）| INT4 |
| `bin_kl_nf4` | 离散 bin KL 散度至 NF4 量化级别均匀分布（论文 Eq 19）| NF4 |

**完整推荐配对**：

| quantizer_type | 推荐 loss | 不推荐 |
|----------------|-----------|--------|
| int4 | whip / swd_unif / kl_unif / bin_kl_unif | swd_gauss / kl_gauss / bin_kl_nf4 |
| nf4 | swd_gauss / kl_gauss / bin_kl_nf4 | whip / swd_unif / kl_unif / bin_kl_unif |

> 不遵循推荐配对会输出警告，但不会报错（允许实验）。

---

### 功能 6：Butterfly Givens 旋转（R3/R4）

**原理**：O(d log d) 复杂度的可学习旋转算法，相比固定 Hadamard 矩阵，可通过训练进一步改善激活分布。

**应用范围（仅 R3 和 R4）**：
- R3：Attention 头内旋转（Q, K 在 RoPE 之后），在线应用
- R4：MLP 中间层旋转（down_proj 输入），离线烘焙到权重 + 在线应用
- 不应用于 R1 / R2（保留 QR-Orth 参数化）

**算法概要**：
```
K = log2(d) 层（例如 d=64 时，K=6 层）
每层 d/2 个 Givens 旋转对
第 l 层配对：(i, i+2^l)，其中 i mod 2^(l+1) < 2^l
总复杂度：O(d log d)
```

**Hadamard 预热（v2 新增）**：
- 初始化：angles = 0（等价单位矩阵）
- 训练前：通过 300 步 Adam 优化将 Butterfly 拟合到随机 Hadamard 矩阵，作为主训练的起点
- 效果：与 DartQuant 原始 Random Hadamard 基线保持一致，避免从全零角度出发的冷启动

**自动选择 Butterfly 损失**（v2 新增）：
- `--quantizer_type int4` → 自动选择 `kl_unif` 作为 Butterfly 损失
- `--quantizer_type nf4`  → 自动选择 `kl_gauss` 作为 Butterfly 损失
- 此选择独立于 R1/R2 使用的 `--loss` 参数

**R4 权重感知重建损失 - Eq 17（v2 新增）**：

R4 会离线烘焙到 `down_proj` 权重中，因此使用联合权重+激活重建损失：

```
L_total = L_dist(B @ x) + λ · L_recon_Eq17

L_recon_Eq17 = ||Wx - Q(W @ B^T) · Q(B @ x)||²
```

其中 Q 为伪量化器（Dequant∘Quant），B 为 Butterfly 矩阵，W 为 down_proj 权重。梯度通过直通估计器（STE）传回旋转参数。

**R3 激活仅重建损失**（R3 在线应用，无权重路径）：

```
L_recon = ||X_in - B^T @ FakeQuant(B @ X_in)||²
```

**维度支持**：
- 2 次方维度（64, 128, 256 ...）：直接使用 `ButterflyRotation`
- 非 2 次方维度（11008, 5120 ...）：自动使用 `ButterflyFactored`（K×m 分解，m 为 2 次方）

**K 因子参数化**（`ButterflyFactored` 专用，用于非 2 次方维度的跨块混合）：

| 参数 | 原理 | 特点 |
|------|------|------|
| `--latent`（默认）| 无约束矩阵 Z，每次前向通过 QR 提取正交因子（OR-Orth，与 R1/R2 方法相同）| 数值稳定，推荐 |
| `--cayley` | Cayley 变换 `Q=(I-A)(I+A)^{-1}`，A 为反对称矩阵 | 严格正交，但需 O(K³) 矩阵求解 |

**与固定 Hadamard 对比**：

| 对比项 | 固定 Hadamard | 可学习 Butterfly |
|--------|--------------|----------------|
| 计算复杂度 | O(d log d) | O(d log d) |
| 参数数量 | 0（固定） | K * d/2 个角度 |
| 可优化 | 否 | 是 |
| 初始化 | 随机 Hadamard | 预热到随机 Hadamard |

**使用示例**：
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_unif \
    --quantizer_type int4 \
    --butterfly \
    --nsamples 128
```

---

## 项目结构

```
int4_quantization_darkquant/
|
|-- dartquant_v2/                         # [核心] 统一量化管道
|   |-- __init__.py                       # 导入时自动注册所有架构
|   |-- run_quantize.py                   # 一键执行脚本（入口）
|   |-- pipeline.py                       # 12 步完整流程
|   |-- args.py                           # 参数解析
|   |-- loss_functions.py                 # 7 种损失函数（whip/swd_*/kl_*/bin_kl_*）
|   |-- unified_model.py                  # UnifiedQuantModel（架构通用化）
|   |-- nf4_quantizer.py                  # NF4 量化（bitsandbytes）
|   |-- int4_quantizer.py                 # INT4 伪量化器（Butterfly 重建损失用）
|   |-- butterfly.py                      # Butterfly Givens 旋转（ButterflyRotation / ButterflyFactored）
|   |-- trainers.py                       # R1（逐层）/ R2 / Butterfly 训练器
|   `-- arch/                             # [架构注册中心] 按公司/类型分组
|       |-- __init__.py                   # 统一入口，触发所有注册
|       |-- dense/                        # Dense（非 MoE）模型
|       |   |-- __init__.py
|       |   |-- llama.py                  # Meta Llama 全家族（1/2/3/3.1/3.2）
|       |   `-- opt.py                    # Meta OPT 全家族（125M~66B）
|       `-- moe/                          # MoE 模型
|           |-- __init__.py
|           |-- mixtral.py                # Mistral AI Mixtral 8x7B / 8x22B
|           `-- qwen_moe.py               # Alibaba Qwen2-MoE（Qwen1.5-MoE / Qwen2-57B）
|
|-- DartQuant/                            # [参考] DartQuant 原始实现
|   |-- fake_quant/                       # 假量化模块（R1-R4 应用、INT4 量化）
|   |   |-- rotation_utils.py             # R1-R4 应用方法
|   |   |-- quant_utils.py                # 激活量化
|   |   |-- model_utils.py                # 模型访问器
|   |   |-- gptq_utils.py                 # GPTQ/RTN 权重量化
|   |   |-- hadamard_utils.py             # Hadamard 矩阵
|   |   |-- data_utils.py                 # 数据加载
|   |   `-- eval_utils.py                 # 困惑度评估
|   `-- calibrater/                       # 校准器
|       |-- r1_base_qr.py                 # R1 QR 分解训练
|       `-- r2_base_qr.py                 # R2 QR 分解训练
|
|-- docs/                                 # [文档] 研究报告
|   |-- SNLP_report_1_v1_en.md            # 第一阶段：SWD_Unif 和 QR-Orth
|   |-- report_2_en.md                    # 第二阶段：Butterfly 和 Gaussian SWD
|   `-- *.md                              # 其他研究笔记
|
|-- scripts/                              # [工具脚本]
|   |-- plot_loss_butterfly.py            # 损失函数 & Butterfly 可视化实验（R1/R2/R3 分布对比）
|   |-- validation.py                     # 损失函数对比验证
|   |-- validation_joint.py              # 联合损失验证
|   `-- *.py                              # 其他辅助脚本
|
`-- requirements.txt                      # Python 依赖列表
```

---

## 支持的模型

所有模型通过 `dartquant_v2/arch/` 中的注册文件实现端到端完整支持（LayerNorm 融合 → R1/R2/R3/R4 训练与应用 → INT4/NF4 量化 → PPL 评估）。

### Dense 模型

#### Meta Llama 系列（`LlamaConfig`，注册于 `arch/dense/llama.py`）

覆盖 Llama-1、Llama-2、Llama-3、Llama-3.1、Llama-3.2 全系列，共享同一配置类名 `LlamaConfig`，所有变体自动支持，包括 Base 和 Instruct 版本。

| 版本 | 典型模型 ID |
|------|------------|
| Llama-1 | `huggyllama/llama-7b`、`huggyllama/llama-13b`、`huggyllama/llama-30b`、`huggyllama/llama-65b` |
| Llama-2 | `meta-llama/Llama-2-7b-hf`、`meta-llama/Llama-2-13b-hf`、`meta-llama/Llama-2-70b-hf` |
| Llama-2 Instruct | `meta-llama/Llama-2-7b-chat-hf`、`meta-llama/Llama-2-13b-chat-hf`、`meta-llama/Llama-2-70b-chat-hf` |
| Llama-3 | `meta-llama/Meta-Llama-3-8B`、`meta-llama/Meta-Llama-3-70B` |
| Llama-3 Instruct | `meta-llama/Meta-Llama-3-8B-Instruct`、`meta-llama/Meta-Llama-3-70B-Instruct` |
| Llama-3.1 | `meta-llama/Llama-3.1-8B`、`meta-llama/Llama-3.1-70B`、`meta-llama/Llama-3.1-405B` |
| Llama-3.1 Instruct | `meta-llama/Llama-3.1-8B-Instruct`、`meta-llama/Llama-3.1-70B-Instruct` |
| Llama-3.2 | `meta-llama/Llama-3.2-1B`、`meta-llama/Llama-3.2-3B` |
| Llama-3.2 Instruct | `meta-llama/Llama-3.2-1B-Instruct`、`meta-llama/Llama-3.2-3B-Instruct` |

#### Meta OPT 系列（`OPTConfig`，注册于 `arch/dense/opt.py`）

覆盖 OPT-125M 到 OPT-66B 所有尺寸。主要用于调试和小规模验证。

| 典型模型 ID |
|------------|
| `facebook/opt-125m` |
| `facebook/opt-1.3b` |
| `facebook/opt-6.7b` |
| `facebook/opt-13b` |
| `facebook/opt-30b` |
| `facebook/opt-66b` |

> **OPT 特殊处理**：无 RoPE（绝对位置嵌入）、无门控 MLP（仅 fc1/fc2）、需要 mean-baking（LayerNorm 均值偏移消除）。这些差异已在注册配置中自动处理。

### MoE 模型

#### Mistral AI Mixtral 系列（`MixtralConfig`，注册于 `arch/moe/mixtral.py`）

| 典型模型 ID |
|------------|
| `mistralai/Mixtral-8x7B-v0.1` |
| `mistralai/Mixtral-8x22B-v0.1` |

> 专家命名：`w1`（up-gate）、`w3`（gate）、`w2`（down）；无共享专家；所有专家共享同一 R4 矩阵。

#### Alibaba Qwen2-MoE 系列（`Qwen2MoeConfig`，注册于 `arch/moe/qwen_moe.py`）

| 典型模型 ID |
|------------|
| `Qwen/Qwen1.5-MoE-A2.7B` |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` |
| `Qwen/Qwen2-57B-A14B` |
| `Qwen/Qwen2-57B-A14B-Instruct` |

> 专家命名：`up_proj`/`gate_proj`/`down_proj`（与 Llama 一致）；有常驻共享专家（`mlp.shared_expert`）；R1 和 R4 同时应用于路由专家和共享专家。

### MoE 旋转应用规则

（依据 `docs/SNLP_report_1_v1_en.md` 第 2.4.4 节）

| 旋转 | 应用方式 |
|------|---------|
| R1 | 应用于每个专家的 up/gate 输入投影（`W @ R1`）和 down 输出投影（`R1^T @ W`），包括 Qwen2-MoE 的共享专家 |
| R4 | 所有专家共享同一 R4 矩阵，离线烘焙到每个专家的 down_proj 权重中，避免重复存储 |

---

## 快速开始

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 fast-hadamard-transform（CUDA 加速的 Hadamard 变换，DartQuant 核心依赖）
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
cd ..
```

核心依赖：
- `torch >= 2.0`
- `transformers >= 4.30`
- `bitsandbytes >= 0.39`（NF4 必需）
- `fast-hadamard-transform`（从源码安装，见上方）
- `numpy`, `scipy`, `datasets`, `accelerate`

### 2. 基础用法（5 个常见场景）

#### 场景 A：INT4 + Whip 损失（原始 DartQuant 行为）
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss whip \
    --quantizer_type int4 \
    --w_bits 4 --a_bits 4 \
    --nsamples 128 \
    --ppl_eval_dataset wikitext2
```

#### 场景 B：INT4 + SWD_Unif（推荐，性能更优）
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_unif \
    --quantizer_type int4 \
    --w_bits 4 --a_bits 4 \
    --nsamples 128 \
    --ppl_eval_dataset wikitext2
```

#### 场景 C：NF4 + Gaussian SWD（低显存、推理优先）
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_gauss \
    --quantizer_type nf4 \
    --nsamples 128 \
    --ppl_eval_dataset wikitext2
```

#### 场景 D：INT4 + KL 散度损失（新增）
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss kl_unif \
    --quantizer_type int4 \
    --w_bits 4 --a_bits 4 \
    --nsamples 128
```

#### 场景 E：带可学习 Butterfly R3/R4 的 INT4
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_unif \
    --quantizer_type int4 \
    --butterfly \
    --w_bits 4 --a_bits 4 \
    --nsamples 128 \
    --ppl_eval_dataset wikitext2
```

### 3. 参数详解

#### 必需参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model` | HF 模型名 | `meta-llama/Llama-3.2-1B`、`facebook/opt-125m` 等 |
| `--loss` | whip / swd_unif / swd_gauss / kl_unif / kl_gauss / bin_kl_unif / bin_kl_nf4 | 旋转训练损失函数，必需（共 7 种） |
| `--quantizer_type` | int4 / nf4 | 量化方法，必需 |

#### 量化配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--w_bits` | 4 | 权重位数 |
| `--a_bits` | 4 | 激活位数 |
| `--w_groupsize` | -1（逐通道） | 权重分组大小 |
| `--a_groupsize` | -1（逐通道） | 激活分组大小 |
| `--w_rtn` | False | 使用 RTN 代替 GPTQ 进行权重量化 |
| `--butterfly` | False | 启用可学习 Butterfly R3/R4 |

#### Butterfly 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--butterfly_epochs` | 100 | Butterfly R3/R4 训练轮数（论文建议 500-700 步，100 epochs × batch 约等于 200 步） |
| `--lambda_recon` | 0.1 | 量化重建损失 L_recon 的权重（仅在 `--butterfly` 启用时生效） |
| `--quant_block_size` | 64 | Butterfly 训练中伪量化器的分块大小 |
| `--latent` | （默认）| ButterflyFactored K 因子使用 QR-Orth 参数化（推荐） |
| `--cayley` | — | ButterflyFactored K 因子使用 Cayley 参数化 |

#### 训练配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--r1_epochs` | 10 | R1 训练轮数（逐层独立训练） |
| `--r2_epochs` | 5 | R2 训练轮数 |
| `--batch_size` | 64 | 训练批大小 |
| `--lr` | 0.001 | 学习率 |
| `--momentum` | 0.9 | SGD 动量 |
| `--cos_lr` | False | 使用余弦退火学习率 |
| `--optim` | sgd | 优化器（sgd / adam） |
| `--accumulation_steps` | 1 | 梯度累积步数 |
| `--rotate_mode` | hadamard | R1/R2 初始化方式（hadamard / random） |

#### 数据与评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nsamples` | 128 | 校准数据样本数 |
| `--seqlen` | 2048 | 序列长度 |
| `--cal_dataset` | wikitext2 | 校准数据集（wikitext2 / c4 / ptb） |
| `--ppl_eval_dataset` | wikitext2 | PPL 评估数据集 |
| `--seed` | 0 | 随机种子 |

---

## 完整量化流程（12 步）

```
步骤 1：参数解析 & 模型加载
  读取 --loss --quantizer_type --butterfly
  UnifiedQuantModel 自动检测模型架构（LlamaConfig / OPTConfig / MixtralConfig / Qwen2MoeConfig）

步骤 2：LayerNorm 融合
  将 LayerNorm 权重吸收到相邻线性层
  使旋转变换对 LayerNorm 透明（必须在旋转前执行）

步骤 3：校准数据加载
  加载 wikitext2/c4/ptb 校准集，--nsamples 条序列

步骤 4：R1 逐层激活收集 & 训练（v2 关键变化）
  为每一层独立收集 mlp.up_proj 和 self_attn.q_proj 的输入激活
  对每层分别使用指定损失函数（Whip / SWD / KL / Bin_KL）训练独立 R1_QR
  输出：dict {layer_idx → (hidden_size, hidden_size) 正交矩阵}

步骤 5：R1 逐层应用（离线）
  对每层独立旋转：输入侧 W = W @ R1_l，输出侧 W = R1_l^T @ W
  涵盖：Q/K/V（输入）、O（输出）、MLP up/gate（输入）、MLP down（输出）

步骤 6：R2 激活收集 & 训练
  Hook self_attn.o_proj 输入（R1 应用后）
  对每层训练 R2_Per_Head（逐 KV 头 head_dim×head_dim 旋转）

步骤 7：R2 离线应用
  旋转 V 投影输出（R2^T @ W_v）、O 投影输入（W_o @ R2）

步骤 8：R3/R4 处理
  若启用 --butterfly：
    R4：收集 down_proj 输入激活 + 采样权重矩阵 bank
        训练 ButterflyRotation/ButterflyFactored（Hadamard 预热 + Eq17 联合损失）
        离线将 Butterfly R4 烘焙到 down_proj 权重
    R3：收集 q_proj 输出激活
        训练 ButterflyRotation（Hadamard 预热 + 激活重建损失）
        在线应用（注册为 buffer）
  否则（默认）：
    固定随机 Hadamard R4：离线烘焙到 down_proj 权重 + 在线应用
    固定随机 Hadamard R3：在线应用

步骤 9：激活量化包装（仅 int4）
  添加 ActQuantWrapper 到所有 Linear 层
  配置 R4 / R3 在线 Hadamard（或 Butterfly）

步骤 10：权重量化
  若 --quantizer_type int4：GPTQ（基于 Hessian）或 RTN 方法
  若 --quantizer_type nf4：bitsandbytes NF4 权重替换

步骤 11：模型设备分配
  单卡（默认）或多卡（--distribute）

步骤 12：困惑度评估 & 报告
  在指定数据集（wikitext2/c4/ptb）上评估困惑度
  输出结果并保存到 output_dir/results.txt
```

---

## 核心算法详解

### Whip Loss（基线损失）

**公式**：
```
L = mean( sum_i exp(-|x_i|) )
```

推动激活分布远离零点，使分布更均匀，降低量化误差。

**优点**：收敛快、实现简单。

---

### SWD_Unif 损失（均匀分布）

**公式**（逐维度独立计算）：
```
b_j    = sqrt(3) * RMS(x[:,j])
target = Uniform[-b_j, b_j] 的等间距分位数
Loss   = mean_j ||sort_j(x) - target_j||^2
```

**优势**：自适应范围、无超参数、特别适合 INT4 均匀量化。

---

### Gaussian SWD 损失（高斯分布）

**公式**（逐维度独立计算）：
```
sigma_j = sqrt(mean(x[:,j]^2))
p_i     = (i - 0.5) / n
target  = Phi^{-1}(p_i) * sigma_j  (正态分布分位数)
Loss    = mean_j ||sort_j(x) - target_j||^2
```

**优势**：与 NF4 高斯假设匹配、自适应标准差估计。

---

### KL_Unif 损失（熵最大化）

**原理**：Vasicek (1976) 间距熵估计器，最大化每维度的微分熵，驱动样本均匀分布。

**公式**（逐维度独立）：
```
H_j ≈ (1/B) Σ_i log(B · Δx_i^j)  （间距估计的微分熵）
L   = -mean_j H_j                  （最小化负熵 = 最大化熵）
```

**优势**：理论上直接最小化与均匀分布的 KL 散度（b_j 旋转不变，可消去）。

---

### KL_Gauss 损失（矩匹配）

**原理**：Gram-Charlier 代理损失，最小化偏度²和峰度²，驱动分布趋向高斯形状。

**公式**（逐维度独立）：
```
L_j = skewness_j²  +  excess_kurtosis_j²
L   = mean_j L_j
```

**优势**：高斯分布时 γ₁=γ₂=0，损失为零；与 NF4 假设精确匹配。

---

### Bin_KL 损失（离散 bin KL，论文 Eq 19）

**原理**：对量化级别使用软直方图，直接最小化激活在每个量化 bin 上的分布与均匀分布的 KL 散度。

- `bin_kl_unif`：基于 INT4 量化级别（15 个等间距 bin，逐维度自适应范围）
- `bin_kl_nf4`：基于 NF4 量化级别（16 个非均匀 bin，来自 QLoRA 论文）

---

### Butterfly Givens 旋转

**算法结构（d=8 示例，K=3 层）**：
```
层 0（stride=2）: (0,1), (2,3), (4,5), (6,7)
层 1（stride=4）: (0,2), (1,3), (4,6), (5,7)
层 2（stride=8）: (0,4), (1,5), (2,6), (3,7)

每对使用 Givens 旋转：
  [cos(θ)  sin(θ)]
  [-sin(θ) cos(θ)]

初始化：θ = 0（等价单位矩阵）
预热：300 步拟合随机 Hadamard 矩阵
训练：通过梯度下降学习最优角度 θ
```

---

## 模型支持与扩展

### 如何添加新的 Dense 架构

**步骤 1**：查看 config 类名
```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("new-model-name")
print(cfg.__class__.__name__)   # 例如：Qwen2Config
```

**步骤 2**：在 `arch/dense/` 下创建新文件
```python
# dartquant_v2/arch/dense/qwen2.py
from dartquant_v2.unified_model import ModelArchConfig, register_arch

register_arch("Qwen2Config", ModelArchConfig(
    embed_tokens_path="model.embed_tokens",
    layers_path="model.layers",
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",
    self_attn_attr="self_attn",
    mlp_up_proj_attr="mlp.up_proj",
    mlp_gate_proj_attr="mlp.gate_proj",
    mlp_down_proj_attr="mlp.down_proj",
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",
    norm_class_name="Qwen2RMSNorm",
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb",
    no_split_module_class="Qwen2DecoderLayer",
    is_moe=False,
))
```

**步骤 3**：在 `arch/dense/__init__.py` 中导入
```python
from dartquant_v2.arch.dense import qwen2 as _qwen2   # noqa: F401
```

**步骤 4**：验证
```bash
python -c "
import dartquant_v2   # 触发注册
from dartquant_v2.unified_model import UnifiedQuantModel
m = UnifiedQuantModel('Qwen/Qwen2-7B')
print(f'Layers: {m.num_layers}, hidden: {m.hidden_size}')
"
```

### 如何添加新的 MoE 架构

在 `arch/moe/` 下创建文件，额外填写 MoE 专用字段：

```python
# dartquant_v2/arch/moe/your_moe.py
from dartquant_v2.unified_model import ModelArchConfig, register_arch

register_arch("YourMoeConfig", ModelArchConfig(
    # --- 与 Dense 相同的公共字段 ---
    embed_tokens_path="model.embed_tokens",
    layers_path="model.layers",
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",
    self_attn_attr="self_attn",
    # Dense MLP 路径设为 None（MoE 层没有独立的 dense MLP）
    mlp_up_proj_attr=None,
    mlp_gate_proj_attr=None,
    mlp_down_proj_attr=None,
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",
    norm_class_name="YourRMSNorm",
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb",
    no_split_module_class="YourDecoderLayer",

    # --- MoE 专用字段 ---
    is_moe=True,
    # 专家列表的路径（相对于 layer）
    experts_attr="mlp.experts",
    # 每个专家内部的投影属性名
    expert_up_proj_attr="up_proj",
    expert_gate_proj_attr="gate_proj",
    expert_down_proj_attr="down_proj",
    # 共享专家（若无，设为 None）
    shared_expert_attr="mlp.shared_expert",
    # 若属性名与上面相同，设为 None 即可复用
    shared_expert_up_attr=None,
    shared_expert_gate_attr=None,
    shared_expert_down_attr=None,
    # 专家中间层维度的 config 属性名
    moe_intermediate_size_attr="moe_intermediate_size",
))
```

然后在 `arch/moe/__init__.py` 中添加导入即可，无需修改任何核心代码。

**MoE 关键字段说明**：

| 字段 | 说明 | Mixtral 示例 | Qwen2-MoE 示例 |
|------|------|------------|--------------|
| `experts_attr` | 专家列表路径（相对于 layer） | `block_sparse_moe.experts` | `mlp.experts` |
| `expert_up_proj_attr` | 专家内 up/gate 输入投影 | `w1` | `up_proj` |
| `expert_gate_proj_attr` | 专家内 gate 投影 | `w3` | `gate_proj` |
| `expert_down_proj_attr` | 专家内 down 投影 | `w2` | `down_proj` |
| `shared_expert_attr` | 常驻专家路径（无则 None） | None | `mlp.shared_expert` |
| `moe_intermediate_size_attr` | config 中专家中间层维度 | `ffn_dim` | `moe_intermediate_size` |

---

## 常见问题（FAQ）

### Q1：为什么 --loss 和 --quantizer_type 是必需的？

这两个参数必须显式指定，原因是：
- `--loss` 决定旋转矩阵的优化方向（激活分布的目标形状）
- `--quantizer_type` 决定最终量化方法（均匀或高斯假设）

二者需要配合使用，因此不提供默认值以避免不合适的配对。

---

### Q2：INT4 和 NF4 应该选哪个？

| 场景 | 推荐 | 理由 |
|------|------|------|
| 追求精度（困惑度） | INT4 | 权重+激活联合量化，精度更高 |
| 显存极其紧张 | NF4 | 仅权重量化，显存占用最少 |
| 推理速度优先 | NF4 | 权重量化的推理速度更快 |
| 不确定 | INT4 | 一般场景的通用选择 |

---

### Q3：7 种损失函数如何选择？

| 量化目标 | 推荐损失 | 说明 |
|---------|---------|------|
| INT4（快速基线） | `whip` | 原始 DartQuant，收敛快 |
| INT4（分布感知） | `swd_unif` | SWD 匹配均匀分布，性能稳定 |
| INT4（信息理论） | `kl_unif` | 最大化微分熵，理论最优 |
| INT4（量化感知） | `bin_kl_unif` | 直接优化量化 bin 分布 |
| NF4（分布感知） | `swd_gauss` | SWD 匹配高斯分布 |
| NF4（信息理论） | `kl_gauss` | 矩匹配驱向高斯 |
| NF4（量化感知） | `bin_kl_nf4` | 直接优化 NF4 级别分布 |

---

### Q4：Butterfly 对性能的影响？

通常能带来 0.1~0.3 困惑度的改善，具体取决于模型大小和数据集。

权衡：
- 优势：性能更优，可学习旋转适应具体模型
- 劣势：训练时间延长约 20%，`--butterfly_epochs` 建议 ≥ 100

建议先用基础设置快速验证，再启用 `--butterfly` 以获得最优结果。

---

### Q5：如何处理显存不足？

按优先级调整：

1. 降低校准数据量：`--nsamples 64` 或更少
2. 减小批大小：`--batch_size 32`
3. 切换到 NF4：`--quantizer_type nf4`
4. 禁用 Butterfly：移除 `--butterfly`

---

### Q6：如何评估量化后的模型？

流程内已自动完成困惑度评估，评估结果会在最后输出：

```
Original Model PPL:   8.12  (WikiText2)
Quantized Model PPL:  8.74  (WikiText2)
```

若要在额外数据集上评估，可在参数中添加 `--ppl_eval_dataset c4`。

---

### Q7：如何运行损失函数和 Butterfly 可视化实验？

使用 `scripts/plot_loss_butterfly.py`：

```bash
python scripts/plot_loss_butterfly.py
```

该脚本完整镜像 dartquant_v2 管道的旋转矩阵计算（跳过权重量化和 PPL 评估），生成三组实验图像：
- **实验 1**：R1 损失函数对比（Whip / SWD_Unif / SWD_Gauss），含激活分布直方图和训练曲线
- **实验 2**：R2 逐头旋转，报告每层最终损失
- **实验 3**：Butterfly R3 与 Hadamard 对比，含分布图和方差均匀性曲线

输出路径：`/root/autodl-tmp/dartquant_v2_plots/`（可在脚本顶部 `PLOT_DIR` 修改）

---

## 完整端到端示例

```bash
# INT4 + SWD_Unif，完整参数示例
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_unif \
    --quantizer_type int4 \
    --w_bits 4 --a_bits 4 \
    --r1_epochs 10 \
    --r2_epochs 5 \
    --batch_size 64 \
    --lr 0.001 \
    --momentum 0.9 \
    --nsamples 128 \
    --cal_dataset wikitext2 \
    --ppl_eval_dataset wikitext2 \
    --seed 42
```

```bash
# INT4 + Butterfly + KL 损失，进阶示例
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss kl_unif \
    --quantizer_type int4 \
    --butterfly \
    --butterfly_epochs 200 \
    --lambda_recon 0.1 \
    --w_bits 4 --a_bits 4 \
    --nsamples 128 \
    --seed 42
```

---

## 技术细节

- **R1**（v2 逐层）：每层独立的 hidden_size × hidden_size 正交矩阵，应用于嵌入、Attention 输入/输出、MLP 输入/输出
- **R2**：head_dim × head_dim 按头旋转，应用于 Attention 的 V 投影输出和 O 投影输入
- **R3**：head_dim × head_dim 旋转，在 RoPE 之后在线应用于 Q, K
- **R4**：intermediate_size × intermediate_size 旋转，离线烘焙到 down_proj 权重 + 在线应用
- **QR-Orth**：通过 QR 分解参数化无约束矩阵，保证正交性（用于 R1、R2 及 ButterflyFactored K 因子）
- **Butterfly**：通过 Givens 角度参数化，O(d log d) 复杂度，天然正交，Hadamard 预热

详见研究报告：
- 第一阶段：`docs/SNLP_report_1_v1_en.md`（SWD_Unif, QR-Orth, 逐层 R1）
- 第二阶段：`docs/report_2_en.md`（Butterfly, Gaussian SWD, KL 损失, Eq 17 重建损失）

---

## 依赖与环境

```bash
# Python >= 3.9
pip install -r requirements.txt

# 安装 fast-hadamard-transform（需要 CUDA 环境）
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
cd ..
```

主要依赖：

```
torch >= 2.0.0
transformers >= 4.30.0
datasets >= 2.16.0
accelerate >= 0.26.0
bitsandbytes >= 0.39.0
numpy >= 1.24.0
scipy >= 1.10.0
fast-hadamard-transform（从源码安装）
```

---

*最后更新：2026 年 2 月*
