# DartQuant v2：大语言模型统一量化框架

> UCL SNLP 项目 — INT4 (W4A4) 权重-激活联合量化研究

DartQuant v2 是一个**一键执行**的统一量化管道，基于 DartQuant 框架构建，集成了 6 项核心创新：支持多种损失函数、NF4 量化器、通用模型接口、以及可学习的 Butterfly 旋转算法。无需复杂配置，即可对任意 LLM 模型进行高效的 INT4 后训练量化。

---

## 快速对比：DartQuant v2 vs v1

| 特性 | DartQuant v1 | DartQuant v2 | 改进 |
|------|------------|------------|------|
| 一键执行 | 否 | 是 | 完整管道编排 |
| 损失函数 | Whip 损失 | Whip / SWD_Unif / SWD_Gauss | 3 种可选 |
| 量化器 | INT4 仅 | INT4 / NF4 | 支持 NF4 |
| 模型支持 | Llama/OPT 硬编码 | 自动检测，可扩展 | 通用架构 |
| R3/R4 旋转 | 固定 Hadamard | 可学习 Butterfly | 性能更优 |
| 使用门槛 | 高（需修改代码） | 低（命令行参数） | 易用性提升 |

---

## 新增 6 大功能详解

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

**默认支持**：Llama（所有版本）、OPT。

**添加新架构示例**：
```python
from dartquant_v2.unified_model import register_arch, ModelArchConfig

config = ModelArchConfig(
    embeddings_path="model.embed_tokens",
    layers_path="model.layers",
    lm_head_path="lm_head",
    # ... 其他路径 ...
)
register_arch("YourModelConfig", config)
```

---

### 功能 4：SWD_Unif 损失函数

**原理**：Sliced Wasserstein Distance (SWD) 匹配均匀分布，自动发现最优量化范围。

**损失公式**：
```
1. 对激活值排序：x_sorted = sort(x, dim=-1)
2. 计算均匀分布范围：b = sqrt(3) * RMS(x)
3. 生成均匀分位数：target ~ Uniform[-b, b]
4. 损失 = ||x_sorted - target||^2
```

**与 Whip 损失对比**：

| 对比项 | Whip | SWD_Unif |
|--------|------|----------|
| 原理 | 指数排斥 exp(-\|x\|) | Wasserstein 匹配 |
| 分布假设 | 无 | 均匀分布 |
| 参数调优 | 少 | 无（自适应） |
| 性能 | 基线 | 更优 |

**推荐配对**：INT4 + SWD_Unif

**使用示例**：
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_unif \
    --quantizer_type int4 \
    --nsamples 128
```

---

### 功能 5：Gaussian SWD 损失 + --quantizer_type 参数

**原理**：匹配高斯分布，特别适合 NF4 假设的高斯分布。

**损失公式**：
```
1. 对激活值排序：x_sorted = sort(x, dim=-1)
2. 估计标准差：sigma = sqrt(mean(x^2))
3. 生成高斯分位数：target = Phi^{-1}((i-0.5)/n) * sigma
   其中 Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
4. 损失 = ||x_sorted - target||^2
```

**--quantizer_type 参数**（必需，无默认值）：
```bash
--quantizer_type int4    # 标准均匀量化（GPTQ/RTN）
--quantizer_type nf4     # NF4 权重量化（bitsandbytes）
```

**推荐配对**：NF4 + SWD_Gauss

**使用示例**：
```bash
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_gauss \
    --quantizer_type nf4 \
    --nsamples 128
```

**参数配对建议**（不遵循会给出警告，不会报错）：

| quantizer_type | 推荐 loss | 不推荐 |
|----------------|-----------|--------|
| int4 | whip / swd_unif | swd_gauss |
| nf4 | swd_gauss | whip / swd_unif |

---

### 功能 6：Butterfly Givens 旋转

**原理**：O(d log d) 复杂度的可学习旋转算法，相比固定 Hadamard 矩阵，可通过训练进一步改善激活分布。

**应用范围（仅 R3 和 R4）**：
- R3：Attention 头内旋转（Q, K 在 RoPE 之后）
- R4：MLP 中间层旋转（down_proj 输入）
- 不应用于 R1 / R2（保留 QR-Orth 参数化）

**算法概要**：
```
K = log2(d) 层（例如 d=64 时，K=6 层）
每层 d/2 个 Givens 旋转对
第 l 层配对：(i, i+2^l)，其中 i mod 2^(l+1) < 2^l
总复杂度：O(d log d)
```

**维度支持**：
- 2 次方维度（64, 128, 256 ...）：直接使用 Butterfly
- 非 2 次方维度（11008, 5120 ...）：自动分解，混合使用 Hadamard + Butterfly

**与固定 Hadamard 对比**：

| 对比项 | 固定 Hadamard | 可学习 Butterfly |
|--------|--------------|----------------|
| 计算复杂度 | O(d log d) | O(d log d) |
| 参数数量 | 0（固定） | K * d/2 个角度 |
| 可优化 | 否 | 是 |

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
|-- dartquant_v2/                   # [新增] 统一量化管道
|   |-- __init__.py
|   |-- run_quantize.py             # 一键执行脚本（入口）
|   |-- pipeline.py                 # 12 步完整流程
|   |-- args.py                     # 参数解析
|   |-- loss_functions.py           # whip / swd_unif / swd_gauss
|   |-- unified_model.py            # UnifiedQuantModel（架构通用化）
|   |-- nf4_quantizer.py            # NF4 量化（bitsandbytes）
|   |-- butterfly.py                # Butterfly Givens 旋转
|   `-- trainers.py                 # R1/R2/Butterfly 训练器
|
|-- DartQuant/                      # [参考] DartQuant 原始实现
|   |-- fake_quant/                 # 假量化模块（R1-R4 应用、INT4 量化）
|   |   |-- rotation_utils.py       # R1-R4 应用方法
|   |   |-- quant_utils.py          # 激活量化
|   |   |-- model_utils.py          # 模型访问器
|   |   |-- gptq_utils.py           # GPTQ/RTN 权重量化
|   |   |-- hadamard_utils.py       # Hadamard 矩阵
|   |   |-- data_utils.py           # 数据加载
|   |   `-- eval_utils.py           # 困惑度评估
|   `-- calibrater/                 # 校准器
|       |-- r1_base_qr.py           # R1 QR 分解训练
|       `-- r2_base_qr.py           # R2 QR 分解训练
|
|-- docs/                           # [文档] 研究报告
|   |-- SNLP_report_1_v1_en.md      # 第一阶段：SWD_Unif 和 QR-Orth
|   |-- report_2_en.md              # 第二阶段：Butterfly 和 Gaussian SWD
|   `-- *.md                        # 其他研究笔记
|
|-- scripts/                        # [工具脚本]
|   |-- validation.py               # 损失函数对比验证
|   `-- *.py                        # 其他辅助脚本
|
`-- requirements.txt                # Python 依赖列表
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：
- `torch >= 2.0`
- `transformers >= 4.30`
- `bitsandbytes >= 0.39`（NF4 必需）
- `numpy`, `scipy`

### 2. 基础用法（4 个常见场景）

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

#### 场景 D：带可学习 Butterfly R3/R4 的 INT4
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
| `--model` | HF 模型名 | meta-llama/Llama-3.2-1B、facebook/opt-125m 等 |
| `--loss` | whip / swd_unif / swd_gauss | 旋转训练损失函数，必需 |
| `--quantizer_type` | int4 / nf4 | 量化方法，必需 |

#### 量化配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--w_bits` | 4 | 权重位数 |
| `--a_bits` | 4 | 激活位数 |
| `--w_groupsize` | 128 | 权重分组大小 |
| `--a_groupsize` | 128 | 激活分组大小 |
| `--butterfly` | False | 启用可学习 Butterfly R3/R4 |

#### 训练配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--r1_epochs` | 10 | R1 训练轮数 |
| `--r2_epochs` | 5 | R2 训练轮数 |
| `--batch_size` | 64 | 训练批大小 |
| `--lr` | 0.001 | 学习率 |
| `--momentum` | 0.9 | SGD 动量 |

#### 数据与评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nsamples` | 128 | 校准数据样本数 |
| `--cal_dataset` | wikitext2 | 校准数据集 |
| `--ppl_eval_dataset` | wikitext2 | PPL 评估数据集 |
| `--seed` | 0 | 随机种子 |

---

## 完整量化流程（12 步）

```
步骤 1：参数解析 & 模型加载
  读取 --loss --quantizer_type --butterfly
  UnifiedQuantModel 自动检测模型架构

步骤 2：LayerNorm 融合
  将 LayerNorm 权重吸收到相邻线性层
  使旋转变换对 LayerNorm 透明（必须在旋转前执行）

步骤 3：R1 激活数据收集
  Hook Attention 输入和 MLP 输入
  收集完整训练数据

步骤 4：R1 全局旋转训练
  使用指定损失函数（Whip / SWD_Unif / SWD_Gauss）
  SGD 优化，得到 d_model x d_model 正交旋转矩阵

步骤 5：R1 离线应用
  旋转词嵌入、Attention 输入/输出、MLP 输入/输出、LM Head
  更新模型权重

步骤 6：R2 激活数据收集
  Hook Q,K,V,O 投影层（R1 应用后）
  逐层收集激活

步骤 7：R2 按头旋转训练
  为每层每个 Attention 头训练 d_head x d_head 旋转
  得到 num_layers x num_heads 个 R2 矩阵

步骤 8：R2 离线应用
  旋转 V 投影输出、O 投影输入
  更新模型权重

步骤 9：R3/R4 处理
  若启用 --butterfly：
    收集 R3（post-RoPE Q,K）和 R4（down_proj 输入）激活
    训练可学习 Butterfly Givens 旋转
    Butterfly R4：离线烘焙到权重 + 在线应用
    Butterfly R3：在线应用
  否则（默认）：
    固定 Hadamard R4：离线烘焙到权重 + 在线应用
    固定 Hadamard R3：在线应用

步骤 10：激活量化包装
  添加 ActQuantWrapper 到所有 Linear 层
  配置按层量化参数

步骤 11：权重量化
  若 --quantizer_type int4：GPTQ 或 RTN 方法
  若 --quantizer_type nf4：bitsandbytes NF4

步骤 12：困惑度评估 & 报告
  在 WikiText2 或 C4 上评估困惑度
  输出量化后模型性能
```

---

## 核心算法详解

### Whip Loss（基线损失）

**公式**：
```
L = sum(exp(-|x|)) / n
```

推动激活分布远离零点，使分布更均匀，降低量化误差。

**优点**：收敛快、实现简单。

---

### SWD_Unif 损失（均匀分布）

**公式**：
```
b      = sqrt(3) * RMS(x)
target = Uniform[-b, b] 的等间距分位数
Loss   = ||sort(x) - target||^2
```

**优势**：
- 自适应范围，无超参数
- 分布感知，性能稳定
- 特别适合 INT4 均匀量化

---

### Gaussian SWD 损失（高斯分布）

**公式**：
```
sigma  = sqrt(mean(x^2))
p_i    = (i - 0.5) / n          (每个排序后元素的概率位置)
target = Phi^{-1}(p_i) * sigma  (正态分布分位数)
Loss   = ||sort(x) - target||^2
```

其中 `Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)`（逆正态分布函数）。

**优势**：
- 与 NF4 高斯假设匹配
- 自适应标准差估计
- 分布感知的量化

---

### Butterfly Givens 旋转

**算法结构**：
```
输入：维度 d（2 的幂）
层数 K = log2(d)

第 l 层配对方式（l = 0, 1, ..., K-1）：
  stride = 2^(l+1)
  配对 (i, i + 2^l)，其中 i mod stride < 2^l

d=8 的例子（K=3）：
  层 0（stride=2）: (0,1), (2,3), (4,5), (6,7)
  层 1（stride=4）: (0,2), (1,3), (4,6), (5,7)
  层 2（stride=8）: (0,4), (1,5), (2,6), (3,7)

每对使用 Givens 旋转：
  [cos(theta)  sin(theta)]
  [-sin(theta) cos(theta)]

初始化：theta = 0（等价于单位矩阵）
训练：通过梯度下降学习最优角度 theta
```

---

## 模型支持与扩展

### 默认支持模型

| 系列 | 具体模型示例 | 备注 |
|------|------------|------|
| Llama | Llama-2-7B、Llama-3-8B、Llama-3.2-1B 等 | 所有版本自动支持 |
| OPT | OPT-125M、OPT-1.3B、OPT-30B 等 | 所有版本自动支持 |

### 如何添加新架构

**步骤 1**：查看新模型的 config 类名
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("new-model-name")
print(config.__class__.__name__)  # 如：MistralConfig
```

**步骤 2**：在代码中注册
```python
from dartquant_v2.unified_model import register_arch, ModelArchConfig

config = ModelArchConfig(
    embeddings_path="model.embed_tokens",
    layers_path="model.layers",
    lm_head_path="lm_head",
    q_proj_path="self_attn.q_proj",
    k_proj_path="self_attn.k_proj",
    v_proj_path="self_attn.v_proj",
    o_proj_path="self_attn.o_proj",
    up_proj_path="mlp.up_proj",
    down_proj_path="mlp.down_proj",
    gate_proj_path="mlp.gate_proj",
    input_ln_path="input_layernorm",
    post_attn_ln_path="post_attention_layernorm",
)
register_arch("MistralConfig", config)
```

**步骤 3**：验证
```bash
python -c "
from dartquant_v2.unified_model import UnifiedQuantModel
m = UnifiedQuantModel('your-model-name')
print(f'Loaded: {m.num_layers} layers, hidden_size={m.hidden_size}')
"
```

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

### Q3：Butterfly 对性能的影响？

通常能带来 0.1~0.3 困惑度的改善，具体取决于模型大小和数据集。

权衡：
- 优势：性能更优
- 劣势：训练时间延长约 20%

建议先用基础设置快速验证，再启用 `--butterfly` 以获得最优结果。

---

### Q4：如何处理显存不足？

按优先级调整：

1. 降低校准数据量：`--nsamples 64` 或更少
2. 减小批大小：`--batch_size 32`
3. 切换到 NF4：`--quantizer_type nf4`
4. 禁用 Butterfly：移除 `--butterfly`

---

### Q5：如何评估量化后的模型？

流程内已自动完成困惑度评估，评估结果会在最后输出：

```
Original Model PPL:   8.12  (WikiText2)
Quantized Model PPL:  8.74  (WikiText2)
```

若要在额外数据集上评估，可在参数中添加 `--ppl_eval_dataset c4`。

---

## 完整端到端示例

```bash
# INT4 + SWD_Unif，完整参数示例
python dartquant_v2/run_quantize.py \
    --model meta-llama/Llama-3.2-1B \
    --loss swd_unif \
    --quantizer_type int4 \
    --w_bits 4 --a_bits 4 \
    --w_groupsize 128 \
    --a_groupsize 128 \
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

---

## 技术细节

- **R1**：d_model x d_model 全局旋转，应用于嵌入、Attention 和 MLP
- **R2**：d_head x d_head 按头旋转，应用于 Attention 的 V 和 O
- **R3**：d_head x d_head 旋转，在 RoPE 之后在线应用于 Q, K
- **R4**：d_inter x d_inter 旋转，离线烘焙到 down_proj 权重 + 在线应用
- **QR-Orth**：通过 QR 分解参数化无约束矩阵，保证正交性

详见研究报告：
- 第一阶段：`docs/SNLP_report_1_v1_en.md`（SWD_Unif, QR-Orth）
- 第二阶段：`docs/report_2_en.md`（Butterfly, Gaussian SWD）

---

## 依赖与环境

```bash
# Python >= 3.9
pip install -r requirements.txt
```

主要依赖：

```
torch >= 2.0.0
transformers >= 4.30.0
bitsandbytes >= 0.39.0
numpy >= 1.21.0
scipy >= 1.7.0
```

---

*最后更新：2026 年 2 月*
