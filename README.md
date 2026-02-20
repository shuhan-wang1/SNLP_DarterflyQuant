# JointQuant: INT4 Quantization with Joint R1 Rotation Optimization

> UCL SNLP Project — LLM Weight-Activation W4A4 Quantization Research

---

## 项目概述

本项目在 **DartQuant** 基础上提出 **JointQuant**，核心创新在于将 R1 旋转矩阵的训练从逐层贪心优化改为跨层联合优化，以在 INT4（W4A4）量化时获得更低的困惑度（Perplexity）损失。

### 核心对比

| 方法 | R1 优化策略 | 特点 |
|------|------------|------|
| **DartQuant**（基线） | 逐层独立贪心 | 各层分别最优，全局不一定最优 |
| **JointQuant**（本项目） | 单个全局 R1 联合优化 | 所有层共享同一旋转，全局最优 |

---

## 项目结构

```
int4_quantization_darkquant/
│
├── main.py                    # [入口] JointQuant 主流程（推荐使用）
├── full_experiment.py         # [入口] 对比实验：Independent vs Joint R1
│
├── joint_quant/               # [核心模块] JointQuant Python 包
│   ├── __init__.py            #   统一对外导出接口
│   ├── config.py              #   配置管理（JointQuantConfig + 模型预设）
│   ├── utils.py               #   基础工具（设备、随机种子、日志）
│   ├── model_utils.py         #   模型加载、层提取、RMSN
│   ├── hadamard_utils.py      #   Hadamard / 正交矩阵生成
│   ├── quant_utils.py         #   量化实现（WeightQuantizer、ActQuantizer、QuantizedLinear）
│   ├── rotation_utils.py      #   旋转应用 + LayerNorm 融合
│   ├── smooth_utils.py        #   SmoothQuant 缩放因子计算
│   ├── data_utils.py          #   数据加载（WikiText2 / C4 / PTB）
│   ├── eval_utils.py          #   困惑度评估
│   ├── joint_training.py      #   [核心创新] 联合 R1 / 独立 R2 训练
│   └── README.md              #   模块文档
│
├── scripts/                   # [工具脚本] 实验、分析、辅助脚本
│   ├── validation.py          #   DartQuant 基线验证（Whip vs SWD 损失对比）
│   ├── validation_joint.py    #   JointQuant 联合旋转可视化实验
│   ├── stat_and_download.py   #   激活值分布分析 + 模型下载
│   ├── clean_cache.py         #   清理 Python 缓存
│   ├── test_imports.py        #   模块导入测试
│   └── draw.py                #   物理轨迹可视化（独立脚本）
│
├── DartQuant/                 # [参考代码] DartQuant 原始实现（独立 git 仓库）
│   ├── fake_quant/            #   假量化模块（原始 DartQuant）
│   ├── calibrater/            #   校准器（R1/R2 QR 分解方法）
│   ├── tests/                 #   测试套件
│   └── NPU_DartQuant/         #   NPU 专用变体
│
├── docs/                      # [文档] 研究报告与笔记
│   ├── notes_dartquant.md     #   DartQuant 研究笔记
│   ├── summary_dartquant.md   #   DartQuant 方法总结
│   ├── report_2_cn.md         #   中文研究报告（第二阶段）
│   ├── report_2_en.md         #   英文研究报告（第二阶段）
│   └── *.pdf                  #   各阶段报告 PDF 版本
│
├── image/                     # [资料] 研究截图与图表
│   ├── notes_darkquant/       #   DartQuant 笔记图片
│   └── summary_of_current_finding/  #   当前发现总结图
│
└── requirements.txt           # Python 依赖列表
```

---

## 分层架构说明

```
┌─────────────────────────────────────────────────────────┐
│                      入口层 (Entry)                      │
│              main.py  │  full_experiment.py              │
└────────────────────────┬────────────────────────────────┘
                         │ 调用
┌────────────────────────▼────────────────────────────────┐
│                    核心模块层 (joint_quant/)              │
│                                                          │
│  config.py      → 超参数配置                             │
│  model_utils.py → 模型加载/层访问                        │
│  data_utils.py  → 校准数据                               │
│  joint_training.py → [创新] 联合 R1 + R2 优化           │
│  rotation_utils.py → 旋转应用 + LayerNorm 融合           │
│  smooth_utils.py   → SmoothQuant 缩放                   │
│  quant_utils.py    → INT4 量化实现                       │
│  eval_utils.py     → 困惑度评估                          │
│  hadamard_utils.py → 正交矩阵初始化                      │
│  utils.py          → 基础工具                            │
└────────────────────────┬────────────────────────────────┘
                         │ 对比/参考
┌────────────────────────▼────────────────────────────────┐
│                  参考实现层 (DartQuant/)                  │
│   fake_quant/  │  calibrater/  │  NPU_DartQuant/         │
└─────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行 JointQuant（推荐）

```bash
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --w_bits 4 \
    --a_bits 4 \
    --nsamples 128 \
    --r1_epochs 10
```

### 运行对比实验（Independent vs Joint）

```bash
python full_experiment.py \
    --model meta-llama/Llama-3.2-1B \
    --cache_dir /path/to/cache
```

---

## 核心算法

### 1. Whip Loss（旋转优化目标）

```
L = Σ exp(-|x|)
```
通过排斥激活值趋向零，推动激活分布更均匀，从而降低量化误差。

### 2. LayerNorm 融合（旋转的前提）

```
[激活] → [LayerNorm] → [Linear]
                   ↓ 融合后
[激活] → [Linear']   (LayerNorm 权重已被吸收)
```
使旋转变换对 LayerNorm 透明，**必须在旋转前调用**。

### 3. SmoothQuant（激活-权重难度迁移）

```
Y = (X · s⁻¹) @ (s · W)
s_j = max|X_j|^α / max|W_j|^(1-α)    α = 0.5
```

### 4. 联合 vs 独立 R1

```
DartQuant（独立）: for each layer_i → optimize R1_i minimizing loss(layer_i)
JointQuant（联合）: optimize single R1 minimizing Σ loss(layer_i) for all layers
```

---

## 实验状态

| Branch | 状态 | 说明 |
|--------|------|------|
| `main` | 稳定 | 原始基线版本 |
| `siph_method` | 已放弃 | SIPH 方法实验，结果随机 |
| `accelerate_quantization` | 探索中 | 量化加速优化 |

---

## 预期性能（W4A4）

| 模型 | FP16 基线 PPL | JointQuant PPL |
|------|--------------|----------------|
| Llama-3.2-1B | ~8.5 | ~10-12 |
| Llama-3-8B | ~6.5 | ~7.5-9 |
