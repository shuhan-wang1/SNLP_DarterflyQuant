# fake_quant — 量化推理与评测模块

本目录负责 DartQuant 流程的**第二阶段**：将训练好的旋转矩阵融入模型权重，执行量化，并评测量化后模型的性能。

---

## 目录结构

```
fake_quant/
├── README.md              # 本文件
│
├── main_for_test.py       # 主入口：量化流程 + PPL/Zero-Shot 评测
├── args_config_gen.py     # 全量参数解析与日志配置
│
├── model_utils.py         # 模型加载、架构适配、层提取
├── rotation_utils.py      # LayerNorm 融合、旋转矩阵应用
├── quant_utils.py         # 激活/权重量化包装器
├── gptq_utils.py          # GPTQ / RTN 权重量化实现
├── hadamard_utils.py      # 快速 Hadamard 变换（R3/R4 在线旋转）
├── eval_utils.py          # PPL 评测
├── data_utils.py          # 数据集加载（wikitext2 / ptb / c4）
├── utils.py               # 通用工具（日志/内存管理/模型分块 I/O）
├── monkeypatch.py         # 推理兼容性补丁
│
└── Script/
    ├── dart_gptq_wxaykvz_test.sh    # 运行量化测试（不保存）
    ├── dart_gptq_wxaykvz_save.sh    # 量化后保存模型
    └── dart_gptq_wxaykvz_load.sh    # 加载已保存的量化模型
```

---

## 模块说明

### `main_for_test.py` — 主流程入口

按以下顺序执行完整的量化评测流程：

```
加载模型
   ↓
融合 LayerNorm（--fuse_norm）
   ↓
应用旋转矩阵 R1 / R2 / R3 / R4
   ↓
添加激活量化包装器（ActQuantWrapper）
   ↓
权重量化：GPTQ 或 RTN（--w_bits）
   ↓
配置激活量化（--a_bits）/ KV-Cache 量化（--k_bits / --v_bits）
   ↓
PPL 评测（--ppl_eval）/ Zero-Shot 评测（--lm_eval）
```

---

### `args_config_gen.py` — 参数配置

全部参数分为以下几组：

#### 基础参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | `meta-llama/Llama-2-7b-hf` | 模型名称或本地路径 |
| `--seed` | `0` | 随机种子 |
| `--hf_token` | `None` | HuggingFace Token（访问受限模型） |

#### 旋转矩阵参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--fuse_norm` | `True` | 融合 LayerNorm 到相邻线性层 |
| `--use_r1` | `True` | 启用 R1（Q/K/V + up/gate-proj 输入旋转） |
| `--r1_path` | `None` | R1 矩阵路径（`.pt` 文件） |
| `--use_r2` | `offline` | 启用 R2：`offline`（离线融合）/ `online`（在线 Hadamard）/ `none` |
| `--r2_path` | `None` | R2 矩阵路径（`.pt` 文件） |
| `--use_r3` | `True` | 启用 R3（K-cache 在线 Hadamard 旋转） |
| `--use_r4` | `True` | 启用 R4（down-proj 输入在线 Hadamard 旋转） |
| `--rotate_mode` | `hadamard` | 未指定 R1/R2 路径时的初始化方式（`hadamard` / `random`） |
| `--fp32_had` | `False` | Hadamard 变换是否使用 FP32 精度 |

#### 激活量化参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--a_bits` | `16` | 激活值量化位宽（`16` 表示不量化） |
| `--a_groupsize` | `-1` | 分组大小（`-1` 表示逐通道） |
| `--a_asym` | `False` | 非对称量化 |
| `--a_clip_ratio` | `1.0` | 激活值裁剪比例 |

#### 权重量化参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--w_bits` | `16` | 权重量化位宽（`16` 表示不量化） |
| `--w_groupsize` | `-1` | 分组大小（`-1` 表示逐通道） |
| `--w_rtn` | `False` | 使用 RTN（默认为 GPTQ） |
| `--w_clip` | `False` | 权重量化裁剪 |
| `--w_asym` | `False` | 非对称量化 |
| `--nsamples` | `128` | GPTQ 校准样本数 |
| `--cal_dataset` | `wikitext2` | GPTQ 校准数据集 |

#### KV-Cache 量化参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--v_bits` | `16` | V-cache 量化位宽 |
| `--v_groupsize` | `-1` | V-cache 分组大小 |
| `--k_bits` | `16` | K-cache 量化位宽（配合 R3 使用） |
| `--k_groupsize` | `-1` | K-cache 分组大小 |

#### 模型存取参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--save_qmodel_path` | `None` | 量化模型保存目录 |
| `--load_qmodel_path` | `None` | 已保存的量化模型加载路径 |

#### 评测参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--ppl_eval` | `False` | 启用 PPL 评测 |
| `--ppl_eval_dataset` | `[wikitext2, ptb, c4]` | PPL 评测数据集 |
| `--lm_eval` | `False` | 启用 Zero-Shot 评测（lm-evaluation-harness） |
| `--tasks` | `[piqa, hellaswag, ...]` | Zero-Shot 评测任务列表 |
| `--distribute` | `False` | 多 GPU 推理 |

---

### `model_utils.py` — 模型工具

| 函数 | 说明 |
|------|------|
| `get_model()` | 统一入口，按模型名称加载 LLaMA 或 OPT |
| `get_layers()` | 获取所有 Transformer 解码层列表 |
| `get_embeddings()` | 获取嵌入层列表 |
| `get_lm_head()` | 获取语言模型输出头 |
| `replace_modules()` | DFS 递归替换指定类型的子模块 |
| `capture_layer_io()` | Hook 方式捕获指定层的输入/输出 |

---

### `rotation_utils.py` — 旋转变换工具

| 函数 | 说明 |
|------|------|
| `fuse_layer_norms()` | 将 LayerNorm 参数融入相邻线性层权重 |
| `rotate_model()` | 按配置应用 R1 / R2 / R3 / R4 |
| `rotate_embeddings()` | 将 R1 应用于 Embedding 层 |
| `rotate_attention_inputs()` | 将 R1 应用于 Q/K/V 投影权重 |
| `rotate_attention_output()` | 将 R2 应用于 O-proj 权重 |
| `rotate_mlp_input()` | 将 R1 应用于 gate/up-proj 权重 |
| `rotate_mlp_output()` | 将 R1 应用于 down-proj 权重 |

---

### `quant_utils.py` — 量化包装器

| 类/函数 | 说明 |
|--------|------|
| `ActQuantizer` | 激活值量化器，支持对称/非对称、逐 token/逐通道 |
| `ActQuantWrapper` | 包装线性层，在前向传播中量化输入/输出 |
| `WQuantizer` | 权重量化器 |
| `add_actquant()` | 将所有线性层替换为 `ActQuantWrapper` |
| `find_qlayers()` | 查找模型中所有量化包装层 |

---

### `gptq_utils.py` — 权重量化实现

| 函数 | 说明 |
|------|------|
| `gptq_fwrd()` | GPTQ：基于 Hessian 矩阵的逐层权重量化 |
| `rtn_fwrd()` | RTN：简单逐元素取整量化（基线方案） |

---

### `hadamard_utils.py` — Hadamard 变换

提供高效的快速 Hadamard 变换，用于 R3（K-cache）和 R4（down-proj）的在线旋转。

| 函数 | 说明 |
|------|------|
| `get_hadK()` | 获取指定维度的 Hadamard 矩阵 |
| `random_hadamard_matrix()` | 生成随机 Hadamard 矩阵 |

---

### `utils.py` — 通用工具

| 函数 | 说明 |
|------|------|
| `cleanup_memory()` | 触发 GC 并清空 GPU 显存缓存 |
| `distribute_model()` | 多 GPU 自动均衡分配模型 |
| `save_model_in_parts()` | 按文件大小分块保存量化模型 |
| `load_model_in_parts()` | 逐块加载分片量化模型 |
| `config_logging()` | 配置同时输出到文件和控制台的日志 |

---

## 使用脚本

### `Script/dart_gptq_wxaykvz_test.sh`

对权重（W）、激活（A）、K-cache 和 V-cache 同时执行量化测试，不保存模型。

```bash
bash Script/dart_gptq_wxaykvz_test.sh \
    <GPU_ID> <MODEL> <W_BITS> <A_BITS> <KV_BITS> <R2_PATH> <R1_PATH>

# 示例：
bash Script/dart_gptq_wxaykvz_test.sh \
    0 meta-llama/Llama-2-7b-hf 4 8 8 \
    /path/to/r2 /path/to/r1
```

### `Script/dart_gptq_wxaykvz_save.sh`

量化后将模型权重保存到磁盘（分块保存，每块约 10GB）。

### `Script/dart_gptq_wxaykvz_load.sh`

加载已保存的量化模型，直接执行评测，跳过 GPTQ 量化步骤。

---

## 典型评测配置

```bash
# W4A8 量化 + PPL 评测
python main_for_test.py \
    --model meta-llama/Llama-2-7b-hf \
    --r1_path /path/to/r1.pt \
    --r2_path /path/to/r2.pt \
    --use_r1 --use_r2 offline --use_r3 --use_r4 \
    --w_bits 4 --w_groupsize 128 --w_clip \
    --a_bits 8 --a_groupsize 128 \
    --k_bits 8 --v_bits 8 \
    --ppl_eval --ppl_eval_dataset wikitext2 ptb c4 \
    --lm_eval
```
