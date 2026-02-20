# tests — 端到端测试脚本

本目录包含三个独立的端到端测试脚本，用于验证量化流程的正确性和评测量化模型的性能。

---

## 脚本说明

### `baseline_ppl_test.py` — FP16 基线评测

在 FP16 精度下评测原始模型的困惑度（PPL），用于建立性能基准。

提供三种评测方式：
- **滑动窗口 PPL**：标准学术评测方法（推荐）
- **顺序 PPL**：非重叠分段计算
- **手动 PPL**：用于调试验证

```bash
python tests/baseline_ppl_test.py \
    --model meta-llama/Llama-2-7b-hf
# 预期结果：LLaMA-2-7B 在 Wikitext-2 上约 5.47
```

---

### `quick_test.py` — 全流程集成测试（Whip Loss）

在 GPU 内存中完整执行 DartQuant 流程，无需将中间结果写入磁盘。

**流程：**
1. 加载模型
2. 融合 LayerNorm
3. 采集激活值数据（内存中）
4. 训练 R1（Whip Loss）
5. 逐层训练 R2（Whip Loss）
6. 应用旋转矩阵
7. GPTQ 权重量化
8. 配置激活量化 / KV-Cache 量化
9. PPL 评测 / Zero-Shot 评测

```bash
python tests/quick_test.py \
    --model meta-llama/Llama-2-7b-hf \
    --w_bits 4 --a_bits 8 \
    --ppl_eval
```

---

### `comparative_test.py` — SWD Loss 对比测试

与 `quick_test.py` 流程完全相同，但使用 **SWD（Sliced Wasserstein Distance）Loss** 替代 Whip Loss，用于对比两种损失函数的量化效果。

**SWD Loss：** 将激活值分布匹配到均匀分布 Uniform[-b, b]，其中 `b = sqrt(3) * RMS(x)`

```bash
python tests/comparative_test.py \
    --model meta-llama/Llama-2-7b-hf \
    --w_bits 4 --a_bits 8 \
    --ppl_eval
```

---

## 适用场景

| 脚本 | 适用场景 |
|------|---------|
| `baseline_ppl_test.py` | 验证评测方法是否正确，建立 FP16 基准 |
| `quick_test.py` | 快速验证完整流程，开发调试 |
| `comparative_test.py` | 损失函数对比实验 |
