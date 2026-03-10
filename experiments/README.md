# HDC-BERT 实验设计方案 (Conference Paper)

## 1. 项目概述

我们实现了四种 BERT 动态计算方法用于文本分类：

| 方法 | 描述 | AG News 结果 |
|------|------|-------------|
| **DenseBERT** | 标准 BERT 微调 baseline | 94.76% |
| **DeeBERT** | Sample-level ACT，基于熵的 early-exit | 94.33%, avg exit layer 4.49/12 |
| **Router-Tuning** | Token-level ACT，每层 router 决定 token 是否跳过 attention | 93.61%, avg keep rate ~0.64 |
| **HDC-BERT** (ours) | 层次化组合 — Stage A (layers 0-5) early-exit + Stage B (layers 6-11) token routing | TBD |

**核心主张：** 层次化组合 sample-level 和 token-level 自适应计算在 accuracy-efficiency tradeoff 上优于单独使用任一方法。

**计算预算：** 1-2 GPUs, ~1 week → 主实验 3 seeds，sweeps 适度精简。

---

## 2. 数据集

| Dataset | Classes | Avg Length | Train / Test | 选择理由 |
|---------|---------|-----------|-------------|---------|
| AG News | 4 | ~38 tokens | 120K / 7.6K | 已实现，中等长度，话题分类 |
| SST-2 | 2 | ~19 tokens | 67K / 872 (val) | 短文本，GLUE benchmark，情感分类 |
| IMDB | 2 | ~230 tokens | 25K / 25K | 长文档，充分展示 token routing 优势 |

三个数据集形成有意义的对比：
- **短 / 中 / 长**文本
- **情感 / 话题**分类
- **2 / 4** 类别数

> **注意：** SST-2 的 test labels 不公开，使用 validation set 作为 test set。IMDB 使用 max_length=256。

---

## 3. 实验列表

### Experiment 1: Main Results Table（核心实验）

**目标：** 在 3 个数据集上对比所有方法的 accuracy 和 efficiency。

#### 表格格式（每个数据集一张表）

| Method | Acc (%) ↑ | FLOPs (G) ↓ | FLOPs Ratio (%) ↓ | Avg Exit Layer | Avg Keep Rate |
|--------|-----------|-------------|-------------------|----------------|---------------|
| DenseBERT | x.xx ± x.xx | x.xx | 100% | 12.0 | 1.00 |
| DeeBERT | x.xx ± x.xx | x.xx | xx% | x.xx | 1.00 |
| Router-Tuning | x.xx ± x.xx | x.xx | xx% | 12.0 | x.xx |
| **HDC-BERT** | x.xx ± x.xx | x.xx | xx% | x.xx | x.xx |

#### 实验配置
- **Seeds:** 42, 123, 2024（report mean ± std）
- **固定超参：**
  - Model: `bert-base-uncased`
  - Batch size: 32
  - Stage 1: LR=2e-5, epochs=3
  - Stage 2/3: LR=1e-3, epochs=5
  - DeeBERT/HDC: entropy_threshold=0.2
  - Router/HDC: target_keep_ratio=0.7, lambda_mod=1e-3, tau=0.5
  - HDC: split_layer=6
- **IMDB 特别设置：** max_length=256
- **总训练量：** 3 datasets × 4 methods × 3 seeds = **36 runs**

#### 预期结论
- AG News / SST-2（短文本）：DeeBERT 效率优势明显（大量样本可在前几层退出）
- IMDB（长文本）：Router-Tuning 效率优势明显（更多 token 可裁剪，attention 复杂度随序列长度二次增长）
- **HDC-BERT 在所有数据集上取得最佳 accuracy-FLOPs 平衡**

---

### Experiment 2: Accuracy-Efficiency Pareto Curves（最重要的图）

**目标：** 可视化证明 HDC-BERT 占据 Pareto 前沿。

#### 如何生成各方法的多个数据点

| Method | Sweep 变量 | 取值 | 需要重训? |
|--------|-----------|------|---------|
| DeeBERT | entropy_threshold | {0.05, 0.1, 0.2, 0.3, 0.5} | 否（推理时调） |
| Router-Tuning | target_keep_ratio | {0.4, 0.6, 0.7, 0.8} | 是（仅重训 Stage 2） |
| HDC-BERT | entropy_threshold | {0.05, 0.1, 0.2, 0.3, 0.5} | 否（推理时调） |
| HDC-BERT | target_keep_ratio | {0.5, 0.7, 0.9} | 是（仅重训 Stage 3） |

#### 计算量估算
- DeeBERT entropy sweep：用同一个训好的模型，换阈值推理 → **几乎免费**
- HDC entropy sweep：同上 → **几乎免费**
- Router-Tuning keep_ratio sweep：需 4 次 Stage 2 训练 → ~4h
- HDC keep_ratio sweep：需 3 次 Stage 3 训练 → ~3h

#### 图表规格
- X 轴：FLOPs (% of DenseBERT)
- Y 轴：Test Accuracy (%)
- 3 个子图（每个数据集一个），或横排 1×3 subplot
- DenseBERT 作为参考点 (100%, max_acc)
- 每个方法一条线 + marker，不同颜色
- **关键：HDC 的曲线应在左上方（高 accuracy + 低 FLOPs）**

---

### Experiment 3: FLOPs 计算（必须先实现）

**目标：** 提供硬件无关的效率度量。

#### 解析 FLOPs 公式

每个 BERT layer（seq_len=L, H=768, I=3072）：

**Attention FLOPs:**
```
QKV projection:  3 × L × H × H
Attention scores: L × L × H
Context vectors:  L × L × H
Output projection: L × H × H
合计: 4·L·H² + 2·L²·H
```

**FFN FLOPs:**
```
FFN layer 1: L × H × I
FFN layer 2: L × I × H
合计: 2·L·H·I
```

#### 各方法 FLOPs 计算方式

| Method | 计算规则 |
|--------|---------|
| DenseBERT | 12 × full_layer_flops(L) |
| DeeBERT | exit_layer × full_layer_flops(L)，对所有样本取平均 |
| Router-Tuning | Σᵢ₌₁¹² [attn_flops(L_kept_i) + ffn_flops(L)]，L_kept_i = keep_rate_i × L |
| HDC-BERT | Stage A 退出：k × full_layer_flops(L)；进入 Stage B：split_layer × full + Σⱼ routed_layer_flops |

> **重要：** Router-Tuning / HDC Stage B 中，FFN 对**所有** token 执行（代码 `_run_post_attention` 无条件执行），仅 attention 部分按 keep_rate 缩减。

#### 额外度量
- **Wall-clock 推理时间：** 使用 `torch.cuda.synchronize()` + `time.time()` 计时，取 3 次平均
- **报告 speedup ratio** = DenseBERT_time / method_time

#### 实现位置
- 新文件 `experiments/flops.py`

---

### Experiment 4: Ablation Studies

#### 4a: Component Ablation（必做）

**目标：** 证明 Stage A (early exit) 和 Stage B (token routing) 缺一不可。

| Config | Early Exit | Token Routing | 说明 |
|--------|-----------|--------------|------|
| DenseBERT | ✗ | ✗ | Full computation baseline |
| DeeBERT | ✓ (all 12 layers) | ✗ | 纯 sample-level ACT |
| Router-only | ✗ | ✓ (all 12 layers) | 纯 token-level ACT |
| **HDC (ours)** | ✓ (layers 0-5) | ✓ (layers 6-11) | 层次化组合 |

- 在 AG News 上运行，seed=42
- Report: accuracy + FLOPs + avg exit layer + avg keep rate

**预期结论：**
- HDC 的 FLOPs 低于 DeeBERT（因为 Stage B 进一步减少计算）
- HDC 的 FLOPs 低于 Router-only（因为 Stage A 让简单样本直接退出）
- HDC 的 accuracy 接近或超过两者

#### 4b: Split Layer 消融

**目标：** 验证 split_layer=6 是合理的选择。

| split_layer | Stage A 层数 | Stage B 层数 | 含义 |
|-------------|-------------|-------------|------|
| 2 | 2 layers | 10 layers | 极少 early exit 机会 |
| 4 | 4 layers | 8 layers | 偏向 token routing |
| **6** | 6 layers | 6 layers | 均衡分割（默认） |
| 8 | 8 layers | 4 layers | 偏向 early exit |
| 10 | 10 layers | 2 layers | 极少 routing 机会 |

- 固定：entropy_threshold=0.2, target_keep_ratio=0.7
- 每个 split_layer 需要完整 3-stage 训练
- 画双 Y 轴折线图：Accuracy (左) + FLOPs (右) vs split_layer

**预期：** 倒 U 型曲线。split_layer 太小 → early exit rate 低，Stage A 没发挥作用；太大 → Stage B 只有几层，token routing 效果差。

#### 4c: 训练策略消融（可选，放 appendix）

**目标：** 验证 3-stage 顺序训练的必要性。

| Config | 训练方式 | 说明 |
|--------|---------|------|
| Joint | 同时训 off-ramps + routers | 梯度冲突 |
| **3-stage (proposed)** | fine-tune → off-ramps → routers | 逐步冻结 |
| 3-stage (reversed) | fine-tune → routers → off-ramps | 颠倒 Stage 2/3 顺序 |

- AG News, seed=42
- 比较：最终 accuracy + 收敛曲线
- **优先级低**：时间紧张可省略

---

### Experiment 5: Hyperparameter Sensitivity

**目标：** 展示 HDC-BERT 对关键超参的敏感度。

在 HDC-BERT 上做单变量 sweep，固定其他超参为默认值。

| 超参 | Sweep 值 | 观察指标 | 需要重训? |
|------|---------|---------|---------|
| entropy_threshold | {0.05, 0.1, 0.2, 0.3, 0.5} | Acc, Stage A exit rate, FLOPs | 否 |
| target_keep_ratio | {0.4, 0.6, 0.7, 0.8} | Acc, actual keep rate, FLOPs | 是 (Stage 3) |
| lambda_mod | {0, 1e-4, 1e-3, 1e-2} | Acc, keep rate | 是 (Stage 3) |

#### 呈现方式
- 每个超参一个子图
- 双 Y 轴折线图：左轴 Accuracy，右轴效率指标

**注意：** entropy_threshold 的 sweep 与 Experiment 2 的 Pareto curve 共享数据，不需额外实验。

---

### Experiment 6: Analysis & Visualization

#### 6a: Exit Layer Distribution（柱状图）

- **内容：** DeeBERT vs HDC Stage A 的退出层分布对比
- **形式：** 并排柱状图（grouped bar），X=layer, Y=exit fraction
- **数据来源：** `evaluate_hdc_inference` 返回的 exit_histogram
- **意义：** 展示 HDC 的 Stage A 如何筛选"简单"样本

#### 6b: Per-Layer Keep Rate 可视化

- **内容：** HDC Stage B 各层（6-11）的平均 keep rate
- **拓展：** 按类别分组（AG News 4 类），看不同类别的 routing pattern 差异
- **形式：** 热力图（X=layer, Y=class, color=keep_rate）或分组柱状图
- **意义：** 展示 router 是否学到了有意义的 pattern（如某些类别更容易裁剪）

#### 6c: Token Routing Case Study（2-3 个例子，可选）

- **内容：** 选取典型样本，可视化 Stage B 各层哪些 token 被保留 / 裁剪
- **形式：** 网格图（行=layers 6-11，列=tokens，颜色=kept(绿) / pruned(红)）
- **实现：** 需要 `forward_hdc_inference_detailed()` 返回 per-layer masks
- **意义：** 提供可解释性，展示 router 学到了什么（预期：padding → 停用词 → 内容相关裁剪）

#### 6d: Easy vs Hard 样本分析（可选）

- **内容：** Stage A 各层退出的样本 accuracy vs 进入 Stage B 的样本 accuracy
- **形式：** 分组柱状图或表格
- **意义：** 验证 entropy threshold 正确分离了"简单" vs "困难"样本

---

## 4. 执行优先级

| Priority | 实验 | 预估时间 | 必要性 |
|----------|------|---------|--------|
| **P0** | 实现 `experiments/flops.py` | 0.5 day | 必须 |
| **P0** | `data.py` 多数据集支持 | 0.5 day | 必须 |
| **P0** | Exp 1: Main Results (36 runs) | 2-3 days | 必须 |
| **P1** | Exp 2: Pareto Curves | 1 day | 强烈建议 |
| **P1** | Exp 4a: Component Ablation | 已有数据 + 少量补充 | 强烈建议 |
| **P1** | Exp 4b: Split Layer Ablation | 0.5 day | 强烈建议 |
| **P2** | Exp 5: Hyperparameter Sensitivity | 0.5 day | 建议 |
| **P2** | Exp 6a-b: Distributions & Heatmaps | 0.5 day | 建议 |
| **P3** | Exp 4c: Training Pipeline Ablation | 0.5 day | 可选 (appendix) |
| **P3** | Exp 6c-d: Case Study & Easy/Hard | 0.5 day | 可选 |

---

## 5. 需要实现的代码变更

### experiments/ 模块新文件
| 文件 | 用途 |
|------|------|
| `experiments/__init__.py` | 模块初始化 |
| `experiments/flops.py` | 解析 FLOPs 计算（各方法） |
| `experiments/visualize.py` | 所有画图函数 |

### 需修改的已有文件
| 文件 | 变更 |
|------|------|
| `data.py` | 添加 `load_dataset_by_name()` 支持 ag_news / sst2 / imdb |
| `eval.py` | 集成 FLOPs 跟踪 + wall-clock timing |
| `models/hdc_bert.py` | 添加 `forward_hdc_inference_detailed()` 返回 per-layer masks |
| `main_*.py` | 添加 `--dataset` 参数 |
| `configs/` | 每个数据集的配置文件 |

---

## 6. 论文实验章节结构建议

```
5. Experiments
   5.1 Experimental Setup
       - Datasets: AG News, SST-2, IMDB（表格描述）
       - Implementation Details: BERT-base, AdamW, 3-stage training, hyperparams
       - Baselines: DenseBERT, DeeBERT, Router-Tuning BERT
       - Metrics: Accuracy, FLOPs, Speedup

   5.2 Main Results (Table 1-3)
       - HDC-BERT achieves best accuracy-efficiency tradeoff across all datasets
       - Token routing 在长文本 (IMDB) 上效果更显著
       - Early exit 在短文本 (SST-2) 上效率更高
       - HDC 兼顾两者优势

   5.3 Accuracy-Efficiency Tradeoff (Figure 1)
       - Pareto curves 证明 HDC 占据前沿

   5.4 Ablation Studies (Table 4 + Figure 2)
       - Component ablation: Stage A + Stage B 缺一不可
       - Split layer 分析: 中间分割点最优

   5.5 Analysis (Figure 3-5)
       - Exit layer distribution
       - Per-layer routing patterns
       - Token-level case study (可选)
```

---

## 7. Reviewer 可能的质疑与对应实验

| Reviewer 质疑 | 对应实验 |
|--------------|---------|
| 只在一个数据集上实验 | Exp 1: 三个数据集 |
| 没有 FLOPs，只有 proxy 指标 | Exp 3: 解析 FLOPs 计算 |
| 组合真的比单独的好吗？ | Exp 4a: Component Ablation |
| 为什么在第 6 层分割？ | Exp 4b: Split Layer 消融 |
| 训练策略的选择有依据吗？ | Exp 4c: Training Pipeline 消融 |
| 模型学到了什么？ | Exp 6c: Token Routing Case Study |
| 结果是否稳定可复现？ | Exp 1: 3 seeds + mean ± std |
| 长文档上效果如何？ | Exp 1: IMDB 实验 |
| 超参怎么选的？ | Exp 5: Sensitivity Analysis |
