# Results and Discussion 准备材料

本文件用于整理本项目后续撰写报告时，`Results` 与 `Discussion` 章节所需的材料、结果字段、图表、待确认问题与后续补充项。

使用建议：

- 每次拿到新的实验输出后，先更新“主结果材料”“结果文件与字段收集”“后续更新与缺失项”三部分。
- 每次和队友同步后，更新“需要向队友确认的问题”。
- 正文写作前，优先确认主表、主图、关键 caveat 是否都已补齐。

---

## 1. Main Result Materials Needed

### 1.1 主结果表

- [ ] 表 1：AG News 主结果比较表
- [ ] 表 2：IMDB 主结果比较表
- [ ] 表中方法至少包含：
  - [ ] Dense baseline
  - [ ] DeeBERT
  - [ ] Router-Tuning
  - [ ] HDC-BERT
- [ ] 表中至少包含以下列：
  - [ ] `Method`
  - [ ] `Main accuracy`
  - [ ] `Seed mean ± std`
  - [ ] `Method-specific efficiency metric`
  - [ ] `Short note`

### 1.2 建议采用的主结果字段

| Method | 主准确率字段 | 主效率字段 | 备注 |
|---|---|---|---|
| Dense baseline | `best_test_acc` | `inference_ms_per_sample` | latency 仅作补充，非统一效率口径 |
| DeeBERT | `best_ee_acc` | `best_avg_exit_layer` | 一定不要误用 `best_step1_acc` |
| Router-Tuning | `final_eval.accuracy` | `final_eval.avg_keep_rate` | 可同时报告 `per_layer_keep_rates` 作补充 |
| HDC-BERT | `hdc_inference.accuracy` | `hdc_inference.stage_a_exit_rate` + `hdc_inference.avg_exit_layer_a` + `hdc_inference.avg_keep_rate_b` | 主结果应使用完整 HDC inference，而非 routing-only |

### 1.3 主结果章节中应准备的对比点

- [ ] 各方法在 AG News 上的准确率比较
- [ ] 各方法在 IMDB 上的准确率比较
- [ ] 各方法的效率 proxy 对比
- [ ] HDC-BERT 是否同时体现 early exit 与 token routing 的双重动态行为
- [ ] 长文本数据集 IMDB 与中等长度数据集 AG News 上，不同动态方法的行为差异

### 1.4 Seed 统计准备

- [ ] 对每个方法、每个数据集整理 3 个 seed 的结果
- [ ] 计算 mean
- [ ] 计算 std
- [ ] 明确注明哪些多 seed 结果不是 fully independent rerun

占位：

- AG News 主表最终版本：
- IMDB 主表最终版本：
- Seed 汇总脚本/表格位置：

---

## 2. Plots and Visualisations Needed

### 2.1 核心图

- [ ] 图 1：主 trade-off / Pareto 图
  - [ ] AG News
  - [ ] IMDB
  - [ ] 若无法统一 FLOPs，图注中说明这是 method-specific efficiency proxy
- [ ] 图 2：DeeBERT `threshold -> accuracy / avg_exit_layer`
- [ ] 图 3：DeeBERT exit histogram
- [ ] 图 4：Router-Tuning `target_keep_ratio -> accuracy / avg_keep_rate`
- [ ] 图 5：Router-Tuning per-layer keep rate 可视化
- [ ] 图 6：HDC-BERT Stage A exit histogram
- [ ] 图 7：HDC-BERT Stage B per-layer keep rate 可视化

### 2.2 可放附录的图

- [ ] Dense baseline 训练曲线
- [ ] DeeBERT Stage 1 / Stage 2 训练曲线
- [ ] Router-Tuning Stage 2 训练曲线
- [ ] HDC-BERT Stage 2 / Stage 3 训练曲线
- [ ] 各 seed 的波动图
- [ ] HDC split-layer ablation 图

### 2.3 图中建议展示的字段

| 图类型 | 推荐字段 |
|---|---|
| 主 trade-off 图 | accuracy + 统一 FLOPs 或 method-specific efficiency proxy |
| DeeBERT 曲线 | `accuracy`, `avg_exit_layer`, `exit_histogram` |
| Router 曲线 | `accuracy`, `avg_keep_rate`, `attn_flops_ratio`, `per_layer_keep_rates` |
| HDC 曲线 | `accuracy`, `stage_a_exit_rate`, `avg_exit_layer_a`, `avg_keep_rate_b`, `per_layer_keep_rates_b` |

### 2.4 需要后续确认的图

- [ ] HDC Pareto 图是否能补齐
- [ ] 是否能补统一 FLOPs 图
- [ ] 是否能补 split-layer ablation 图
- [ ] 是否能补 wall-clock latency 图

占位：

- 正文主图清单：
- 附录图清单：
- 计划由谁出图：

---

## 3. Method-Specific Discussion Materials

### 3.1 Dense baseline

用途：

- 作为 full-compute reference
- 作为 accuracy 上限/参考点
- 作为 latency 参考

需要准备：

- [ ] `best_test_acc`
- [ ] `inference_ms_per_sample`
- [ ] 训练曲线
- [ ] 不同 seed 的稳定性

讨论重点：

- [ ] 作为所有动态方法的 reference point
- [ ] 动态方法相对 Dense 的精度损失或保留程度
- [ ] 动态方法相对 Dense 的效率收益

### 3.2 DeeBERT

用途：

- 样本级 early exit 方法
- 重点讨论“是否能在保持准确率的同时提早退出”

需要准备：

- [ ] `best_ee_acc`
- [ ] `best_avg_exit_layer`
- [ ] `step2_test_acc_ee`
- [ ] `step2_avg_exit_layer`
- [ ] `exit_histogram`
- [ ] Pareto threshold sweep

讨论重点：

- [ ] AG News 与 IMDB 上退出层是否明显不同
- [ ] 准确率与平均退出层数之间是否存在平滑 trade-off
- [ ] 默认阈值下的 operating point 是否合理

### 3.3 Router-Tuning

用途：

- token-level routing 方法
- 重点讨论“保留率、注意力跳过比例、层级裁剪模式”

需要准备：

- [ ] `final_eval.accuracy`
- [ ] `final_eval.avg_keep_rate`
- [ ] `final_eval.per_layer_keep_rates`
- [ ] `step2_avg_keep_rate`
- [ ] `step2_eval_keep_rates`
- [ ] Pareto keep-ratio sweep

讨论重点：

- [ ] `target_keep_ratio` 与 `actual avg_keep_rate` 的关系
- [ ] 哪些层更容易被裁剪
- [ ] AG News 与 IMDB 在 keep-rate 模式上的差异
- [ ] accuracy 与 keep-rate 的平衡关系

### 3.4 HDC-BERT

用途：

- 本项目核心方法
- 重点讨论“Stage A 先筛 easy samples，Stage B 再对 hard samples 做 token routing”

需要准备：

- [ ] `hdc_inference.accuracy`
- [ ] `hdc_inference.stage_a_exit_rate`
- [ ] `hdc_inference.avg_exit_layer_a`
- [ ] `hdc_inference.avg_keep_rate_b`
- [ ] `hdc_inference.exit_histogram`
- [ ] `final_routing_acc`
- [ ] `final_eval_keep_rates`
- [ ] `parameter_counts`
- [ ] `flops_estimation_data`

讨论重点：

- [ ] Stage A 是否真的筛掉了足够多 easy samples
- [ ] Stage B 是否还保留了足够强的 token routing 行为
- [ ] 相比 DeeBERT，HDC 是否保留了 early exit 优势
- [ ] 相比 Router-Tuning，HDC 是否保留了 routing 优势
- [ ] 相比 Dense，HDC 的 accuracy-efficiency trade-off 是否更优

### 3.5 方法内部行为分析清单

- [ ] Dense：仅作为 reference
- [ ] DeeBERT：exit histogram、avg exit layer、threshold sweep
- [ ] Router：per-layer keep rate、avg keep rate、keep-ratio sweep
- [ ] HDC：Stage A exit histogram、Stage B keep rate、split-layer sensitivity

占位：

- 每个方法的关键一句话结论：
- 每个方法最强证据图/表：

---

## 4. Exact Result Files / Fields / Outputs To Collect

### 4.1 Dense baseline

文件位置：

- [ ] [experiments/baseline/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/baseline/agnews)
- [ ] [experiments/baseline/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/baseline/imdb)

应收集字段：

- [ ] `best_test_acc`
- [ ] `final_test_acc`
- [ ] `inference_ms_per_sample`
- [ ] `history.test_acc`
- [ ] `history.train_loss`
- [ ] `args.seed`

### 4.2 DeeBERT

文件位置：

- [ ] [experiments/deebert/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/deebert/agnews)
- [ ] [experiments/deebert/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/deebert/imdb)
- [ ] [experiments_pareto/deebert](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments_pareto/deebert)

应收集字段：

- [ ] `best_step1_acc`
- [ ] `best_ee_acc`
- [ ] `best_avg_exit_layer`
- [ ] `history_step2.step2_test_acc_ee`
- [ ] `history_step2.step2_avg_exit_layer`
- [ ] `history_step2.step2_test_acc_last`
- [ ] `args.resume_step1_ckpt`

额外要收集：

- [ ] `training.log`
- [ ] Pareto `pareto.json`
- [ ] `exit_histogram`

### 4.3 Router-Tuning

文件位置：

- [ ] [experiments/router_tuning/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/router_tuning/agnews)
- [ ] [experiments/router_tuning/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/router_tuning/imdb)
- [ ] [experiments_pareto/routerbert](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments_pareto/routerbert)

应收集字段：

- [ ] `final_eval.accuracy`
- [ ] `final_eval.avg_keep_rate`
- [ ] `final_eval.per_layer_keep_rates`
- [ ] `best_test_acc`
- [ ] `history_step2.step2_eval_keep_rates`
- [ ] `history_step2.step2_avg_keep_rate`
- [ ] `parameter_counts`
- [ ] `flops_estimation_data`
- [ ] `model_config.routed_layers`
- [ ] `args.resume_step1_ckpt`

额外要收集：

- [ ] `training.log`
- [ ] Pareto `pareto.json`

### 4.4 HDC-BERT

文件位置：

- [ ] [experiments/hdc/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/hdc/agnews)
- [ ] [experiments/hdc/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/hdc/imdb)
- [ ] [experiments_pareto/hdcbert](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments_pareto/hdcbert)（若后续跑完）

应收集字段：

- [ ] `hdc_inference.accuracy`
- [ ] `hdc_inference.stage_a_exit_rate`
- [ ] `hdc_inference.avg_exit_layer_a`
- [ ] `hdc_inference.avg_keep_rate_b`
- [ ] `hdc_inference.exit_histogram`
- [ ] `final_routing_acc`
- [ ] `final_eval_keep_rates`
- [ ] `final_avg_keep_rate`
- [ ] `parameter_counts`
- [ ] `flops_estimation_data`
- [ ] `model_config.split_layer`
- [ ] `args.resume_step1_ckpt`
- [ ] `args.resume_step2_ckpt`

额外要收集：

- [ ] `training.log`
- [ ] Pareto `pareto.json`（如果补跑）
- [ ] split-layer ablation 输出（如果补跑）

### 4.5 通用补充输出

- [ ] 所有使用到的 config 文件
- [ ] 所有 shell script 版本
- [ ] 最终实验对应的 commit hash
- [ ] 若有离线表格汇总，收集 CSV / Excel
- [ ] 若有后处理脚本，记录路径

占位：

- 统一结果汇总表位置：
- 日志汇总位置：
- 队友提供结果打包位置：

---

## 5. Caveats and Limitations

### 5.1 实验设计层面的 caveat

- [ ] 没有 validation split
- [ ] checkpoint selection 使用 test set
- [ ] 不同方法看 test set 的次数不同
- [ ] 动态方法的多 seed 并非 fully independent
- [ ] HDC 的主结果与 checkpoint 选择指标并不完全一致

### 5.2 指标层面的 caveat

- [ ] 不同方法的效率指标口径不统一
- [ ] `avg_keep_rate` 不是总 FLOPs 比例
- [ ] HDC 的 `avg_exit_layer_a` 不能直接和 DeeBERT 的 `avg_exit_layer` 等价比较
- [ ] 现有 latency 结果不完整，且 wall-clock timing 可能较粗糙

### 5.3 输出完整性层面的 caveat

- [ ] HDC Pareto 输出目前看起来缺失
- [ ] split-layer ablation 结果目前看起来缺失
- [ ] README 与当前实验实现存在新旧不一致
- [ ] 一些计划中的实验仅出现在实验设计文档中，未必已经跑完

### 5.4 写作时需要谨慎的表述

- [ ] 不要直接写“严格 Pareto 优于”除非 HDC Pareto 补齐且口径统一
- [ ] 不要把 keep rate 直接写成“总计算量下降比例”
- [ ] 不要把 HDC routing-only 结果当作最终方法结果
- [ ] 不要默认多 seed 完全独立

占位：

- 正文需要显式承认的 limitation：
- 可以在附录解释的 limitation：

---

## 6. Questions To Ask My Teammate

### 6.1 关于结果完整性

- [ ] HDC Pareto 实验是否已经跑完？
- [ ] split-layer ablation 是否已经跑完？
- [ ] 是否还有未提交的 `experiments_split/` 或 `experiments_pareto/hdcbert/` 输出？
- [ ] 是否有统一 FLOPs 后处理脚本或结果表？

### 6.2 关于结果口径

- [ ] 最终报告里是否默认用 `best checkpoint` 还是 `last epoch`？
- [ ] HDC 的最终主结果是否确认使用 `hdc_inference.accuracy`？
- [ ] Router 的最终主结果是否确认使用 `final_eval.accuracy`？
- [ ] DeeBERT 的最终主结果是否确认使用 `best_ee_acc`？
- [ ] Dense latency 是否需要进入主文还是附录？

### 6.3 关于多 seed 设计

- [ ] 动态方法的 seed 设计是否有意复用 Step 1 checkpoint？
- [ ] 报告中是否需要明确写出这一点？
- [ ] 是否有任何 fully independent rerun 的补充结果？

### 6.4 关于图表与写作

- [ ] 他是否已经有成品图或草图？
- [ ] 是否有偏好的主图/主表格式？
- [ ] 是否打算把某些分析放到 appendix？
- [ ] 论文/报告最想突出的是 HDC 的哪一条核心结论？

### 6.5 关于实验环境与可复现性

- [ ] 最终采用的 commit hash 是什么？
- [ ] 最终运行环境、GPU、运行时间有记录吗？
- [ ] 是否有额外未写进 config 的命令行覆盖参数？

占位：

- 已确认的问题：
- 待确认的问题：
- 与队友下次同步时间：

---

## 7. Later Updates and Missing Items

### 7.1 尚缺结果

- [ ] HDC Pareto JSON
- [ ] split-layer ablation 结果
- [ ] 统一 FLOPs 汇总
- [ ] 最终主表的 mean ± std
- [ ] 最终图表初稿

### 7.2 后续更新记录

#### Update Log

- [ ] `YYYY-MM-DD`：新增/更新内容
- [ ] `YYYY-MM-DD`：新增/更新内容
- [ ] `YYYY-MM-DD`：新增/更新内容

### 7.3 写作前最终检查

- [ ] 主表字段已最终确认
- [ ] 主图数据已最终确认
- [ ] Discussion 中 caveat 已明确列出
- [ ] 与队友达成一致的结果版本已锁定
- [ ] 正文与附录材料划分已确定

### 7.4 最终待填空内容

主结果一句话总结：

> TODO

最关键主图一句话结论：

> TODO

Discussion 里最重要的 limitation：

> TODO

本项目最核心的 method claim：

> TODO

