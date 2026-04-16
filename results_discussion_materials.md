# Results and Discussion Preparation Materials

This file is used to organise the materials, result fields, figures, open questions, and follow-up items needed when writing the `Results` and `Discussion` sections for this project.

Suggested usage:

- After receiving new experiment outputs, update the three sections on "main result materials", "result files and field collection", and "later updates / missing items" first.
- After syncing with teammates, update the section on "questions to confirm with my teammate".
- Before writing the main text, prioritise confirming that the main tables, main figures, and key caveats have all been filled in.

---

## 1. Main Result Materials Needed

### 1.1 Main Results Tables

- [ ] Table 1: AG News main results comparison table
- [ ] Table 2: IMDB main results comparison table
- [ ] The methods shown in the tables should include at least:
  - [ ] Dense baseline
  - [ ] DeeBERT
  - [ ] Router-Tuning
  - [ ] HDC-BERT
- [ ] The table should include at least these columns:
  - [ ] `Method`
  - [ ] `Main accuracy`
  - [ ] `Seed mean +/- std`
  - [ ] `Method-specific efficiency metric`
  - [ ] `Short note`

### 1.2 Recommended Main Result Fields

| Method | Main Accuracy Field | Main Efficiency Field | Notes |
|---|---|---|---|
| Dense baseline | `best_test_acc` | `inference_ms_per_sample` | Latency is only a supplementary metric, not a unified efficiency definition |
| DeeBERT | `best_ee_acc` | `best_avg_exit_layer` | Do not accidentally use `best_step1_acc` |
| Router-Tuning | `final_eval.accuracy` | `final_eval.avg_keep_rate` | `per_layer_keep_rates` can also be reported as a supplement |
| HDC-BERT | `hdc_inference.accuracy` | `hdc_inference.stage_a_exit_rate` + `hdc_inference.avg_exit_layer_a` + `hdc_inference.avg_keep_rate_b` | The main result should use full HDC inference rather than routing-only output |

### 1.3 Comparison Points to Prepare in the Main Results Section

- [ ] Accuracy comparison of all methods on AG News
- [ ] Accuracy comparison of all methods on IMDB
- [ ] Efficiency proxy comparison of all methods
- [ ] Whether HDC-BERT simultaneously shows both early exit and token routing as dynamic behaviours
- [ ] Behaviour differences across methods on long-text IMDB versus medium-length AG News

### 1.4 Seed Statistics Preparation

- [ ] Gather results for 3 seeds for each method and each dataset
- [ ] Compute the mean
- [ ] Compute the standard deviation
- [ ] Clearly mark which multi-seed results are not fully independent reruns

Placeholders:

- Final AG News main table version:
- Final IMDB main table version:
- Seed summary script / spreadsheet location:

---

## 2. Plots and Visualisations Needed

### 2.1 Core Figures

- [ ] Figure 1: main trade-off / Pareto figure
  - [ ] AG News
  - [ ] IMDB
  - [ ] If unified FLOPs cannot be provided, state in the caption that this is a method-specific efficiency proxy
- [ ] Figure 2: DeeBERT `threshold -> accuracy / avg_exit_layer`
- [ ] Figure 3: DeeBERT exit histogram
- [ ] Figure 4: Router-Tuning `target_keep_ratio -> accuracy / avg_keep_rate`
- [ ] Figure 5: Router-Tuning per-layer keep-rate visualisation
- [ ] Figure 6: HDC-BERT Stage A exit histogram
- [ ] Figure 7: HDC-BERT Stage B per-layer keep-rate visualisation

### 2.2 Figures That Can Go Into the Appendix

- [ ] Dense baseline training curves
- [ ] DeeBERT Stage 1 / Stage 2 training curves
- [ ] Router-Tuning Stage 2 training curves
- [ ] HDC-BERT Stage 2 / Stage 3 training curves
- [ ] Variability plots across seeds
- [ ] HDC split-layer ablation figure

### 2.3 Fields Suggested for the Figures

| Figure Type | Recommended Fields |
|---|---|
| Main trade-off figure | accuracy + unified FLOPs or method-specific efficiency proxy |
| DeeBERT curve | `accuracy`, `avg_exit_layer`, `exit_histogram` |
| Router curve | `accuracy`, `avg_keep_rate`, `attn_flops_ratio`, `per_layer_keep_rates` |
| HDC curve | `accuracy`, `stage_a_exit_rate`, `avg_exit_layer_a`, `avg_keep_rate_b`, `per_layer_keep_rates_b` |

### 2.4 Figures That Still Need Confirmation

- [ ] Can the HDC Pareto figure be completed?
- [ ] Can a unified FLOPs figure be added?
- [ ] Can a split-layer ablation figure be added?
- [ ] Can a wall-clock latency figure be added?

Placeholders:

- Main-text figure list:
- Appendix figure list:
- Planned owner for each figure:

---

## 3. Method-Specific Discussion Materials

### 3.1 Dense Baseline

Purpose:

- Serve as the full-compute reference
- Serve as the accuracy upper bound / reference point
- Serve as the latency reference

Need to prepare:

- [ ] `best_test_acc`
- [ ] `inference_ms_per_sample`
- [ ] Training curves
- [ ] Stability across seeds

Discussion focus:

- [ ] Use it as the reference point for all dynamic methods
- [ ] Accuracy loss or preservation relative to Dense
- [ ] Efficiency gains relative to Dense

### 3.2 DeeBERT

Purpose:

- Sample-level early-exit method
- Focus on whether it can exit earlier while preserving accuracy

Need to prepare:

- [ ] `best_ee_acc`
- [ ] `best_avg_exit_layer`
- [ ] `step2_test_acc_ee`
- [ ] `step2_avg_exit_layer`
- [ ] `exit_histogram`
- [ ] Pareto threshold sweep

Discussion focus:

- [ ] Whether exit layers differ clearly between AG News and IMDB
- [ ] Whether there is a smooth trade-off between accuracy and average exit layer
- [ ] Whether the default-threshold operating point is reasonable

### 3.3 Router-Tuning

Purpose:

- Token-level routing method
- Focus on keep rate, attention skip ratio, and layer-wise pruning patterns

Need to prepare:

- [ ] `final_eval.accuracy`
- [ ] `final_eval.avg_keep_rate`
- [ ] `final_eval.per_layer_keep_rates`
- [ ] `step2_avg_keep_rate`
- [ ] `step2_eval_keep_rates`
- [ ] Pareto keep-ratio sweep

Discussion focus:

- [ ] Relationship between `target_keep_ratio` and `actual avg_keep_rate`
- [ ] Which layers are easier to prune
- [ ] Differences in keep-rate patterns between AG News and IMDB
- [ ] Balance between accuracy and keep rate

### 3.4 HDC-BERT

Purpose:

- Core method of this project
- Focus on the story that Stage A filters easy samples first and Stage B performs token routing on hard samples

Need to prepare:

- [ ] `hdc_inference.accuracy`
- [ ] `hdc_inference.stage_a_exit_rate`
- [ ] `hdc_inference.avg_exit_layer_a`
- [ ] `hdc_inference.avg_keep_rate_b`
- [ ] `hdc_inference.exit_histogram`
- [ ] `final_routing_acc`
- [ ] `final_eval_keep_rates`
- [ ] `parameter_counts`
- [ ] `flops_estimation_data`

Discussion focus:

- [ ] Whether Stage A truly filters out enough easy samples
- [ ] Whether Stage B still shows sufficiently strong token-routing behaviour
- [ ] Whether HDC preserves the early-exit advantage relative to DeeBERT
- [ ] Whether HDC preserves the routing advantage relative to Router-Tuning
- [ ] Whether HDC offers a better accuracy-efficiency trade-off than Dense

### 3.5 Internal Behaviour Analysis Checklist

- [ ] Dense: reference only
- [ ] DeeBERT: exit histogram, avg exit layer, threshold sweep
- [ ] Router: per-layer keep rate, avg keep rate, keep-ratio sweep
- [ ] HDC: Stage A exit histogram, Stage B keep rate, split-layer sensitivity

Placeholders:

- One-sentence conclusion for each method:
- Strongest evidence figure/table for each method:

---

## 4. Exact Result Files / Fields / Outputs To Collect

### 4.1 Dense Baseline

File locations:

- [ ] [experiments/baseline/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/baseline/agnews)
- [ ] [experiments/baseline/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/baseline/imdb)

Fields to collect:

- [ ] `best_test_acc`
- [ ] `final_test_acc`
- [ ] `inference_ms_per_sample`
- [ ] `history.test_acc`
- [ ] `history.train_loss`
- [ ] `args.seed`

### 4.2 DeeBERT

File locations:

- [ ] [experiments/deebert/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/deebert/agnews)
- [ ] [experiments/deebert/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/deebert/imdb)
- [ ] [experiments_pareto/deebert](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments_pareto/deebert)

Fields to collect:

- [ ] `best_step1_acc`
- [ ] `best_ee_acc`
- [ ] `best_avg_exit_layer`
- [ ] `history_step2.step2_test_acc_ee`
- [ ] `history_step2.step2_avg_exit_layer`
- [ ] `history_step2.step2_test_acc_last`
- [ ] `args.resume_step1_ckpt`

Extra items to collect:

- [ ] `training.log`
- [ ] Pareto `pareto.json`
- [ ] `exit_histogram`

### 4.3 Router-Tuning

File locations:

- [ ] [experiments/router_tuning/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/router_tuning/agnews)
- [ ] [experiments/router_tuning/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/router_tuning/imdb)
- [ ] [experiments_pareto/routerbert](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments_pareto/routerbert)

Fields to collect:

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

Extra items to collect:

- [ ] `training.log`
- [ ] Pareto `pareto.json`

### 4.4 HDC-BERT

File locations:

- [ ] [experiments/hdc/agnews](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/hdc/agnews)
- [ ] [experiments/hdc/imdb](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments/hdc/imdb)
- [ ] [experiments_pareto/hdcbert](C:/Msc_DSML/NLP/token-level-mixture-of-depth/experiments_pareto/hdcbert) (if completed later)

Fields to collect:

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

Extra items to collect:

- [ ] `training.log`
- [ ] Pareto `pareto.json` (if rerun later)
- [ ] Split-layer ablation outputs (if rerun later)

### 4.5 General Supplementary Outputs

- [ ] All config files that were used
- [ ] All shell script versions
- [ ] Commit hash for the final experiments
- [ ] Collect CSV / Excel files if there is an offline summary table
- [ ] Record paths for any post-processing scripts

Placeholders:

- Unified results summary table location:
- Log summary location:
- Location of result packages provided by teammates:

---

## 5. Caveats and Limitations

### 5.1 Experimental-Design Caveats

- [ ] No validation split
- [ ] Checkpoint selection uses the test set
- [ ] Different methods inspect the test set a different number of times
- [ ] Multi-seed runs for dynamic methods are not fully independent
- [ ] HDC main-result selection is not perfectly aligned with the checkpoint-selection criterion

### 5.2 Metric-Level Caveats

- [ ] Efficiency metrics are not defined consistently across methods
- [ ] `avg_keep_rate` is not the total FLOPs ratio
- [ ] HDC's `avg_exit_layer_a` is not directly comparable to DeeBERT's `avg_exit_layer`
- [ ] Existing latency results are incomplete, and wall-clock timing may be rough

### 5.3 Output-Completeness Caveats

- [ ] HDC Pareto outputs currently appear to be missing
- [ ] Split-layer ablation results currently appear to be missing
- [ ] README and the current experiment implementation are not fully aligned
- [ ] Some planned experiments appear only in the experiment-design document and may not actually have been run

### 5.4 Claims That Need Care in the Final Writing

- [ ] Do not directly claim "strict Pareto superiority" unless HDC Pareto results are complete and the metric definition is unified
- [ ] Do not describe keep rate directly as "total computation reduction"
- [ ] Do not treat HDC routing-only results as the final method result
- [ ] Do not assume all multi-seed runs are fully independent

Placeholders:

- Limitations that must be stated explicitly in the main text:
- Limitations that can be explained in the appendix:

---

## 6. Questions To Ask My Teammate

### 6.1 About Result Completeness

- [ ] Has the HDC Pareto experiment already finished?
- [ ] Has the split-layer ablation already finished?
- [ ] Are there any uncommitted outputs under `experiments_split/` or `experiments_pareto/hdcbert/`?
- [ ] Is there a unified FLOPs post-processing script or result table?

### 6.2 About Result Conventions

- [ ] Should the final report use the `best checkpoint` or the `last epoch` by default?
- [ ] Is the HDC main result confirmed to use `hdc_inference.accuracy`?
- [ ] Is the Router main result confirmed to use `final_eval.accuracy`?
- [ ] Is the DeeBERT main result confirmed to use `best_ee_acc`?
- [ ] Should Dense latency appear in the main text or in the appendix?

### 6.3 About Multi-Seed Design

- [ ] Was the seed design for dynamic methods intentionally reusing the Step 1 checkpoint?
- [ ] Does this need to be stated explicitly in the report?
- [ ] Are there any supplementary fully independent reruns?

### 6.4 About Figures and Writing

- [ ] Do they already have finished figures or draft figures?
- [ ] Is there a preferred main-figure / main-table format?
- [ ] Are some analyses intended to go into the appendix?
- [ ] Which core HDC conclusion should the paper / report emphasise most strongly?

### 6.5 About Experimental Environment and Reproducibility

- [ ] What is the final commit hash being used?
- [ ] Are the final runtime environment, GPU, and total runtime recorded?
- [ ] Are there any command-line override arguments not written into the config?

Placeholders:

- Confirmed issues:
- Pending issues:
- Next sync time with teammate:

---

## 7. Later Updates and Missing Items

### 7.1 Missing Results

- [ ] HDC Pareto JSON
- [ ] Split-layer ablation results
- [ ] Unified FLOPs summary
- [ ] Final main-table mean +/- std
- [ ] First draft of the final figures

### 7.2 Later Update Log

#### Update Log

- [ ] `YYYY-MM-DD`: added / updated item
- [ ] `YYYY-MM-DD`: added / updated item
- [ ] `YYYY-MM-DD`: added / updated item

### 7.3 Final Checklist Before Writing

- [ ] Main-table fields confirmed
- [ ] Main-figure data confirmed
- [ ] Caveats explicitly listed in the Discussion section
- [ ] Result version aligned with teammate and locked
- [ ] Main-text versus appendix material split confirmed

### 7.4 Final Fill-in-the-Blank Items

One-sentence summary of the main result:

> TODO

One-sentence conclusion of the key main figure:

> TODO

Most important limitation in the Discussion:

> TODO

Core method claim of this project:

> TODO
