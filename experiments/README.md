# HDC-BERT Experiment Design Plan (Conference Paper)

## 1. Project Overview

We implement four dynamic BERT computation methods for text classification:

| Method | Description | AG News Result |
|------|------|-------------|
| **DenseBERT** | Standard fine-tuned BERT baseline | 94.76% |
| **DeeBERT** | Sample-level ACT with entropy-based early exit | 94.33%, avg exit layer 4.49/12 |
| **Router-Tuning** | Token-level ACT where each layer's router decides whether a token skips attention | 93.61%, avg keep rate ~0.64 |
| **HDC-BERT** (ours) | Hierarchical combination: Stage A (layers 0-5) early exit + Stage B (layers 6-11) token routing | TBD |

**Core claim:** Hierarchically combining sample-level and token-level adaptive computation provides a better accuracy-efficiency trade-off than using either method alone.

**Compute budget:** 1-2 GPUs, about 1 week. The main experiments use 3 seeds, while sweeps are kept reasonably compact.

---

## 2. Datasets

| Dataset | Classes | Avg Length | Train / Test | Rationale |
|---------|---------|-----------|-------------|---------|
| AG News | 4 | ~38 tokens | 120K / 7.6K | Already implemented, medium-length topic classification |
| SST-2 | 2 | ~19 tokens | 67K / 872 (val) | Short text, GLUE benchmark, sentiment classification |
| IMDB | 2 | ~230 tokens | 25K / 25K | Long documents, clearly highlights the benefit of token routing |

These three datasets create a meaningful contrast:
- **Short / medium / long** texts
- **Sentiment / topic** classification
- **2 / 4** classes

> **Note:** SST-2 test labels are not public, so the validation split is used as the test set. IMDB uses `max_length=256`.

---

## 3. Experiment List

### Experiment 1: Main Results Table (Core Experiment)

**Goal:** Compare the accuracy and efficiency of all methods on three datasets.

#### Table format (one table per dataset)

| Method | Acc (%) up | FLOPs (G) down | FLOPs Ratio (%) down | Avg Exit Layer | Avg Keep Rate |
|--------|-----------|-------------|-------------------|----------------|---------------|
| DenseBERT | x.xx +/- x.xx | x.xx | 100% | 12.0 | 1.00 |
| DeeBERT | x.xx +/- x.xx | x.xx | xx% | x.xx | 1.00 |
| Router-Tuning | x.xx +/- x.xx | x.xx | xx% | 12.0 | x.xx |
| **HDC-BERT** | x.xx +/- x.xx | x.xx | xx% | x.xx | x.xx |

#### Experimental setup

- **Seeds:** 42, 123, 2024 (report mean +/- std)
- **Fixed hyperparameters:**
  - Model: `bert-base-uncased`
  - Batch size: 32
  - Stage 1: LR=2e-5, epochs=3
  - Stage 2/3: LR=1e-3, epochs=5
  - DeeBERT/HDC: `entropy_threshold=0.2`
  - Router/HDC: `target_keep_ratio=0.7`, `lambda_mod=1e-3`, `tau=0.5`
  - HDC: `split_layer=6`
- **Special IMDB setting:** `max_length=256`
- **Total training volume:** 3 datasets x 4 methods x 3 seeds = **36 runs**

#### Expected conclusions

- AG News / SST-2 (short text): DeeBERT should show a clear efficiency advantage because many samples can exit in early layers.
- IMDB (long text): Router-Tuning should show a clear efficiency advantage because more tokens can be pruned and attention complexity grows quadratically with sequence length.
- **HDC-BERT should deliver the best overall accuracy-FLOPs balance across all datasets.**

---

### Experiment 2: Accuracy-Efficiency Pareto Curves (Most Important Figure)

**Goal:** Visualise evidence that HDC-BERT lies on the Pareto frontier.

#### How to generate multiple data points for each method

| Method | Sweep Variable | Values | Retraining Needed? |
|--------|-----------|------|---------|
| DeeBERT | `entropy_threshold` | {0.05, 0.1, 0.2, 0.3, 0.5} | No (adjusted at inference time) |
| Router-Tuning | `target_keep_ratio` | {0.4, 0.6, 0.7, 0.8} | Yes (retrain Stage 2 only) |
| HDC-BERT | `entropy_threshold` | {0.05, 0.1, 0.2, 0.3, 0.5} | No (adjusted at inference time) |
| HDC-BERT | `target_keep_ratio` | {0.5, 0.7, 0.9} | Yes (retrain Stage 3 only) |

#### Estimated compute cost

- DeeBERT entropy sweep: reuse one trained model and change the threshold at inference time -> **almost free**
- HDC entropy sweep: same idea -> **almost free**
- Router-Tuning keep-ratio sweep: 4 Stage 2 retraining runs -> about 4 hours
- HDC keep-ratio sweep: 3 Stage 3 retraining runs -> about 3 hours

#### Figure specification

- X-axis: FLOPs (% of DenseBERT)
- Y-axis: Test Accuracy (%)
- 3 subplots (one per dataset), or a horizontal 1x3 layout
- DenseBERT serves as the reference point `(100%, max_acc)`
- One curve plus markers per method, using different colours
- **Key expectation:** the HDC curve should sit in the upper-left region (higher accuracy + lower FLOPs)

---

### Experiment 3: FLOPs Computation (Must Be Implemented First)

**Goal:** Provide a hardware-independent efficiency metric.

#### FLOPs formula breakdown

For each BERT layer (`seq_len=L, H=768, I=3072`):

**Attention FLOPs:**
```text
QKV projection:   3 x L x H x H
Attention scores: L x L x H
Context vectors:  L x L x H
Output projection: L x H x H
Total: 4 x L x H^2 + 2 x L^2 x H
```

**FFN FLOPs:**
```text
FFN layer 1: L x H x I
FFN layer 2: L x I x H
Total: 2 x L x H x I
```

#### FLOPs computation for each method

| Method | Computation Rule |
|--------|---------|
| DenseBERT | `12 x full_layer_flops(L)` |
| DeeBERT | `exit_layer x full_layer_flops(L)`, averaged over all samples |
| Router-Tuning | `sum_i [attn_flops(L_kept_i) + ffn_flops(L)]`, where `L_kept_i = keep_rate_i x L` |
| HDC-BERT | Stage A exits: `k x full_layer_flops(L)`; Stage B entries: `split_layer x full + sum_j routed_layer_flops` |

> **Important:** In Router-Tuning / HDC Stage B, the FFN is applied to **all** tokens (`_run_post_attention` always executes). Only the attention part is reduced according to `keep_rate`.

#### Additional metrics

- **Wall-clock inference time:** use `torch.cuda.synchronize()` + `time.time()` and average over 3 runs
- **Report speedup ratio** = `DenseBERT_time / method_time`

#### Implementation location

- New file: `experiments/flops.py`

---

### Experiment 4: Ablation Studies

#### 4a: Component Ablation (Required)

**Goal:** Show that both Stage A (early exit) and Stage B (token routing) are necessary.

| Config | Early Exit | Token Routing | Description |
|--------|-----------|--------------|------|
| DenseBERT | x | x | Full computation baseline |
| DeeBERT | check (all 12 layers) | x | Pure sample-level ACT |
| Router-only | x | check (all 12 layers) | Pure token-level ACT |
| **HDC (ours)** | check (layers 0-5) | check (layers 6-11) | Hierarchical combination |

- Run on AG News, `seed=42`
- Report: accuracy + FLOPs + avg exit layer + avg keep rate

**Expected conclusions:**

- HDC should have lower FLOPs than DeeBERT because Stage B further reduces computation.
- HDC should have lower FLOPs than Router-only because Stage A lets easy samples exit immediately.
- HDC accuracy should be close to, or better than, both baselines.

#### 4b: Split Layer Ablation

**Goal:** Verify that `split_layer=6` is a sensible choice.

| split_layer | Stage A Layers | Stage B Layers | Interpretation |
|-------------|-------------|-------------|------|
| 2 | 2 layers | 10 layers | Very few early-exit opportunities |
| 4 | 4 layers | 8 layers | More biased toward token routing |
| **6** | 6 layers | 6 layers | Balanced split (default) |
| 8 | 8 layers | 4 layers | More biased toward early exit |
| 10 | 10 layers | 2 layers | Very few routing opportunities |

- Fix `entropy_threshold=0.2` and `target_keep_ratio=0.7`
- Each `split_layer` requires a full 3-stage training run
- Plot a dual-axis curve: Accuracy (left) + FLOPs (right) vs `split_layer`

**Expected result:** an inverted U-shape. If `split_layer` is too small, Stage A cannot contribute much because exit rates stay low. If it is too large, Stage B has too few layers to make routing effective.

#### 4c: Training Strategy Ablation (Optional, Appendix)

**Goal:** Verify the necessity of the 3-stage training order.

| Config | Training Strategy | Description |
|--------|---------|------|
| Joint | Train off-ramps + routers simultaneously | Gradient conflict |
| **3-stage (proposed)** | `fine-tune -> off-ramps -> routers` | Progressive freezing |
| 3-stage (reversed) | `fine-tune -> routers -> off-ramps` | Reverse the Stage 2/3 order |

- AG News, `seed=42`
- Compare final accuracy + convergence curves
- **Low priority:** can be skipped if time is limited

---

### Experiment 5: Hyperparameter Sensitivity

**Goal:** Show how sensitive HDC-BERT is to the key hyperparameters.

Run single-variable sweeps on HDC-BERT while holding the other hyperparameters at their default values.

| Hyperparameter | Sweep Values | Metrics to Observe | Retraining Needed? |
|------|---------|---------|---------|
| `entropy_threshold` | {0.05, 0.1, 0.2, 0.3, 0.5} | Acc, Stage A exit rate, FLOPs | No |
| `target_keep_ratio` | {0.4, 0.6, 0.7, 0.8} | Acc, actual keep rate, FLOPs | Yes (Stage 3) |
| `lambda_mod` | {0, 1e-4, 1e-3, 1e-2} | Acc, keep rate | Yes (Stage 3) |

#### Presentation

- One subplot per hyperparameter
- Dual-axis line chart: Accuracy on the left axis, efficiency metric on the right axis

**Note:** the `entropy_threshold` sweep shares data with the Pareto curves in Experiment 2, so it does not require additional experiments.

---

### Experiment 6: Analysis & Visualization

#### 6a: Exit Layer Distribution (Bar Chart)

- **Content:** compare the exit-layer distribution of DeeBERT and HDC Stage A
- **Format:** grouped bar chart, with `X=layer` and `Y=exit fraction`
- **Data source:** `exit_histogram` returned by `evaluate_hdc_inference`
- **Purpose:** show how HDC Stage A filters out "easy" samples

#### 6b: Per-Layer Keep Rate Visualisation

- **Content:** average keep rate for each HDC Stage B layer (6-11)
- **Extension:** group by class (AG News has 4 classes) to inspect differences in routing patterns
- **Format:** heatmap (`X=layer, Y=class, colour=keep_rate`) or grouped bar chart
- **Purpose:** show whether the router has learned meaningful patterns, for example whether some classes are easier to prune

#### 6c: Token Routing Case Study (2-3 Examples, Optional)

- **Content:** choose representative samples and visualise which tokens are kept / pruned at each Stage B layer
- **Format:** grid plot (`rows=layers 6-11`, `columns=tokens`, `colours=kept(green) / pruned(red)`)
- **Implementation:** requires `forward_hdc_inference_detailed()` to return per-layer masks
- **Purpose:** provide interpretability and show what the router has learned, for example pruning padding first, then stopwords, then content-related tokens

#### 6d: Easy vs Hard Sample Analysis (Optional)

- **Content:** compare the accuracy of samples that exit at each Stage A layer versus samples that continue into Stage B
- **Format:** grouped bar chart or table
- **Purpose:** verify that the entropy threshold properly separates "easy" and "hard" samples

---

## 4. Execution Priority

| Priority | Experiment | Estimated Time | Necessity |
|----------|------|---------|--------|
| **P0** | Implement `experiments/flops.py` | 0.5 day | Required |
| **P0** | Multi-dataset support in `data.py` | 0.5 day | Required |
| **P0** | Exp 1: Main Results (36 runs) | 2-3 days | Required |
| **P1** | Exp 2: Pareto Curves | 1 day | Strongly recommended |
| **P1** | Exp 4a: Component Ablation | Existing data + minor additions | Strongly recommended |
| **P1** | Exp 4b: Split Layer Ablation | 0.5 day | Strongly recommended |
| **P2** | Exp 5: Hyperparameter Sensitivity | 0.5 day | Recommended |
| **P2** | Exp 6a-b: Distributions & Heatmaps | 0.5 day | Recommended |
| **P3** | Exp 4c: Training Pipeline Ablation | 0.5 day | Optional (appendix) |
| **P3** | Exp 6c-d: Case Study & Easy/Hard | 0.5 day | Optional |

---

## 5. Code Changes Required

### New files in the `experiments/` module

| File | Purpose |
|------|------|
| `experiments/__init__.py` | Module initialisation |
| `experiments/flops.py` | FLOPs analysis for all methods |
| `experiments/visualize.py` | All plotting functions |

### Existing files that need to be modified

| File | Change |
|------|------|
| `data.py` | Add `load_dataset_by_name()` to support `ag_news / sst2 / imdb` |
| `eval.py` | Integrate FLOPs tracking + wall-clock timing |
| `models/hdc_bert.py` | Add `forward_hdc_inference_detailed()` to return per-layer masks |
| `main_*.py` | Add the `--dataset` argument |
| `configs/` | Provide configuration files for each dataset |

---

## 6. Suggested Structure for the Paper's Experiments Section

```text
5. Experiments
   5.1 Experimental Setup
       - Datasets: AG News, SST-2, IMDB (described in a table)
       - Implementation details: BERT-base, AdamW, 3-stage training, hyperparameters
       - Baselines: DenseBERT, DeeBERT, Router-Tuning BERT
       - Metrics: Accuracy, FLOPs, Speedup

   5.2 Main Results (Table 1-3)
       - HDC-BERT achieves the best accuracy-efficiency trade-off across all datasets
       - Token routing is more effective on long text (IMDB)
       - Early exit is more efficient on short text (SST-2)
       - HDC combines the strengths of both

   5.3 Accuracy-Efficiency Trade-off (Figure 1)
       - Pareto curves show that HDC occupies the frontier

   5.4 Ablation Studies (Table 4 + Figure 2)
       - Component ablation: Stage A + Stage B are both necessary
       - Split-layer analysis: the middle split point is optimal

   5.5 Analysis (Figure 3-5)
       - Exit-layer distribution
       - Per-layer routing patterns
       - Token-level case study (optional)
```

---

## 7. Likely Reviewer Questions and Matching Experiments

| Reviewer Concern | Matching Experiment |
|--------------|---------|
| Only evaluated on one dataset | Exp 1: Three datasets |
| No FLOPs, only proxy metrics | Exp 3: FLOPs analysis |
| Is the combination really better than each part alone? | Exp 4a: Component Ablation |
| Why split at layer 6? | Exp 4b: Split Layer Ablation |
| Is the training strategy justified? | Exp 4c: Training Pipeline Ablation |
| What has the model actually learned? | Exp 6c: Token Routing Case Study |
| Are the results stable and reproducible? | Exp 1: 3 seeds + mean +/- std |
| How does it behave on long documents? | Exp 1: IMDB experiments |
| How were the hyperparameters selected? | Exp 5: Sensitivity Analysis |
