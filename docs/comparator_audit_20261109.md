# DeeBERT / Router-Tuning correctness and latency audit

## Preregistered 2026-11-09, before new training and measurements

Question: The previous paired experiment measured Dense and HDC only. Does the standalone Router-Tuning implementation have semantic or execution problems, and how do standalone DeeBERT and Router-Tuning compare under matched measurement conditions?

Start from clean, fetched, synced branch `codex/latency-20261108`, commit `558cee6`. Preserve all previous measurements and original team-project source. New files and outputs extend the experiment; no resume into the old result directory and no resume edits to submitted CVs.

### Correctness and bounded engineering

1. Reproduce whether historical token routing trains with all valid K/V context but removes unselected K/V at inference. Test mixed, all-keep, zero-keep and padding cases, separating selected-query semantics from historical compact K-by-K semantics.
2. Audit `.item()` / nonzero synchronization, per-sample loops, always-executed FFN, router initialization and FLOPs formulas. A design choice or overhead is not automatically a correctness defect.
3. Implement a separately labelled batched query-sparse backend, with full K/V and padded Kmax queries. Validate numerical equivalence before timing. Count the actual padded batch shapes, not an ideal sum of individual K values. A dense-gate reference is allowed for semantic checking and bounded diagnosis, but cannot claim sparse compute savings.
4. Preserve the historical runtime as a diagnostic. New training-consistent inference is explicitly not historical inference. Do not repair semantics by altering test labels or claiming the old project used these fixes.

### New baseline weights

- Reuse the exact frozen dense task-finetuned backbone and final classifier from `outputs/latency_20261108/stage1_best.pt`. Check shared tensor hashes. No need to retrain the common backbone.
- Reuse the exact 114,000 / 6,000 / 7,600 train / validation / official-test tensors, seed 42, padded length 128. Test labels are never used for training, checkpoint selection, threshold selection or performance tuning.
- Standalone Router-Tuning: route all 12 layers, original AG News configuration target keep 0.7, tau 0.5, lambda 0.001. Train routers for five epochs with frozen backbone in eval mode, BF16 compute, batch 32, AdamW LR 0.001, weight decay 0.01, 10% warmup, clip 1. Select highest validation accuracy using training-consistent query gating in FP32; ties keep earlier epoch. Dense-gated validation is mathematically equivalent on valid positions and may be used to avoid per-sample sparse overhead. Verify this equivalence on selected weights. Its target differs from the earlier HDC run's 0.5: report this explicit architecture/configuration comparison, not identical learned sparsity.
- Standalone DeeBERT: same dense backbone and final classifier; train 11 intermediate classifiers using frozen backbone representations, five epochs with the same optimizer schedule and dropout as HDC's intermediate heads. Train features BF16 compute / validation FP32; select mean validation CE. Use bounded memory-mapped caches because current available host RAM is limited.
- HDC: reuse selected Stage 3 weights and locked threshold 0.3 from previous validation. Do not retune to improve its test results.
- DeeBERT threshold: candidates 0.0 through 0.6 in steps of 0.1, lowest executed matrix FLOPs within 1 percentage point of dense validation accuracy at batch 1; fallback highest validation accuracy with explicit constraint failure. Lock threshold before official-test evaluation. Router-Tuning has no entropy exit. Report if its selected validation accuracy does not meet the same quality envelope; do not conceal it as an equal-quality comparison.

### Matched measurements

- Same local RTX 5050 Laptop, FP32 eager attention, TF32 off, four torch CPU threads; all variants omit unused pooler, same fixed padding and same input groups, one active model and one GPU workload at a time.
- Primary models: Dense; DeeBERT; standalone Router-Tuning with corrected query semantics; HDC. Label loop and batched implementations separately for the routing models, using identical weights for their before/after comparison.
- Pre-measurement causal ablation: `hdc_exitonly` keeps exactly HDC's first-six heads, shared dense weights and threshold 0.3, but runs the remaining layers densely without routers. It isolates the routing contribution from independently trained DeeBERT head differences. It is labelled an ablation, not standalone DeeBERT or a separately tuned model.
- Batch sizes 1 / 8 / 32. Full 7,600-example quality evaluated with the actual path and batch grouping. Whole-batch early exit retained for both DeeBERT and HDC to avoid an asymmetric optimization.
- Same 1,024 unique seeded test inputs as previous run; at least 256 timed batch calls, 30 warmups, three repetitions with rotated/reversed path order. Batch 1 naturally has 1,024 calls. This smaller minimum than the first run is an explicit new matched protocol for all competitors, not a splice of old and new times. Separate continuous throughput window; synchronized wall-clock per-batch P50/P95.
- Matrix FLOPs: one multiply-add = two operations; actual executed padded shapes, including full FFN/router/head work. Separate untimed count on exactly the timing groups. Indexing, memory movement, synchronization and non-matmul operations are excluded from this count but included in measured latency.
- Bounded diagnostics may compare original compact attention and a dense-gate reference on the same new Router weights. Label them separately from the corrected four-model comparison. No profiler kernel timing claim when CUDA profiling is unavailable.
- Before measurements, narrow the required semantic diagnostic to full-test batch-32 query versus legacy K-by-K predictions on the same new Router weights. Report disagreement counts and accuracy changes; this answers the correctness question without a redundant timing sweep. The broader `--phase diagnose` timing sweep remains optional and is not required if the primary loop/batched measurements already resolve the overhead question.

### Evidence / stopping

Configuration: `configs/comparator_audit_20261109.json`. Local output: `outputs/comparator_audit_20261109/`. Record source/config/data/checkpoint hashes before each phase. Preserve per-call times, predictions, routes, train history and new weights locally; commit only source, tests, protocol and safe aggregates. Re-run only comparisons invalidated by a demonstrated error. Stop after validated semantic findings and matched measured trade-offs; further sparsity sweeps, fused kernels, multi-seed retraining and other datasets require a separately justified study.

## Preflight validation and launch

The GPU resource smoke passed on 2026-11-09 at 08:37 Asia/Shanghai. After startup,
router training steps took approximately 0.123 seconds at batch 32, peak allocated
memory 1.97 GiB. FP32 dense-gated and sparse-query logits differed by at most
1.79e-6 on the smoke batch. Frozen shared weights remained identical. These are
resource/correctness checks, not classification results or inference timings.

The substantive serial Router/Dee training began at 08:39. Training uses
memory-mapped inputs and bounded feature-cache writes; no global environment or
persistent power-plan change was made. CPU tests separately verify historical
bugs, corrected query equivalence, padded-batch FLOPs, final-head mapping,
validation-only selection, paired timing orchestration and stale-cache rejection.
Before the primary timings, also check the actual selected Router/HDC weights
on the GPU: first validation inputs at B1/8/32, query-loop versus batched-query
and dense-gate logits, predictions, retained counts and exits. HDC is checked
with and without early exits. This is a correctness check, not a quality result.

Commands:

```text
python -m benchmarks.train_comparators --phase smoke
python -m benchmarks.train_comparators --phase all
python -m benchmarks.compare_baselines --phase all
python -m benchmarks.compare_baselines --phase semantic
python -m unittest discover -s tests -p "test_*.py"
```

The optional diagnosis is separate from primary measurements. Source and input
identities are checked on resume; do not edit training dependencies mid-run.

## Additional attribution check planned during measurement

The corrected sparse paths omit padding query rows even if every valid token is
retained. Therefore total FLOPs reduction must not be attributed entirely to the
learned gate. After primary timing finishes, a CPU-only counterfactual analysis
will use the recorded timing groups and valid lengths to count the same runtime
with `force_keep_all`: unchanged early exits, heads, router projections and full
FFN; per-sample valid counts for loop execution, batch maximum valid count for
packed execution. The difference from actual recorded FLOPs isolates the extra
matrix-compute saving from learned token decisions. No latency or accuracy is
claimed for this unmeasured counterfactual, and no checkpoint, threshold or
primary measurement setting is changed. This analysis was added after code
inspection during the run, not retroactively described as the initial protocol.

## Completed training and measurements — 2026-11-09

New standalone RouterBERT selected epoch 2: validation accuracy 94.95%, a 0.05 percentage-point drop relative to Dense validation accuracy 95.00%. DeeBERT selected epoch 5 by mean intermediate-exit validation CE (0.248483); its batch-1 threshold was locked at 0.3, validation accuracy 94.05%, within the preregistered 1 percentage-point envelope. Its final exit remains a frozen copy of the dense classifier. All seven paths share 201 identical `bert.*` / `classifier.*` tensors, fingerprint `052aa801387ff73fff5280ce7f85c9fc0370fd224c78dd545bc087b4ac1df5c2`.

The primary benchmark completed at 2026-11-09 10:45:04 Asia/Shanghai. All table timings below were collected afresh in this run; none are copied from the November 8 benchmark. Independent artifact audit passed on 21 full-test evaluations (7,600 examples each), 63 JSON/CSV timing pairs, and 32,256 timed batch calls. It independently recomputed predictions, execution-shape FLOPs, grouped inputs, quantiles, summary arithmetic, shared weights and semantic disagreement counts. The audit is not a new physical replay of elapsed time.

Machine-readable reviewed results: [comparator_results_20261109.json](comparator_results_20261109.json). Code findings and tiny correctness reproductions: [router_code_audit_20261109.md](router_code_audit_20261109.md).

| Path | Batch | Test accuracy | Timed GFLOPs/text | FLOPs saved | P50 ms/batch | P95 ms/batch | Texts/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 1 | 94.6316% | 22.347 | 0.000% | 8.058 | 9.389 | 128.02 |
| deebert | 1 | 93.8947% | 6.571 | 70.597% | 2.467 | 11.411 | 265.90 |
| router_loop | 1 | 94.6579% | 19.150 | 14.309% | 12.591 | 15.835 | 78.58 |
| router_batched | 1 | 94.6579% | 19.150 | 14.309% | 15.851 | 18.937 | 63.16 |
| hdc_loop | 1 | 93.9474% | 6.844 | 69.376% | 2.368 | 12.375 | 246.66 |
| hdc_batched | 1 | 93.9474% | 6.844 | 69.376% | 2.380 | 14.514 | 230.35 |
| hdc_exitonly | 1 | 93.8158% | 7.096 | 68.245% | 2.376 | 10.061 | 270.41 |
| dense | 8 | 94.6316% | 22.347 | 0.000% | 37.945 | 45.172 | 215.80 |
| deebert | 8 | 94.6053% | 17.692 | 20.833% | 40.184 | 46.702 | 233.52 |
| router_loop | 8 | 94.6579% | 19.150 | 14.309% | 73.770 | 80.023 | 109.11 |
| router_batched | 8 | 94.6579% | 19.735 | 11.687% | 41.942 | 45.267 | 191.61 |
| hdc_loop | 8 | 94.5789% | 17.638 | 21.075% | 55.014 | 61.287 | 174.74 |
| hdc_batched | 8 | 94.5789% | 17.847 | 20.139% | 41.610 | 45.912 | 223.27 |
| hdc_exitonly | 8 | 94.5263% | 18.972 | 15.104% | 41.855 | 46.386 | 222.60 |
| dense | 32 | 94.6316% | 22.347 | 0.000% | 149.120 | 151.173 | 215.20 |
| deebert | 32 | 94.6184% | 22.347 | -0.000% | 151.714 | 153.779 | 211.04 |
| router_loop | 32 | 94.6579% | 19.150 | 14.309% | 291.162 | 313.770 | 108.43 |
| router_batched | 32 | 94.6579% | 20.223 | 9.505% | 149.274 | 155.972 | 215.17 |
| hdc_loop | 32 | 94.6842% | 20.583 | 7.895% | 222.798 | 241.645 | 141.88 |
| hdc_batched | 32 | 94.6842% | 21.036 | 5.868% | 152.933 | 158.731 | 212.52 |
| hdc_exitonly | 32 | 94.6316% | 22.347 | -0.000% | 150.351 | 152.010 | 213.02 |

Accuracy is on the complete test split. Matrix FLOPs refer to exactly the grouped timing inputs, including batch Kmax padding. P50/P95 are means of three separately computed per-repeat quantiles, not a pooled percentile. A negative tiny saving for full-depth DeeBERT / HDC-exitonly reflects extra exit-head computation.

### Correctness findings and implementation effects

- The historical token path trains selected query outputs against all valid K/V, but inference removes unselected K/V as well. On the same newly trained RouterBERT checkpoint and 7,600 test examples, query-consistent execution gives **7,194 correct (94.6579%)**, while historical compact K-by-K execution gives **6,957 correct (91.5395%)**: **+3.1184 percentage points**. Of 416 prediction disagreements, 316 change from wrong to correct, 79 from correct to wrong, and 21 remain wrong under both predictions. This is the measured consequence on these new weights; unavailable historical weights cannot be assigned the same loss. This semantic diagnostic is accuracy-only, not another timing sweep.
- Per-sample sparse attention creates serial index lookups and small matrix calls. Batched query packing reduces these calls while preserving full K/V. All 7,600 predictions match between loop and batched backends at every measured batch size for RouterBERT and HDC. On batch 32, Router median falls from 291.162 to 149.274 ms; HDC from 222.798 to 152.933 ms. Batched packing increases FLOPs because dummy queries to Kmax are actually executed.
- Batching is not universally better: Router batch-1 median increases from 12.591 to 15.851 ms. At batch 32, batched Router (149.274 ms) is approximately equal to Dense (149.120 ms): Router repeat medians 146.315–153.054 ms overlap Dense 148.913–149.274 ms. There is no reliable speedup claim from the 0.153 ms mean difference.
- Original `forward` also computes discarded training statistics/budget loss with per-layer `.item()` synchronization. New timing adapters omit those statistics; the legacy semantic diagnostic is not a verbatim timing of the entire historical forward. Attention-only routing deliberately leaves the complete FFN, approximately 64.86% of dense-layer matrix FLOPs at length 128. With nonempty selected queries, full K/V projections account for another approximately 16.22%. These are arithmetic shares, not measured latency shares.

### Is routing adding value beyond early exit?

HDC with routing and HDC-exitonly share the same early heads, threshold and backbone; only later-layer routing is removed in the ablation. At batch 1, median latency is virtually unchanged (2.380 versus 2.376 ms), while batched HDC P95 is higher (14.514 versus 10.061 ms). HDC accuracy is 93.9474% versus exit-only 93.8158%, a 0.1316 percentage-point difference (10 examples), not an established significant improvement. The measured median gain over full Dense is primarily associated with early exits, not a demonstrated extra latency gain from token routing.

Standalone DeeBERT reaches 2.467 ms median at batch 1 with 93.8947% accuracy and 70.600% fewer timed matrix FLOPs. HDC batched reaches 2.380 ms with 93.9474% accuracy and 69.376% fewer FLOPs. Their small accuracy and median differences do not establish one method as superior; repeat medians overlap and independently trained heads/configurations differ. All three early-exit variants have batch-1 P95 above Dense in this new run. Large batches mostly lose the early-exit advantage under the retained whole-batch exit policy.

### Analytical attribution: gate, PAD queries and early exit

A post-measurement CPU-only calculation holds **the same early-exit paths and executed layer counts** fixed. It substitutes all-valid queries and then all-padded queries into execution-shape formulas, retaining Kmax packing, full padded K/V and FFN, head/router cost. No counterfactual quality or latency run was performed. The components use complete Dense FLOPs as their common denominator and add to the observed total saving.

- Standalone Router batch 1: total 14.309% saving = 3.168% additional learned-gate saving + 11.152% PAD-query omission − 0.011% router overhead. Thus most of its arithmetic reduction at batch 1 is not evidence that learned gates selected useful words to skip. The learned component is 22.137% of the total reduction, a different denominator.
- HDC batch 1: total 69.376% = 0.299% additional learned-gate saving + 0.833% PAD-query omission + 68.245% early-exit saving net of extra heads/routers. This reinforces the early-exit attribution at the arithmetic level.
- Packed batch 32: Router total saving is 9.505%, of which learned gates contribute 6.139% of Dense and PAD omission 3.376%, less router overhead. HDC total is 5.868%, including 4.185% learned-gate and 1.688% PAD omission, less small net extra-head/router cost. Packing changes both the total and attribution; the loop FLOPs must not be attached to batched latency.

These counterfactuals describe this implementation, not the best achievable padding-aware Dense baseline. They omit non-matmul operations, so they cannot explain measured latency by arithmetic alone. Exact values, definitions and limitations are included in the safe aggregate JSON.

### Validation, artifacts and operational note

The expanded offline test suite, actual-checkpoint GPU equivalence checks and independent result audit passed. Report generation additionally checks complete paired paths, raw-versus-summary P50/P95, measured threshold identity, semantic checkpoint/data identity and FLOPs decomposition arithmetic. The report-generator synthetic fixture test passed after final schema integration.

The orchestration wrapper recorded `failed` at 10:47:38 because PowerShell exposed a null process ExitCode; the comparison process had already written complete valid primary and semantic summaries. The preserved wrapper status is an orchestration false alarm, not a failed model experiment. The independent audit at 10:48:36 passed. No measurements were rerun or discarded to obtain the reported outcome.

Local-only outputs retain weights, memmap features, exact data references, per-call timing/route records, prediction CSVs and logs under `outputs/comparator_audit_20261109/`. Readable artifact: `benchmark/阅读版.html`; chart: `benchmark/comparison.png`; main tables: `benchmark/summary.json` / `summary.csv`; semantic comparison: `semantic_diagnostic/summary.json`; compute attribution: `padding_compute_analysis.json`; independent audit: `independent_audit.json`. Large artifacts and raw records remain excluded from Git.

Post-measurement commands:

```text
python -m benchmarks.analyze_padding_compute
python -m benchmarks.build_comparator_report --self-test
python -m benchmarks.build_comparator_report
```

### Interpretation limits and stopping rationale

This is one new training seed, one dataset and one laptop, with frozen shared backbone and declared, different routing/exit configurations. It is not an exhaustive or equal-FLOPs Pareto search, an official Router-Tuning paper reproduction, or evidence of cross-seed statistical superiority. The original report implementation is a BERT adaptation; the [final Router-Tuning paper](https://aclanthology.org/2025.emnlp-main.99.pdf) covers both token and sequence routers, with different main architecture/task/inference settings. Fixed batch-1-selected thresholds are not separately optimized for larger batches. Eager FP32 results do not compare production serving, quantization, SDPA, compiled/fused kernels, concurrent requests or other input lengths. Laptop power/clock variation remains possible, and repeat ranges are not confidence intervals.

The requested gap is closed: standalone baselines have fresh matched measurements, the semantic mismatch has both reproduction and full-test evidence, and batching overhead has a same-weight intervention. No broad retraining sweep is justified merely to make routing look better. Further latency optimization requires a separately specified experiment and an appropriately optimized dense comparator. Historical source/results and submitted CVs remain unchanged.
