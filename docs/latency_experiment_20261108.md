# HDC-BERT local latency and executed-compute experiment

## Preregistered before execution — 2026-11-08

Question: Does HDC reduce measured inference latency and executed matrix-multiplication FLOPs relative to its own dense task-finetuned backbone, and at what accuracy cost? This is a new supplementary experiment, not a recovery of historical model weights or a confirmation of the old 65.4% estimate.

Repository baseline: `957e8350d03be6ab516e3973cd39ec0eda49ab4f`. Original branch fetched and synced; existing untracked report/build files preserved in original worktree. Work takes place on isolated `codex/latency-20261108` worktree. No original result files are overwritten.

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB, Windows WDDM, driver 610.60, plugged into AC. CUDA matrix-multiplication smoke passed. Python 3.13.11, torch 2.10.0+cu128, transformers 4.57.6, datasets 5.0.0 available. Isolated `.venv` reads existing system site packages without modifying them. Capture full versions and hardware before running.

## Data and training

- AG News, pinned cached revision in config. Stratified 6,000-example validation split from the original 120,000-example training set; remaining 114,000 examples for training. Official 7,600 test examples remain unused for checkpoint/threshold selection.
- Seed 42; fixed padding/truncation to 128 tokens. Preserve exact split indices and sample hashes locally.
- Train dense backbone once (3 epochs, LR 2e-5). HDC starts from that selected dense checkpoint and freezes backbone/final head. This paired baseline isolates dynamic-computation changes; it is not the historical published baseline.
- Off-ramps: five epochs, LR 1e-3. Frozen backbone runs in eval mode. Cache deterministic CLS representations from the first six layers, then train only off-ramp heads with dropout 0.1. This is an explicit efficiency modification to historical training, whose frozen backbone still used dropout.
- Routers: five epochs, LR 1e-3, target keep 0.5, tau 0.5, lambda 0.001, using the existing straight-through dense-query-gating training objective. Frozen backbone/dropout in eval mode; model remains in training mode for differentiable gates. Full batch 32, BF16 autocast. Do not split the unnormalised ReLU budget across microbatches. A resource smoke precedes full execution; record any resource-driven changes before training.
- AdamW weight decay 0.01, 10% linear warmup, gradient clipping 1.0. Select dense by validation accuracy, off-ramps by average validation cross entropy, routers by no-early-exit query-routing validation accuracy; ties keep earlier epoch.

## Corrected and historical inference

Historical routing trains attention over all keys/values, but inference restricts both queries and keys to kept tokens. Add a separate query-only sparse runtime preserving full keys/values so the gated-query semantics match training. Keep historical runtime for explicitly labelled diagnostics. Never silently replace historical results.

Validate all-keep/no-exit against dense logits, selected-query sparse attention against dense attention followed by gate, all-zero routing, padding, and FLOPs on tiny models. Both timed paths use eager attention and omit the unused BERT pooler. No torch.compile or mixed-precision advantage for only one competitor.

Whole-batch early exit remains explicit. Evaluate accuracy/exit statistics separately at batch 1, 8, and 32. Fix threshold using validation at batch 1 from [0,.1,.2,.3,.4,.5,.6], minimising executed matmul FLOPs subject to at most 1 percentage point dense accuracy loss. If no threshold qualifies, report that failure and select highest accuracy, then lowest compute. No test-set retuning.

## Measurement

- Primary FP32, TF32 disabled, same model weights/backbone, padded tensors, order, GPU, attention backend, thread count. Batch 1 / 8 / 32 subject to available memory. Warm up 30 batches, measure 1,024 seeded test examples, repeat three rounds with alternating model order.
- Synchronized wall-clock model-only latency with inputs already on GPU; batch 1 median and p95; separate continuous-window throughput; memory allocated/reserved with clear scope. Optional local text-to-label time includes tokenizer and transfers, never presented as hosted-service latency.
- Accuracy on full official test for each primary batch configuration using exactly its inference path. Historical key-pruned path is a batch-1 diagnostic, not interchangeable accuracy evidence.
- FLOPs are an analytical count of matrix multiplications actually executed for observed exits and per-layer per-sample kept counts (multiply + add = 2 FLOPs). Count padded full FFN and full KV projections, routing and classification heads. Exclude softmax, LayerNorm, GELU, masking/indexing, memory movement and control-flow operations, all documented. Never infer speedup from this count. Independently cross-check selected tiny/dense cases against profiler or hand calculations where supported.
- Retain per-call timing, routes, predictions, hashes, checkpoints, logs locally under ignored outputs. Store only safe scripts/configs/aggregate findings in Git.

## Stop and interpretation rules

First run small correctness/resource smokes; then the declared full training/measurement. Do not promise speedup. If latency does not improve, profile representative requests and consider only clearly justified bounded implementation changes; preserve before/after quality and timing. One-seed results are hardware-specific and do not establish cross-seed or production-service gains. No expensive broad sweeps or second dataset unless the primary experiment leaves a concrete unresolved issue.

After completion record exact commands, runtime, metrics, limitations, test results, and safe commit/push status here. No private data, raw weights or raw outputs committed.

## Pre-measurement amendments and resource check

- Resource smoke on the actual GPU passed full batch 32 in BF16: Stage 1 peak allocation 3.53 GiB, steady step approximately 0.18–0.19 s; Stage 3 peak 1.21 GiB, approximately 0.08 s. These are resource checks on repeated training inputs, not model-quality or inference results.
- Corrected a dense-validation metadata counter before the substantive training run: it previously returned zero FLOPs although classification was correct. The initial few minutes were stopped before any selected checkpoint was produced; fresh training restarted from the same seed. Preserve those preliminary logs locally.
- Benchmark amendment before measurement: cycle the same 1,024 unique seeded test inputs to at least 1,024 timed **batch calls per configuration per repeat**, including larger batches. Report both unique inputs and repeated executions. This improves batch-level quantile stability without inventing independent examples.
- Whole-batch early exit depends on grouping. In addition to full-test FLOPs/accuracy, collect FLOPs and routes in a separate untimed traversal of exactly the same random groups used for latency. Do not use official-test sequential grouping FLOPs as if they described the random timed grouping.
- Original project FLOPs use a different MAC convention and aggregate routing approximations; this run's absolute GFLOPs and savings are recomputed with the declared two-operations convention and observed shapes. Historical 65.4% is not an expected target.
- Independent CPU profiler cross-check passed: a tiny BERT with B=2,T=6,H=16,I=32,C=3,N=3,A=1, mixed kept counts and padding produced 161,472 dense, 139,904 query, and 121,600 legacy matmul FLOPs. Runtime accounting, independent arithmetic and profiler mm/addmm/bmm sums agreed exactly. Eleven runtime tests and five protocol tests passed, including a complete offline smoke/resume cycle. Smoke results are not model evaluation evidence.

## Training and validation selection — completed 2026-11-09

The implementation was committed as `d905c12` on `codex/latency-20261108` while training continued; local training source SHA256 records preserve the exact files used at process start. The original project worktree and historical reports were not changed.

Selected dense epoch 2: validation accuracy 95.0000%; epochs 1/2/3 yielded 94.1333%, 95.0000%, 94.8167%. Selected off-ramp epoch 5 by validation CE (0.296803). Feature extraction used BF16 autocast for training examples and FP32 for validation examples; stored CLS tensors were FP32 in both cases. In local stage2 metadata, "feature_cache_precision" describes extraction compute precision, not tensor storage dtype.

Router validation with no early exit, using the corrected query-sparse runtime: epochs 1–5 yielded 95.0167%, 94.9667%, 95.0333%, 95.1000%, 94.9167%. Epoch 4 was selected. These small validation differences are selection observations, not evidence of statistically significant accuracy improvement.

Both checkpoints' 201 shared `bert.*` and `classifier.*` tensors were verified identical before benchmarking (SHA256 `052aa801387ff73fff5280ce7f85c9fc0370fd224c78dd545bc087b4ac1df5c2`). Off-ramps/routers are the only trained additions after dense fine-tuning.

Validation threshold selection at batch 1:

| Entropy threshold | Validation accuracy | Mean executed matmul GFLOPs/text | Within 1 pp of dense? |
| ---: | ---: | ---: | :--- |
| 0.0 | 95.1000% | 20.5824 | Yes |
| 0.1 | 94.7833% | 10.2933 | Yes |
| 0.2 | 94.5500% | 8.0753 | Yes |
| 0.3 | 94.1667% | 6.8223 | Yes — selected |
| 0.4 | 93.9167% | 5.8824 | No |
| 0.5 | 93.5167% | 5.0793 | No |
| 0.6 | 92.8833% | 4.3377 | No |

The threshold was locked at 0.3 before official-test evaluation. This validation constraint is not a guarantee on test data or on other batch sizes. No test labels were used to retune it.

## Post-measurement diagnosis plan

Batch 8 primary measurements show fewer matmul FLOPs but longer wall-clock time. After all primary timings finish, `benchmarks/profile_latency.py` can inspect a fixed batch from the already-recorded modal routing case. It refuses incomplete primary artifacts and checks experiment identity before GPU use. Operator counts and CPU/device profiler attribution diagnose the implementation; profiler timings never replace the primary measurements. No post-test architecture tuning is part of the primary result.

## Completed measurements

The serial training/selection/measurement pipeline finished at 2026-11-09 02:06:15 Asia/Shanghai, in 8,646 seconds (2.40 hours, excluding initial setup and subsequent reporting/diagnosis). Additional hardware identification: Intel Core 7 240H, 10 cores / 16 logical processors; torch CPU threads fixed at four. All seven complete-test evaluations contain 7,600 examples; all 21 timing JSON/CSV pairs contain 1,024 calls. Independent checks reproduced their quantiles and aggregate results, and confirmed paired input order.

Machine-readable safe aggregates: [latency_results_20261108.json](latency_results_20261108.json). FLOPs below are the mean for the exact timing input groups; accuracy is on the complete official test split. Latencies are the mean of three separately calculated per-repeat quantiles, in milliseconds **per batch**.

| Path | Batch | Test accuracy | Mean GFLOPs/text | FLOPs saved | P50 ms/batch | P95 ms/batch | Texts/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 1 | 94.6316% | 22.347 | — | 9.887 | 18.062 | 98.24 |
| HDC query | 1 | 93.9474% | 6.844 | 69.38% | 3.534 | 17.671 | 184.44 |
| Dense | 8 | 94.6316% | 22.347 | — | 43.463 | 46.482 | 205.47 |
| HDC query | 8 | 94.5789% | 17.638 | 21.07% | 62.441 | 77.286 | 151.93 |
| Dense | 32 | 94.6316% | 22.347 | — | 157.431 | 158.871 | 203.74 |
| HDC query | 32 | 94.6842% | 20.583 | 7.89% | 281.820 | 326.007 | 113.18 |

For batch 1, P50 decreases 64.26% (2.798x speedup), with a 0.6842 percentage-point accuracy loss. This is not an equivalent tail-latency improvement: P95 ranges overlap (Dense 16.702–19.276 ms; HDC 17.052–18.429 ms), and one repeat has a worse HDC P95. Among the 1,024 fixed timing inputs, 870 exit early (HDC median 2.91–3.07 ms across repeats); 154 reach full routing (14.99–16.83 ms, versus 9.32–10.51 ms in Dense for those same inputs).

For batch 8 and 32, HDC P50 is respectively **43.67% and 79.01% higher**, despite fewer matrix FLOPs. Whole-batch early exit means most large batches continue through the full model. The threshold remains the one selected at validation batch 1; these are controlled measurements of that fixed model/configuration, not a claim that all possible per-batch tuning or other implementations fail.

Batch 32 HDC P50 varies from 274.640 to 295.070 ms across repeats; Dense varies from 157.357 to 157.563 ms. Even the fastest HDC repeat is 74.30% slower than the slowest Dense repeat, so the observed variation does not reverse the conclusion. Device snapshots show power/clock variation on the laptop; report the range rather than treating a single run as a constant.

The legacy K-by-K diagnostic gives 93.3421% accuracy, P50 3.400 ms and P95 19.226 ms at batch 1 on these newly trained weights. It is a different attention function and is not substituted for the corrected runtime in the main comparison.

## Operator diagnosis, limitations and stopping rationale

After the primary benchmark completed, five profiled calls on the first modal full-depth batch of 32 produced the following CPU-side operator counts **per model call**:

| Operator | Dense | HDC query |
| --- | ---: | ---: |
| `aten::linear` | 73 | 829 |
| `aten::bmm` | 24 | 396 |
| `aten::nonzero` | 0 | 192 |

The sparse attention implementation loops over samples, making many small matrix calls and index lookups while the full padded FFN still runs. These counts support an implementation-overhead explanation; they are not a CUDA kernel-time decomposition. CUPTI reported `CUPTI_ERROR_INVALID_DEVICE` on this setup and did not capture CUDA activity timings. Valid CPU counters were retained. The primary synchronized wall-clock timing and execution-shape FLOPs accounting do not rely on CUPTI and remain valid.

No additional broad sweep or architecture rewrite was performed: the requested paired accuracy/latency/FLOPs result is complete, repeated differences are large, and the main limitation is identified. Follow-up engineering would evaluate per-sample exit handling and genuinely batched sparse attention, while comparing against an appropriately optimized dense backend. No speedup from those unimplemented changes is claimed.

Best Stage 1/2/3 weights, exact data/split identities, logs, raw measurements, diagnosis and reading report are retained locally. Safe source/configuration/aggregate records are committed; weights, raw inputs and raw measurement traces remain excluded from Git.
