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
