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

Commands:

```text
python -m benchmarks.train_comparators --phase smoke
python -m benchmarks.train_comparators --phase all
python -m benchmarks.compare_baselines --phase all
python -m benchmarks.compare_baselines --phase diagnose
python -m unittest discover -s tests -p "test_*.py"
```

The optional diagnosis is separate from primary measurements. Source and input
identities are checked on resume; do not edit training dependencies mid-run.
