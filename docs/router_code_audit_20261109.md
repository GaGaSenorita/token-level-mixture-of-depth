# RouterBERT implementation audit — 9 November 2026

Scope: audit of the repository implementation and new paired-runtime adapters.
Historical task checkpoints remain unavailable. Tiny CPU reproductions establish
code behaviour, not historical accuracy loss or a measured GPU speedup. Original
model files and historical results were preserved.

## Relation to the published method

The repository's RouterBERT is a BERT adaptation, not the original authors'
decoder-inference experiment. The [final EMNLP 2025 Router-Tuning paper](https://aclanthology.org/2025.emnlp-main.99.pdf)
studies both token and sequence routing; it uses sequence-level attention routing
as its default because token routing creates unequal lengths and padding work.
It mainly routes the deeper half of layers while preserving the last layer.
The local AG News configuration instead routes all 12 BERT layers at token level.
Do not confuse the paper's reported generation speedup with this classification
implementation, or claim that the paper never studied token routing.

The performance concern about `nonzero` is also an API-level fact: the
[PyTorch 2.10 documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nonzero.html)
states that CUDA `nonzero` synchronizes host and device. The new batched helper
reduces this to one Kmax scalar synchronization per routed layer; it does not
eliminate sorting, padding or all host synchronization.

## What was and was not measured previously

The November 8 supplementary experiment compared Dense and HDC, plus a labelled
historical K-by-K HDC diagnostic. It did not time standalone DeeBERT or RouterBERT.
The historical results aggregator explicitly sets their latency fields to `None`
(`experiments/flops.py:151,171`). Its FLOPs numbers are analytical estimates and
cannot supply the missing latency comparison.

## Confirmed findings

| Finding | Evidence and implication |
| --- | --- |
| Token routing changes the attention function between training and inference. | Training computes dense attention, then gates query outputs (`models/router_tuning_bert.py:278`). Inference selects a subsequence before Q/K/V projection (`:209–232`), removing unselected keys/values as well. HDC shares the same historical mismatch (`models/hdc_bert.py:154,257`). This is a correctness inconsistency, not merely a slower implementation. |
| Per-sample inference serializes sparse attention calls. | The token inference helper loops over batch samples and invokes `nonzero` and attention separately (`models/router_tuning_bert.py:219–232`). The November 8 corrected query implementation preserves training semantics but retains that per-sample loop (`benchmarks/hdc_runtime.py:165`). Thus the semantic repair alone does not remove batching overhead. |
| Ordinary RouterBERT inference still computes training statistics and budget loss. | `forward` calls `forward_with_routing`; every routed layer calls `_compute_budget_loss`, including `.item()` (`models/router_tuning_bert.py:203,275,350`). On CUDA this creates a host synchronization per layer even when the caller discards the returned statistics. The new HDC/comparator timing paths separate optional statistics from inference. |
| Attention skipping does not skip the FFN. | Residual, normalization and the full padded FFN intentionally execute for every token (`models/router_tuning_bert.py:167–178`). With BERT-base and padded length 128, FFN matrix work is about 64.86% of a dense layer's matrix FLOPs. Even eliminating all attention leaves this work; routing half the layers limits savings further. This is the implemented architecture, not a missing conditional to add without retraining. |
| Original sparse scatter is not autocast-safe. | `attn_out` uses hidden-state dtype, while projected attention can be BF16; indexed assignment at `models/router_tuning_bert.py:232` raises a dtype mismatch. Reproduced on CPU BF16. This does not explain FP32 timings. New runtime helpers cast before scatter. |
| Initialization comment is misleading, but dense initialization is valid. | Zero router weights produce probability 0.5, and thresholding uses `>= 0.5` (`models/router_tuning_bert.py:41,54`): all tokens are retained. The comment claiming roughly half are kept is false. Dense stage-one training intentionally relies on all-keep initialization (`train.py:237–240`). |

## Training and reporting limitations

- The historical router budget is a sum of excess kept tokens, while task cross
  entropy is averaged (`models/router_tuning_bert.py:183–205`, `train.py:341–342`).
  Duplicating a batch doubles its budget penalty. Batch size, valid-token count
  and routed-layer count therefore affect the meaning of the same lambda. This
  is an objective scaling choice, not evidence of absent router gradients.
- Historical router-only training calls `model.train()` after freezing weights
  (`train.py:318`). Frozen backbone dropout consequently stays active. Weight
  freezing does not mean eval mode; this can be a training design choice, and is
  not independently established as a cause of poor quality.
- Historical checkpoint selection uses test accuracy (`main_routerbert.py:171,184`,
  `train.py:382,406`). New comparisons must select on the held-out validation split.
- Historical keep ratios are averaged over batches (`eval.py:107–122`) and then
  substituted into an analytical attention formula (`experiments/flops.py:37,55`).
  Keep ratio is measured over valid tokens but the formula uses padded length;
  squaring a mean also differs from averaging squared per-sample retained counts.
  Exact execution-shaped FLOPs require actual per-call counts and packing shapes.
- Historical final run summaries do not show complete router collapse: AG News
  mean keep ratios are 0.5477/0.6062/0.6084, with all layer means between 0.4097
  and 0.6784. IMDB means are 0.6459/0.6549/0.6140. These aggregate logs do not
  establish per-example behaviour or current newly trained checkpoint quality.

## Reproductions and new-runtime checks

Run from the repository root:

```text
python -m unittest tests.test_router_audit tests.test_comparator_runtime -v
```

`tests/test_router_audit.py` contains six offline CPU tests. With dropout disabled
and fixed mixed token gates, the original train/eval maximum valid-hidden-state
difference is **0.0086125** in the seeded tiny case; corrected full-K/V query
attention differs from the training reference by **3.73e-9**. All-keep and
sample-routing controls agree, isolating the token-compaction mismatch. Other
tests verify all-skipped FFN execution, initialization/budget scaling and the
historical BF16 scatter failure. A tiny three-layer, batch-two original forward
records three `aten::item`, six `aten::nonzero` and twelve `aten::bmm` CPU calls.
These counts are observations, not GPU timings.

`tests/test_comparator_runtime.py` checks full-runtime predictions, optional-stat
invariance, HDC compatibility and all first-six early exits in a twelve-layer
model, plus DeeBERT's final-head contract. The standalone DeeBERT adapter must
map its last off-ramp to the frozen dense final classifier; this mapping is a
checkpoint-construction responsibility. The final head executes once.

An independent CPU operator profiler cross-check uses real router projections
with fixed gate hooks, B=2, T=6, H=16, I=32, C=3, and three routed layers. Retained
counts are [3,0], [1,2], [0,0]. The full runtime's analytical matrix count and
the profiler agree exactly:

| Backend | Executed matrix FLOPs |
| --- | ---: |
| Per-sample full-K/V query loop | 101,952 |
| Batched full-K/V query packing | 113,728 |
| Full dense attention followed by gate | 162,624 |

The batched implementation correctly counts Kmax=[3,2,0], including dummy query
work on shorter/empty rows; it must not report the loop's lower FLOPs. All three
backends agree in final logits to floating-point tolerance. CUDA profiling was
not used for these CPU equivalence/count checks. GPU speed and quality require
the separate paired benchmark; no speedup is inferred from these tests.
