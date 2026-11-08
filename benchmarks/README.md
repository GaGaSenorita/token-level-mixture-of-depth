# Supplementary HDC latency experiment

This workflow trains new AG News checkpoints and measures the dense and dynamic paths under matched conditions. It does not reuse historical reported accuracies or claim that analytical FLOPs predict wall-clock speed. The preregistration and findings live in [the experiment log](../docs/latency_experiment_20261108.md).

## Environment and cached inputs

Recorded execution: Windows, Python 3.13.11, torch 2.10.0+cu128, transformers 4.57.6, datasets 5.0.0, numpy 2.2.6, pyarrow 23.0.0, matplotlib 3.10.9, huggingface-hub 0.36.0, safetensors 0.7.0. Actual environment metadata is saved under the local run directory. Training requires a CUDA GPU supporting BF16; benchmark precision is FP32.

Use an appropriate isolated environment. For a fresh cache, retrieve the pinned public model and data before invoking the offline experiment:

```python
from huggingface_hub import snapshot_download
snapshot_download("bert-base-uncased", revision="86b5e0934494bd15c9632b12f734a8a67f723594",
                  allow_patterns=["*.json", "model.safetensors", "vocab.txt"])
snapshot_download("ag_news", repo_type="dataset", revision="eb185aade064a813bc0b7f42de02595523103ca4",
                  allow_patterns=["data/*.parquet", "README.md"])
```

## Commands from repository root

```text
python -m benchmarks.run_experiment --phase prepare
python -m benchmarks.run_experiment --phase smoke
python -m benchmarks.run_experiment --phase all
python -m benchmarks.measure_latency --config outputs/latency_20261108/config_resolved.json --run-dir outputs/latency_20261108
python -m benchmarks.build_latency_report --run-dir outputs/latency_20261108
```

Alternatively `python -m benchmarks.run_pipeline` executes training, validation selection and the benchmark serially. On Windows it prevents idle system sleep only while the process is alive; it does not change persistent power settings. Check `outputs/latency_20261108/progress.jsonl` and `pipeline_status.json` for progress. Redirect stdout/stderr to files when running detached.

The first stage trains a dense backbone; its selected checkpoint is the paired dense baseline. HDC freezes that backbone and final head and adds trained off-ramps/routers. Stage 2 caches eval-mode representations; Stage 3 uses the original differentiable query-gating objective. Corrected sparse inference retains all keys/values for selected queries. The historical key-pruned path is labelled as a diagnostic.

Do not change configuration and silently resume into the same run directory. Training supports epoch-boundary resume; off-ramp training restarts its small head-only stage if interrupted. A new model/data/configuration needs a new output directory. The benchmark additionally verifies config/source/checkpoint hashes and refuses mixed-protocol resumes.

## Outputs and checks

- Local only: tokenized data, split indices, cached features, checkpoints, optimizer resume state, detailed timing/route records, environment snapshots and logs. They are under ignored `outputs/`.
- Benchmark: `benchmark/summary.json`, `manifest.json`, `report.md`, per-repeat JSON/CSV, predictions and routes. Timing is measured without the route counters; observed-shape FLOPs are collected in separate passes.
- Safe to commit: scripts, unit tests, configuration, preregistration and reviewed aggregate findings.

```text
python -m unittest discover -s tests -p "test_*.py"
```

Tests use tiny offline CPU models. They check state compatibility, gated-query equivalence with padding, all/zero/mixed retained tokens, early exit, router gradients, profiler FLOP agreement, sampling/quantiles, and a complete benchmark smoke/resume cycle. Smoke artifacts are never production experiment results.

Report batch latency as milliseconds per batch. Only batch 1 approximates a single local model call. Throughput is measured in a separate continuous window. Neither measure includes a production serving/network stack. The FLOPs convention counts multiply and add separately and covers matrix multiplications, not all instructions or memory traffic.
