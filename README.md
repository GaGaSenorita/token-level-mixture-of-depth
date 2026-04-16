# Token-Level Mixture of Depth

This repository implements four BERT-based text classification pipelines for dynamic-computation experiments:

- `DenseBERT`: standard dense fine-tuning of `bert-base-uncased`.
- `DeeBERT`: sample-level early exit with multiple off-ramp classifiers.
- `Router-Tuning`: token-level routing that learns which tokens can skip computation.
- `HDC-BERT`: a hierarchical design that combines early exit in shallow layers with token routing in deeper layers.

The current codebase supports two datasets through checked-in YAML configs:

- `AG News`
- `IMDB`

## Repository Layout

- `main_densebert.py`: dense BERT training entrypoint
- `main_deebert.py`: two-stage DeeBERT training entrypoint
- `main_routerbert.py`: two-stage Router-Tuning training entrypoint
- `main_hdcbert.py`: three-stage HDC-BERT training entrypoint
- `configs/`: runnable experiment configs for AG News and IMDB
- `models/`: model implementations
- `train.py`: training loops for all methods
- `eval.py`: evaluation utilities
- `experiments/`: experiment outputs and planning notes

## Setup

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Available Configurations

| Method | AG News config | IMDB config |
| --- | --- | --- |
| DenseBERT | `configs/bert_baseline_agnews.yaml` | `configs/bert_baseline_imdb.yaml` |
| DeeBERT | `configs/deebert_agnews.yaml` | `configs/deebert_imdb.yaml` |
| Router-Tuning | `configs/router_tuning_agnews.yaml` | `configs/router_tuning_imdb.yaml` |
| HDC-BERT | `configs/hdc_agnews.yaml` | `configs/hdc_imdb.yaml` |

## Run Experiments

DenseBERT:

```bash
python main_densebert.py --config configs/bert_baseline_agnews.yaml
python main_densebert.py --config configs/bert_baseline_imdb.yaml
```

DeeBERT:

```bash
python main_deebert.py --config configs/deebert_agnews.yaml
python main_deebert.py --config configs/deebert_imdb.yaml
```

Router-Tuning:

```bash
python main_routerbert.py --config configs/router_tuning_agnews.yaml
python main_routerbert.py --config configs/router_tuning_imdb.yaml
```

HDC-BERT:

```bash
python main_hdcbert.py --config configs/hdc_agnews.yaml
python main_hdcbert.py --config configs/hdc_imdb.yaml
```

## Outputs

Each run writes results to `Path(output_root) / run_name`, where these values come from the selected YAML config or CLI overrides.

Typical run artifacts include:

- training logs
- best-model checkpoints such as `best_model.pt`, `best_model_step1.pt`, `best_model_step2.pt`, or `best_model_step3.pt`
- JSON result summaries written through `save_results(...)`

Examples of default output locations in the checked-in configs:

- DenseBERT AG News: `experiments/baseline/agnews`
- DenseBERT IMDB: `experiments/baseline/imdb`
- DeeBERT AG News: `experiments/deebert/agnews/run_01`
- Router-Tuning AG News: `outputs/router_tuning_agnews`
- HDC-BERT AG News: `outputs/hdc_bert_agnews`

## Notes

- Datasets are loaded with Hugging Face `datasets`.
- The default backbone is `bert-base-uncased`.
- You can override YAML fields from the command line, for example `--batch_size 16` or `--seed 123`.
- Some evaluation scripts in the repository are intended for Pareto-style analysis after training finishes.
