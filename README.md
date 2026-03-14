# DenseBERT & DeeBERT on AG News

This project implements two BERT-based text classification baselines on the AG News dataset:

- **DenseBERT**: standard dense fine-tuning of `bert-base-uncased` for 4-way news classification.
- **DeeBERT**: a two-stage training pipeline that adds multiple early-exit “off-ramps” to reduce average inference cost while keeping accuracy competitive.

The code is organized around lightweight entrypoints (`main_densebert.py`, `main_deebert.py`) that load YAML configs, build data loaders, initialize models, run training, and save results to `./outputs`.

## Quickstart

Install dependencies (from the project root):

```bash
pip install -r requirements.txt
```

## Run DenseBERT

Uses the existing config at `configs/bert_baseline.yaml`:

```bash
python main_densebert.py --config configs/bert_baseline.yaml
```

Outputs will be written under `./outputs/dense_bert_finetune_agnews/` by default.

## Run DeeBERT

Uses the existing config at `configs/deebert.yaml`:

```bash
python main_deebert.py --config configs/deebert.yaml
```

Outputs will be written under `./outputs/deebert_training_agnews/` by default.

### Optional: evaluate early-exit metrics

Enable early-exit evaluation during/after stage 2:

```bash
python main_deebert.py --config configs/deebert.yaml --eval_early_exit
```

## Notes

- Both entrypoints download AG News via Hugging Face `datasets` and cache it automatically.
- You can override any YAML field via CLI flags (e.g., `--batch_size 16`).
