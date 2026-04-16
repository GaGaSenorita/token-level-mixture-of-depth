# Token-Level Mixture of Depth

This repository studies dynamic-computation variants of BERT for text classification. The codebase compares a standard dense baseline against progressively more adaptive models that save computation by exiting early, routing tokens, or combining both ideas in a hierarchical design.

The project currently focuses on two datasets:

- `AG News`
- `IMDB`

## Experiment Goal

The main research question is whether adaptive computation can improve the accuracy-efficiency trade-off of `bert-base-uncased` on short and long text classification tasks.

The repository is organised around three experiment families:

- Main comparison runs: compare DenseBERT, DeeBERT, Router-Tuning, and HDC-BERT on AG News and IMDB.
- Pareto experiments: sweep inference or routing settings to trace accuracy-vs-compute trade-off curves.
- Split-layer ablation: test different HDC split points to see where early exit should end and token routing should begin.

## Implemented Methods

| Method | Idea | Entry point |
| --- | --- | --- |
| `DenseBERT` | Standard dense fine-tuning with no adaptive computation. | `main_densebert.py` |
| `DeeBERT` | Adds multiple off-ramp classifiers and exits early for easy samples. | `main_deebert.py` |
| `Router-Tuning` | Learns token-level routing so some tokens skip work in later layers. | `main_routerbert.py` |
| `HDC-BERT` | Uses early exit in shallow layers and token routing in deeper layers. | `main_hdcbert.py` |

## Script-First Workflow

The intended workflow in this repository is to launch experiments through the checked-in Bash scripts rather than invoking the Python entry points by hand every time.

If you are on Windows, run the scripts with Git Bash or WSL.

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run a script from the project root, for example:

```bash
bash run_baseline.sh
bash run_hdcbert.sh
```

## What Each `.sh` Script Does

| Script | Purpose | What it actually runs |
| --- | --- | --- |
| `run_baseline.sh` | Dense baseline runs for the main comparison table. | Runs `main_densebert.py` on AG News and IMDB for seeds `42`, `123`, and `456`. |
| `run_deebert.sh` | Main DeeBERT experiment runs. | Runs Stage 1 and Stage 2 once per dataset with `seed=42`. |
| `run_routerbert.sh` | Main Router-Tuning experiment runs. | Runs full Step 1 + Step 2 once, then reuses the saved Step 1 checkpoint to train Step 2 for seeds `123` and `456`. |
| `run_hdcbert.sh` | Main HDC-BERT experiment runs. | Runs Stage 1 + Stage 2 + Stage 3 for AG News and IMDB, then reuses the Stage 1 checkpoint for additional seeds. |
| `run_pareto_deebert.sh` | DeeBERT Pareto frontier generation. | Trains one model per dataset, then sweeps early-exit entropy thresholds at evaluation time. |
| `run_pareto_routerbert.sh` | Router-Tuning Pareto frontier generation. | Trains a shared Step 1 checkpoint once, sweeps `target_keep_ratio`, evaluates each setting, and removes intermediate checkpoints to save space. |
| `run_pareto_hdcbert.sh` | HDC-BERT Pareto frontier generation. | Reuses shared Stage 1 and Stage 2 checkpoints, sweeps `target_keep_ratio`, and evaluates multiple entropy thresholds for each trained checkpoint. |
| `run_best_split.sh` | HDC split-layer ablation. | Reuses one shared Step 1 checkpoint and evaluates several `split_layer` settings such as `2`, `4`, `6`, `8`, and `10`. |

## Where Experiment Results Go

The curated experiment scripts write into versioned experiment folders instead of a generic runtime output folder:

- `experiments/`: main experiment runs
- `experiments_pareto/`: Pareto sweeps
- `experiments_split/`: split-layer ablations

Several scripts intentionally reuse earlier checkpoints and delete intermediate `.pt` files after evaluation to reduce storage cost.

## Configurations

| Method | AG News config | IMDB config |
| --- | --- | --- |
| DenseBERT | `configs/bert_baseline_agnews.yaml` | `configs/bert_baseline_imdb.yaml` |
| DeeBERT | `configs/deebert_agnews.yaml` | `configs/deebert_imdb.yaml` |
| Router-Tuning | `configs/router_tuning_agnews.yaml` | `configs/router_tuning_imdb.yaml` |
| HDC-BERT | `configs/hdc_agnews.yaml` | `configs/hdc_imdb.yaml` |

## Optional Direct Entry-Point Usage

If you want to bypass the shell scripts and run the Python entry points directly, these are the main commands:

```bash
python main_densebert.py --config configs/bert_baseline_agnews.yaml
python main_deebert.py --config configs/deebert_agnews.yaml
python main_routerbert.py --config configs/router_tuning_agnews.yaml
python main_hdcbert.py --config configs/hdc_agnews.yaml
```

All entry points also accept CLI overrides such as `--seed`, `--batch_size`, `--target_keep_ratio`, or `--resume_step1_ckpt`.

## Repository Layout

- `configs/`: runnable YAML configurations
- `models/`: model definitions
- `train.py`: training loops for all methods
- `eval.py`: shared evaluation utilities
- `eval_pareto_*.py`: evaluation scripts for Pareto sweeps
- `experiments/`, `experiments_pareto/`, `experiments_split/`: experiment records and outputs
