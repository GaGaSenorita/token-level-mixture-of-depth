"""
DeeBERT Pareto sweep: scan multiple entropy thresholds on a trained model
and record the accuracy and avg_exit_layer at each threshold for plotting the
trade-off curve.

Usage:
    python eval_pareto_deebert.py \
        --ckpt experiments/deebert/agnews/run_01/best_model_step2.pt \
        --dataset ag_news \
        --run_name run_01 \
        --output_root ./experiments_pareto/deebert/agnews
"""
import argparse
import json
from pathlib import Path

import torch

from data import load_data
from eval import evaluate_early_exit
from models import DeeBERTClassifier
from utils import set_seed, ensure_dir, get_device


# max entropy = ln(C): IMDB(2-class)=0.693, AGNews(4-class)=1.386
THRESHOLDS = {
    "imdb":    [0.05, 0.2, 0.4, 0.6, 0.69],   # strict -> close to maximum entropy
    "ag_news": [0.05, 0.2, 0.5, 0.9, 1.38],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         type=str, required=True,  help="Path to best_model_step2.pt")
    parser.add_argument("--dataset",      type=str, required=True,  help="ag_news or imdb")
    parser.add_argument("--run_name",     type=str, default="run_01")
    parser.add_argument("--output_root",  type=str, required=True)
    parser.add_argument("--model_name",   type=str, default="bert-base-uncased")
    parser.add_argument("--max_length",   type=int, default=128)
    parser.add_argument("--seed",         type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_root) / args.run_name
    ensure_dir(output_dir)

    # ---- Data (batch_size=1 for sample-level exit) ----
    _, test_loader, num_labels = load_data(
        dataset=args.dataset,
        tokenizer_name=args.model_name,
        batch_size=1,
        max_length=args.max_length
    )

    # ---- Model ----
    model = DeeBERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # ---- Threshold sweep ----
    sweep = []
    for thr in THRESHOLDS[args.dataset]:
        ee_acc, avg_exit, exit_hist = evaluate_early_exit(
            model, test_loader, device,
            entropy_threshold=thr,
            fn_name="forward_early_exit_batchwise"
        )
        entry = {
            "threshold":      thr,
            "accuracy":       round(ee_acc, 4),
            "avg_exit_layer": round(avg_exit, 4),
            "exit_histogram": exit_hist,
        }
        sweep.append(entry)
        print(f"thr={thr:.3f} | acc={ee_acc:.4f} | avg_exit={avg_exit:.2f}")

    # ---- Save ----
    results = {
        "ckpt":    args.ckpt,
        "dataset": args.dataset,
        "seed":    args.seed,
        "sweep":   sweep,
    }
    out_file = output_dir / "pareto.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
