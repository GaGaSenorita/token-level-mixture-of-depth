"""
HDC-BERT Pareto evaluation: evaluate a single Step 3 checkpoint across
multiple entropy_threshold values and append every (keep_ratio, threshold)
combination to pareto.json.

Usage:
    python eval_pareto_hdcbert.py \
        --ckpt               ./experiments_pareto/hdcbert/agnews/keep_0.7/best_model_step3.pt \
        --keep_ratio         0.7 \
        --dataset            ag_news \
        --output_root        ./experiments_pareto/hdcbert/agnews \
        --max_length         128 \
        --entropy_thresholds 0.05 0.2 0.5 0.9 1.38
"""
import argparse
import json
from pathlib import Path

import torch

from data import load_data
from eval import evaluate_hdc_routing_full, evaluate_hdc_inference
from models.hdc_bert import HDCBERTClassifier
from utils import set_seed, ensure_dir, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",               type=str,   required=True)
    parser.add_argument("--keep_ratio",         type=float, required=True)
    parser.add_argument("--dataset",            type=str,   required=True, help="ag_news or imdb")
    parser.add_argument("--output_root",        type=str,   required=True)
    parser.add_argument("--model_name",         type=str,   default="bert-base-uncased")
    parser.add_argument("--max_length",         type=int,   default=128)
    parser.add_argument("--split_layer",        type=int,   default=6)
    parser.add_argument("--tau",                type=float, default=0.5)
    parser.add_argument("--entropy_thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--seed",               type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    ensure_dir(Path(args.output_root))

    # ---- Data ----
    _, test_loader, num_labels = load_data(
        dataset=args.dataset,
        tokenizer_name=args.model_name,
        batch_size=32,
        max_length=args.max_length,
    )
    _, test_loader_ee, _ = load_data(
        dataset=args.dataset,
        tokenizer_name=args.model_name,
        batch_size=1,
        max_length=args.max_length,
    )

    # ---- Model (load once, sweep threshold at eval) ----
    model = HDCBERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        split_layer=args.split_layer,
        tau=args.tau,
        target_keep_ratio=args.keep_ratio,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # ---- Stage B routing stats (fixed and unaffected by the threshold) ----
    _, eval_keep_rates_b, avg_keep_rate_b = evaluate_hdc_routing_full(model, test_loader, device)
    H, L = 768, args.max_length
    k = avg_keep_rate_b
    attn_flops_ratio_b = round(
        (k * 8 * L * H**2 + k**2 * 4 * L**2 * H) /
        (8 * L * H**2 + 4 * L**2 * H), 4
    )

    # ---- Load / init pareto.json ----
    out_file = Path(args.output_root) / "pareto.json"
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
    else:
        results = {"dataset": args.dataset, "seed": args.seed, "sweep": []}

    # ---- Sweep entropy_threshold ----
    for threshold in args.entropy_thresholds:
        hdc_results = evaluate_hdc_inference(
            model, test_loader_ee, device,
            entropy_threshold=threshold,
        )
        entry = {
            "target_keep_ratio":      args.keep_ratio,
            "entropy_threshold":      threshold,
            "accuracy":               round(hdc_results["accuracy"], 4),
            "stage_a_exit_rate":      round(hdc_results["stage_a_exit_rate"], 4),
            "avg_exit_layer_a":       round(hdc_results["avg_exit_layer_a"], 4),
            "exit_histogram":         hdc_results["exit_histogram"],
            "avg_keep_rate_b":        round(avg_keep_rate_b, 4),
            "attn_flops_ratio_b":     attn_flops_ratio_b,
            "per_layer_keep_rates_b": [round(r, 4) for r in (eval_keep_rates_b or [])],
        }
        print(f"keep_ratio={args.keep_ratio:.2f} | threshold={threshold:.2f} | "
              f"acc={hdc_results['accuracy']:.4f} | exit_rate={hdc_results['stage_a_exit_rate']:.4f} | "
              f"keep_rate_b={avg_keep_rate_b:.4f}")
        results["sweep"].append(entry)

    results["sweep"].sort(key=lambda x: (x["target_keep_ratio"], x["entropy_threshold"]))

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Appended {len(args.entropy_thresholds)} entries to {out_file}")


if __name__ == "__main__":
    main()
