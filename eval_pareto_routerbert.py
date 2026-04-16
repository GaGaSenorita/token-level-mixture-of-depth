"""
RouterBERT Pareto evaluation: evaluate a single checkpoint and append the
result to pareto.json. Call it immediately after training each keep_ratio
model; once evaluation is complete, the .pt file can be deleted to save disk
space.

Usage:
    python eval_pareto_routerbert.py \
        --ckpt        ./experiments_pareto/routerbert/agnews/keep_0.7/best_model_step2.pt \
        --keep_ratio  0.7 \
        --dataset     ag_news \
        --output_root ./experiments_pareto/routerbert/agnews \
        --max_length  128
"""
import argparse
import json
from pathlib import Path

import torch

from data import load_data
from eval import evaluate_router_full
from models.router_tuning_bert import RouterTuningBERTClassifier
from utils import set_seed, ensure_dir, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          type=str,  required=True)
    parser.add_argument("--keep_ratio",    type=float, required=True)
    parser.add_argument("--dataset",       type=str,  required=True,  help="ag_news or imdb")
    parser.add_argument("--output_root",   type=str,  required=True)
    parser.add_argument("--model_name",    type=str,  default="bert-base-uncased")
    parser.add_argument("--max_length",    type=int,  default=128)
    parser.add_argument("--routing_mode",  type=str,  default="token", choices=["token", "sample"])
    parser.add_argument("--seed",          type=int,  default=42)
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
        max_length=args.max_length
    )

    # ---- Model ----
    model = RouterTuningBERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        routing_mode=args.routing_mode,
        target_keep_ratio=args.keep_ratio,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # ---- Eval ----
    acc, eval_keep_rates, avg_keep_rate = evaluate_router_full(model, test_loader, device)

    # Attention FLOPs ratio: only attention is skipped, estimated using the formula in main_routerbert.py
    # baseline_attn = 2L(3H^2 + 2LH + H^2), routed = same with K=k*L
    # ratio = (k*8LH^2 + k^2*4L^2H) / (8LH^2 + 4L^2H)
    H, L = 768, args.max_length
    if avg_keep_rate is not None:
        k = avg_keep_rate
        attn_flops_ratio = round((k * 8 * L * H**2 + k**2 * 4 * L**2 * H) /
                                 (8 * L * H**2 + 4 * L**2 * H), 4)
    else:
        attn_flops_ratio = None

    entry = {
        "target_keep_ratio":    args.keep_ratio,
        "accuracy":             round(acc, 4),
        "avg_keep_rate":        round(avg_keep_rate, 4) if avg_keep_rate is not None else None,
        "attn_flops_ratio":     attn_flops_ratio,
        "per_layer_keep_rates": [
            round(r, 4) if r is not None else None
            for r in (eval_keep_rates or [])
        ],
    }
    print(f"keep_ratio={args.keep_ratio:.2f} | acc={acc:.4f} | avg_keep_rate={avg_keep_rate:.4f}"
          if avg_keep_rate is not None else
          f"keep_ratio={args.keep_ratio:.2f} | acc={acc:.4f} | avg_keep_rate=N/A")

    # ---- Append to pareto.json ----
    out_file = Path(args.output_root) / "pareto.json"
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
    else:
        results = {"dataset": args.dataset, "routing_mode": args.routing_mode, "sweep": []}

    results["seed"] = args.seed
    results["sweep"].append(entry)
    results["sweep"].sort(key=lambda x: x["target_keep_ratio"])

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Appended to {out_file}")


if __name__ == "__main__":
    main()
