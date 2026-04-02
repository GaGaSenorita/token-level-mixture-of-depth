"""
Compute per-seed accuracy and FLOPs/sample for all 4 models × 2 datasets × 3 seeds.
Prints a summary table and saves results_summary.json.

Reuses FLOPs formulas and data loaders from flops.py.
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

# Allow importing from experiments/ directory
sys.path.insert(0, str(Path(__file__).parent))
from flops import (
    compute_baseline_flops,
    compute_deebert_flops,
    compute_routerbert_flops,
    compute_hdc_flops,
    load_baseline_results,
    load_model_results,
    to_gflops,
    EXPERIMENTS_DIR,
)


def compute_per_seed(dataset):
    """
    Returns a dict with per-seed + mean/std for all 4 models on a dataset.
    """
    baseline_runs = load_baseline_results(dataset)
    seq_len = baseline_runs[0]["args"]["max_length"]
    base_flops = compute_baseline_flops(seq_len)
    models = {}

    # ── DenseBERT ──────────────────────────────────────────────────────────
    acc_list = [r["best_test_acc"] for r in baseline_runs]
    gflops_list = [to_gflops(base_flops)] * len(baseline_runs)  # constant across seeds
    models["DenseBERT"] = {
        "acc_per_seed": acc_list,
        "acc_mean": float(np.mean(acc_list)),
        "acc_std": float(np.std(acc_list)),
        "gflops_per_seed": gflops_list,
        "gflops_mean": float(np.mean(gflops_list)),
        "gflops_std": float(np.std(gflops_list)),
        "flops_ratio": 1.0,
        "flops_saved": 0.0,
    }

    # ── DeeBERT ────────────────────────────────────────────────────────────
    deebert_runs = load_model_results("deebert", dataset)
    if deebert_runs:
        acc_list, gflops_list, exit_list = [], [], []
        for r in deebert_runs:
            avg_exit = r["best_avg_exit_layer"]
            f = compute_deebert_flops(seq_len, avg_exit)
            acc_list.append(r["best_ee_acc"])
            gflops_list.append(to_gflops(f))
            exit_list.append(avg_exit)
        mean_flops = np.mean([compute_deebert_flops(seq_len, r["best_avg_exit_layer"]) for r in deebert_runs])
        models["DeeBERT"] = {
            "acc_per_seed": acc_list,
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list)),
            "gflops_per_seed": gflops_list,
            "gflops_mean": float(np.mean(gflops_list)),
            "gflops_std": float(np.std(gflops_list)),
            "flops_ratio": float(mean_flops / base_flops),
            "flops_saved": float(1 - mean_flops / base_flops),
            "avg_exit_layer_per_seed": exit_list,
            "avg_exit_layer_mean": float(np.mean(exit_list)),
            "avg_exit_layer_std": float(np.std(exit_list)),
        }

    # ── RouterBERT ─────────────────────────────────────────────────────────
    router_runs = load_model_results("router_tuning", dataset)
    if router_runs:
        acc_list, gflops_list, keep_list = [], [], []
        for r in router_runs:
            keep_rates = r["flops_estimation_data"]["eval_keep_rates_per_layer"]
            f = compute_routerbert_flops(seq_len, keep_rates)
            acc_list.append(r["final_eval"]["accuracy"])
            gflops_list.append(to_gflops(f))
            keep_list.append(r["final_eval"]["avg_keep_rate"])
        mean_flops = np.mean([
            compute_routerbert_flops(seq_len, r["flops_estimation_data"]["eval_keep_rates_per_layer"])
            for r in router_runs
        ])
        models["RouterBERT"] = {
            "acc_per_seed": acc_list,
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list)),
            "gflops_per_seed": gflops_list,
            "gflops_mean": float(np.mean(gflops_list)),
            "gflops_std": float(np.std(gflops_list)),
            "flops_ratio": float(mean_flops / base_flops),
            "flops_saved": float(1 - mean_flops / base_flops),
            "avg_keep_rate_per_seed": keep_list,
            "avg_keep_rate_mean": float(np.mean(keep_list)),
            "avg_keep_rate_std": float(np.std(keep_list)),
        }

    # ── HDC-BERT ───────────────────────────────────────────────────────────
    hdc_runs = load_model_results("hdc", dataset)
    if hdc_runs:
        acc_list, gflops_list = [], []
        exit_rate_list, keep_b_list, exit_layer_list = [], [], []
        for r in hdc_runs:
            fed = r["flops_estimation_data"]
            hist = dict(fed["exit_histogram"])
            keep_rates = fed["stage_b_eval_keep_rates"]
            split = fed.get("split_layer", 6)
            f = compute_hdc_flops(seq_len, hist, keep_rates, split)
            acc_list.append(r["hdc_inference"]["accuracy"])
            gflops_list.append(to_gflops(f))
            exit_rate_list.append(fed["stage_a_exit_rate"])
            keep_b_list.append(fed["stage_b_avg_keep_rate"])
            exit_layer_list.append(fed["avg_exit_layer_a"])
        mean_flops = np.mean([
            compute_hdc_flops(
                seq_len,
                dict(r["flops_estimation_data"]["exit_histogram"]),
                r["flops_estimation_data"]["stage_b_eval_keep_rates"],
                r["flops_estimation_data"].get("split_layer", 6),
            )
            for r in hdc_runs
        ])
        models["HDC-BERT"] = {
            "acc_per_seed": acc_list,
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list)),
            "gflops_per_seed": gflops_list,
            "gflops_mean": float(np.mean(gflops_list)),
            "gflops_std": float(np.std(gflops_list)),
            "flops_ratio": float(mean_flops / base_flops),
            "flops_saved": float(1 - mean_flops / base_flops),
            "stage_a_exit_rate_per_seed": exit_rate_list,
            "stage_a_exit_rate_mean": float(np.mean(exit_rate_list)),
            "stage_a_exit_rate_std": float(np.std(exit_rate_list)),
            "avg_exit_layer_a_per_seed": exit_layer_list,
            "avg_exit_layer_a_mean": float(np.mean(exit_layer_list)),
            "avg_exit_layer_a_std": float(np.std(exit_layer_list)),
            "avg_keep_rate_b_per_seed": keep_b_list,
            "avg_keep_rate_b_mean": float(np.mean(keep_b_list)),
            "avg_keep_rate_b_std": float(np.std(keep_b_list)),
        }

    return {"seq_len": seq_len, "models": models}


def print_table(dataset, data):
    seq_len = data["seq_len"]
    models = data["models"]
    n_seeds = max(len(v["acc_per_seed"]) for v in models.values())

    print(f"\n{'='*100}")
    print(f"  {dataset.upper()}  (seq_len={seq_len})")
    print(f"{'='*100}")

    # Header
    seed_acc_hdrs = "  ".join(f"S{i+1} Acc" for i in range(n_seeds))
    seed_gf_hdrs  = "  ".join(f"S{i+1} GF" for i in range(n_seeds))
    header = (
        f"{'Method':<14}  "
        f"{seed_acc_hdrs}  {'Acc Mean±Std':<16}  "
        f"{seed_gf_hdrs}  {'GFLOPs Mean±Std':<18}  "
        f"{'FLOPs%':>7}  {'Saved%':>7}"
    )
    print(header)
    print("-" * len(header))

    for method, m in models.items():
        acc_seeds = "  ".join(f"{a*100:6.2f}" for a in m["acc_per_seed"])
        acc_mean_str = f"{m['acc_mean']*100:.2f}±{m['acc_std']*100:.2f}"
        gf_seeds = "  ".join(f"{g:6.3f}" for g in m["gflops_per_seed"])
        gf_mean_str = f"{m['gflops_mean']:.3f}±{m['gflops_std']:.3f}"
        flops_pct = f"{m['flops_ratio']*100:.1f}%"
        saved_pct = f"{m['flops_saved']*100:.1f}%"
        print(
            f"{method:<14}  "
            f"{acc_seeds}  {acc_mean_str:<16}  "
            f"{gf_seeds}  {gf_mean_str:<18}  "
            f"{flops_pct:>7}  {saved_pct:>7}"
        )

    # Model-specific extras
    print()
    print("  Model-specific efficiency metrics:")
    for method, m in models.items():
        extras = []
        if "avg_exit_layer_per_seed" in m:
            seeds_str = ", ".join(f"{v:.2f}" for v in m["avg_exit_layer_per_seed"])
            extras.append(f"avg_exit_layer=[{seeds_str}]  mean={m['avg_exit_layer_mean']:.2f}±{m['avg_exit_layer_std']:.2f}")
        if "avg_keep_rate_per_seed" in m:
            seeds_str = ", ".join(f"{v:.3f}" for v in m["avg_keep_rate_per_seed"])
            extras.append(f"avg_keep_rate=[{seeds_str}]  mean={m['avg_keep_rate_mean']:.3f}±{m['avg_keep_rate_std']:.3f}")
        if "stage_a_exit_rate_per_seed" in m:
            seeds_str = ", ".join(f"{v:.3f}" for v in m["stage_a_exit_rate_per_seed"])
            extras.append(f"stage_a_exit_rate=[{seeds_str}]  mean={m['stage_a_exit_rate_mean']:.3f}±{m['stage_a_exit_rate_std']:.3f}")
        if "avg_keep_rate_b_per_seed" in m:
            seeds_str = ", ".join(f"{v:.3f}" for v in m["avg_keep_rate_b_per_seed"])
            extras.append(f"avg_keep_rate_b=[{seeds_str}]  mean={m['avg_keep_rate_b_mean']:.3f}±{m['avg_keep_rate_b_std']:.3f}")
        if extras:
            print(f"  {method:<14}: " + "  |  ".join(extras))


if __name__ == "__main__":
    all_results = {}
    for ds in ["agnews", "imdb"]:
        try:
            data = compute_per_seed(ds)
            all_results[ds] = data
            print_table(ds, data)
        except FileNotFoundError as e:
            print(f"Skipping {ds}: {e}")

    # Save JSON summary
    out_path = EXPERIMENTS_DIR / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
