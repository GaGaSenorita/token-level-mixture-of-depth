"""
Plot Pareto curves for DeeBERT, RouterBERT, and HDC-BERT.

Data sources:
  - DeeBERT:    experiments_pareto/deebert/{ds}/run_01/pareto.json
  - RouterBERT: experiments_pareto/routerbert/{ds}/pareto.json
  - HDC-BERT:   experiments_split/{ds}/split_{2,4,6,8,10}/results.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── project root ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── FLOPs formulas (same as experiments/flops.py) ────────────────────
H, I_FFN = 768, 3072

def attn_flops(L):
    return 4 * L * H**2 + 2 * L**2 * H

def ffn_flops(L):
    return 2 * L * H * I_FFN

def layer_flops(L):
    return attn_flops(L) + ffn_flops(L)

def routed_layer_flops(L, k):
    return (4 * k * L * H**2 + 2 * k**2 * L**2 * H) + ffn_flops(L)

def baseline_flops(L):
    return 12 * layer_flops(L)

def deebert_flops_from_hist(L, exit_histogram):
    """Compute average FLOPs/sample from exit histogram dict."""
    lf = layer_flops(L)
    total_samples = sum(exit_histogram.values())
    total_flops = 0
    for key, count in exit_histogram.items():
        total_flops += count * int(key) * lf
    return total_flops / total_samples

def routerbert_flops_from_keeps(L, per_layer_keep_rates):
    return sum(routed_layer_flops(L, k) for k in per_layer_keep_rates)

def hdc_flops(L, exit_histogram, stage_b_keep_rates, split_layer):
    lf = layer_flops(L)
    total_samples = sum(exit_histogram.values())
    total_flops = 0
    for key, count in exit_histogram.items():
        if key == "final":
            stage_a = split_layer * lf
            stage_b = sum(routed_layer_flops(L, k) for k in stage_b_keep_rates)
            total_flops += count * (stage_a + stage_b)
        else:
            total_flops += count * int(key) * lf
    return total_flops / total_samples

# ── Data loading ──────────────────────────────────────────────────────

def load_deebert_pareto(dataset):
    f = ROOT / "experiments_pareto/deebert" / dataset / "run_01/pareto.json"
    data = json.loads(f.read_text())
    seq_len = 128 if dataset == "agnews" else 256
    base = baseline_flops(seq_len)
    points = []
    for pt in data["sweep"]:
        hist = {k: v for k, v in pt["exit_histogram"].items()}
        flops = deebert_flops_from_hist(seq_len, hist)
        points.append((flops / base * 100, pt["accuracy"] * 100))
    return points

def load_routerbert_pareto(dataset):
    f = ROOT / "experiments_pareto/routerbert" / dataset / "pareto.json"
    data = json.loads(f.read_text())
    seq_len = 128 if dataset == "agnews" else 256
    base = baseline_flops(seq_len)
    points = []
    for pt in data["sweep"]:
        flops = routerbert_flops_from_keeps(seq_len, pt["per_layer_keep_rates"])
        points.append((flops / base * 100, pt["accuracy"] * 100))
    return points

def load_hdc_pareto(dataset):
    split_dir = ROOT / "experiments_split" / dataset
    seq_len = 128 if dataset == "agnews" else 256
    base = baseline_flops(seq_len)
    points = []
    for split in [2, 4, 6, 8, 10]:
        f = split_dir / f"split_{split}/results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        fed = r["flops_estimation_data"]
        hist = fed["exit_histogram"]
        keeps = fed["stage_b_eval_keep_rates"]
        flops = hdc_flops(seq_len, hist, keeps, split)
        acc = r["hdc_inference"]["accuracy"] * 100
        points.append((flops / base * 100, acc))
    return points

# ── Plotting ──────────────────────────────────────────────────────────

DATASET_LABELS = {"agnews": "AG News (L=128)", "imdb": "IMDB (L=256)"}
COLORS = {
    "DeeBERT":   "#2196F3",   # blue
    "RouterBERT": "#FF5722",  # orange-red
    "HDC-BERT":  "#4CAF50",   # green
}
MARKERS = {"DeeBERT": "o", "RouterBERT": "s", "HDC-BERT": "^"}

def plot_dataset(ax, dataset, show_legend=True):
    deebert  = sorted(load_deebert_pareto(dataset),   key=lambda x: x[0])
    router   = sorted(load_routerbert_pareto(dataset), key=lambda x: x[0])
    hdc      = sorted(load_hdc_pareto(dataset),        key=lambda x: x[0])

    # also add baseline point
    base_acc = {"agnews": 94.69, "imdb": 92.32}[dataset]

    for name, pts in [("DeeBERT", deebert), ("RouterBERT", router), ("HDC-BERT", hdc)]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c  = COLORS[name]
        m  = MARKERS[name]
        ax.plot(xs, ys, color=c, marker=m, linewidth=2,
                markersize=7, label=name, zorder=3)

    # baseline reference
    ax.axhline(base_acc, color="gray", linestyle="--", linewidth=1.2,
               label="DenseBERT (100%)", zorder=1)
    ax.scatter([100], [base_acc], color="gray", marker="*", s=120, zorder=4)

    ax.set_title(DATASET_LABELS[dataset], fontsize=13, fontweight="bold")
    ax.set_xlabel("FLOPs (% of DenseBERT)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    if show_legend:
        ax.legend(fontsize=10, loc="lower right")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_dataset(axes[0], "agnews", show_legend=True)
plot_dataset(axes[1], "imdb",   show_legend=False)

fig.suptitle("Accuracy–Efficiency Pareto Curves", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()

out = ROOT / "experiments_pareto/pareto_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
