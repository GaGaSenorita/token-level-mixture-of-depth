"""
Pareto trade-off analysis: FLOPs vs Accuracy for DeeBERT, RouterBERT, HDC-BERT.

Outputs:
  - pareto_summary.json  : all pareto points + per-model Pareto frontiers
  - pareto_curves.png    : publication-quality trade-off figure (2 datasets)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).parent.parent
PARETO_DIR = Path(__file__).parent

# ── FLOPs formulas ────────────────────────────────────────────────────
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
    lf = layer_flops(L)
    total_samples = sum(exit_histogram.values())
    total_flops = sum(count * int(key) * lf for key, count in exit_histogram.items())
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

SEQ_LEN   = {"agnews": 128, "imdb": 256}
SPLIT_LAYER = {"agnews": 6, "imdb": 8}

# ── Pareto frontier extraction ────────────────────────────────────────

def pareto_frontier(points):
    """
    Given list of (flops_pct, acc_pct) points, return the non-dominated
    subset (lower FLOPs is better, higher accuracy is better), sorted by FLOPs.
    """
    pts = sorted(points, key=lambda p: p[0])  # sort by FLOPs ascending
    frontier = []
    best_acc = -1
    for p in pts:
        if p[1] > best_acc:
            frontier.append(p)
            best_acc = p[1]
    return frontier

# ── Data loading ──────────────────────────────────────────────────────

def load_deebert_points(dataset):
    f = PARETO_DIR / "deebert" / dataset / "run_01/pareto.json"
    data = json.loads(f.read_text())
    L = SEQ_LEN[dataset]
    base = baseline_flops(L)
    points = []
    for pt in data["sweep"]:
        hist = dict(pt["exit_histogram"])
        f_val = deebert_flops_from_hist(L, hist)
        points.append({
            "flops_pct": f_val / base * 100,
            "gflops": f_val / 1e9,
            "accuracy": pt["accuracy"] * 100,
            "avg_exit_layer": pt["avg_exit_layer"],
            "threshold": pt["threshold"],
        })
    return points

def load_routerbert_points(dataset):
    f = PARETO_DIR / "routerbert" / dataset / "pareto.json"
    data = json.loads(f.read_text())
    L = SEQ_LEN[dataset]
    base = baseline_flops(L)
    points = []
    for pt in data["sweep"]:
        f_val = routerbert_flops_from_keeps(L, pt["per_layer_keep_rates"])
        points.append({
            "flops_pct": f_val / base * 100,
            "gflops": f_val / 1e9,
            "accuracy": pt["accuracy"] * 100,
            "avg_keep_rate": pt["avg_keep_rate"],
            "target_keep_ratio": pt["target_keep_ratio"],
        })
    return points

def load_hdc_points(dataset):
    f = PARETO_DIR / "hdcbert" / dataset / "pareto.json"
    data = json.loads(f.read_text())
    L = SEQ_LEN[dataset]
    split = SPLIT_LAYER[dataset]
    base = baseline_flops(L)
    points = []
    for pt in data["sweep"]:
        hist = dict(pt["exit_histogram"])
        keeps = pt["per_layer_keep_rates_b"]
        f_val = hdc_flops(L, hist, keeps, split)
        points.append({
            "flops_pct": f_val / base * 100,
            "gflops": f_val / 1e9,
            "accuracy": pt["accuracy"] * 100,
            "target_keep_ratio": pt["target_keep_ratio"],
            "entropy_threshold": pt["entropy_threshold"],
            "stage_a_exit_rate": pt["stage_a_exit_rate"],
            "avg_keep_rate_b": pt["avg_keep_rate_b"],
        })
    return points

# ── Baseline accuracy (from main experiments) ─────────────────────────
BASELINE_ACC = {"agnews": 94.69, "imdb": 92.32}

# ── Styling ───────────────────────────────────────────────────────────
STYLE = {
    "DeeBERT":    {"color": "#2563EB", "marker": "o", "zorder": 4},
    "RouterBERT": {"color": "#DC2626", "marker": "s", "zorder": 4},
    "HDC-BERT":   {"color": "#16A34A", "marker": "^", "zorder": 5},
}

# ── Plot one dataset ──────────────────────────────────────────────────

def plot_dataset(ax, dataset, dee_pts, rt_pts, hdc_pts, show_legend=True, show_ylabel=True):
    L = SEQ_LEN[dataset]
    base_acc = BASELINE_ACC[dataset]
    base_gf  = baseline_flops(L) / 1e9

    model_data = [
        ("DeeBERT",    dee_pts),
        ("RouterBERT", rt_pts),
        ("HDC-BERT",   hdc_pts),
    ]

    for name, pts in model_data:
        s = STYLE[name]
        xs = [p["flops_pct"] for p in pts]
        ys = [p["accuracy"]  for p in pts]

        # For HDC: show all 25 points faintly, then overlay Pareto frontier
        if name == "HDC-BERT":
            ax.scatter(xs, ys, color=s["color"], alpha=0.18, s=22, zorder=2)
            frontier = pareto_frontier(list(zip(xs, ys)))
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, color=s["color"], marker=s["marker"],
                    linewidth=2.2, markersize=8, zorder=s["zorder"],
                    label=name, markeredgecolor="white", markeredgewidth=0.8)
        else:
            sorted_pts = sorted(zip(xs, ys), key=lambda p: p[0])
            sx = [p[0] for p in sorted_pts]
            sy = [p[1] for p in sorted_pts]
            ax.plot(sx, sy, color=s["color"], marker=s["marker"],
                    linewidth=2.2, markersize=8, zorder=s["zorder"],
                    label=name, markeredgecolor="white", markeredgewidth=0.8)

    # DenseBERT baseline
    ax.axhline(base_acc, color="#6B7280", linestyle="--", linewidth=1.4,
               alpha=0.7, zorder=1)
    ax.scatter([100], [base_acc], color="#6B7280", marker="*",
               s=160, zorder=6, label="DenseBERT (100% FLOPs)")

    # Axis labels and title
    ds_labels = {"agnews": "AG News  (seq_len=128)", "imdb": "IMDB  (seq_len=256)"}
    ax.set_title(ds_labels[dataset], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("FLOPs  (% of DenseBERT)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)

    ax.grid(True, color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0, right=110)

    # Shade "better region" (top-left is ideal)
    ax.fill_betweenx([base_acc - 0.5, base_acc + 0.5],
                     0, 100, alpha=0.04, color="#16A34A", zorder=0)

    if show_legend:
        ax.legend(fontsize=9.5, loc="lower right", framealpha=0.9,
                  edgecolor="#D1D5DB", fancybox=True)

    # Annotate the best HDC point (highest acc on frontier)
    frontier = pareto_frontier([(p["flops_pct"], p["accuracy"]) for p in hdc_pts])
    # Annotate leftmost HDC point (most aggressive compression)
    min_pt = min(frontier, key=lambda p: p[0])
    ax.annotate(
        f"HDC\n{min_pt[0]:.0f}% FLOPs\n{min_pt[1]:.1f}% Acc",
        xy=min_pt, xytext=(min_pt[0] + 4, min_pt[1] - 2.5),
        fontsize=7.5, color=STYLE["HDC-BERT"]["color"],
        arrowprops=dict(arrowstyle="->", color=STYLE["HDC-BERT"]["color"],
                        lw=0.9, connectionstyle="arc3,rad=0.1"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=STYLE["HDC-BERT"]["color"],
                  alpha=0.85, lw=0.8),
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    summary = {}

    for dataset in ["agnews", "imdb"]:
        dee_pts = load_deebert_points(dataset)
        rt_pts  = load_routerbert_points(dataset)
        hdc_pts = load_hdc_points(dataset)

        dee_frontier = pareto_frontier([(p["flops_pct"], p["accuracy"]) for p in dee_pts])
        rt_frontier  = pareto_frontier([(p["flops_pct"], p["accuracy"]) for p in rt_pts])
        hdc_frontier = pareto_frontier([(p["flops_pct"], p["accuracy"]) for p in hdc_pts])

        summary[dataset] = {
            "seq_len": SEQ_LEN[dataset],
            "baseline_acc": BASELINE_ACC[dataset],
            "baseline_gflops": round(baseline_flops(SEQ_LEN[dataset]) / 1e9, 3),
            "DeeBERT": {
                "all_points": dee_pts,
                "pareto_frontier": [{"flops_pct": p[0], "accuracy": p[1]} for p in dee_frontier],
            },
            "RouterBERT": {
                "all_points": rt_pts,
                "pareto_frontier": [{"flops_pct": p[0], "accuracy": p[1]} for p in rt_frontier],
            },
            "HDC-BERT": {
                "all_points": hdc_pts,
                "pareto_frontier": [{"flops_pct": p[0], "accuracy": p[1]} for p in hdc_frontier],
            },
        }

    # Save JSON
    out_json = PARETO_DIR / "pareto_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#FAFAFA")

    for i, dataset in enumerate(["agnews", "imdb"]):
        dee_pts = [p for p in summary[dataset]["DeeBERT"]["all_points"]]
        rt_pts  = [p for p in summary[dataset]["RouterBERT"]["all_points"]]
        hdc_pts = [p for p in summary[dataset]["HDC-BERT"]["all_points"]]
        plot_dataset(
            axes[i], dataset, dee_pts, rt_pts, hdc_pts,
            show_legend=(i == 0),
            show_ylabel=(i == 0),
        )

    # Shared legend on right subplot
    legend_elements = [
        Line2D([0], [0], color=STYLE["DeeBERT"]["color"],    marker="o", linewidth=2,
               markersize=7, markeredgecolor="white", label="DeeBERT"),
        Line2D([0], [0], color=STYLE["RouterBERT"]["color"], marker="s", linewidth=2,
               markersize=7, markeredgecolor="white", label="RouterBERT"),
        Line2D([0], [0], color=STYLE["HDC-BERT"]["color"],   marker="^", linewidth=2,
               markersize=7, markeredgecolor="white", label="HDC-BERT (frontier)"),
        Line2D([0], [0], color=STYLE["HDC-BERT"]["color"],   marker="o", linewidth=0,
               markersize=6, alpha=0.35, label="HDC-BERT (all configs)"),
        Line2D([0], [0], color="#6B7280", linestyle="--", linewidth=1.4,
               label="DenseBERT baseline"),
        Line2D([0], [0], color="#6B7280", marker="*", linewidth=0,
               markersize=10, label="DenseBERT (100% FLOPs)"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=9.5, loc="lower right",
                   framealpha=0.92, edgecolor="#D1D5DB", fancybox=True)

    fig.suptitle("Accuracy–Efficiency Pareto Trade-off", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1])

    out_png = PARETO_DIR / "pareto_curves.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out_png}")
    plt.close()

    # ── Print summary table ───────────────────────────────────────────
    for dataset in ["agnews", "imdb"]:
        print(f"\n{'='*65}")
        print(f"  {dataset.upper()}  —  Pareto Points")
        print(f"{'='*65}")
        for model in ["DeeBERT", "RouterBERT", "HDC-BERT"]:
            pts = summary[dataset][model]["all_points"]
            print(f"\n  {model}  ({len(pts)} points)")
            print(f"  {'FLOPs%':>7}  {'GFLOPs':>7}  {'Acc%':>7}  Notes")
            print(f"  {'-'*50}")
            for p in sorted(pts, key=lambda x: x["flops_pct"]):
                notes = ""
                if "avg_exit_layer" in p:
                    notes = f"exit_layer={p['avg_exit_layer']:.2f}"
                elif "target_keep_ratio" in p and "entropy_threshold" in p:
                    notes = f"keep={p['target_keep_ratio']:.1f} thr={p['entropy_threshold']:.2f}"
                elif "target_keep_ratio" in p:
                    notes = f"keep={p['target_keep_ratio']:.1f}"
                print(f"  {p['flops_pct']:>7.1f}  {p['gflops']:>7.3f}  {p['accuracy']:>7.2f}  {notes}")


if __name__ == "__main__":
    main()
