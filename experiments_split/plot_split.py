"""
Split layer ablation — clean publication-style figure.

Three panels (horizontal):
  1. Accuracy vs Split Layer
  2. Stage A Exit Rate vs Split Layer  (key parameter)
  3. FLOPs breakdown (Stage A + Stage B stacked) per dataset
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent

# ── FLOPs ─────────────────────────────────────────────────────────────
H, I_FFN = 768, 3072

def layer_flops(L):
    return (4*L*H**2 + 2*L**2*H) + (2*L*H*I_FFN)

def routed_layer_flops(L, k):
    return (4*k*L*H**2 + 2*k**2*L**2*H) + (2*L*H*I_FFN)

def hdc_flops(L, hist, keeps, split):
    lf = layer_flops(L)
    N  = sum(hist.values())
    fa = fb = 0
    for key, cnt in hist.items():
        if key == "final":
            fa += cnt * split * lf
            fb += cnt * sum(routed_layer_flops(L, k) for k in keeps)
        else:
            fa += cnt * int(key) * lf
    return fa / N, fb / N          # per-sample Stage A / Stage B FLOPs

# ── Data ──────────────────────────────────────────────────────────────
def load(dataset):
    seq = 128 if dataset == "agnews" else 256
    base = 12 * layer_flops(seq)
    rows = []
    for s in [2, 4, 6, 8, 10]:
        f = ROOT / f"experiments_split/{dataset}/split_{s}/results.json"
        r = json.loads(f.read_text())
        fed = r["flops_estimation_data"]
        fa, fb = hdc_flops(seq, fed["exit_histogram"],
                            fed["stage_b_eval_keep_rates"], s)
        rows.append({
            "split":     s,
            "acc":       r["hdc_inference"]["accuracy"] * 100,
            "exit_rate": fed["stage_a_exit_rate"] * 100,
            "fa_pct":    fa / base * 100,
            "fb_pct":    fb / base * 100,
            "total_pct": (fa + fb) / base * 100,
        })
    return rows

agnews = load("agnews")
imdb   = load("imdb")

SPLITS     = [2, 4, 6, 8, 10]
C_AG       = "#2196F3"
C_IMDB     = "#E91E63"
C_STAGE_A  = "#42A5F5"
C_STAGE_B  = "#66BB6A"
BASE_ACC   = {"AG News": 94.69, "IMDB": 92.32}

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("HDC-BERT Split Layer Ablation", fontsize=13, fontweight="bold", y=1.02)

# ── Panel 1: Accuracy vs Split Layer ─────────────────────────────────
ax = axes[0]
for label, rows, color in [("AG News", agnews, C_AG), ("IMDB", imdb, C_IMDB)]:
    sp = [r["split"] for r in rows]
    ac = [r["acc"]   for r in rows]
    ax.plot(sp, ac, marker="o", color=color, linewidth=2.2,
            markersize=8, label=label, zorder=3)
    for s, a in zip(sp, ac):
        ax.annotate(f"{a:.2f}%", (s, a),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha="center", color=color)
    ax.axhline(BASE_ACC[label], color=color, linestyle=":", linewidth=1, alpha=0.5)

ax.set_xticks(SPLITS)
ax.set_xlabel("Split Layer", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("(a) Accuracy", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# ── Panel 2: Stage A Exit Rate vs Split Layer ─────────────────────────
ax = axes[1]
x  = np.arange(len(SPLITS))
w  = 0.35
for i, (label, rows, color) in enumerate([("AG News", agnews, C_AG),
                                           ("IMDB",    imdb,   C_IMDB)]):
    rates = [r["exit_rate"] for r in rows]
    bars  = ax.bar(x + (i - 0.5) * w, rates, w, label=label,
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"s={s}" for s in SPLITS])
ax.set_ylabel("Stage A Exit Rate (%)", fontsize=11)
ax.set_title("(b) Stage A Exit Rate", fontsize=11, fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(fontsize=9)

# ── Panel 3: FLOPs breakdown per split, side-by-side datasets ────────
ax = axes[2]
n_splits = len(SPLITS)
group_w  = 0.38          # width of each dataset's bar
gap      = 0.08          # gap between the two datasets within a group
xs_ag   = np.arange(n_splits) - (group_w + gap) / 2
xs_imdb = np.arange(n_splits) + (group_w + gap) / 2

for xs, rows, alpha_a, alpha_b, label_suffix in [
        (xs_ag,   agnews, 1.0, 0.85, " (AG)"),
        (xs_imdb, imdb,   0.6, 0.5,  " (IMDB)")]:
    fa = [r["fa_pct"] for r in rows]
    fb = [r["fb_pct"] for r in rows]
    ax.bar(xs, fa, group_w, color=C_STAGE_A, alpha=alpha_a,
           edgecolor="white", linewidth=0.4,
           label="Stage A (early-exit)" + label_suffix)
    ax.bar(xs, fb, group_w, bottom=fa, color=C_STAGE_B, alpha=alpha_b,
           edgecolor="white", linewidth=0.4,
           label="Stage B (routing)" + label_suffix)
    for xi, (a, b) in enumerate(zip(fa, fb)):
        ax.text(xs[xi], a + b + 0.6, f"{a+b:.0f}%",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# dataset labels under groups
for xi in range(n_splits):
    ax.text(xs_ag[xi],   -6, "AG",   ha="center", fontsize=7.5, color=C_AG)
    ax.text(xs_imdb[xi], -6, "IMDB", ha="center", fontsize=7.5, color=C_IMDB)

ax.set_xticks(np.arange(n_splits))
ax.set_xticklabels([f"s={s}" for s in SPLITS])
ax.set_ylabel("FLOPs (% of DenseBERT)", fontsize=11)
ax.set_title("(c) FLOPs Breakdown", fontsize=11, fontweight="bold")

# deduplicated legend
handles = [
    plt.Rectangle((0,0),1,1, color=C_STAGE_A),
    plt.Rectangle((0,0),1,1, color=C_STAGE_B),
]
ax.legend(handles, ["Stage A (early-exit)", "Stage B (routing)"],
          fontsize=8.5, loc="upper right")

plt.tight_layout()
out = ROOT / "experiments_split/split_ablation.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ── Figure 2: FLOPs vs Accuracy (improved) ───────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5.5))

# manual label offsets to avoid overlapping:  (dx, dy) in points
OFFSETS_AG   = {2: (-8, -14), 4: (-8, -14), 6: (6, 7), 8: (6, 7), 10: (6, 7)}
OFFSETS_IMDB = {2: (6, 7), 4: (6, -12), 6: (-10, -14), 8: (6, 7), 10: (6, 7)}

for label, rows, color, base_acc, offsets in [
        ("AG News", agnews, C_AG,   94.66, OFFSETS_AG),
        ("IMDB",    imdb,   C_IMDB, 92.34, OFFSETS_IMDB)]:
    flops_pct = [r["total_pct"] for r in rows]
    accs      = [r["acc"]       for r in rows]

    # manually chosen best split per dataset
    best_split = {"AG News": 6, "IMDB": 8}[label]
    best_idx = next(i for i, r in enumerate(rows) if r["split"] == best_split)

    # plot all points as scatter (no connecting line — avoids zigzag)
    ax2.scatter(flops_pct, accs, color=color, s=70, zorder=3, alpha=0.7,
                edgecolors="white", linewidths=0.8, label=label)

    # highlight best split with a star
    ax2.scatter(flops_pct[best_idx], accs[best_idx], color=color,
                marker="*", s=350, zorder=5, edgecolors="white", linewidths=0.8)

    for r in rows:
        dx, dy = offsets[r["split"]]
        is_best = (r["split"] == rows[best_idx]["split"])
        txt = f"s={r['split']}"
        if is_best:
            txt += " (best)"
        ax2.annotate(txt, (r["total_pct"], r["acc"]),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=8.5, color=color,
                     fontweight="bold" if is_best else "normal")

    # baseline reference: 100% FLOPs
    ax2.plot(100, base_acc, marker="D", markersize=9, color=color,
             zorder=4, markeredgecolor="white", markeredgewidth=1.0)
    ax2.annotate(f"Dense\n{base_acc:.2f}%", (100, base_acc),
                 textcoords="offset points", xytext=(-45, -8),
                 fontsize=8, color=color, fontstyle="italic",
                 ha="center")

    # dashed line connecting best to baseline to show savings
    ax2.plot([flops_pct[best_idx], 100], [accs[best_idx], base_acc],
             color=color, linestyle="--", linewidth=1.0, alpha=0.4, zorder=2)

ax2.set_xlabel("FLOPs (% of Dense BERT)", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("HDC-BERT: FLOPs vs Accuracy Trade-off",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="lower right")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(True, alpha=0.25, linestyle="--")

fig2.tight_layout()
out2 = ROOT / "experiments_split/split_flops_accuracy.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
