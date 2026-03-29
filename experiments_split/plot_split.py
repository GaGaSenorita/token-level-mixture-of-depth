"""
HDC-BERT Split Layer Ablation — publication-quality figure.

Three-panel figure embedding key conclusions:
  (a) FLOPs vs Accuracy trade-off  → "sweet spot" is clear
  (b) Stage A exit rate vs split   → AG News exits early, IMDB barely does
  (c) FLOPs breakdown (Stage A / Stage B stacked) → where computation goes
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

ROOT     = Path(__file__).parent.parent
SPLIT_DIR = Path(__file__).parent

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
    return fa / N, fb / N

# ── Data loading ───────────────────────────────────────────────────────
SEQ      = {"agnews": 128, "imdb": 256}
BASELINE = {"agnews": 94.69, "imdb": 92.32}
BEST_SPLIT = {"agnews": 6, "imdb": 8}   # chosen from ablation
SPLITS   = [2, 4, 6, 8, 10]

def load(dataset):
    L    = SEQ[dataset]
    base = 12 * layer_flops(L)
    rows = []
    for s in SPLITS:
        f = SPLIT_DIR / dataset / f"split_{s}/results.json"
        r = json.loads(f.read_text())
        fed = r["flops_estimation_data"]
        fa, fb = hdc_flops(L, fed["exit_histogram"],
                           fed["stage_b_eval_keep_rates"], s)
        rows.append({
            "split":      s,
            "acc":        r["hdc_inference"]["accuracy"] * 100,
            "exit_rate":  fed["stage_a_exit_rate"] * 100,
            "keep_b":     fed["stage_b_avg_keep_rate"],
            "avg_exit_a": fed["avg_exit_layer_a"],
            "fa_pct":     fa / base * 100,
            "fb_pct":     fb / base * 100,
            "total_pct":  (fa + fb) / base * 100,
        })
    return rows

agnews = load("agnews")
imdb   = load("imdb")

# ── Colours ────────────────────────────────────────────────────────────
C_AG      = "#2563EB"   # blue
C_IMDB    = "#DC2626"   # red
C_STA     = "#60A5FA"   # light blue – Stage A
C_STB     = "#34D399"   # teal-green  – Stage B
BEST_CLR  = "#F59E0B"   # amber – highlights best split

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("#FAFAFA")

# ══════════════════════════════════════════════════════════════════════
# Panel (a): FLOPs % vs Accuracy — "sweet spot"
# ══════════════════════════════════════════════════════════════════════
ax = axes[0]

for label, rows, color, base_acc in [
        ("AG News", agnews, C_AG,   BASELINE["agnews"]),
        ("IMDB",    imdb,   C_IMDB, BASELINE["imdb"])]:

    flops = [r["total_pct"] for r in rows]
    accs  = [r["acc"]       for r in rows]
    splits = [r["split"] for r in rows]
    best_s = BEST_SPLIT[label.split()[0].lower() if "AG" not in label else "agnews"]
    best_s = BEST_SPLIT["agnews" if "AG" in label else "imdb"]

    # connect all points with a thin line
    ax.plot(flops, accs, color=color, linewidth=1.4, alpha=0.5, zorder=2)

    # scatter each point
    for x, y, s in zip(flops, accs, splits):
        is_best = (s == best_s)
        ax.scatter(x, y,
                   color=BEST_CLR if is_best else color,
                   s=130 if is_best else 60,
                   marker="*" if is_best else "o",
                   zorder=5 if is_best else 3,
                   edgecolors="white", linewidths=0.8)
        offset = (4, 5) if x < 70 else (-22, 5)
        weight = "bold" if is_best else "normal"
        txt = f"s={s}{'★' if is_best else ''}"
        ax.annotate(txt, (x, y),
                    textcoords="offset points", xytext=offset,
                    fontsize=8, color=BEST_CLR if is_best else color,
                    fontweight=weight)

    # baseline (100% FLOPs)
    ax.scatter(100, base_acc, marker="D", s=80, color=color,
               zorder=4, edgecolors="white", linewidths=0.8)
    ax.annotate(f"Dense\n{base_acc:.1f}%", (100, base_acc),
                textcoords="offset points", xytext=(-6, -22),
                fontsize=7.5, color=color, ha="center", style="italic")

ax.set_xlabel("FLOPs  (% of DenseBERT)", fontsize=10)
ax.set_ylabel("Test Accuracy (%)", fontsize=10)
ax.set_title("(a)  Accuracy–FLOPs Trade-off\n(★ = chosen split)", fontsize=10, fontweight="bold")
ax.grid(True, color="#E5E7EB", linewidth=0.7)

legend_elems = [
    Line2D([0],[0], color=C_AG,   marker="o", linewidth=1.4, markersize=6, label="AG News"),
    Line2D([0],[0], color=C_IMDB, marker="o", linewidth=1.4, markersize=6, label="IMDB"),
    Line2D([0],[0], color=BEST_CLR, marker="*", linewidth=0, markersize=10, label="Best split"),
]
ax.legend(handles=legend_elems, fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor="#D1D5DB")

# ══════════════════════════════════════════════════════════════════════
# Panel (b): Stage A Exit Rate vs Split Layer
# — Key story: AG News exits aggressively; IMDB barely does at low splits
# ══════════════════════════════════════════════════════════════════════
ax = axes[1]

for label, rows, color in [("AG News", agnews, C_AG), ("IMDB", imdb, C_IMDB)]:
    sp    = [r["split"]     for r in rows]
    rates = [r["exit_rate"] for r in rows]
    ax.plot(sp, rates, color=color, marker="o", linewidth=2.2,
            markersize=8, label=label, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(sp, rates, alpha=0.10, color=color)
    for s, v in zip(sp, rates):
        ax.annotate(f"{v:.0f}%", (s, v),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha="center", color=color, fontweight="bold")

# Annotate the key insight: IMDB barely exits at s=2
ax.annotate(
    "IMDB: ~0% exit\nat split=2\n(too few layers\nto decide)",
    xy=(2, imdb[0]["exit_rate"]), xytext=(3.2, 18),
    fontsize=7.5, color=C_IMDB,
    arrowprops=dict(arrowstyle="->", color=C_IMDB, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_IMDB, alpha=0.85, lw=0.8),
)
ax.annotate(
    "AG News: 85%+ exit\neven at split=6",
    xy=(6, agnews[2]["exit_rate"]), xytext=(6.8, 65),
    fontsize=7.5, color=C_AG,
    arrowprops=dict(arrowstyle="->", color=C_AG, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_AG, alpha=0.85, lw=0.8),
)

ax.set_xticks(SPLITS)
ax.set_xlabel("Split Layer", fontsize=10)
ax.set_ylabel("Stage A Exit Rate (%)", fontsize=10)
ax.set_title("(b)  Stage A Exit Rate vs Split\n(early exit behavior)", fontsize=10, fontweight="bold")
ax.set_ylim(-5, 108)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9, edgecolor="#D1D5DB")
ax.grid(True, color="#E5E7EB", linewidth=0.7)

# ══════════════════════════════════════════════════════════════════════
# Panel (c): Stacked FLOPs breakdown — where does computation go?
# ══════════════════════════════════════════════════════════════════════
ax = axes[2]

x     = np.arange(len(SPLITS))
w     = 0.32
gap   = 0.06
xs_ag   = x - (w + gap) / 2
xs_imdb = x + (w + gap) / 2

for xs, rows, lbl_suffix, alpha in [
        (xs_ag,   agnews, "AG",   1.0),
        (xs_imdb, imdb,   "IMDB", 0.65)]:

    fa_vals = [r["fa_pct"] for r in rows]
    fb_vals = [r["fb_pct"] for r in rows]
    splits  = [r["split"]  for r in rows]

    bars_a = ax.bar(xs, fa_vals, w,
                    color=C_STA, alpha=alpha,
                    edgecolor="white", linewidth=0.5,
                    label=f"Stage A (early-exit)  {lbl_suffix}")
    bars_b = ax.bar(xs, fb_vals, w, bottom=fa_vals,
                    color=C_STB, alpha=alpha,
                    edgecolor="white", linewidth=0.5,
                    label=f"Stage B (routing)  {lbl_suffix}")

    for xi, (a, b, s) in enumerate(zip(fa_vals, fb_vals, splits)):
        total = a + b
        # total label on top
        ax.text(xs[xi], total + 1.0, f"{total:.0f}%",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color="#374151")
        # highlight best split with amber border
        best_s = BEST_SPLIT["agnews" if lbl_suffix == "AG" else "imdb"]
        if s == best_s:
            for bar in [bars_a[xi], bars_b[xi]]:
                bar.set_edgecolor(BEST_CLR)
                bar.set_linewidth(2.0)

# dataset labels below x-axis
for xi in range(len(SPLITS)):
    ax.text(xs_ag[xi],   -5, "AG",   ha="center", fontsize=7.5, color=C_AG,   fontweight="bold")
    ax.text(xs_imdb[xi], -5, "IMDB", ha="center", fontsize=7.5, color=C_IMDB, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"s={s}" for s in SPLITS])
ax.set_ylabel("FLOPs  (% of DenseBERT)", fontsize=10)
ax.set_title("(c)  FLOPs Breakdown: Stage A + B\n(amber border = chosen split)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 105)
ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.7)

# Simplified legend
handles = [
    mpatches.Patch(color=C_STA, label="Stage A  (early-exit)"),
    mpatches.Patch(color=C_STB, label="Stage B  (token routing)"),
    mpatches.Patch(facecolor="none", edgecolor=BEST_CLR, linewidth=2, label="Chosen split"),
]
ax.legend(handles=handles, fontsize=8.5, loc="upper right",
          framealpha=0.9, edgecolor="#D1D5DB")

# ── Global title & layout ──────────────────────────────────────────────
fig.suptitle("HDC-BERT: Split Layer Ablation Study", fontsize=13,
             fontweight="bold", y=1.03)
plt.tight_layout()

out = SPLIT_DIR / "split_ablation.png"
plt.savefig(out, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved: {out}")
plt.close()
