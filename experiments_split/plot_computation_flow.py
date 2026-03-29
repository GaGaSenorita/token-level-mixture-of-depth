"""
HDC-BERT Computational Flow Visualization (MoD-style).

For each dataset, shows ALL 12 BERT layers as horizontal bars:
  Stage A: how many samples are still active vs already exited at each layer
  Stage B: of active samples, what fraction of tokens are kept (routing sparsity)

Inspired by the Mixture-of-Depths (MoD) paper visualization style.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

SPLIT_DIR = Path(__file__).parent

# ── Data ──────────────────────────────────────────────────────────────

def load_flow_data(dataset, split):
    f = SPLIT_DIR / dataset / f"split_{split}/results.json"
    r = json.loads(f.read_text())
    fed = r["flops_estimation_data"]

    hist = {(int(k) if k != "final" else split): v
            for k, v in fed["exit_histogram"].items()}
    total = sum(hist.values())
    keep_b = fed["stage_b_eval_keep_rates"]
    n_layers = 12

    # Build per-layer flow
    layers = []
    remaining = total
    for layer in range(1, n_layers + 1):
        if layer <= split:
            # Stage A: sample-level early exit
            exited = hist.get(layer, 0)
            active_before = remaining
            remaining -= exited
            layers.append({
                "layer":    layer,
                "stage":    "A",
                "active":   active_before / total,   # fraction of samples entering
                "exited":   exited / total,           # fraction exiting HERE
                "keep":     None,
            })
        else:
            # Stage B: token-level routing
            b_idx = layer - split - 1
            keep = keep_b[b_idx] if b_idx < len(keep_b) else keep_b[-1]
            active_frac = remaining / total           # fraction of samples reaching Stage B
            layers.append({
                "layer":  layer,
                "stage":  "B",
                "active": active_frac,
                "exited": 0.0,
                "keep":   keep,
            })

    return layers, {
        "total":          total,
        "split":          split,
        "exit_rate":      fed["stage_a_exit_rate"],
        "avg_keep_b":     fed["stage_b_avg_keep_rate"],
        "acc":            r["hdc_inference"]["accuracy"] * 100,
        "avg_exit_layer": fed["avg_exit_layer_a"],
    }


AG_SPLIT   = 6
IMDB_SPLIT = 8

ag_layers,   ag_meta   = load_flow_data("agnews", AG_SPLIT)
imdb_layers, imdb_meta = load_flow_data("imdb",   IMDB_SPLIT)

# ── Colours ────────────────────────────────────────────────────────────
# Stage A exit colours: gradient from teal to amber
EXIT_CMAP = plt.cm.YlOrRd
ACTIVE_A   = "#BFDBFE"   # light blue  – samples entering layer (Stage A)
ACTIVE_B   = "#6EE7B7"   # light green – tokens kept (Stage B)
DROPPED_B  = "#D1D5DB"   # light gray  – tokens dropped (Stage B)
EXITED_A   = "#F87171"   # soft red    – samples that exit here
GHOST      = "#F3F4F6"   # very light  – samples already exited (past layers)
SPLIT_LINE = "#1D4ED8"   # dark blue   – separator line color

# ── Helper: draw one dataset panel ────────────────────────────────────

def draw_panel(ax, layers, meta, title):
    split = meta["split"]
    n     = len(layers)

    for i, ly in enumerate(layers):
        y    = n - i - 1          # top = layer 1
        lnum = ly["layer"]

        if ly["stage"] == "A":
            already_exited = 1.0 - ly["active"]
            still_in       = ly["active"] - ly["exited"]
            exits_here     = ly["exited"]

            # ghost: samples already gone
            ax.barh(y, already_exited, left=0,
                    height=0.72, color=GHOST, zorder=2)
            # samples that exit at THIS layer
            ax.barh(y, exits_here, left=already_exited,
                    height=0.72, color=EXITED_A, zorder=3,
                    alpha=0.85)
            # samples that continue
            ax.barh(y, still_in, left=already_exited + exits_here,
                    height=0.72, color=ACTIVE_A, zorder=3)

            # annotate exit count
            if exits_here > 0.005:
                ax.text(already_exited + exits_here / 2, y,
                        f"exit\n{exits_here*100:.1f}%",
                        ha="center", va="center",
                        fontsize=6.8, color="#7F1D1D", fontweight="bold")

        else:  # Stage B
            already_exited = 1.0 - ly["active"]
            keep           = ly["keep"]
            tokens_kept    = ly["active"] * keep
            tokens_dropped = ly["active"] * (1 - keep)

            # ghost: samples exited in Stage A
            ax.barh(y, already_exited, left=0,
                    height=0.72, color=GHOST, zorder=2)
            # dropped tokens
            ax.barh(y, tokens_dropped, left=already_exited,
                    height=0.72, color=DROPPED_B, zorder=3)
            # kept tokens
            ax.barh(y, tokens_kept, left=already_exited + tokens_dropped,
                    height=0.72, color=ACTIVE_B, zorder=3)

            # annotate keep rate
            ax.text(already_exited + tokens_dropped + tokens_kept / 2, y,
                    f"{keep*100:.0f}%\nkept",
                    ha="center", va="center",
                    fontsize=6.8, color="#064E3B", fontweight="bold")

    # ── Split separator line ──────────────────────────────────────────
    split_y = n - split - 0.5
    ax.axhline(split_y, color=SPLIT_LINE, linewidth=2.0, linestyle="--",
               zorder=5, alpha=0.8)
    ax.text(1.02, split_y, f"  split\n  layer {split}",
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8,
            color=SPLIT_LINE, fontweight="bold")

    # Stage labels on left
    mid_a = n - split / 2 - 0.5
    mid_b = n - split - (n - split) / 2 - 0.5
    ax.text(-0.01, mid_a, "Stage A\n(early exit)",
            transform=ax.get_yaxis_transform(),
            va="center", ha="right", fontsize=8.5,
            color="#1D4ED8", fontweight="bold", rotation=0)
    ax.text(-0.01, mid_b, "Stage B\n(token routing)",
            transform=ax.get_yaxis_transform(),
            va="center", ha="right", fontsize=8.5,
            color="#065F46", fontweight="bold", rotation=0)

    # Y-axis: layer numbers
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"L{n - i}" for i in range(n)], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xlabel("Fraction of Total Computation", fontsize=9)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title with stats
    ax.set_title(
        f"{title}\n"
        f"Stage A exit rate: {meta['exit_rate']*100:.1f}%  |  "
        f"Avg keep (B): {meta['avg_keep_b']*100:.1f}%  |  "
        f"Acc: {meta['acc']:.2f}%",
        fontsize=10, fontweight="bold", pad=10
    )

# ── Main figure ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.left":  True,
    "axes.spines.bottom":True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("#FAFAFA")

draw_panel(axes[0], ag_layers,   ag_meta,   "AG News  (split=6, seq_len=128)")
draw_panel(axes[1], imdb_layers, imdb_meta, "IMDB  (split=8, seq_len=256)")

# ── Shared legend ──────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=ACTIVE_A,  label="Sample active (Stage A)"),
    mpatches.Patch(color=EXITED_A,  label="Sample exits here"),
    mpatches.Patch(color=GHOST,     label="Sample already exited"),
    mpatches.Patch(color=ACTIVE_B,  label="Token kept (Stage B routing)"),
    mpatches.Patch(color=DROPPED_B, label="Token dropped (Stage B routing)"),
    Line2D([0],[0], color=SPLIT_LINE, linewidth=2, linestyle="--",
           label="Stage A / B boundary"),
]
fig.legend(handles=legend_handles,
           loc="lower center", ncol=3,
           fontsize=9, framealpha=0.95,
           edgecolor="#D1D5DB", fancybox=True,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle("HDC-BERT: Layer-wise Computational Flow\n"
             "(MoD-style — how computation shrinks across layers)",
             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout(rect=[0, 0.07, 1, 1])
out = SPLIT_DIR / "computation_flow.png"
plt.savefig(out, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved: {out}")
plt.close()
