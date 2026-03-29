"""
MoD-style schematic comparison figure.
Shows routing decisions (computed vs skipped) for 4 methods:
  Dense | Early-Exit | Token Routing | HDC-BERT

Rows = BERT layers (1-12), Columns = samples/tokens.
Purple = computed, Orange = skipped / routed around.
No real data needed — illustrative schematic only.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

np.random.seed(7)

N_LAYERS = 12
N_COLS   = 36       # columns in visualization
SPLIT    = 6        # HDC split layer (Stage A / Stage B boundary)

PURPLE = "#7C3AED"   # computed
ORANGE = "#F97316"   # skipped / routed around
GRAY   = "#D1D5DB"   # already exited (past early-exit layer) in HDC Stage B view

CMAP = ListedColormap([ORANGE, PURPLE])   # 0 = orange, 1 = purple

# ── Grid generators ────────────────────────────────────────────────────

def grid_dense():
    return np.ones((N_LAYERS, N_COLS), dtype=float)

def grid_early_exit():
    """Scattered orange stripes: most columns purple, some exit at various layers.
    Columns NOT sorted — random order like the MoD paper figure."""
    g = np.ones((N_LAYERS, N_COLS), dtype=float)
    # ~40% of columns exit early, rest go all the way (purple)
    # Exit layers: spread across L2–L9
    exit_assignments = (
        [(2, [1, 8, 20])]          +   # exits at L2
        [(3, [3, 14, 25, 31])]     +   # exits at L3
        [(4, [6, 17, 28])]         +   # exits at L4
        [(5, [10, 22])]            +   # exits at L5
        [(6, [0, 33])]             +   # exits at L6
        [(8, [12, 26])]            +   # exits at L8
        [(9, [4, 19])]                 # exits at L9
    )
    for exit_l, cols in exit_assignments:
        for col in cols:
            if col < N_COLS:
                g[exit_l:, col] = 0.0   # orange from exit_layer to top
    return g

def grid_router():
    """Per-layer token routing: ~45-60% tokens kept at each layer."""
    g = np.zeros((N_LAYERS, N_COLS), dtype=float)
    for layer in range(N_LAYERS):
        keep = np.random.uniform(0.42, 0.62)
        g[layer] = (np.random.rand(N_COLS) < keep).astype(float)
    return g

def grid_hdc():
    """
    Stage A (layers 0..SPLIT-1): sample-level early exit.
    Stage B (layers SPLIT..11): token-level routing for surviving samples.
    Samples sorted: early-exit left, survivors right.
    """
    g = np.ones((N_LAYERS, N_COLS), dtype=float)

    n_exit = int(N_COLS * 0.62)   # ~62% exit in Stage A
    n_stay = N_COLS - n_exit

    # Stage A exits — sorted
    choices_a = ([2]*3 + [2]*2 + [3]*5 + [3]*3 + [4]*4 +
                 [4]*2 + [5]*3 + [6]*2 + [6]*2)
    exit_ls = sorted(np.random.choice(choices_a, size=n_exit, replace=True))
    for col, el in enumerate(exit_ls):
        g[el:, col] = 0.0   # orange after exit layer

    # Stage B token routing for surviving columns
    for layer in range(SPLIT, N_LAYERS):
        keep = np.random.uniform(0.35, 0.55)
        for col in range(n_exit, N_COLS):
            g[layer, col] = float(np.random.rand() < keep)

    return g, n_exit   # also return boundary for annotation


# ── Build grids ────────────────────────────────────────────────────────
G_dense = grid_dense()
G_ee    = grid_early_exit()
G_rt    = grid_router()
G_hdc, hdc_boundary = grid_hdc()


# ── Figure ─────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(15, 7.5), facecolor="white")

# Layout: 2 rows × 4 cols of axes, plus room for titles
gs = fig.add_gridspec(2, 4,
                      left=0.04, right=0.97,
                      top=0.88,  bottom=0.12,
                      hspace=0.55, wspace=0.18)

TITLES = [
    ("Dense\n(DenseBERT)", "All tokens, all layers\n— no savings"),
    ("Early-Exit\n(DeeBERT)",  "Samples exit at different layers\n— saves depth, not width"),
    ("Token Routing\n(RouterBERT)", "Tokens selectively skipped\nat each layer — saves width"),
    ("Hierarchical Dynamic\n(HDC-BERT  ours)", "Stage A: sample exits  +\nStage B: token routing"),
]
GRIDS  = [G_dense, G_ee, G_rt, G_hdc]

axes = []
for col, (grid, (title, subtitle)) in enumerate(zip(GRIDS, TITLES)):
    ax = fig.add_subplot(gs[:, col])
    axes.append(ax)

    # Draw grid
    ax.imshow(grid, aspect="auto", cmap=CMAP, vmin=0, vmax=1,
              interpolation="nearest", origin="lower")

    # ── HDC only: draw split boundary and annotations ─────────────────
    if col == 3:
        # Horizontal dashed line between Stage A and Stage B
        ax.axhline(SPLIT - 0.5, color="white", linewidth=2.5,
                   linestyle="--", zorder=5)
        # Vertical boundary between exited and surviving samples
        ax.axvline(hdc_boundary - 0.5, color="white", linewidth=1.5,
                   linestyle=":", alpha=0.7, zorder=5)

        # Stage A / Stage B labels — with origin="lower", Stage A is bottom half
        ax.text(N_COLS * 0.5, SPLIT * 0.5 - 0.5,
                "Stage A\n(early exit)", color="white",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                alpha=0.95)
        ax.text(N_COLS * 0.5, SPLIT + (N_LAYERS - SPLIT) * 0.5 - 0.5,
                "Stage B\n(token routing)", color="white",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                alpha=0.95)

    # ── Axis formatting ───────────────────────────────────────────────
    # Y-axis: layer numbers (L1 at bottom, L12 at top)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i+1}" for i in range(N_LAYERS)],
                       fontsize=7.5)
    ax.set_ylabel("Layer", fontsize=9) if col == 0 else ax.set_yticklabels([])

    # X-axis label
    ax.set_xlabel("Tokens / Samples →", fontsize=8.5)
    ax.set_xticks([])

    # Title
    color = "#7C2D12" if col == 3 else "black"   # highlight HDC
    weight = "bold" if col == 3 else "semibold"
    ax.set_title(title, fontsize=10.5, fontweight=weight,
                 color=color, pad=6)

    # Subtitle (below each panel)
    fig.text(
        ax.get_position().x0 + ax.get_position().width / 2,
        ax.get_position().y0 - 0.06,
        subtitle,
        ha="center", va="top", fontsize=8, color="#4B5563",
        style="italic"
    )

    # Border highlight for HDC
    if col == 3:
        for spine in ax.spines.values():
            spine.set_edgecolor("#7C2D12")
            spine.set_linewidth(2.0)


# ── Shared legend ──────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=PURPLE, label="Computed (use block)"),
    mpatches.Patch(color=ORANGE, label="Skipped (route around)"),
    Line2D([0],[0], color="white", linewidth=2.5, linestyle="--",
           label="HDC Stage A / B boundary",
           markerfacecolor="gray", markeredgecolor="gray"),
]
fig.legend(handles=legend_handles,
           loc="lower center", ncol=3,
           fontsize=10, framealpha=0.95,
           edgecolor="#D1D5DB", fancybox=True,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle("Routing Decisions: Four Methods Compared",
             fontsize=14, fontweight="bold", y=0.97)

out = Path(__file__).parent / "mod_style_comparison.png"
plt.savefig(out, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved: {out}")
plt.close()


