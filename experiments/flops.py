"""
FLOPs calculation for BERT dynamic computation methods.

Computes analytical FLOPs (multiply-accumulate) for:
  - DenseBERT (baseline)
  - DeeBERT (early-exit)
  - RouterBERT (token-level routing)
  - HDC-BERT (hierarchical: early-exit + token routing)
"""

import json
import glob
import os
import numpy as np
from pathlib import Path

# ── BERT-base constants ──────────────────────────────────────────────
H = 768       # hidden_size
I = 3072      # intermediate_size (FFN)
N_LAYERS = 12
N_HEADS = 12

# ── FLOPs formulas ───────────────────────────────────────────────────

def attn_flops(L, h=H):
    """Attention FLOPs for one layer: QKV + scores + context + output proj."""
    return 4 * L * h**2 + 2 * L**2 * h

def ffn_flops(L, h=H, i=I):
    """FFN FLOPs for one layer: two linear projections."""
    return 2 * L * h * i

def layer_flops(L, h=H, i=I):
    """Total FLOPs for one dense BERT layer."""
    return attn_flops(L, h) + ffn_flops(L, h, i)

def routed_attn_flops(L, k, h=H):
    """Attention FLOPs with token routing (keep_rate=k). FFN excluded."""
    return 4 * k * L * h**2 + 2 * k**2 * L**2 * h

def routed_layer_flops(L, k, h=H, i=I):
    """One routed layer: reduced attention + full FFN."""
    return routed_attn_flops(L, k, h) + ffn_flops(L, h, i)

# ── Per-model FLOPs calculators ──────────────────────────────────────

def compute_baseline_flops(seq_len, n_layers=N_LAYERS):
    """DenseBERT: all layers, all tokens."""
    return n_layers * layer_flops(seq_len)

def compute_deebert_flops(seq_len, avg_exit_layer):
    """DeeBERT: avg_exit_layer × layer_flops (exact since all layers identical)."""
    return avg_exit_layer * layer_flops(seq_len)

def compute_routerbert_flops(seq_len, keep_rates_per_layer):
    """RouterBERT: per-layer routed attention + full FFN."""
    total = 0
    for k in keep_rates_per_layer:
        total += routed_layer_flops(seq_len, k)
    return total

def compute_hdc_flops(seq_len, exit_histogram, stage_b_keep_rates, split_layer=6):
    """HDC-BERT: histogram-weighted Stage A + routed Stage B."""
    lf = layer_flops(seq_len)

    # Parse histogram: keys are "1","2",...,"6","final"
    total_samples = sum(exit_histogram.values())
    total_flops = 0

    for key, count in exit_histogram.items():
        if key == "final":
            # These samples run all Stage A layers + Stage B
            stage_a = split_layer * lf
            stage_b = sum(routed_layer_flops(seq_len, k) for k in stage_b_keep_rates)
            total_flops += count * (stage_a + stage_b)
        else:
            # Early exit at layer int(key)
            exit_layer = int(key)
            total_flops += count * (exit_layer * lf)

    return total_flops / total_samples

# ── Data loading ─────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path(__file__).parent

def load_baseline_results(dataset):
    """Load baseline results for a dataset (3 seeds)."""
    ds_dir = EXPERIMENTS_DIR / "baseline" / dataset
    results = []
    for f in sorted(ds_dir.glob("*_densebert_results.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results

def load_model_results(model_dir, dataset):
    """Load results for deebert/router_tuning/hdc (3 runs)."""
    ds_dir = EXPERIMENTS_DIR / model_dir / dataset
    results = []
    for run_dir in sorted(ds_dir.glob("run_*")):
        f = run_dir / "results.json"
        if f.exists():
            with open(f) as fp:
                results.append(json.load(fp))
    return results

# ── Main ─────────────────────────────────────────────────────────────

def to_gflops(flops):
    return flops / 1e9

def compute_all(dataset):
    """Compute FLOPs for all models on a given dataset. Returns list of dicts."""
    # Determine seq_len from baseline
    baseline_runs = load_baseline_results(dataset)
    seq_len = baseline_runs[0]["args"]["max_length"]

    rows = []

    # ── Baseline ──
    base_flops = compute_baseline_flops(seq_len)
    accs = [r["best_test_acc"] for r in baseline_runs]
    ms_list = [r.get("inference_ms_per_sample") for r in baseline_runs]
    rows.append({
        "method": "DenseBERT",
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "gflops": to_gflops(base_flops),
        "flops_ratio": 1.0,
        "flops_saved": 0.0,
        "ms_per_sample": np.mean([m for m in ms_list if m is not None]) if any(ms_list) else None,
    })

    # ── DeeBERT ──
    deebert_runs = load_model_results("deebert", dataset)
    if deebert_runs:
        dee_flops_list = []
        dee_accs = []
        for r in deebert_runs:
            avg_exit = r["best_avg_exit_layer"]
            dee_flops_list.append(compute_deebert_flops(seq_len, avg_exit))
            dee_accs.append(r["best_ee_acc"])
        avg_dee_flops = np.mean(dee_flops_list)
        rows.append({
            "method": "DeeBERT",
            "acc_mean": np.mean(dee_accs),
            "acc_std": np.std(dee_accs),
            "gflops": to_gflops(avg_dee_flops),
            "flops_ratio": avg_dee_flops / base_flops,
            "flops_saved": 1 - avg_dee_flops / base_flops,
            "ms_per_sample": None,
        })

    # ── RouterBERT ──
    router_runs = load_model_results("router_tuning", dataset)
    if router_runs:
        rt_flops_list = []
        rt_accs = []
        for r in router_runs:
            keep_rates = r["flops_estimation_data"]["eval_keep_rates_per_layer"]
            rt_flops_list.append(compute_routerbert_flops(seq_len, keep_rates))
            rt_accs.append(r["final_eval"]["accuracy"])
        avg_rt_flops = np.mean(rt_flops_list)
        rows.append({
            "method": "RouterBERT",
            "acc_mean": np.mean(rt_accs),
            "acc_std": np.std(rt_accs),
            "gflops": to_gflops(avg_rt_flops),
            "flops_ratio": avg_rt_flops / base_flops,
            "flops_saved": 1 - avg_rt_flops / base_flops,
            "ms_per_sample": None,
        })

    # ── HDC-BERT ──
    hdc_runs = load_model_results("hdc", dataset)
    if hdc_runs:
        hdc_flops_list = []
        hdc_accs = []
        for r in hdc_runs:
            fed = r["flops_estimation_data"]
            hist = {k: v for k, v in fed["exit_histogram"].items()}
            keep_rates = fed["stage_b_eval_keep_rates"]
            split = fed.get("split_layer", 6)
            hdc_flops_list.append(compute_hdc_flops(seq_len, hist, keep_rates, split))
            hdc_accs.append(r["hdc_inference"]["accuracy"])
        avg_hdc_flops = np.mean(hdc_flops_list)
        rows.append({
            "method": "HDC-BERT",
            "acc_mean": np.mean(hdc_accs),
            "acc_std": np.std(hdc_accs),
            "gflops": to_gflops(avg_hdc_flops),
            "flops_ratio": avg_hdc_flops / base_flops,
            "flops_saved": 1 - avg_hdc_flops / base_flops,
            "ms_per_sample": None,
        })

    return rows, seq_len

def print_table(dataset):
    rows, seq_len = compute_all(dataset)
    print(f"\n{'='*70}")
    print(f"  {dataset.upper()} (seq_len={seq_len})")
    print(f"{'='*70}")
    header = f"{'Method':<14} {'Acc(%)':<14} {'GFLOPs':>8} {'FLOPs%':>8} {'Saved%':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        acc_str = f"{r['acc_mean']*100:.2f}±{r['acc_std']*100:.2f}"
        print(f"{r['method']:<14} {acc_str:<14} {r['gflops']:>8.3f} {r['flops_ratio']*100:>7.1f}% {r['flops_saved']*100:>7.1f}%")

if __name__ == "__main__":
    for ds in ["agnews", "imdb"]:
        try:
            print_table(ds)
        except FileNotFoundError as e:
            print(f"Skipping {ds}: {e}")
    print()
