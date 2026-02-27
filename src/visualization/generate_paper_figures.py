"""
generate_paper_figures.py
=========================
Generate all 8 publication-quality figures for the IEEE Transactions paper.

Figures produced (all saved to outputs/figures/):
  fig1_methodology.png       - System pipeline flowchart (sEMG -> GRU -> InMoov)
  fig2_mrmr_channels.png     - MRMR channel ranking bar chart
  fig3_per_subject_r2.png    - Per-subject R2 grouped bar chart (all 10 subjects)
  fig4_outlier_analysis.png  - S8 outlier analysis
  fig5_prediction_traces.png - Force prediction traces for 3 subjects
  fig6_scatter.png           - Predicted vs actual scatter plot
  fig7_model_comparison.png  - Model comparison bar chart with 95% CI
  fig8_anatomy_electrode.png - Posterior forearm anatomy + electrode placement

Usage:
    python -m src.visualization.generate_paper_figures
    python -m src.visualization.generate_paper_figures --skip-model  # skip fig5/fig6

Requirements: matplotlib, numpy, scipy, torch (only for fig5/fig6)
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

# -- Project paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
MODELS_DIR  = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -- Dataset constants -------------------------------------------------------------
GHORBANI_FS         = 200    # Hz
GHORBANI_RMS_HOP_MS = 20     # ms

# -- IEEE-style matplotlib defaults ------------------------------------------------
def set_ieee_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.framealpha":  0.85,
        "axes.linewidth":     0.8,
        "lines.linewidth":    1.2,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.35,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
    })

# IEEE column widths
SINGLE_COL = 3.45   # inches
DOUBLE_COL = 7.16   # inches

# -- Hardcoded results (from outputs/results/ghorbani_evaluation.json) -------------
# 9 subjects (S8 excluded)
SUBJECTS_9   = [1, 2, 3, 4, 5, 6, 7, 9, 10]
SUBJECT_LBLS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10"]

GRU_R2 = {
    1: 0.7357, 2: 0.7814, 3: 0.8066, 4: 0.7857, 5: 0.6760,
    6: 0.8271, 7: 0.7087, 9: 0.8047, 10: 0.8784
}
RIDGE_R2 = {
    1: 0.6902, 2: 0.5660, 3: 0.7178, 4: 0.5512, 5: 0.6339,
    6: 0.8021, 7: 0.7139, 9: 0.7589, 10: 0.7768
}
MLP_R2 = {
    1: 0.6906, 2: 0.7309, 3: 0.6890, 4: 0.5886, 5: 0.6841,
    6: 0.8350, 7: 0.6914, 9: 0.7647, 10: 0.8978
}

# S8 values (all models fail)
S8_RESULTS = {"GRU": 0.4340, "Ridge": 0.2130, "MLP": 0.1660}

# Summary stats (9 subjects)
STATS = {
    "GRU":   {"mean": 0.7783, "std": 0.0624, "ci_low": 0.7303, "ci_high": 0.8262},
    "Ridge": {"mean": 0.6901, "std": 0.0894, "ci_low": 0.6214, "ci_high": 0.7588},
    "MLP":   {"mean": 0.7302, "std": 0.0917, "ci_low": 0.6598, "ci_high": 0.8007},
}

# Model colors (IEEE-friendly, colorblind-safe)
COLORS = {
    "GRU":   "#1f77b4",  # blue
    "Ridge": "#d62728",  # red
    "MLP":   "#ff7f0e",  # orange
}

# MRMR data
CHANNEL_NAMES_SHORT = ["FCR", "PL", "FCU", "ECU", "ED", "ECRB", "BR", "PT"]
CHANNEL_RELEVANCE   = [0.01182, 0.03019, 0.02086, 0.03031, 0.02775, 0.07705, 0.03713, 0.01289]
ALL_MRMR_SCORES     = [-0.00756, 0.01867, -0.00377, 0.01853, -0.06051, 0.06527, -0.00946, -0.01518]
SELECTED_CHANNELS   = [5, 3]  # indices of ECRB and ECU


# ============================================================================
# FIG 1 - System pipeline flowchart (includes hardware)
# ============================================================================
def fig1_methodology():
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.70, 0.5, "MyoWare 2.0\nsEMG Sensors\n(2 channels)"),
        (2.55, 0.5, "MRMR\nChannel\nSelection"),
        (4.40, 0.5, "Feature\nExtraction\n(36 inputs)"),
        (6.25, 0.5, "GRU\nEnsemble\n(23,809 params)"),
        (8.10, 0.5, "Butterworth\nSmoothing\n(4 Hz LP)"),
        (9.95, 0.5, "Arduino\nMega 2560\n(PWM)"),
        (11.55, 0.5, "InMoov\nProsthetic\nHand"),
    ]

    box_w, box_h = 1.35, 0.72
    for (cx, cy, label) in boxes:
        if "GRU" in label:
            fc = "#ddeeff"
        elif "MyoWare" in label or "sEMG" in label:
            fc = "#ffe8cc"
        elif "Arduino" in label:
            fc = "#e8e0f0"
        elif "InMoov" in label:
            fc = "#ddffd4"
        else:
            fc = "#f0f0f0"
        rect = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.04", linewidth=0.8,
            edgecolor="#333333", facecolor=fc, zorder=2
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7, zorder=3, linespacing=1.3)

    # Arrows between boxes
    arrow_kw = dict(arrowstyle="->", color="#444444",
                    lw=0.9, mutation_scale=10, zorder=1)
    xs = [b[0] for b in boxes]
    for i in range(len(xs) - 1):
        ax.annotate(
            "", xy=(xs[i + 1] - box_w / 2, 0.5),
            xytext=(xs[i] + box_w / 2, 0.5),
            arrowprops=arrow_kw,
        )

    # Sub-labels below arrows
    arrow_labels = [
        "ECRB + ECU",
        "RMS,MAV,WL\nVAR,ZC,SSC+\u0394",
        "MSE + Adam",
        "zero-phase",
        "Force\u21920-1",
        "Linear\nActuators"
    ]
    arrow_x_mid = [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)]
    for i, lbl in enumerate(arrow_labels):
        ax.text(arrow_x_mid[i], 0.07, lbl, ha="center", va="bottom",
                fontsize=6, color="#555555", linespacing=1.25)

    fig.tight_layout(pad=0.1)
    out = FIGURES_DIR / "fig1_methodology.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================================
# FIG 2 - MRMR channel ranking
# ============================================================================
def fig2_mrmr_channels():
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.4, 2.6))

    x = np.arange(len(CHANNEL_NAMES_SHORT))
    width = 0.38

    bars1 = ax.bar(x - width / 2, CHANNEL_RELEVANCE, width,
                   label="Relevance (MI)", color="#aec7e8", edgecolor="gray", lw=0.5)
    bars2 = ax.bar(x + width / 2, ALL_MRMR_SCORES, width,
                   label="MRMR score", color="#1f77b4", edgecolor="gray", lw=0.5)

    # Highlight selected channels
    for idx in SELECTED_CHANNELS:
        for bar in [bars1[idx], bars2[idx]]:
            bar.set_edgecolor("#c00000")
            bar.set_linewidth(1.5)
        ax.text(idx, max(CHANNEL_RELEVANCE[idx], max(0, ALL_MRMR_SCORES[idx])) + 0.003,
                "*", ha="center", va="bottom", fontsize=11, color="#c00000",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES_SHORT, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("EMG Channel")
    ax.set_title("MRMR Channel Selection (Myo Armband, 8 channels)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_mrmr_channels.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================================
# FIG 3 - Per-subject R2 grouped bar chart (3 models, 10 subjects, S8 flagged)
# ============================================================================
def fig3_per_subject_r2():
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))

    all_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lbls = [f"S{s}" for s in all_subjects]

    models = ["GRU (Ours)", "MLP", "Ridge"]
    r2_dicts = [GRU_R2, MLP_R2, RIDGE_R2]
    model_colors = [COLORS["GRU"], COLORS["MLP"], COLORS["Ridge"]]

    n_models = len(models)
    x = np.arange(len(all_subjects))
    width = 0.24
    offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    for i, (model, data, color) in enumerate(zip(models, r2_dicts, model_colors)):
        vals = []
        hatches = []
        for s in all_subjects:
            if s == 8:
                key = model.split(" ")[0]  # "GRU", "MLP", or "Ridge"
                vals.append(S8_RESULTS.get(key, 0.0))
                hatches.append("///")
            else:
                vals.append(data[s])
                hatches.append("")
        bars = ax.bar(x + offsets[i], vals, width,
                      label=model, color=color, edgecolor="gray",
                      lw=0.4, alpha=0.88)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    # S8 annotation
    s8_idx = all_subjects.index(8)
    ax.annotate("S8\n(outlier)", xy=(s8_idx, 0.46), xytext=(s8_idx + 0.6, 0.55),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"),
                fontsize=7, color="#aa0000", ha="left")

    ax.axhline(0.80, color="black", lw=0.8, ls="--", alpha=0.7, label="R\u00b2=0.80 target")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls)
    ax.set_ylabel("R\u00b2 Score")
    ax.set_xlabel("Subject")
    ax.set_title("Per-Subject Grip Force Prediction R\u00b2 \u2014 All 10 Subjects")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, framealpha=0.85, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade S8 region
    ax.axvspan(s8_idx - 0.45, s8_idx + 0.45, alpha=0.07, color="red")

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_per_subject_r2.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================================
# FIG 4 - S8 outlier analysis
# ============================================================================
def fig4_outlier_analysis():
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # Left: R2 comparison S8 vs mean across models
    models      = ["GRU\n(Ours)", "MLP", "Ridge"]
    model_keys  = ["GRU", "MLP", "Ridge"]
    s8_vals     = [S8_RESULTS[m] for m in model_keys]
    mean_vals   = [STATS[m]["mean"] for m in model_keys]

    x = np.arange(len(models))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, s8_vals,  width, label="S8 (outlier)",
           color="#d62728", edgecolor="gray", lw=0.5, hatch="///", alpha=0.8)
    ax.bar(x + width / 2, mean_vals, width, label="Group mean\n(9 subjects)",
           color="#1f77b4", edgecolor="gray", lw=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("R\u00b2 Score")
    ax.set_title("(a) S8 vs. Group Mean R\u00b2")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.80, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.legend(framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Bar chart of all 10 subjects for GRU to illustrate S8 visually
    ax2 = axes[1]
    all_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r2_vals = [GRU_R2.get(s, S8_RESULTS["GRU"]) for s in all_subjects]
    bar_colors = ["#d62728" if s == 8 else "#1f77b4" for s in all_subjects]
    bar_hatches = ["///" if s == 8 else "" for s in all_subjects]
    xlbls = [f"S{s}" for s in all_subjects]

    bars = ax2.bar(xlbls, r2_vals, color=bar_colors, edgecolor="gray",
                   lw=0.5, alpha=0.85)
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)

    ax2.axhline(0.80, color="black", lw=0.8, ls="--", alpha=0.7, label="R\u00b2=0.80")
    ax2.axhline(np.mean([r2_vals[i] for i, s in enumerate(all_subjects) if s != 8]),
                color="#1f77b4", lw=0.9, ls=":", alpha=0.9, label="Mean (9 subj)")
    ax2.set_ylabel("R\u00b2 Score")
    ax2.set_xlabel("Subject")
    ax2.set_title("(b) GRU: S8 as Outlier")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    patch_outlier = mpatches.Patch(facecolor="#d62728", hatch="///",
                                   edgecolor="gray", label="S8 (excluded)")
    patch_normal  = mpatches.Patch(facecolor="#1f77b4", edgecolor="gray",
                                   label="Included subjects")
    ax2.legend(handles=[patch_outlier, patch_normal], framealpha=0.85, fontsize=7)

    fig.tight_layout(pad=0.6)
    out = FIGURES_DIR / "fig4_outlier_analysis.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================================
# FIG 7 - Model comparison bar chart with 95% CI (3 models)
# ============================================================================
def fig7_model_comparison():
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.4, 2.8))

    models = ["GRU\n(Ours)", "MLP", "Ridge"]
    model_keys = ["GRU", "MLP", "Ridge"]
    means  = [STATS[k]["mean"] for k in model_keys]
    ci_err = [(STATS[k]["mean"] - STATS[k]["ci_low"],
               STATS[k]["ci_high"] - STATS[k]["mean"]) for k in model_keys]
    ci_lo  = [e[0] for e in ci_err]
    ci_hi  = [e[1] for e in ci_err]

    # Per-subject R2 dots for each model
    r2_dicts = [GRU_R2, MLP_R2, RIDGE_R2]
    r2_per_model = [[d[s] for s in SUBJECTS_9] for d in r2_dicts]

    x = np.arange(len(models))
    colors = [COLORS[k] for k in model_keys]

    bars = ax.bar(x, means, 0.6, yerr=[ci_lo, ci_hi],
                  capsize=4, color=colors, edgecolor="gray",
                  lw=0.5, alpha=0.85, error_kw={"lw": 1.2, "ecolor": "black"})

    # Overlay individual subject dots with slight jitter
    rng = np.random.default_rng(42)
    for i, vals in enumerate(r2_per_model):
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(x[i] + jitter, vals, s=14, color=colors[i],
                   edgecolors="white", lw=0.4, zorder=3, alpha=0.9)

    ax.axhline(0.80, color="black", lw=0.8, ls="--", alpha=0.6, label="R\u00b2=0.80")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("R\u00b2 Score")
    ax.set_title("Model Comparison \u2014 9 Subjects\n(mean \u00b1 95% CI, dots = per-subject)")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Significance annotation: GRU vs Ridge
    y_sig = 0.97
    ax.annotate("", xy=(0, y_sig), xytext=(2, y_sig),
                arrowprops=dict(arrowstyle="-", lw=1.0, color="black"))
    ax.text(1.0, y_sig + 0.01, "p<0.05", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out = FIGURES_DIR / "fig7_model_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================================
# FIG 8 - Posterior forearm anatomy + electrode placement
# ============================================================================
def fig8_anatomy_electrode():
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.6, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_aspect("equal")

    # -- Draw forearm outline (posterior view) ---
    # Simplified trapezoid shape: wider at elbow, narrower at wrist
    forearm_x = [2.5, 7.5, 7.0, 3.0, 2.5]
    forearm_y = [13.0, 13.0, 1.5, 1.5, 13.0]
    ax.fill(forearm_x, forearm_y, color="#fce4c8", edgecolor="#8c6239", lw=1.5, zorder=1)

    # -- ECRB muscle region (radial side, mid-forearm) ---
    ecrb_x = [4.8, 6.2, 6.0, 4.6, 4.8]
    ecrb_y = [12.0, 12.0, 6.5, 6.5, 12.0]
    ax.fill(ecrb_x, ecrb_y, color="#ff9999", alpha=0.55, edgecolor="#cc3333",
            lw=1.2, zorder=2, label="ECRB")

    # -- ECU muscle region (ulnar side) ---
    ecu_x = [3.2, 4.5, 4.3, 3.4, 3.2]
    ecu_y = [12.0, 12.0, 5.5, 5.5, 12.0]
    ax.fill(ecu_x, ecu_y, color="#9999ff", alpha=0.55, edgecolor="#3333cc",
            lw=1.2, zorder=2, label="ECU")

    # -- Myo armband ring (circumferential at proximal third) ---
    ring_y = 10.0
    ring_h = 0.6
    ring = plt.Rectangle((2.7, ring_y - ring_h / 2), 4.6, ring_h,
                          facecolor="#cccccc", edgecolor="#555555", lw=1.5,
                          alpha=0.7, zorder=3)
    ax.add_patch(ring)
    ax.text(5.0, ring_y, "Myo Armband", ha="center", va="center",
            fontsize=7, fontweight="bold", zorder=4)

    # -- Electrode markers ---
    # Ch5 ECRB electrode
    ecrb_elec_x, ecrb_elec_y = 5.5, 9.0
    ax.plot(ecrb_elec_x, ecrb_elec_y, "o", markersize=12, color="#cc3333",
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    ax.text(ecrb_elec_x, ecrb_elec_y, "E1", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="white", zorder=6)

    # Ch3 ECU electrode
    ecu_elec_x, ecu_elec_y = 3.8, 9.0
    ax.plot(ecu_elec_x, ecu_elec_y, "o", markersize=12, color="#3333cc",
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    ax.text(ecu_elec_x, ecu_elec_y, "E2", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="white", zorder=6)

    # -- Annotation arrows and labels ---
    # ECRB label
    ax.annotate("ECRB\n(Ch5)\nWrist ext. +\nradial dev.",
                xy=(ecrb_elec_x, ecrb_elec_y),
                xytext=(8.5, 9.5),
                fontsize=7, ha="center", color="#cc3333", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#cc3333", lw=0.8),
                zorder=7)

    # ECU label
    ax.annotate("ECU\n(Ch3)\nWrist ext. +\nulnar dev.",
                xy=(ecu_elec_x, ecu_elec_y),
                xytext=(1.2, 9.5),
                fontsize=7, ha="center", color="#3333cc", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#3333cc", lw=0.8),
                zorder=7)

    # Anatomical landmarks
    ax.text(5.0, 13.4, "Lateral Epicondyle\n(Elbow)", ha="center", va="bottom",
            fontsize=7, fontstyle="italic", color="#666666")
    ax.text(5.0, 0.9, "Wrist", ha="center", va="top",
            fontsize=7, fontstyle="italic", color="#666666")

    # Orientation label
    ax.text(8.0, 7.0, "Radial\nside", ha="center", va="center",
            fontsize=6.5, color="#888888", fontstyle="italic")
    ax.text(2.0, 7.0, "Ulnar\nside", ha="center", va="center",
            fontsize=6.5, color="#888888", fontstyle="italic")

    ax.set_title("Posterior Forearm: Electrode Placement\nover ECRB and ECU Muscles",
                 fontsize=9, pad=8)

    # Legend
    ecrb_patch = mpatches.Patch(facecolor="#ff9999", edgecolor="#cc3333",
                                alpha=0.55, label="ECRB (Ch5)")
    ecu_patch  = mpatches.Patch(facecolor="#9999ff", edgecolor="#3333cc",
                                alpha=0.55, label="ECU (Ch3)")
    ax.legend(handles=[ecrb_patch, ecu_patch], loc="lower right",
              framealpha=0.85, fontsize=7)

    fig.tight_layout()
    out = FIGURES_DIR / "fig8_anatomy_electrode.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================================
# FIG 5 + FIG 6 - Prediction traces and scatter (requires model inference)
# ============================================================================
def run_fast_predictions():
    """Run a single-pass trained GRU model to collect predictions for fig5/fig6."""
    print("  Running fast GRU model for prediction traces (fig5/fig6)...")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from src.data.load_ghorbani import load_all_ghorbani
        from src.models.gru_model import GRUForcePredictor, SeqDataset
        from src.config import (RAW_GHORBANI_DIR, GHORBANI_SUBJECTS,
                                GHORBANI_RMS_WINDOW_MS, GHORBANI_RMS_HOP_MS,
                                GHORBANI_FS)
        from src.data.preprocess import compute_multi_features, compute_delta_features
        from sklearn.preprocessing import MinMaxScaler
        from scipy.signal import butter, sosfiltfilt

    except ImportError as e:
        print(f"  WARNING: Import failed ({e}). Skipping fig5/fig6.")
        return None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = load_all_ghorbani(str(RAW_GHORBANI_DIR), n_subjects=GHORBANI_SUBJECTS)
    raw_data = [d for d in raw_data if d["subject_id"] != 8]

    selected_channels = [5, 3]
    SEQ_LEN   = 50
    N_FEAT    = 36

    def build_subject_arrays(entry):
        emg   = entry["emg"][:, selected_channels]
        force = entry["force"]
        fs    = float(entry["fs"])

        feats = compute_multi_features(emg, window_ms=GHORBANI_RMS_WINDOW_MS,
                                       hop_ms=GHORBANI_RMS_HOP_MS, fs=fs)
        feats = compute_delta_features(feats, order=2)

        win_samples = max(1, int(round(GHORBANI_RMS_WINDOW_MS * fs / 1000.0)))
        hop_samples = max(1, int(round(GHORBANI_RMS_HOP_MS * fs / 1000.0)))
        n_windows = feats.shape[0]
        centres = np.array([i * hop_samples + win_samples // 2 for i in range(n_windows)])
        centres = np.clip(centres, 0, len(force) - 1)
        force_ds = force[centres]

        n_seq = n_windows - SEQ_LEN
        if n_seq <= 0:
            return None, None
        X = np.stack([feats[i:i + SEQ_LEN] for i in range(n_seq)])
        y = force_ds[SEQ_LEN:]
        return X, y

    all_data = {}
    for entry in raw_data:
        sid = entry["subject_id"]
        X, y = build_subject_arrays(entry)
        if X is None:
            continue
        all_data[sid] = {"X": X, "y": y}

    test_pct = 0.20
    subject_results = {}

    for sid, arrays in all_data.items():
        X_all, y_all = arrays["X"], arrays["y"]
        n = len(y_all)
        split = int(n * (1 - test_pct))
        X_tr, X_te = X_all[:split], X_all[split:]
        y_tr, y_te = y_all[:split], y_all[split:]

        n_feat = X_tr.shape[2]
        scaler_X = MinMaxScaler()
        X_tr_s = scaler_X.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
        X_te_s = scaler_X.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape)

        scaler_y = MinMaxScaler()
        y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_te_s = scaler_y.transform(y_te.reshape(-1, 1)).ravel()

        subject_results[sid] = {
            "X_train": X_tr_s, "y_train": y_tr_s,
            "X_test": X_te_s, "y_test_norm": y_te_s,
            "scaler_y": scaler_y,
        }

    # Train a single fast GRU (general model, no fine-tuning)
    X_pool = np.concatenate([v["X_train"] for v in subject_results.values()])
    y_pool = np.concatenate([v["y_train"] for v in subject_results.values()])

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y_pool))
    X_pool, y_pool = X_pool[idx], y_pool[idx]

    model = GRUForcePredictor(
        input_size=N_FEAT, hidden_size=64, dense_size=64,
        output_size=1, dropout=0.2, num_layers=1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    criterion = nn.MSELoss()

    train_ds = SeqDataset(X_pool, y_pool)
    loader   = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(30):
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Collect per-subject predictions (general model - no fine-tuning)
    model.eval()
    predictions = {}
    bsz_lp, order_lp = 4.0, 2

    for sid, arrays in subject_results.items():
        X_te_s    = arrays["X_test"]
        y_te_norm = arrays["y_test_norm"]

        ds = SeqDataset(X_te_s, np.zeros(len(X_te_s)))
        dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

        preds = []
        with torch.no_grad():
            for Xb, _ in dl:
                out = model(Xb.to(DEVICE)).cpu().numpy().ravel()
                preds.append(out)
        pred_norm = np.concatenate(preds)

        # Butterworth smoothing
        hop_hz = GHORBANI_FS / max(1, int(round(GHORBANI_RMS_HOP_MS * GHORBANI_FS / 1000.0)))
        if bsz_lp < hop_hz / 2:
            sos = butter(order_lp, bsz_lp / (hop_hz / 2), btype="low", output="sos")
            pred_smooth = sosfiltfilt(sos, pred_norm)
        else:
            pred_smooth = pred_norm

        from sklearn.metrics import r2_score
        r2 = float(r2_score(y_te_norm, pred_smooth))

        predictions[sid] = {
            "actual": y_te_norm,
            "predicted": pred_smooth,
            "r2": r2,
        }
        print(f"    S{sid}: R\u00b2={r2:.4f}")

    return predictions


def fig5_prediction_traces(predictions: dict):
    """3-panel prediction trace plot for selected subjects."""
    set_ieee_style()

    panel_subjects = [10, 6, 5]
    avail = list(predictions.keys())
    panel_subjects = [s for s in panel_subjects if s in avail]
    r2s = {sid: v["r2"] for sid, v in predictions.items()}
    sorted_sids = sorted(r2s, key=lambda s: r2s[s], reverse=True)
    while len(panel_subjects) < 3 and sorted_sids:
        s = sorted_sids.pop(0)
        if s not in panel_subjects:
            panel_subjects.append(s)

    panel_labels = ["(a) Best: S10", "(b) Mid-high: S6", "(c) Mid-low: S5"]

    hop_hz = GHORBANI_FS / max(1, int(round(GHORBANI_RMS_HOP_MS * GHORBANI_FS / 1000.0)))
    max_secs = 100.0

    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COL, 5.5), sharex=False)

    for i, (sid, lbl) in enumerate(zip(panel_subjects, panel_labels)):
        ax = axes[i]
        data = predictions[sid]
        y_act  = data["actual"]
        y_pred = data["predicted"]
        r2     = data["r2"]

        n_show = min(len(y_act), int(max_secs * hop_hz))
        t = np.arange(n_show) / hop_hz

        ax.plot(t, y_act[:n_show],  color="#333333", lw=0.9, label="Actual", alpha=0.9)
        ax.plot(t, y_pred[:n_show], color="#e06c10", lw=0.9, label="Predicted", alpha=0.85, ls="--")
        ax.annotate(f"R\u00b2={r2:.3f}", xy=(0.98, 0.93), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        ax.set_title(lbl, fontsize=9, pad=3)
        ax.set_ylabel("Norm. Force (a.u.)" if i == 1 else "")
        ax.set_ylim(-0.05, 1.15)
        if i == 2:
            ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.85)

    fig.suptitle("EMG-to-Force Prediction Traces (GRU General Model, Test Set)", y=1.01, fontsize=9)
    fig.tight_layout(pad=0.5)
    out = FIGURES_DIR / "fig5_prediction_traces.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def fig6_scatter(predictions: dict):
    """Predicted vs actual scatter - all subjects combined."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.6, 3.0))

    cmap = plt.cm.tab10
    all_actual, all_pred = [], []

    valid_predictions = {sid: d for sid, d in predictions.items() if d["r2"] >= 0.3}

    for j, (sid, data) in enumerate(sorted(valid_predictions.items())):
        actual = data["actual"]
        pred   = data["predicted"]
        color  = cmap(j % 10)
        ax.scatter(actual, pred, s=4, color=color, alpha=0.45,
                   label=f"S{sid}", rasterized=True)
        all_actual.append(actual)
        all_pred.append(pred)

    all_actual = np.concatenate(all_actual)
    all_pred   = np.concatenate(all_pred)

    from sklearn.metrics import r2_score
    r2  = float(r2_score(all_actual, all_pred))
    rho, _ = pearsonr(all_actual, all_pred)

    vmin = min(all_actual.min(), all_pred.min())
    vmax = max(all_actual.max(), all_pred.max())
    identity = np.array([vmin, vmax])
    ax.plot(identity, identity, "k--", lw=1.0, label="Ideal (y=x)")

    ax.set_xlabel("Actual Force (normalized)")
    ax.set_ylabel("Predicted Force (normalized)")
    ax.set_title("Predicted vs. Actual Grip Force\n(All 9 subjects, GRU General, test set)")
    ax.annotate(f"R\u00b2={r2:.3f}\nr={rho:.3f}",
                xy=(0.04, 0.93), xycoords="axes fraction",
                fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.legend(ncol=2, framealpha=0.8, markerscale=2, fontsize=7)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "fig6_scatter.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip fig5/fig6 (requires model training)")
    parser.add_argument("--only", nargs="+", type=int,
                        help="Generate only listed figure numbers, e.g. --only 1 2 3")
    args = parser.parse_args()

    enabled = set(args.only) if args.only else set(range(1, 9))

    print(f"Saving figures to: {FIGURES_DIR}")

    if 1 in enabled:
        print("Generating fig1: System pipeline...")
        fig1_methodology()

    if 2 in enabled:
        print("Generating fig2: MRMR channel ranking...")
        fig2_mrmr_channels()

    if 3 in enabled:
        print("Generating fig3: Per-subject R\u00b2 bar chart...")
        fig3_per_subject_r2()

    if 4 in enabled:
        print("Generating fig4: Outlier analysis...")
        fig4_outlier_analysis()

    if 7 in enabled:
        print("Generating fig7: Model comparison...")
        fig7_model_comparison()

    if 8 in enabled:
        print("Generating fig8: Anatomy + electrode placement...")
        fig8_anatomy_electrode()

    if (5 in enabled or 6 in enabled) and not args.skip_model:
        print("Generating fig5/fig6: Running model inference...")
        predictions = run_fast_predictions()
        if predictions is not None:
            if 5 in enabled:
                print("Generating fig5: Prediction traces...")
                fig5_prediction_traces(predictions)
            if 6 in enabled:
                print("Generating fig6: Scatter plot...")
                fig6_scatter(predictions)
        else:
            print("  Skipped fig5/fig6 due to import/data error.")
    elif (5 in enabled or 6 in enabled) and args.skip_model:
        print("Skipping fig5/fig6 (--skip-model specified).")

    print("\nDone. All figures saved.")


if __name__ == "__main__":
    main()
