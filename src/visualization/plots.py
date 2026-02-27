"""
Visualization functions for EMG-to-Force prediction project.
All plots are saved to the specified path (not displayed interactively).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mrmr_ranking(relevance, mrmr_scores, channel_names, selected_indices, save_path):
    """
    Bar chart of MRMR channel scores showing relevance and final MRMR score.

    Parameters
    ----------
    relevance : array of shape (n_channels,)
    mrmr_scores : array of shape (n_channels,) — final MRMR score per channel
    channel_names : list of str
    selected_indices : list of int — indices of selected channels
    save_path : Path or str
    """
    n = len(relevance)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, relevance, width, label="Relevance (MI with Force)",
                   color="#4C72B0", alpha=0.8)
    bars2 = ax.bar(x + width / 2, mrmr_scores, width, label="MRMR Score",
                   color="#DD8452", alpha=0.8)

    # Highlight selected channels
    for idx in selected_indices:
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.15, color="green")

    ax.set_xlabel("EMG Channel", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("MRMR Channel Selection — Relevance vs MRMR Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate selected
    for idx in selected_indices:
        ax.annotate("SELECTED", (idx, max(relevance[idx], mrmr_scores[idx])),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, fontweight="bold", color="green")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_redundancy_heatmap(redundancy_matrix, channel_names, save_path):
    """
    Heatmap of pairwise mutual information between channels.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        redundancy_matrix,
        xticklabels=channel_names,
        yticklabels=channel_names,
        annot=True, fmt=".3f", cmap="YlOrRd",
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mutual Information"},
    )
    ax.set_title("Pairwise Redundancy (MI between EMG Channels)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(train_losses, val_losses, dataset_name, save_path):
    """Plot training and validation MSE loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE", linewidth=2, color="#4C72B0")
    ax.plot(epochs, val_losses, label="Val MSE", linewidth=2, color="#DD8452")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(f"GRU Training Curve — {dataset_name.upper()}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Best epoch: {best_epoch}")
    ax.scatter([best_epoch], [best_val], color="green", s=80, zorder=5)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_prediction_vs_actual(y_true, y_pred, dataset_name, save_path, n_points=3000):
    """Time-series overlay of predicted vs actual force."""
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(n_points, len(y_true))
    t = np.arange(n)
    ax.plot(t, y_true[:n], label="Actual", linewidth=0.8, color="orange", alpha=0.9)
    ax.plot(t, y_pred[:n], label="Predicted", linewidth=0.8, color="green",
            linestyle="--", alpha=0.8)
    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("Force (normalized)", fontsize=12)
    ax.set_title(f"Prediction vs Actual — {dataset_name.upper()}", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter(y_true, y_pred, r2, dataset_name, save_path, n_points=5000):
    """Scatter plot with identity line and R2 annotation."""
    fig, ax = plt.subplots(figsize=(7, 7))
    idx = np.random.choice(len(y_true), min(n_points, len(y_true)), replace=False)
    ax.scatter(y_true[idx], y_pred[idx], alpha=0.2, s=5, color="#4C72B0")

    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Identity")

    ax.set_xlabel("Actual Force", fontsize=12)
    ax.set_ylabel("Predicted Force", fontsize=12)
    ax.set_title(f"Scatter — {dataset_name.upper()} (R² = {r2:.4f})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_subject_r2(subject_metrics, dataset_name, save_path):
    """Bar chart of R2 per subject."""
    sids = [s["subject_id"] for s in subject_metrics]
    r2s = [s["R2"] for s in subject_metrics]

    fig, ax = plt.subplots(figsize=(max(10, len(sids) * 0.4), 5))
    colors = ["#C44E52" if r < 0.5 else "#DD8452" if r < 0.7 else "#4C72B0" for r in r2s]
    bars = ax.bar(range(len(sids)), r2s, color=colors, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"Per-Subject R² — {dataset_name.upper()}", fontsize=14)
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels([f"S{s}" for s in sids], rotation=45, ha="right", fontsize=8)
    ax.axhline(np.mean(r2s), color="green", linestyle="--", lw=1.5,
               label=f"Mean R² = {np.mean(r2s):.4f}")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(0, min(r2s) - 0.05), 1.05)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_dataset_comparison(results_dict, save_path):
    """
    Side-by-side comparison of metrics across datasets.

    Parameters
    ----------
    results_dict : dict mapping dataset_name → metrics dict
    save_path : Path or str
    """
    metric_keys = ["R2", "RMSE", "MAE", "correlation"]
    labels = list(results_dict.keys())
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, metric in enumerate(metric_keys):
        ax = axes[i]
        vals = [results_dict[name].get(metric, 0) for name in labels]
        bars = ax.bar(labels, vals, color=colors[:len(labels)], alpha=0.85)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Cross-Dataset Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
