"""
Evaluation metrics and per-subject analysis for EMG-to-Force prediction.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.

    Returns
    -------
    dict with keys: R2, RMSE, NRMSE (%), MAE, correlation
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = y_true.max() - y_true.min()
    nrmse = (rmse / y_range * 100) if y_range > 0 else 0.0
    mae = mean_absolute_error(y_true, y_pred)
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "NRMSE_pct": float(nrmse),
        "MAE": float(mae),
        "correlation": float(corr),
    }


def per_subject_evaluation(y_true, y_pred, subjects):
    """
    Compute metrics for each subject individually.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    subjects : array-like of subject IDs (same length as y_true)

    Returns
    -------
    list of dicts, each with 'subject_id' and metric keys
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    subjects = np.asarray(subjects).ravel()

    results = []
    for sid in np.unique(subjects):
        mask = subjects == sid
        if mask.sum() < 2:
            continue
        metrics = compute_metrics(y_true[mask], y_pred[mask])
        metrics["subject_id"] = int(sid)
        metrics["n_samples"] = int(mask.sum())
        results.append(metrics)

    results.sort(key=lambda x: x["subject_id"])
    return results


def print_results(metrics, dataset_name=""):
    """Pretty-print a results table."""
    title = f"Results -- {dataset_name}" if dataset_name else "Results"
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  {'R2':<15} {metrics['R2']:.4f}")
    print(f"  {'RMSE':<15} {metrics['RMSE']:.4f}")
    print(f"  {'NRMSE':<15} {metrics['NRMSE_pct']:.2f}%")
    print(f"  {'MAE':<15} {metrics['MAE']:.4f}")
    print(f"  {'Correlation':<15} {metrics['correlation']:.4f}")
    print(f"{'=' * 50}\n")


def print_per_subject_results(subject_results):
    """Print per-subject metrics table."""
    print(f"\n{'Subject':<10} {'R2':<10} {'RMSE':<10} {'Corr':<10} {'N':<8}")
    print("-" * 48)
    for sr in subject_results:
        print(
            f"  S{sr['subject_id']:<7} "
            f"{sr['R2']:<10.4f} "
            f"{sr['RMSE']:<10.4f} "
            f"{sr['correlation']:<10.4f} "
            f"{sr['n_samples']:<8}"
        )
    # Summary
    r2s = [sr["R2"] for sr in subject_results]
    print("-" * 48)
    print(f"  {'Mean':<7} {np.mean(r2s):.4f}")
    print(f"  {'Std':<7}  {np.std(r2s):.4f}")
    print(f"  {'Min':<7}  {np.min(r2s):.4f}")
    print(f"  {'Max':<7}  {np.max(r2s):.4f}")
