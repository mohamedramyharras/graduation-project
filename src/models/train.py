"""
Main training pipeline for EMG-to-Force prediction.

Usage:
    python -m src.models.train --dataset ninapro
    python -m src.models.train --dataset hyser
"""
import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BANDPASS_HIGH, BANDPASS_LOW, BANDPASS_ORDER,
    BATCH_SIZE, EARLY_STOPPING_PATIENCE, ENVELOPE_HOP_MS,
    ENVELOPE_WINDOW_MS, EPOCHS, FIGURES_DIR, GRU_DENSE_SIZE,
    GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, LEARNING_RATE, MODELS_DIR,
    MRMR_N_SELECT, NINAPRO_CHANNEL_NAMES, NINAPRO_FS, NINAPRO_SUBJECTS,
    RANDOM_SEED, RAW_HYSER_DIR, RAW_NINAPRO_DIR, RAW_GHORBANI_DIR,
    RESULTS_DIR, SEQ_LEN, TEST_SPLIT, VAL_SPLIT,
    HYSER_FS, GHORBANI_FS, GHORBANI_CHANNEL_NAMES, GHORBANI_SUBJECTS,
    GHORBANI_RMS_WINDOW_MS, GHORBANI_RMS_HOP_MS,
)
from src.data.load_ninapro import load_all_ninapro
from src.data.preprocess import (
    bandpass_filter, compute_rms_envelope, create_sequences,
    temporal_block_split,
)
from src.features.mrmr import run_mrmr_analysis
from src.models.evaluate import (
    compute_metrics, per_subject_evaluation, print_per_subject_results,
    print_results,
)
from src.models.gru_model import GRUForcePredictor, SeqDataset, count_parameters

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"


def _seed_worker(worker_id):
    """Ensure reproducible DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def _process_one_segment(emg, force, window_ms, hop_ms, fs):
    """Process a single EMG/force segment: RMS envelope + sequences."""
    win_samp = int(window_ms * fs / 1000)
    hop_samp = int(hop_ms * fs / 1000)
    envelope = compute_rms_envelope(emg, window_ms, hop_ms, fs)

    n_windows = len(envelope)
    force_env = np.zeros(n_windows, dtype=np.float32)
    for j in range(n_windows):
        start = j * hop_samp
        end = min(start + win_samp, len(force))
        force_env[j] = np.mean(force[start:end])

    X, y = create_sequences(envelope, force_env, seq_len=SEQ_LEN, pred_horizon=1)
    return X, y


def preprocess_subjects(subject_data, selected_channels, fs, dataset_name,
                        window_ms=ENVELOPE_WINDOW_MS, hop_ms=ENVELOPE_HOP_MS):
    """
    For each subject: select channels -> compute RMS envelope -> create sequences.

    Returns X, y, subject_ids as concatenated arrays.
    """
    all_X, all_y, all_s = [], [], []

    for i, subj in enumerate(subject_data):
        emg = subj["emg"][:, selected_channels]
        force = subj["force"]
        sid = subj["subject_id"]

        X, y = _process_one_segment(emg, force, window_ms, hop_ms, fs)

        if len(X) == 0:
            continue

        all_X.append(X)
        all_y.append(y)
        all_s.append(np.full(len(y), sid, dtype=np.int32))

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(subject_data)} subjects...")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    s = np.concatenate(all_s, axis=0)
    print(f"    Total: {len(y):,} sequences, shape: {X.shape}")
    return X, y, s


def preprocess_subjects_by_trial(subject_data, selected_channels, fs, dataset_name,
                                  window_ms=ENVELOPE_WINDOW_MS, hop_ms=ENVELOPE_HOP_MS):
    """
    Process each trial independently to avoid cross-trial boundary sequences.

    Returns X, y, subject_ids, trial_ids as concatenated arrays.
    Each sequence is labeled with its trial number (0-indexed).
    """
    all_X, all_y, all_s, all_t = [], [], [], []

    for i, subj in enumerate(subject_data):
        emg_full = subj["emg"][:, selected_channels]
        force_full = subj["force"]
        sid = subj["subject_id"]
        boundaries = subj.get("trial_boundaries", [(0, len(emg_full))])

        for trial_idx, (start, end) in enumerate(boundaries):
            emg_trial = emg_full[start:end]
            force_trial = force_full[start:end]

            X, y = _process_one_segment(emg_trial, force_trial, window_ms, hop_ms, fs)

            if len(X) == 0:
                continue

            all_X.append(X)
            all_y.append(y)
            all_s.append(np.full(len(y), sid, dtype=np.int32))
            all_t.append(np.full(len(y), trial_idx, dtype=np.int32))

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(subject_data)} subjects...")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    s = np.concatenate(all_s, axis=0)
    t = np.concatenate(all_t, axis=0)
    print(f"    Total: {len(y):,} sequences, shape: {X.shape}")
    return X, y, s, t


def train_model(train_loader, val_loader, model, epochs, lr, patience, device,
                weight_decay=0.0, noise_std=0.0):
    """
    Train the GRU model with early stopping, LR scheduler, and gradient clipping.

    Returns trained model, train_losses, val_losses.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if noise_std > 0:
                Xb = Xb + torch.randn_like(Xb) * noise_std
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss_sum += criterion(model(Xb), yb).item()
                n_val += 1
        val_loss = val_loss_sum / n_val
        val_losses.append(val_loss)

        # LR scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"    Epoch {epoch + 1:3d}/{epochs}  "
                f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  "
                f"lr={current_lr:.1e}"
                f"{'  *best*' if patience_counter == 0 else ''}"
            )

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def _numpy_safe(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_per_subject_models(X, y, subjects, args, selected_channels, selected_names,
                              rms_window_ms, rms_hop_ms, device, trial_ids=None):
    """
    Train individual GRU models per subject (subject-dependent approach).

    If trial_ids is provided (Ghorbani dataset), uses trial-based splitting:
    train on trials 0+1, test on last trial. Otherwise falls back to temporal
    block split (first 80% train, last 20% test).
    """
    unique_subjects = sorted(np.unique(subjects).astype(int))
    n_ch = X.shape[2]
    dropout = 0.15 if args.dataset == "ghorbani" else 0.3
    num_layers = getattr(args, 'num_layers', 1)
    use_trial_split = trial_ids is not None

    all_subj_metrics = []
    all_y_test, all_y_pred, all_s_test = [], [], []
    per_subj_models = {}
    per_subj_scalers = {}
    total_train_time = 0
    total_epochs = 0
    total_train = total_val = total_test = 0

    split_mode = "per-trial temporal (80/20 within each trial)" if use_trial_split else "temporal block (80/20)"
    print(f"\n[4] Per-subject GRU training ({len(unique_subjects)} subjects)...")
    print(f"    Split mode: {split_mode}")
    print(f"    Architecture: GRU({n_ch}->{args.hidden_size}, {num_layers}L, relu) -> Drop({dropout}) "
          f"-> Dense({args.dense_size}, relu) -> Drop({dropout}) -> Dense({GRU_OUTPUT_SIZE})")

    model_template = GRUForcePredictor(
        input_size=n_ch, hidden_size=args.hidden_size,
        dense_size=args.dense_size, output_size=GRU_OUTPUT_SIZE, dropout=dropout,
        num_layers=num_layers,
    )
    params = count_parameters(model_template)
    print(f"    Parameters per model: {params['total']:,}")
    print(f"    Config: MSE loss, Adam (lr={args.lr}), {args.epochs} epochs, patience={args.patience}")

    t0_total = time.time()

    for sid_int in unique_subjects:
        mask = (subjects == sid_int)
        X_s = X[mask].copy()
        y_s = y[mask].copy()

        print(f"\n    --- Subject {sid_int} ({len(X_s):,} sequences) ---")

        # Split: per-trial temporal or temporal block
        if use_trial_split:
            t_s = trial_ids[mask]
            unique_trials = sorted(np.unique(t_s).astype(int))
            # Within each trial: first 80% train, last 20% test
            train_indices = []
            test_indices = []
            for trial_idx in unique_trials:
                trial_mask = np.where(t_s == trial_idx)[0]
                n_trial = len(trial_mask)
                split_pt = int(n_trial * (1 - TEST_SPLIT))
                train_indices.extend(trial_mask[:split_pt].tolist())
                test_indices.extend(trial_mask[split_pt:].tolist())
            train_indices = np.array(train_indices, dtype=int)
            test_indices = np.array(test_indices, dtype=int)
            X_train_s = X_s[train_indices]
            y_train_s = y_s[train_indices]
            X_test_s = X_s[test_indices]
            y_test_s = y_s[test_indices]
            print(f"      Per-trial temporal split: {len(unique_trials)} trials, "
                  f"80/20 within each trial")
        else:
            split_idx = int(len(X_s) * (1 - TEST_SPLIT))
            X_train_s = X_s[:split_idx]
            y_train_s = y_s[:split_idx]
            X_test_s = X_s[split_idx:]
            y_test_s = y_s[split_idx:]

        # Per-subject normalization (fit on train only)
        X_flat_tr = X_train_s.reshape(-1, n_ch)
        X_flat_te = X_test_s.reshape(-1, n_ch)
        emg_scaler = MinMaxScaler(feature_range=(0, 1))
        X_flat_tr = emg_scaler.fit_transform(X_flat_tr).astype(np.float32)
        X_flat_te = emg_scaler.transform(X_flat_te).astype(np.float32)
        X_train_s = X_flat_tr.reshape(X_train_s.shape)
        X_test_s = X_flat_te.reshape(X_test_s.shape)

        force_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train_s = force_scaler.fit_transform(y_train_s.reshape(-1, 1)).ravel().astype(np.float32)
        y_test_s = force_scaler.transform(y_test_s.reshape(-1, 1)).ravel().astype(np.float32)

        per_subj_scalers[sid_int] = {'emg': emg_scaler, 'force': force_scaler}
        print(f"      Force range: [{force_scaler.data_min_[0]:.1f}, {force_scaler.data_max_[0]:.1f}] N")

        # Train/val split (shuffle train, split 85/15)
        indices = np.arange(len(X_train_s))
        rng = np.random.RandomState(RANDOM_SEED + sid_int)
        rng.shuffle(indices)
        X_train_s = X_train_s[indices]
        y_train_s = y_train_s[indices]

        val_idx = int(len(X_train_s) * (1 - VAL_SPLIT))
        X_val_s = X_train_s[val_idx:]
        y_val_s = y_train_s[val_idx:]
        X_train_s = X_train_s[:val_idx]
        y_train_s = y_train_s[:val_idx]

        total_train += len(y_train_s)
        total_val += len(y_val_s)
        total_test += len(y_test_s)

        print(f"      Train: {len(y_train_s):,}, Val: {len(y_val_s):,}, Test: {len(y_test_s):,}")

        # DataLoaders
        g = torch.Generator()
        g.manual_seed(int(RANDOM_SEED + sid_int))
        train_loader = DataLoader(
            SeqDataset(X_train_s, y_train_s), batch_size=args.batch_size,
            shuffle=True, num_workers=0, worker_init_fn=_seed_worker, generator=g,
            pin_memory=PIN_MEMORY,
        )
        val_loader = DataLoader(SeqDataset(X_val_s, y_val_s), batch_size=256,
                                shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
        test_loader = DataLoader(SeqDataset(X_test_s, y_test_s), batch_size=256,
                                 shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)

        # Build model
        torch.manual_seed(int(RANDOM_SEED + sid_int))
        model = GRUForcePredictor(
            input_size=n_ch, hidden_size=args.hidden_size,
            dense_size=args.dense_size, output_size=GRU_OUTPUT_SIZE,
            dropout=dropout, num_layers=num_layers,
        ).to(device)

        # Train
        t0 = time.time()
        model, train_losses, val_losses = train_model(
            train_loader, val_loader, model, args.epochs, args.lr,
            args.patience, device,
        )
        subj_train_time = time.time() - t0
        total_train_time += subj_train_time
        total_epochs += len(train_losses)

        per_subj_models[sid_int] = model.state_dict()

        # Evaluate
        model.eval()
        preds = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                preds.append(model(Xb.to(device)).cpu().numpy())
        y_pred_s = np.concatenate(preds).ravel()

        metrics_s = compute_metrics(y_test_s, y_pred_s)
        metrics_s['subject_id'] = int(sid_int)
        metrics_s['n_samples'] = int(len(y_test_s))
        # Ensure all metric values are JSON-safe
        metrics_s = {k: _numpy_safe(v) for k, v in metrics_s.items()}
        all_subj_metrics.append(metrics_s)

        all_y_test.append(y_test_s)
        all_y_pred.append(y_pred_s)
        all_s_test.append(np.full(len(y_test_s), sid_int))

        print(f"      => R2={metrics_s['R2']:.4f}, RMSE={metrics_s['RMSE']:.4f}, "
              f"Corr={metrics_s['correlation']:.4f} ({subj_train_time:.0f}s, {len(train_losses)} ep)")

    # Aggregate
    y_test_all = np.concatenate(all_y_test)
    y_pred_all = np.concatenate(all_y_pred)
    s_test_all = np.concatenate(all_s_test)
    overall_metrics = compute_metrics(y_test_all, y_pred_all)

    r2_values = [m['R2'] for m in all_subj_metrics]
    mean_r2 = float(np.mean(r2_values))

    total_train_time_total = time.time() - t0_total
    print(f"\n    Total training time: {total_train_time_total:.1f}s")

    # Print summary
    print_results(overall_metrics, f"{args.dataset.upper()} -- Per-Subject GRU ({n_ch}-channel MRMR)")
    print(f"  Mean per-subject R2: {mean_r2:.4f}")
    print(f"\n  Per-subject breakdown:")
    print_per_subject_results(all_subj_metrics)

    # Save outputs
    print(f"\n[5] Saving outputs...")

    model_path = MODELS_DIR / f"gru_{args.dataset}_best.pt"
    torch.save({
        "per_subject_models": per_subj_models,
        "architecture": f"GRU({n_ch}->{args.hidden_size},{num_layers}L,relu)->Drop({dropout})->Dense({args.dense_size},relu)->Drop({dropout})->Dense({GRU_OUTPUT_SIZE})",
        "selected_channels": selected_channels,
        "selected_channel_names": selected_names,
        "input_size": n_ch,
        "training_mode": "per_subject",
        "num_layers": num_layers,
    }, model_path)
    print(f"    Model -> {model_path}")

    joblib.dump(per_subj_scalers, MODELS_DIR / f"scalers_{args.dataset}.joblib")
    print(f"    Scalers -> {MODELS_DIR}")

    # Use mean per-subject R2 as the primary metric
    metrics_report = {k: _numpy_safe(v) for k, v in overall_metrics.items()}
    metrics_report['mean_per_subject_R2'] = mean_r2

    results = {
        "dataset": args.dataset,
        "model": "GRU (per-subject)",
        "n_channels": int(n_ch),
        "selected_channels": [int(c) for c in selected_channels],
        "selected_channel_names": selected_names,
        "metrics": metrics_report,
        "per_subject": all_subj_metrics,
        "training": {
            "epochs_run": int(total_epochs),
            "epochs_max": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "train_time_s": round(total_train_time_total, 1),
            "best_val_loss": None,
            "final_train_loss": None,
            "total_params": int(params["total"]),
            "mode": "per_subject",
        },
        "preprocessing": {
            "seq_len": int(SEQ_LEN),
            "envelope_window_ms": int(rms_window_ms),
            "envelope_hop_ms": int(rms_hop_ms),
            "bandpass_hz": [float(BANDPASS_LOW), float(BANDPASS_HIGH)],
            "test_split": float(TEST_SPLIT),
            "val_split": float(VAL_SPLIT),
        },
        "n_train": int(total_train),
        "n_val": int(total_val),
        "n_test": int(total_test),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = RESULTS_DIR / f"{args.dataset}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Results -> {results_path}")

    # Generate plots
    print(f"\n[6] Generating plots...")
    try:
        from src.visualization.plots import (
            plot_training_curves, plot_prediction_vs_actual,
            plot_scatter, plot_per_subject_r2,
        )
        plot_prediction_vs_actual(
            y_test_all, y_pred_all, args.dataset,
            FIGURES_DIR / f"prediction_vs_actual_{args.dataset}.png",
        )
        plot_scatter(
            y_test_all, y_pred_all, overall_metrics["R2"], args.dataset,
            FIGURES_DIR / f"scatter_{args.dataset}.png",
        )
        plot_per_subject_r2(
            all_subj_metrics, args.dataset,
            FIGURES_DIR / f"per_subject_r2_{args.dataset}.png",
        )
        print(f"    Plots -> {FIGURES_DIR}")
    except Exception as e:
        warnings.warn(f"Could not generate plots: {e}")

    print(f"\n{'=' * 60}")
    print(f"  DONE -- Overall R2 = {overall_metrics['R2']:.4f}, Mean per-subject R2 = {mean_r2:.4f}")
    print(f"{'=' * 60}")

    return metrics_report


def main():
    parser = argparse.ArgumentParser(description="Train GRU for EMG-to-Force prediction")
    parser.add_argument("--dataset", type=str, default="ninapro",
                        choices=["ninapro", "hyser", "ghorbani"],
                        help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--n-channels", type=int, default=MRMR_N_SELECT,
                        help="Number of channels to select via MRMR")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="GRU hidden size (default: config value or 128 for ghorbani)")
    parser.add_argument("--dense-size", type=int, default=None,
                        help="Dense layer size (default: config value or 128 for ghorbani)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (default: config value or 25 for ghorbani)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of GRU layers (default: 1, or 2 for ghorbani)")
    parser.add_argument("--skip-mrmr", action="store_true",
                        help="Skip MRMR and use channels from saved results")
    parser.add_argument("--all-channels", action="store_true",
                        help="Use all EMG channels (skip MRMR selection)")
    parser.add_argument("--per-subject", action="store_true",
                        help="Train separate model per subject (subject-dependent approach)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="L2 regularization weight decay for Adam optimizer")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (default: 0.2 for general, 0.15 for per-subject ghorbani)")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise std for training augmentation (0 = disabled)")
    parser.add_argument("--random-split", action="store_true",
                        help="Use shuffled random split instead of temporal block split")
    args = parser.parse_args()

    # Dataset-specific defaults
    if args.dataset == "ghorbani":
        if args.hidden_size is None:
            args.hidden_size = GRU_HIDDEN_SIZE   # 50 (Arduino-compatible)
        if args.dense_size is None:
            args.dense_size = GRU_DENSE_SIZE     # 100 (Arduino-compatible)
        if args.patience is None:
            args.patience = 10
        if args.num_layers is None:
            args.num_layers = 1                  # single layer (Arduino-compatible)
        if args.epochs == EPOCHS:
            args.epochs = 30
        rms_window_ms = GHORBANI_RMS_WINDOW_MS
        rms_hop_ms = GHORBANI_RMS_HOP_MS
    else:
        if args.hidden_size is None:
            args.hidden_size = GRU_HIDDEN_SIZE
        if args.dense_size is None:
            args.dense_size = GRU_DENSE_SIZE
        if args.patience is None:
            args.patience = EARLY_STOPPING_PATIENCE
        if args.num_layers is None:
            args.num_layers = 1
        rms_window_ms = ENVELOPE_WINDOW_MS
        rms_hop_ms = ENVELOPE_HOP_MS

    # Default dropout if not specified
    if args.dropout is None:
        args.dropout = 0.2

    print("=" * 60)
    print(f"EMG-TO-FORCE PREDICTION -- GRU with MRMR Channel Selection")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Ensure output directories exist
    for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── [1] Load raw data ──────────────────────────────────────────
    print("\n[1] Loading raw data...")
    if args.dataset == "ninapro":
        subject_data = load_all_ninapro(str(RAW_NINAPRO_DIR), n_subjects=NINAPRO_SUBJECTS)
        fs = NINAPRO_FS
        channel_names = NINAPRO_CHANNEL_NAMES
    elif args.dataset == "hyser":
        try:
            from src.data.load_hyser import load_all_hyser
            subject_data = load_all_hyser(str(RAW_HYSER_DIR))
            fs = HYSER_FS
            channel_names = [f"HD_Ch{i}" for i in range(len(subject_data[0]["emg"][0]))]
        except Exception as e:
            print(f"ERROR: Could not load Hyser dataset: {e}")
            print("Make sure to download it first: python -m src.data.download_hyser")
            return
    elif args.dataset == "ghorbani":
        try:
            from src.data.load_ghorbani import load_all_ghorbani, print_ghorbani_summary
            subject_data = load_all_ghorbani(str(RAW_GHORBANI_DIR), n_subjects=GHORBANI_SUBJECTS)
            fs = GHORBANI_FS
            channel_names = GHORBANI_CHANNEL_NAMES
            print_ghorbani_summary(subject_data)
        except Exception as e:
            print(f"ERROR: Could not load Ghorbani dataset: {e}")
            print("Ensure the data is in: data/ghorbani_raw/Dataset/Filtered/")
            return

    if len(subject_data) == 0:
        print("ERROR: No subject data loaded.")
        return

    n_channels = subject_data[0]["emg"].shape[1]
    print(f"    Loaded {len(subject_data)} subjects, {n_channels} EMG channels")

    # ── [2] Channel selection ────────────────────────────────────────
    mrmr_path = MODELS_DIR / f"mrmr_results_{args.dataset}.json"

    if args.all_channels:
        print(f"\n[2] Using ALL {n_channels} EMG channels (--all-channels)")
        selected_channels = list(range(n_channels))
        selected_names = channel_names[:n_channels] if isinstance(channel_names, list) else [f"Ch{i}" for i in range(n_channels)]
    elif args.skip_mrmr and mrmr_path.exists():
        print(f"\n[2] Loading saved MRMR results from {mrmr_path}")
        with open(mrmr_path) as f:
            mrmr_saved = json.load(f)
        selected_channels = mrmr_saved["selected_indices"]
        selected_names = mrmr_saved["selected_names"]
    else:
        print(f"\n[2] Running MRMR channel selection (selecting {args.n_channels} from {n_channels})...")
        print(f"    NOTE: Using only first {int((1-TEST_SPLIT)*100)}% of each subject's data to prevent leakage")
        mrmr_result = run_mrmr_analysis(
            subject_data, channel_names=channel_names, n_select=args.n_channels,
            train_ratio=1.0 - TEST_SPLIT,
        )
        selected_channels = mrmr_result["selected_indices"]
        selected_names = mrmr_result["selected_names"]

        # Save MRMR results
        mrmr_save = {
            "selected_indices": selected_channels,
            "selected_names": selected_names,
            "relevance": mrmr_result["relevance"].tolist(),
            "redundancy_matrix": mrmr_result["redundancy_matrix"].tolist(),
            "mrmr_scores": mrmr_result["mrmr_scores"],
            "all_channel_mrmr_scores": mrmr_result["all_channel_mrmr_scores"].tolist(),
        }
        with open(mrmr_path, "w") as f:
            json.dump(mrmr_save, f, indent=2)
        print(f"    MRMR results saved to {mrmr_path}")

    print(f"    Selected channels ({len(selected_channels)}): {selected_names}")

    # ── [3] Preprocess ─────────────────────────────────────────────
    print(f"\n[3] Preprocessing (RMS envelope -> sequences)...")
    print(f"    RMS window={rms_window_ms}ms, hop={rms_hop_ms}ms")

    # For Ghorbani: always use trial-based processing to avoid cross-trial leakage
    trial_ids = None
    if args.dataset == "ghorbani":
        has_trials = any("trial_boundaries" in s and len(s.get("trial_boundaries", [])) > 1
                         for s in subject_data)
        if has_trials:
            print(f"    Processing each trial independently (no cross-boundary sequences)")
            X, y, subjects, trial_ids = preprocess_subjects_by_trial(
                subject_data, selected_channels, fs,
                args.dataset, rms_window_ms, rms_hop_ms)
        else:
            X, y, subjects = preprocess_subjects(subject_data, selected_channels, fs,
                                                 args.dataset, rms_window_ms, rms_hop_ms)
    else:
        X, y, subjects = preprocess_subjects(subject_data, selected_channels, fs,
                                             args.dataset, rms_window_ms, rms_hop_ms)

    # ── Per-subject training mode ──────────────────────────────────
    if args.per_subject:
        metrics = train_per_subject_models(
            X, y, subjects, args, selected_channels, selected_names,
            rms_window_ms, rms_hop_ms, DEVICE, trial_ids=trial_ids,
        )
        return metrics

    # ── [4] Train/test split ─────────────────────────────────────
    if args.random_split:
        print(f"\n[4] Stratified random split (test={TEST_SPLIT*100:.0f}%)...")
        X, y, subjects = shuffle(X, y, subjects, random_state=RANDOM_SEED)
        train_idx_list, test_idx_list = [], []
        for sid in np.unique(subjects):
            sid_idx = np.where(subjects == sid)[0]
            n_test = max(1, int(len(sid_idx) * TEST_SPLIT))
            test_idx_list.extend(sid_idx[:n_test].tolist())
            train_idx_list.extend(sid_idx[n_test:].tolist())
        train_idx = np.array(train_idx_list)
        test_idx = np.array(test_idx_list)
        X_train, y_train, s_train = X[train_idx], y[train_idx], subjects[train_idx]
        X_test, y_test, s_test = X[test_idx], y[test_idx], subjects[test_idx]
        split_method = "random_stratified"
    elif trial_ids is not None:
        # Trial-based split: train on first N-1 trials, test on last trial per subject
        print(f"\n[4] Trial-based split (train on first trials, test on last trial)...")
        train_idx_list, test_idx_list = [], []
        for sid in np.unique(subjects):
            sid_mask = np.where(subjects == sid)[0]
            sid_trials = trial_ids[sid_mask]
            max_trial = sid_trials.max()
            for idx, t in zip(sid_mask, sid_trials):
                if t == max_trial:
                    test_idx_list.append(idx)
                else:
                    train_idx_list.append(idx)
        train_idx = np.array(train_idx_list, dtype=int)
        test_idx = np.array(test_idx_list, dtype=int)
        X_train, y_train, s_train = X[train_idx], y[train_idx], subjects[train_idx]
        X_test, y_test, s_test = X[test_idx], y[test_idx], subjects[test_idx]
        split_method = "trial_based"
    else:
        print(f"\n[4] Temporal block split (test={TEST_SPLIT*100:.0f}% per subject)...")
        X_train, y_train, s_train, X_test, y_test, s_test = temporal_block_split(
            X, y, subjects, test_ratio=TEST_SPLIT
        )
        split_method = "temporal_block"
    print(f"    Train: {len(y_train):,}, Test: {len(y_test):,}")

    # ── [5] Per-subject normalization (fit on train portion per subject) ──
    print(f"\n[5] Per-subject normalization (MinMaxScaler per subject, EMG + force)...")
    n_ch = X_train.shape[2]
    per_subj_emg_scalers = {}
    per_subj_force_scalers = {}

    for subj_id in np.unique(s_train):
        sid = int(subj_id)
        train_mask = (s_train == subj_id)

        # EMG: per-subject per-channel MinMaxScaler
        X_subj = X_train[train_mask]
        shape = X_subj.shape
        X_flat = X_subj.reshape(-1, n_ch)
        emg_scaler = MinMaxScaler(feature_range=(0, 1))
        X_flat_scaled = emg_scaler.fit_transform(X_flat).astype(np.float32)
        X_train[train_mask] = X_flat_scaled.reshape(shape)
        per_subj_emg_scalers[sid] = emg_scaler

        # Force: per-subject MinMaxScaler
        force_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train[train_mask] = force_scaler.fit_transform(
            y_train[train_mask].reshape(-1, 1)
        ).ravel().astype(np.float32)
        per_subj_force_scalers[sid] = force_scaler

    # Apply train-fitted scalers to test set
    for subj_id in np.unique(s_test):
        sid = int(subj_id)
        test_mask = (s_test == subj_id)

        if sid in per_subj_emg_scalers:
            X_subj = X_test[test_mask]
            shape = X_subj.shape
            X_flat = X_subj.reshape(-1, n_ch)
            X_flat_scaled = per_subj_emg_scalers[sid].transform(X_flat).astype(np.float32)
            X_test[test_mask] = X_flat_scaled.reshape(shape)

        if sid in per_subj_force_scalers:
            y_test[test_mask] = per_subj_force_scalers[sid].transform(
                y_test[test_mask].reshape(-1, 1)
            ).ravel().astype(np.float32)

    print(f"    Normalized {len(per_subj_force_scalers)} subjects (EMG + force per-subject)")
    for sid in sorted(per_subj_force_scalers.keys()):
        sc = per_subj_force_scalers[sid]
        print(f"      S{sid}: force range [{sc.data_min_[0]:.1f}, {sc.data_max_[0]:.1f}] N")

    # ── [6] Train/val split ──
    print(f"\n[6] Train/val split ({(1-VAL_SPLIT)*100:.0f}/{VAL_SPLIT*100:.0f})...")
    X_train, y_train, s_train = shuffle(X_train, y_train, s_train, random_state=RANDOM_SEED)

    val_idx = int(len(X_train) * (1 - VAL_SPLIT))
    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]
    X_train = X_train[:val_idx]
    y_train = y_train[:val_idx]
    s_train_final = s_train[:val_idx]

    print(f"    Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")

    # ── [7] Create DataLoaders ─────────────────────────────────────
    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    test_ds = SeqDataset(X_test, y_test)

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=_seed_worker, generator=g,
                              pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0,
                            pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0,
                             pin_memory=PIN_MEMORY)

    # ── [8] Train GRU ──────────────────────────────────────────────
    print(f"\n[7] Training GRU...")
    print(f"    Architecture: GRU({n_ch}->{args.hidden_size}, {args.num_layers}L, relu) -> Drop({args.dropout}) -> Dense({args.dense_size}, relu) -> Drop({args.dropout}) -> Dense({GRU_OUTPUT_SIZE})")

    model = GRUForcePredictor(
        input_size=n_ch,
        hidden_size=args.hidden_size,
        dense_size=args.dense_size,
        output_size=GRU_OUTPUT_SIZE,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(DEVICE)

    params = count_parameters(model)
    print(f"    Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    print(f"    Config: MSE loss, Adam (lr={args.lr}), {args.epochs} epochs, batch={args.batch_size}")
    print(f"    Early stopping: patience={args.patience}")
    print(f"    LR scheduler: ReduceLROnPlateau(factor=0.5, patience=7)")
    print(f"    Gradient clipping: max_norm=1.0")
    if args.weight_decay > 0:
        print(f"    Weight decay: {args.weight_decay}")
    if args.noise_std > 0:
        print(f"    Training noise: std={args.noise_std}")

    t0 = time.time()
    model, train_losses, val_losses = train_model(
        train_loader, val_loader, model, args.epochs, args.lr,
        args.patience, DEVICE, weight_decay=args.weight_decay,
        noise_std=args.noise_std,
    )
    train_time = time.time() - t0
    print(f"    Training time: {train_time:.1f}s")

    # ── [9] Evaluate ───────────────────────────────────────────────
    print(f"\n[8] Evaluating on test set...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(DEVICE)
            all_preds.append(model(Xb).cpu().numpy())
    y_pred = np.concatenate(all_preds).ravel()

    # Overall metrics
    metrics = compute_metrics(y_test, y_pred)
    print_results(metrics, f"{args.dataset.upper()} -- GRU ({n_ch}-channel MRMR)")

    # Per-subject metrics
    print("  Per-subject breakdown:")
    subj_metrics = per_subject_evaluation(y_test, y_pred, s_test)
    print_per_subject_results(subj_metrics)

    # ── [10] Save outputs ──────────────────────────────────────────
    print(f"\n[9] Saving outputs...")

    # Model
    model_path = MODELS_DIR / f"gru_{args.dataset}_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": f"GRU({n_ch}->{args.hidden_size},relu)->Drop(0.2)->Dense({args.dense_size},relu)->Drop(0.2)->Dense({GRU_OUTPUT_SIZE})",
        "selected_channels": selected_channels,
        "selected_channel_names": selected_names,
        "input_size": n_ch,
    }, model_path)
    print(f"    Model -> {model_path}")

    # Scalers (per-subject)
    joblib.dump(per_subj_emg_scalers, MODELS_DIR / f"emg_scalers_{args.dataset}.joblib")
    joblib.dump(per_subj_force_scalers, MODELS_DIR / f"force_scalers_{args.dataset}.joblib")
    print(f"    Scalers (per-subject) -> {MODELS_DIR}")

    # Results JSON
    results = {
        "dataset": args.dataset,
        "model": "GRU",
        "n_channels": n_ch,
        "selected_channels": selected_channels,
        "selected_channel_names": selected_names,
        "metrics": metrics,
        "per_subject": subj_metrics,
        "training": {
            "epochs_run": len(train_losses),
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_time_s": round(train_time, 1),
            "best_val_loss": float(min(val_losses)),
            "final_train_loss": float(train_losses[-1]),
            "total_params": params["total"],
            "split_method": split_method,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
        },
        "preprocessing": {
            "seq_len": SEQ_LEN,
            "envelope_window_ms": rms_window_ms,
            "envelope_hop_ms": rms_hop_ms,
            "bandpass_hz": [BANDPASS_LOW, BANDPASS_HIGH],
            "test_split": TEST_SPLIT,
            "val_split": VAL_SPLIT,
        },
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = RESULTS_DIR / f"{args.dataset}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Results -> {results_path}")

    # ── [11] Generate plots ────────────────────────────────────────
    print(f"\n[10] Generating plots...")
    try:
        from src.visualization.plots import (
            plot_training_curves, plot_prediction_vs_actual,
            plot_scatter, plot_per_subject_r2,
        )

        plot_training_curves(
            train_losses, val_losses, args.dataset,
            FIGURES_DIR / f"training_curves_{args.dataset}.png",
        )
        plot_prediction_vs_actual(
            y_test, y_pred, args.dataset,
            FIGURES_DIR / f"prediction_vs_actual_{args.dataset}.png",
        )
        plot_scatter(
            y_test, y_pred, metrics["R2"], args.dataset,
            FIGURES_DIR / f"scatter_{args.dataset}.png",
        )
        plot_per_subject_r2(
            subj_metrics, args.dataset,
            FIGURES_DIR / f"per_subject_r2_{args.dataset}.png",
        )
        print(f"    Plots -> {FIGURES_DIR}")
    except ImportError:
        warnings.warn("Visualization module not available, skipping plots.")

    print(f"\n{'=' * 60}")
    print(f"  DONE -- R2 = {metrics['R2']:.4f} ({args.dataset.upper()})")
    print(f"{'=' * 60}")

    return metrics


if __name__ == "__main__":
    main()
