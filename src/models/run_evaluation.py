"""
Comprehensive evaluation for EMG-to-Force prediction (Ghorbani dataset).

Produces review-ready results:
  1. Per-subject GRU evaluation with within-trial temporal split (80/20)
     + baseline comparisons (Ridge Regression, MLP)
  2. Cross-trial generalization (train first N-1 trials, test last trial)
  3. Leave-One-Subject-Out (LOSO) cross-validation
  4. Channel count ablation (1, 2, 4, 8 channels)
  5. Mean +/- std with 95% confidence intervals

Features:
  Six time-domain features per channel (Phinyomark et al., 2012):
  RMS, MAV, WL, VAR, ZC, SSC  =>  2 channels x 6 = 12 input features

Usage:
    python -m src.models.run_evaluation
    python -m src.models.run_evaluation --skip-loso          # skip slow LOSO
    python -m src.models.run_evaluation --skip-ablation      # skip channel ablation
    python -m src.models.run_evaluation --skip-cross-trial   # skip cross-trial eval
"""
import argparse
import copy
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE, EPOCHS, FIGURES_DIR, GRU_OUTPUT_SIZE,
    LEARNING_RATE, MODELS_DIR, MRMR_N_SELECT, RANDOM_SEED,
    RAW_GHORBANI_DIR, RESULTS_DIR, SEQ_LEN, TEST_SPLIT, VAL_SPLIT,
    GHORBANI_FS, GHORBANI_CHANNEL_NAMES, GHORBANI_SUBJECTS,
    GHORBANI_RMS_WINDOW_MS, GHORBANI_RMS_HOP_MS,
)
from src.data.preprocess import (
    compute_multi_features, compute_delta_features, create_sequences,
    MULTI_FEATURE_NAMES, N_FEATURES_PER_CHANNEL,
)
from src.models.evaluate import compute_metrics
from src.models.gru_model import (
    GRUForcePredictor, AttentionGRUPredictor, SeqDataset, count_parameters,
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ──────────────────────────────────────────────────────────────

def process_trial(emg, force, selected_channels, fs, window_ms, hop_ms,
                   delta_order=0):
    """Process one trial: channel select -> multi-feature extraction -> delta -> sequences."""
    emg_sel = emg[:, selected_channels]
    win_samp = int(window_ms * fs / 1000)
    hop_samp = int(hop_ms * fs / 1000)

    # Extract 6 time-domain features per channel
    features = compute_multi_features(emg_sel, window_ms, hop_ms, fs)

    # Optionally add temporal derivatives (delta, delta-delta)
    if delta_order > 0:
        features = compute_delta_features(features, order=delta_order)

    n_windows = len(features)

    # Down-sample force to match feature windows (mean over each window)
    force_env = np.zeros(n_windows, dtype=np.float32)
    for j in range(n_windows):
        start = j * hop_samp
        end = min(start + win_samp, len(force))
        force_env[j] = np.mean(force[start:end])

    X, y = create_sequences(features, force_env, seq_len=SEQ_LEN, pred_horizon=1)
    return X, y


def smooth_predictions(y_pred, window=9, cutoff_hz=4.0, hop_ms=20.0):
    """Lowpass filter prediction smoothing (standard in EMG-force literature).

    Uses zero-phase 2nd order Butterworth lowpass filter at cutoff_hz,
    which matches the expected bandwidth of voluntary grip force (~3-5 Hz).
    Falls back to moving average if signal is too short.
    """
    from scipy.signal import butter, filtfilt
    if len(y_pred) < 20:
        return y_pred
    fs_pred = 1000.0 / hop_ms  # prediction sampling rate (50 Hz for 20ms hop)
    nyq = fs_pred / 2.0
    if cutoff_hz >= nyq:
        cutoff_hz = nyq * 0.9
    b, a = butter(2, cutoff_hz / nyq, btype='low')
    # filtfilt needs at least 3*max(len(a),len(b)) samples
    min_len = 3 * max(len(a), len(b))
    if len(y_pred) < min_len:
        # Fall back to moving average for very short signals
        kernel = np.ones(window) / window
        smoothed = np.convolve(y_pred, kernel, mode='same')
        return smoothed.astype(y_pred.dtype)
    smoothed = filtfilt(b, a, y_pred).astype(y_pred.dtype)
    return smoothed


def select_best_channels(emg, force, n_select, fs, window_ms, hop_ms):
    """Select the best n_select channels per subject based on RMS-force correlation.

    Uses only training data (first 80%) to avoid data leakage.
    Returns indices of selected channels sorted by correlation.
    """
    n_channels = emg.shape[1]
    win_samp = int(window_ms * fs / 1000)
    hop_samp = int(hop_ms * fs / 1000)

    # Use first 80% for channel selection (same as train split)
    n_train = int(len(emg) * 0.8)
    emg_train = emg[:n_train]
    force_train = force[:n_train]

    correlations = np.zeros(n_channels)
    for ch in range(n_channels):
        # Compute RMS envelope for this channel
        n_windows = 1 + (len(emg_train) - win_samp) // hop_samp
        rms = np.zeros(n_windows)
        for j in range(n_windows):
            start = j * hop_samp
            end = start + win_samp
            rms[j] = np.sqrt(np.mean(emg_train[start:end, ch] ** 2))

        # Downsample force to match
        force_ds = np.zeros(n_windows)
        for j in range(n_windows):
            start = j * hop_samp
            end = min(start + win_samp, len(force_train))
            force_ds[j] = np.mean(force_train[start:end])

        # Absolute correlation
        if np.std(rms) > 0 and np.std(force_ds) > 0:
            correlations[ch] = abs(np.corrcoef(rms, force_ds)[0, 1])
        else:
            correlations[ch] = 0.0

    # Select top n_select channels
    selected = np.argsort(correlations)[::-1][:n_select].tolist()
    return selected, correlations


def build_trial_data(subject_data, selected_channels, fs, window_ms, hop_ms,
                     delta_order=0):
    """
    Process all subjects trial-by-trial.
    Returns dict: {subject_id: [(X_trial0, y_trial0), ...]}
    """
    data = {}
    for subj in subject_data:
        sid = subj["subject_id"]
        boundaries = subj.get("trial_boundaries", [(0, len(subj["emg"]))])
        trials = []
        for start, end in boundaries:
            emg_t = subj["emg"][start:end]
            force_t = subj["force"][start:end]
            X, y = process_trial(emg_t, force_t, selected_channels, fs,
                                 window_ms, hop_ms, delta_order=delta_order)
            if len(X) > 0:
                trials.append((X.astype(np.float32), y.astype(np.float32)))
        data[sid] = trials
    return data


def normalize_and_split_temporal(trials, test_split=TEST_SPLIT,
                                  val_split=VAL_SPLIT, seed=RANDOM_SEED):
    """
    Within-trial temporal split with gap to prevent leakage.
    Last test_split% of each trial = test. Gap of SEQ_LEN sequences discarded.
    Scalers fit on train only.
    """
    gap = 10  # Small gap to prevent boundary leakage
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for X_t, y_t in trials:
        n = len(X_t)
        split_pt = int(n * (1 - test_split))
        train_end = max(0, split_pt - gap)
        test_start = split_pt

        if train_end > 0:
            X_train_list.append(X_t[:train_end])
            y_train_list.append(y_t[:train_end])
        if test_start < n:
            X_test_list.append(X_t[test_start:])
            y_test_list.append(y_t[test_start:])

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    n_feat = X_train.shape[2]

    emg_scaler = MinMaxScaler(feature_range=(0, 1))
    X_flat = emg_scaler.fit_transform(X_train.reshape(-1, n_feat)).astype(np.float32)
    X_train = X_flat.reshape(X_train.shape)

    force_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train = force_scaler.fit_transform(
        y_train.reshape(-1, 1)).ravel().astype(np.float32)

    X_test_flat = emg_scaler.transform(
        X_test.reshape(-1, n_feat)).astype(np.float32)
    X_test = X_test_flat.reshape(X_test.shape)
    y_test = force_scaler.transform(
        y_test.reshape(-1, 1)).ravel().astype(np.float32)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]
    val_n = int(len(X_train) * val_split)
    X_val = X_train[-val_n:]
    y_val = y_train[-val_n:]
    X_train = X_train[:-val_n]
    y_train = y_train[:-val_n]

    return X_train, y_train, X_val, y_val, X_test, y_test, emg_scaler, force_scaler


def normalize_and_split_cross_trial(trials, val_split=VAL_SPLIT, seed=RANDOM_SEED):
    """Cross-trial split: train first N-1 trials, test last trial."""
    n_trials = len(trials)
    train_trials = trials[:n_trials - 1]
    test_trial = trials[n_trials - 1]

    X_train = np.concatenate([t[0] for t in train_trials])
    y_train = np.concatenate([t[1] for t in train_trials])
    X_test, y_test = test_trial

    n_feat = X_train.shape[2]

    emg_scaler = MinMaxScaler(feature_range=(0, 1))
    X_flat = emg_scaler.fit_transform(X_train.reshape(-1, n_feat)).astype(np.float32)
    X_train = X_flat.reshape(X_train.shape)

    force_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train = force_scaler.fit_transform(
        y_train.reshape(-1, 1)).ravel().astype(np.float32)

    X_test_flat = emg_scaler.transform(
        X_test.reshape(-1, n_feat)).astype(np.float32)
    X_test = X_test_flat.reshape(X_test.shape)
    y_test = force_scaler.transform(
        y_test.reshape(-1, 1)).ravel().astype(np.float32)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]
    val_n = int(len(X_train) * val_split)
    X_val = X_train[-val_n:]
    y_val = y_train[-val_n:]
    X_train = X_train[:-val_n]
    y_train = y_train[:-val_n]

    return X_train, y_train, X_val, y_val, X_test, y_test, emg_scaler, force_scaler


def train_gru_model(X_train, y_train, X_val, y_val, n_input,
                    hidden_size, dense_size, num_layers, dropout,
                    epochs, lr, patience, batch_size, weight_decay=1e-4,
                    seed=RANDOM_SEED, bidirectional=False,
                    use_attention=False):
    """Train a GRU model and return it."""
    torch.manual_seed(seed)
    if use_attention:
        model = AttentionGRUPredictor(
            input_size=n_input, hidden_size=hidden_size,
            dense_size=dense_size, output_size=GRU_OUTPUT_SIZE,
            dropout=dropout, num_layers=num_layers,
        ).to(DEVICE)
    else:
        model = GRUForcePredictor(
            input_size=n_input, hidden_size=hidden_size,
            dense_size=dense_size, output_size=GRU_OUTPUT_SIZE,
            dropout=dropout, num_layers=num_layers,
            bidirectional=bidirectional,
        ).to(DEVICE)

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(3, patience // 3), min_lr=1e-6)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            # Data augmentation: add small Gaussian noise to inputs
            noise = torch.randn_like(Xb) * 0.02
            Xb_aug = Xb + noise
            optimizer.zero_grad()
            loss = criterion(model(Xb_aug), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item()
                n += 1
        val_loss /= max(n, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_gru(model, X_test):
    """Run GRU inference, return predictions as numpy array."""
    model.eval()
    ds = SeqDataset(X_test, np.zeros(len(X_test), dtype=np.float32))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            preds.append(model(Xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds).ravel()


def fine_tune_model(model, X_train, y_train, X_val, y_val,
                    ft_epochs=15, ft_lr=0.0002, batch_size=64):
    """Fine-tune a pre-trained model on subject-specific data.

    Uses lower learning rate and fewer epochs to avoid catastrophic
    forgetting of the general knowledge.
    """
    model = copy.deepcopy(model)
    model.train()
    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(ft_epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(Xb), yb).item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    model.load_state_dict(best_state)
    return model


def compute_ci(values, confidence=0.95):
    """Compute mean, std, and 95% CI via t-distribution."""
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        ci_low = ci_high = mean
    return {"mean": mean, "std": std, "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high), "n": n}


def _print_comparison_table(title, model_results):
    """Print a formatted comparison table."""
    print(f"\n{'='*78}")
    print(title)
    print(f"{'='*78}")
    print(f"{'Model':<20} {'Mean R2':>10} {'Std':>8} {'95% CI':>22} {'Min':>8} {'Max':>8}")
    print("-" * 78)
    for name, ci_data, r2_vals in model_results:
        print(f"{name:<20} {ci_data['mean']:>10.4f} {ci_data['std']:>8.4f} "
              f"[{ci_data['ci_95_low']:>8.4f}, {ci_data['ci_95_high']:>8.4f}] "
              f"{min(r2_vals):>8.4f} {max(r2_vals):>8.4f}")


# ── Main Evaluation Functions ────────────────────────────────────────────

def run_per_subject_evaluation(subject_data, selected_channels, fs,
                                window_ms, hop_ms, hidden_size, dense_size,
                                num_layers, dropout, epochs, lr, patience,
                                batch_size, weight_decay, n_ensemble=1,
                                bidirectional=False, use_attention=False,
                                delta_order=0):
    """
    Per-subject GRU + baselines with within-trial temporal split.
    Each trial split 80/20 temporally with a small gap.
    When n_ensemble > 1, trains multiple GRU models with different seeds
    and averages their predictions (ensemble averaging).
    """
    trial_data = build_trial_data(subject_data, selected_channels,
                                  fs, window_ms, hop_ms,
                                  delta_order=delta_order)
    # Derive actual input dimension from data (n_channels * n_features)
    first_trials = next(iter(trial_data.values()))
    n_input = first_trials[0][0].shape[2]

    gru_results, ridge_results, mlp_results = [], [], []

    for sid in sorted(trial_data.keys()):
        trials = trial_data[sid]
        if len(trials) < 1:
            print(f"    S{sid}: no valid trials, skipping")
            continue

        X_tr, y_tr, X_val, y_val, X_te, y_te, _, _ = \
            normalize_and_split_temporal(trials, seed=RANDOM_SEED + sid)
        print(f"    S{sid}: train={len(y_tr):,}, val={len(y_val):,}, "
              f"test={len(y_te):,}", end="")

        # GRU ensemble
        ensemble_preds = []
        for k in range(n_ensemble):
            seed_k = RANDOM_SEED + sid * 100 + k * 7
            model = train_gru_model(
                X_tr, y_tr, X_val, y_val, n_input,
                hidden_size, dense_size, num_layers, dropout,
                epochs, lr, patience, batch_size, weight_decay,
                seed=seed_k, bidirectional=bidirectional,
                use_attention=use_attention)
            ensemble_preds.append(predict_gru(model, X_te))
        y_pred_gru = smooth_predictions(np.mean(ensemble_preds, axis=0), window=9)
        m_gru = compute_metrics(y_te, y_pred_gru)
        m_gru["subject_id"] = sid
        gru_results.append(m_gru)

        # Ridge
        X_tr_flat = X_tr.reshape(len(X_tr), -1)
        X_te_flat = X_te.reshape(len(X_te), -1)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_flat, y_tr)
        m_ridge = compute_metrics(y_te, smooth_predictions(
            ridge.predict(X_te_flat), window=9))
        m_ridge["subject_id"] = sid
        ridge_results.append(m_ridge)

        # MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_SEED, learning_rate_init=0.001,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_tr_flat, y_tr)
        m_mlp = compute_metrics(y_te, smooth_predictions(
            mlp.predict(X_te_flat), window=9))
        m_mlp["subject_id"] = sid
        mlp_results.append(m_mlp)

        print(f"  GRU={m_gru['R2']:.3f}  Ridge={m_ridge['R2']:.3f}  "
              f"MLP={m_mlp['R2']:.3f}")

    return gru_results, ridge_results, mlp_results


def run_general_model_evaluation(subject_data, selected_channels, fs,
                                  window_ms, hop_ms, hidden_size, dense_size,
                                  num_layers, dropout, epochs, lr, patience,
                                  batch_size, weight_decay, n_ensemble=1,
                                  use_attention=False, delta_order=0):
    """
    Subject-independent (general) model evaluation.

    1. For each subject, split trials temporally (80/20) and normalize
       per-subject (scaler fit on train only).
    2. Pool all subjects' normalized training data into one dataset.
    3. Train a single model (or ensemble) on the pooled data.
    4. Evaluate on each subject's held-out test data.

    This tests whether one model can generalize across all subjects,
    which is the practical deployment scenario (no per-subject retraining).
    """
    trial_data = build_trial_data(subject_data, selected_channels,
                                  fs, window_ms, hop_ms,
                                  delta_order=delta_order)
    first_trials = next(iter(trial_data.values()))
    n_input = first_trials[0][0].shape[2]

    # ── Step 1: per-subject temporal split + normalization ─────────
    per_subject = {}
    for sid in sorted(trial_data.keys()):
        trials = trial_data[sid]
        if len(trials) < 1:
            continue
        X_tr, y_tr, X_val, y_val, X_te, y_te, _, _ = \
            normalize_and_split_temporal(trials, seed=RANDOM_SEED + sid)
        per_subject[sid] = dict(X_tr=X_tr, y_tr=y_tr,
                                X_val=X_val, y_val=y_val,
                                X_te=X_te, y_te=y_te)

    # ── Step 2: pool training / validation across subjects ────────
    X_train_pool = np.concatenate([d['X_tr'] for d in per_subject.values()])
    y_train_pool = np.concatenate([d['y_tr'] for d in per_subject.values()])
    X_val_pool = np.concatenate([d['X_val'] for d in per_subject.values()])
    y_val_pool = np.concatenate([d['y_val'] for d in per_subject.values()])

    print(f"    Pooled data: train={len(y_train_pool):,}, "
          f"val={len(y_val_pool):,}")

    # ── Step 3: train general GRU ensemble ────────────────────────
    ensemble_models = []
    for k in range(n_ensemble):
        seed_k = RANDOM_SEED + k * 7
        model = train_gru_model(
            X_train_pool, y_train_pool, X_val_pool, y_val_pool, n_input,
            hidden_size, dense_size, num_layers, dropout,
            epochs, lr, patience, batch_size, weight_decay,
            seed=seed_k, use_attention=use_attention)
        ensemble_models.append(model)
        print(f"      Ensemble member {k+1}/{n_ensemble} trained")

    # ── Step 4: train baselines on pooled data ────────────────────
    X_tr_flat = X_train_pool.reshape(len(X_train_pool), -1)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_flat, y_train_pool)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128), max_iter=500,
        early_stopping=True, validation_fraction=0.15,
        random_state=RANDOM_SEED, learning_rate_init=0.001,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(X_tr_flat, y_train_pool)

    # ── Step 5: evaluate per subject (fine-tune + ensemble) ─────
    gru_results, gru_ft_results = [], []
    ridge_results, mlp_results = [], []

    for sid in sorted(per_subject.keys()):
        d = per_subject[sid]
        X_tr, y_tr = d['X_tr'], d['y_tr']
        X_val, y_val = d['X_val'], d['y_val']
        X_te, y_te = d['X_te'], d['y_te']

        # GRU general ensemble (no fine-tuning)
        preds = [predict_gru(m, X_te) for m in ensemble_models]
        y_pred_gru = smooth_predictions(np.mean(preds, axis=0), window=9)
        m_gru = compute_metrics(y_te, y_pred_gru)
        m_gru["subject_id"] = sid
        gru_results.append(m_gru)

        # GRU fine-tuned ensemble (pre-train + fine-tune per subject)
        # Adaptive: use validation MSE to select general vs fine-tuned per model
        ft_preds = []
        val_preds_general = [predict_gru(m, X_val) for m in ensemble_models]
        for k, m in enumerate(ensemble_models):
            ft_model = fine_tune_model(m, X_tr, y_tr, X_val, y_val)
            ft_pred_val = predict_gru(ft_model, X_val)
            gen_mse = float(np.mean((val_preds_general[k] - y_val) ** 2))
            ft_mse = float(np.mean((ft_pred_val - y_val) ** 2))
            # Use fine-tuned only if it improves validation loss
            if ft_mse < gen_mse:
                ft_preds.append(predict_gru(ft_model, X_te))
            else:
                ft_preds.append(preds[k])
        y_pred_ft = smooth_predictions(np.mean(ft_preds, axis=0), window=9)
        m_ft = compute_metrics(y_te, y_pred_ft)
        m_ft["subject_id"] = sid
        gru_ft_results.append(m_ft)

        # Ridge
        X_te_flat = X_te.reshape(len(X_te), -1)
        X_tr_flat_subj = X_tr.reshape(len(X_tr), -1)
        ridge_subj = Ridge(alpha=1.0)
        ridge_subj.fit(X_tr_flat_subj, y_tr)
        m_ridge = compute_metrics(y_te, smooth_predictions(
            ridge_subj.predict(X_te_flat), window=9))
        m_ridge["subject_id"] = sid
        ridge_results.append(m_ridge)

        # MLP per-subject
        mlp_subj = MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_SEED, learning_rate_init=0.001,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp_subj.fit(X_tr_flat_subj, y_tr)
        m_mlp = compute_metrics(y_te, smooth_predictions(
            mlp_subj.predict(X_te_flat), window=9))
        m_mlp["subject_id"] = sid
        mlp_results.append(m_mlp)

        print(f"    S{sid}: test={len(y_te):,}  "
              f"GRU={m_gru['R2']:.3f}  GRU+FT={m_ft['R2']:.3f}  "
              f"Ridge={m_ridge['R2']:.3f}  MLP={m_mlp['R2']:.3f}")

    return gru_results, gru_ft_results, ridge_results, mlp_results


def run_cross_trial_evaluation(subject_data, selected_channels, fs,
                                window_ms, hop_ms, hidden_size, dense_size,
                                num_layers, dropout, epochs, lr, patience,
                                batch_size, weight_decay, use_attention=False,
                                delta_order=0):
    """Cross-trial generalization: train first N-1 trials, test last."""
    trial_data = build_trial_data(subject_data, selected_channels,
                                  fs, window_ms, hop_ms,
                                  delta_order=delta_order)
    first_trials = next(iter(trial_data.values()))
    n_input = first_trials[0][0].shape[2]
    results = []

    for sid in sorted(trial_data.keys()):
        trials = trial_data[sid]
        if len(trials) < 2:
            print(f"    S{sid}: only {len(trials)} trial(s), skipping")
            continue

        X_tr, y_tr, X_val, y_val, X_te, y_te, _, _ = \
            normalize_and_split_cross_trial(trials, seed=RANDOM_SEED + sid)
        print(f"    S{sid}: train={len(y_tr):,}, val={len(y_val):,}, "
              f"test={len(y_te):,}", end="")

        torch.manual_seed(RANDOM_SEED + sid)
        model = train_gru_model(
            X_tr, y_tr, X_val, y_val, n_input,
            hidden_size, dense_size, num_layers, dropout,
            epochs, lr, patience, batch_size, weight_decay,
            use_attention=use_attention)
        y_pred = smooth_predictions(predict_gru(model, X_te), window=9)
        m = compute_metrics(y_te, y_pred)
        m["subject_id"] = sid
        results.append(m)
        print(f"  R2={m['R2']:.3f}, Corr={m['correlation']:.3f}")

    return results


def run_loso(subject_data, selected_channels, fs, window_ms, hop_ms,
             hidden_size, dense_size, num_layers, dropout, epochs, lr,
             patience, batch_size, weight_decay, use_attention=False,
             delta_order=0):
    """Leave-One-Subject-Out cross-validation."""
    trial_data = build_trial_data(subject_data, selected_channels,
                                  fs, window_ms, hop_ms,
                                  delta_order=delta_order)
    first_trials = next(iter(trial_data.values()))
    n_input = first_trials[0][0].shape[2]
    sids = sorted(trial_data.keys())
    results = []

    for test_sid in sids:
        X_train_all, y_train_all = [], []
        for sid in sids:
            if sid == test_sid:
                continue
            for X_t, y_t in trial_data[sid]:
                X_train_all.append(X_t)
                y_train_all.append(y_t)
        X_train = np.concatenate(X_train_all)
        y_train = np.concatenate(y_train_all)

        X_test_all, y_test_all = [], []
        for X_t, y_t in trial_data[test_sid]:
            X_test_all.append(X_t)
            y_test_all.append(y_t)
        X_test = np.concatenate(X_test_all)
        y_test = np.concatenate(y_test_all)

        emg_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = emg_scaler.fit_transform(
            X_train.reshape(-1, n_input)).astype(np.float32).reshape(X_train.shape)
        X_test = emg_scaler.transform(
            X_test.reshape(-1, n_input)).astype(np.float32).reshape(X_test.shape)

        force_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = force_scaler.fit_transform(
            y_train.reshape(-1, 1)).ravel().astype(np.float32)
        y_test = force_scaler.transform(
            y_test.reshape(-1, 1)).ravel().astype(np.float32)

        rng = np.random.RandomState(RANDOM_SEED + test_sid)
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        val_n = int(len(X_train) * VAL_SPLIT)
        X_val = X_train[-val_n:]
        y_val = y_train[-val_n:]
        X_train = X_train[:-val_n]
        y_train = y_train[:-val_n]

        torch.manual_seed(RANDOM_SEED + test_sid)
        model = train_gru_model(
            X_train, y_train, X_val, y_val, n_input,
            hidden_size, dense_size, num_layers, dropout,
            epochs, lr, patience, batch_size, weight_decay,
            use_attention=use_attention)
        y_pred = smooth_predictions(predict_gru(model, X_test), window=9)
        m = compute_metrics(y_test, y_pred)
        m["subject_id"] = test_sid
        results.append(m)
        print(f"    LOSO hold-out S{test_sid}: R2={m['R2']:.4f}, "
              f"Corr={m['correlation']:.4f}")

    return results


def run_channel_ablation(subject_data, fs, window_ms, hop_ms,
                          hidden_size, dense_size, num_layers, dropout,
                          epochs, lr, patience, batch_size, weight_decay,
                          mrmr_path, use_attention=False, delta_order=0):
    """Ablation over channel counts: 1, 2, 4, 8."""
    n_total = subject_data[0]["emg"].shape[1]
    channel_counts = [c for c in [1, 2, 4, 8] if c <= n_total]
    results = {}

    try:
        if mrmr_path.exists():
            with open(mrmr_path) as f:
                mrmr_data = json.load(f)
            all_scores = np.array(mrmr_data["all_channel_mrmr_scores"])
            ranked = np.argsort(all_scores)[::-1].tolist()
        else:
            ranked = list(range(n_total))
    except (json.JSONDecodeError, KeyError):
        ranked = list(range(n_total))

    for n_ch_phys in channel_counts:
        selected = ranked[:n_ch_phys]
        ch_names = [GHORBANI_CHANNEL_NAMES[i] for i in selected]
        print(f"\n    --- {n_ch_phys} channel(s): {ch_names} ---")

        trial_data = build_trial_data(subject_data, selected,
                                      fs, window_ms, hop_ms,
                                      delta_order=delta_order)
        first_trials = next(iter(trial_data.values()))
        n_input = first_trials[0][0].shape[2]
        r2_values = []

        for sid in sorted(trial_data.keys()):
            trials = trial_data[sid]
            if len(trials) < 1:
                continue
            X_tr, y_tr, X_val, y_val, X_te, y_te, _, _ = \
                normalize_and_split_temporal(trials, seed=RANDOM_SEED + sid)
            torch.manual_seed(RANDOM_SEED + sid)
            model = train_gru_model(
                X_tr, y_tr, X_val, y_val, n_input,
                hidden_size, dense_size, num_layers, dropout,
                epochs, lr, patience, batch_size, weight_decay,
                use_attention=use_attention)
            y_pred = smooth_predictions(predict_gru(model, X_te), window=9)
            m = compute_metrics(y_te, y_pred)
            r2_values.append(m["R2"])
            print(f"      S{sid}: R2={m['R2']:.4f}")

        ci = compute_ci(r2_values)
        results[n_ch_phys] = {
            "channels": selected,
            "channel_names": ch_names,
            "n_input_features": n_input,
            "per_subject_r2": r2_values,
            "stats": ci,
        }
        print(f"    {n_ch_phys}ch ({n_input} features): R2 = "
              f"{ci['mean']:.4f} +/- {ci['std']:.4f} "
              f"(95% CI: [{ci['ci_95_low']:.4f}, {ci['ci_95_high']:.4f}])")

    return results


# ── Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ghorbani EMG-to-Force evaluation")
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-cross-trial", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--n-ensemble", type=int, default=5,
                        help="Number of GRU models to ensemble (default: 5)")
    parser.add_argument("--hop-ms", type=float, default=20,
                        help="Feature hop in ms (default: 20)")
    parser.add_argument("--exclude-subjects", type=int, nargs="+", default=[],
                        help="Subject IDs to exclude (e.g., --exclude-subjects 8)")
    args = parser.parse_args()

    # Architecture -- general model
    hidden_size = 64
    dense_size = 64
    num_layers = 1
    dropout = 0.2
    use_attention = False
    delta_order = 2  # base + delta + delta-delta (3x features)
    n_phys_channels = MRMR_N_SELECT  # 2 physical electrodes
    n_base_features = n_phys_channels * N_FEATURES_PER_CHANNEL  # 2 * 6 = 12
    n_input = n_base_features * (1 + delta_order)  # 12 * 3 = 36

    print("=" * 70)
    print("COMPREHENSIVE EVALUATION -- Ghorbani EMG-to-Force (Review-Ready)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Approach: GENERAL (subject-independent) model")
    print(f"Features: {N_FEATURES_PER_CHANNEL} per channel "
          f"({', '.join(MULTI_FEATURE_NAMES)})")
    print(f"Delta order: {delta_order} "
          f"(base + {'delta + delta-delta' if delta_order == 2 else 'delta'})")
    print(f"Input:  {n_phys_channels} channels x {N_FEATURES_PER_CHANNEL} "
          f"features x {1 + delta_order} (delta) = {n_input} features")
    model_type = "AttentionGRU" if use_attention else "GRU"
    print(f"Model:  {model_type}({n_input}->{hidden_size}, {num_layers}L) -> "
          f"Dense({dense_size}) -> Dense(1)")
    if use_attention:
        params = count_parameters(AttentionGRUPredictor(
            input_size=n_input, hidden_size=hidden_size,
            dense_size=dense_size, num_layers=num_layers))
    else:
        params = count_parameters(GRUForcePredictor(
            input_size=n_input, hidden_size=hidden_size,
            dense_size=dense_size, num_layers=num_layers))
    print(f"Params: {params['total']:,}")
    print(f"Config: epochs={args.epochs}, patience={args.patience}, "
          f"lr={args.lr}, batch={args.batch_size}, "
          f"dropout={dropout}, wd={args.weight_decay}, "
          f"ensemble={args.n_ensemble}")

    for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n[1] Loading Ghorbani dataset...")
    from src.data.load_ghorbani import load_all_ghorbani
    subject_data = load_all_ghorbani(str(RAW_GHORBANI_DIR),
                                      n_subjects=GHORBANI_SUBJECTS)
    # Exclude subjects if requested
    if args.exclude_subjects:
        subject_data = [s for s in subject_data
                        if s["subject_id"] not in args.exclude_subjects]
        print(f"    Excluded subjects: {args.exclude_subjects}")
    print(f"    Loaded {len(subject_data)} subjects")

    fs = GHORBANI_FS
    window_ms = GHORBANI_RMS_WINDOW_MS
    hop_ms = args.hop_ms

    # ── MRMR channels ─────────────────────────────────────────────
    mrmr_path = MODELS_DIR / "mrmr_results_ghorbani.json"
    try:
        if mrmr_path.exists():
            with open(mrmr_path) as f:
                mrmr_data = json.load(f)
            selected_channels = mrmr_data["selected_indices"][:n_phys_channels]
            selected_names = mrmr_data["selected_names"][:n_phys_channels]
        else:
            raise FileNotFoundError("MRMR results file not found")
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"WARNING: Could not load MRMR results ({e}), using first "
              f"{n_phys_channels} channels")
        selected_channels = list(range(n_phys_channels))
        selected_names = GHORBANI_CHANNEL_NAMES[:n_phys_channels]
    print(f"    MRMR channels: {selected_names}")

    # ══════════════════════════════════════════════════════════════
    # [2] Primary: general (subject-independent) model
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[2] General model evaluation (pooled training, per-subject test)")
    print(f"    Split: last {TEST_SPLIT*100:.0f}% of each trial = test, "
          f"gap=10 sequences")
    print(f"    Training: pooled normalized data from ALL subjects")
    print(f"    Features: {MULTI_FEATURE_NAMES} + delta(order={delta_order})")
    print(f"    Model: {model_type}, Ensemble: {args.n_ensemble} models")
    print(f"    Data augmentation: Gaussian noise (std=0.02)")
    print(f"    Post-processing: Butterworth lowpass filter (4Hz, 2nd order, zero-phase)")
    print(f"{'='*70}")
    t0 = time.time()
    gru_results, gru_ft_results, ridge_results, mlp_results = \
        run_general_model_evaluation(
            subject_data, selected_channels, fs, window_ms, hop_ms,
            hidden_size, dense_size, num_layers, dropout,
            args.epochs, args.lr, args.patience, args.batch_size,
            args.weight_decay, n_ensemble=args.n_ensemble,
            use_attention=use_attention, delta_order=delta_order,
        )
    eval_time = time.time() - t0

    gru_r2 = [m["R2"] for m in gru_results]
    gru_ft_r2 = [m["R2"] for m in gru_ft_results]
    ridge_r2 = [m["R2"] for m in ridge_results]
    mlp_r2 = [m["R2"] for m in mlp_results]

    gru_ci = compute_ci(gru_r2)
    gru_ft_ci = compute_ci(gru_ft_r2)
    ridge_ci = compute_ci(ridge_r2)
    mlp_ci = compute_ci(mlp_r2)

    _print_comparison_table(
        "COMPARISON TABLE (per-subject R2, general model)",
        [("GRU (general)", gru_ci, gru_r2),
         ("GRU+FT (ours)", gru_ft_ci, gru_ft_r2),
         ("Ridge Regression", ridge_ci, ridge_r2),
         ("MLP (128-64)", mlp_ci, mlp_r2)],
    )
    print(f"\n    Evaluation time: {eval_time:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # [3] Cross-trial generalization
    # ══════════════════════════════════════════════════════════════
    cross_trial_results = None
    cross_trial_ci = None
    if not args.skip_cross_trial:
        print(f"\n{'='*70}")
        print("[3] Cross-trial generalization (train first N-1, test last)")
        print(f"{'='*70}")
        t0 = time.time()
        cross_trial_results = run_cross_trial_evaluation(
            subject_data, selected_channels, fs, window_ms, hop_ms,
            hidden_size, dense_size, num_layers, dropout,
            args.epochs, args.lr, args.patience, args.batch_size,
            args.weight_decay, use_attention=use_attention,
            delta_order=delta_order,
        )
        ct_time = time.time() - t0
        ct_r2 = [m["R2"] for m in cross_trial_results]
        cross_trial_ci = compute_ci(ct_r2)
        print(f"\n    Cross-trial R2: {cross_trial_ci['mean']:.4f} "
              f"+/- {cross_trial_ci['std']:.4f}")
        print(f"    Time: {ct_time:.1f}s")
    else:
        print(f"\n[3] Cross-trial: SKIPPED")

    # ══════════════════════════════════════════════════════════════
    # [4] LOSO cross-validation
    # ══════════════════════════════════════════════════════════════
    loso_results = None
    loso_ci = None
    if not args.skip_loso:
        print(f"\n{'='*70}")
        print("[4] Leave-One-Subject-Out (LOSO) cross-validation")
        print(f"{'='*70}")
        t0 = time.time()
        loso_results = run_loso(
            subject_data, selected_channels, fs, window_ms, hop_ms,
            hidden_size, dense_size, num_layers, dropout,
            args.epochs, args.lr, args.patience, args.batch_size,
            args.weight_decay, use_attention=use_attention,
            delta_order=delta_order,
        )
        loso_time = time.time() - t0
        loso_r2 = [m["R2"] for m in loso_results]
        loso_ci = compute_ci(loso_r2)
        print(f"\n    LOSO R2: {loso_ci['mean']:.4f} +/- {loso_ci['std']:.4f}")
        print(f"    Time: {loso_time:.1f}s")
    else:
        print(f"\n[4] LOSO: SKIPPED")

    # ══════════════════════════════════════════════════════════════
    # [5] Channel ablation
    # ══════════════════════════════════════════════════════════════
    ablation_results = None
    if not args.skip_ablation:
        print(f"\n{'='*70}")
        print("[5] Channel count ablation (1, 2, 4, 8 channels)")
        print(f"{'='*70}")
        t0 = time.time()
        ablation_results = run_channel_ablation(
            subject_data, fs, window_ms, hop_ms,
            hidden_size, dense_size, num_layers, dropout,
            args.epochs, args.lr, args.patience, args.batch_size,
            args.weight_decay, mrmr_path,
            use_attention=use_attention, delta_order=delta_order,
        )
        abl_time = time.time() - t0

        print(f"\n    {'Channels':<12} {'Features':>8} {'Mean R2':>10} "
              f"{'Std':>8} {'95% CI':>22}")
        print("    " + "-" * 62)
        for n_ch in sorted(ablation_results.keys()):
            s = ablation_results[n_ch]["stats"]
            nf = ablation_results[n_ch]["n_input_features"]
            print(f"    {n_ch:<12} {nf:>8} {s['mean']:>10.4f} {s['std']:>8.4f} "
                  f"[{s['ci_95_low']:>8.4f}, {s['ci_95_high']:>8.4f}]")
        print(f"    Time: {abl_time:.1f}s")
    else:
        print(f"\n[5] Channel ablation: SKIPPED")

    # ══════════════════════════════════════════════════════════════
    # [6] Save results
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[6] Saving results")
    print(f"{'='*70}")

    feature_names_full = []
    for ch_name in selected_names:
        for feat in MULTI_FEATURE_NAMES:
            feature_names_full.append(f"{ch_name}_{feat}")
    if delta_order >= 1:
        for ch_name in selected_names:
            for feat in MULTI_FEATURE_NAMES:
                feature_names_full.append(f"{ch_name}_d{feat}")
    if delta_order >= 2:
        for ch_name in selected_names:
            for feat in MULTI_FEATURE_NAMES:
                feature_names_full.append(f"{ch_name}_dd{feat}")

    report = {
        "dataset": "ghorbani",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "architecture": (
                f"{model_type}({n_input}->{hidden_size},{num_layers}L,relu)->"
                f"Drop({dropout})->Dense({dense_size},relu)->"
                f"Drop({dropout})->Dense(1)"
            ),
            "params": params["total"],
            "hidden_size": hidden_size,
            "dense_size": dense_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "weight_decay": args.weight_decay,
            "use_attention": use_attention,
        },
        "channels": {
            "n_physical_channels": n_phys_channels,
            "selected_indices": selected_channels,
            "selected_names": selected_names,
            "selection_method": "MRMR",
        },
        "features": {
            "n_features_per_channel": N_FEATURES_PER_CHANNEL,
            "feature_names": MULTI_FEATURE_NAMES,
            "delta_order": delta_order,
            "total_input_features": n_input,
            "full_feature_names": feature_names_full,
        },
        "primary_evaluation": {
            "approach": "general (subject-independent) model + fine-tuning",
            "split_method": (
                f"within-trial temporal (last {TEST_SPLIT*100:.0f}% "
                f"of each trial, gap=10 sequences)"
            ),
            "training": "pooled normalized data from all subjects",
            "fine_tuning": "per-subject with lower lr (0.0002), 15 epochs",
            "normalization": "per-subject MinMaxScaler (fit on train only)",
            "val_split": VAL_SPLIT,
            "test_split": TEST_SPLIT,
            "gru_general": {"per_subject": gru_results, "stats": gru_ci},
            "gru_finetuned": {"per_subject": gru_ft_results, "stats": gru_ft_ci},
            "ridge_regression": {"per_subject": ridge_results, "stats": ridge_ci},
            "mlp": {"per_subject": mlp_results, "stats": mlp_ci},
        },
        "preprocessing": {
            "seq_len": SEQ_LEN,
            "feature_window_ms": window_ms,
            "feature_hop_ms": hop_ms,
            "sampling_rate_hz": fs,
            "prediction_smoothing": "Butterworth lowpass 4Hz, order=2, zero-phase",
        },
        "training_config": {
            "epochs_max": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "Adam",
            "loss": "MSE",
            "lr_scheduler": "ReduceLROnPlateau(factor=0.5)",
            "grad_clip": 1.0,
            "n_ensemble": args.n_ensemble,
        },
    }

    if cross_trial_results is not None:
        report["cross_trial_generalization"] = {
            "split_method": "train first N-1 trials, test last trial",
            "per_subject": cross_trial_results,
            "stats": cross_trial_ci,
        }

    if loso_results is not None:
        report["loso_cross_validation"] = {
            "per_subject": loso_results,
            "stats": loso_ci,
        }

    if ablation_results is not None:
        abl_json = {}
        for k, v in ablation_results.items():
            abl_json[str(k)] = {
                "channels": v["channels"],
                "channel_names": v["channel_names"],
                "n_input_features": v["n_input_features"],
                "per_subject_r2": [float(x) for x in v["per_subject_r2"]],
                "stats": v["stats"],
            }
        report["channel_ablation"] = abl_json

    results_path = RESULTS_DIR / "ghorbani_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Results -> {results_path}")

    # ══════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  PRIMARY (general model, pooled training):")
    print(f"    GRU (general):    R2 = {gru_ci['mean']:.4f} "
          f"+/- {gru_ci['std']:.4f}")
    print(f"    GRU+FT (ours):    R2 = {gru_ft_ci['mean']:.4f} "
          f"+/- {gru_ft_ci['std']:.4f}")
    print(f"    Ridge baseline:   R2 = {ridge_ci['mean']:.4f} "
          f"+/- {ridge_ci['std']:.4f}")
    print(f"    MLP baseline:     R2 = {mlp_ci['mean']:.4f} "
          f"+/- {mlp_ci['std']:.4f}")
    if cross_trial_ci is not None:
        print(f"  CROSS-TRIAL generalization:")
        print(f"    GRU:              R2 = {cross_trial_ci['mean']:.4f} "
              f"+/- {cross_trial_ci['std']:.4f}")
    if loso_ci is not None:
        print(f"  LOSO cross-validation:")
        print(f"    GRU:              R2 = {loso_ci['mean']:.4f} "
              f"+/- {loso_ci['std']:.4f}")
    if ablation_results is not None:
        print(f"  CHANNEL ABLATION:")
        for n_ch in sorted(ablation_results.keys()):
            s = ablation_results[n_ch]["stats"]
            nf = ablation_results[n_ch]["n_input_features"]
            print(f"    {n_ch} ch ({nf} feat):   R2 = {s['mean']:.4f} "
                  f"+/- {s['std']:.4f}")
    print(f"  Model size: {params['total']:,} parameters")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
