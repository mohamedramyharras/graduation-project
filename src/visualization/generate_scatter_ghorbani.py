"""
generate_scatter_ghorbani.py
============================
Generate the Ghorbani scatter plot using GRU General model (subject-independent).
No per-subject fine-tuning - the model works for all users without calibration.

Saves: outputs/figures/scatter_ghorbani_gru.png
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from scipy.signal import butter, sosfiltfilt

from src.data.load_ghorbani import load_all_ghorbani
from src.data.preprocess import compute_multi_features, compute_delta_features
from src.models.gru_model import GRUForcePredictor, SeqDataset
from src.config import (
    RAW_GHORBANI_DIR, GHORBANI_SUBJECTS,
    GHORBANI_RMS_WINDOW_MS, GHORBANI_RMS_HOP_MS, GHORBANI_FS,
)

FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
SAVE_PATH     = FIGURES_DIR / "scatter_ghorbani_gru.png"

EXCLUDE_SUBJECTS = [8]
SELECTED_CH      = [5, 3]       # ECRB, ECU  (MRMR top-2)
SEQ_LEN          = 50
N_FEAT           = 36
TRAIN_EPOCHS     = 50
PATIENCE         = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- helpers ----------------------------------------------------------------

def build_sequences(entry):
    emg   = entry["emg"][:, SELECTED_CH]
    force = entry["force"]
    fs    = float(entry["fs"])

    feats = compute_multi_features(
        emg,
        window_ms=GHORBANI_RMS_WINDOW_MS,
        hop_ms=GHORBANI_RMS_HOP_MS,
        fs=fs,
    )
    feats = compute_delta_features(feats, order=2)

    win_s = max(1, int(round(GHORBANI_RMS_WINDOW_MS * fs / 1000.0)))
    hop_s = max(1, int(round(GHORBANI_RMS_HOP_MS * fs / 1000.0)))
    n_win = feats.shape[0]
    centres = np.clip(
        np.array([i * hop_s + win_s // 2 for i in range(n_win)]),
        0, len(force) - 1,
    )
    force_ds = force[centres]

    n_seq = n_win - SEQ_LEN
    if n_seq <= 0:
        return None, None
    X = np.stack([feats[i : i + SEQ_LEN] for i in range(n_seq)])
    y = force_ds[SEQ_LEN:]
    return X, y


def smooth(signal, fs_feat, cutoff=4.0, order=2):
    nyq = fs_feat / 2
    if cutoff < nyq:
        sos = butter(order, cutoff / nyq, btype="low", output="sos")
        return sosfiltfilt(sos, signal)
    return signal


# -- load & preprocess ------------------------------------------------------

print("Loading Ghorbani dataset ...")
raw_all = load_all_ghorbani(str(RAW_GHORBANI_DIR), n_subjects=GHORBANI_SUBJECTS)
raw_all = [d for d in raw_all if d["subject_id"] not in EXCLUDE_SUBJECTS]
print(f"  {len(raw_all)} subjects after exclusion")

TEST_PCT = 0.20
VAL_FRAC = 0.15
GAP      = 10

subject_data = {}
for entry in raw_all:
    sid    = entry["subject_id"]
    X, y   = build_sequences(entry)
    if X is None:
        continue

    n      = len(y)
    n_test = int(n * TEST_PCT)
    n_tv   = n - n_test - GAP
    n_val  = int(n_tv * VAL_FRAC)
    n_tr   = n_tv - n_val

    X_tr, y_tr = X[:n_tr],                     y[:n_tr]
    X_va, y_va = X[n_tr : n_tr + n_val],       y[n_tr : n_tr + n_val]
    X_te, y_te = X[n_tr + n_val + GAP :],      y[n_tr + n_val + GAP :]

    # Per-subject feature scaling (fit on train only)
    sc_X = MinMaxScaler()
    X_tr_s = sc_X.fit_transform(X_tr.reshape(-1, N_FEAT)).reshape(X_tr.shape)
    X_va_s = sc_X.transform(X_va.reshape(-1, N_FEAT)).reshape(X_va.shape)
    X_te_s = sc_X.transform(X_te.reshape(-1, N_FEAT)).reshape(X_te.shape)

    # Per-subject force scaling (fit on train only)
    sc_y = MinMaxScaler()
    y_tr_s = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_va_s = sc_y.transform(y_va.reshape(-1, 1)).ravel()
    y_te_s = sc_y.transform(y_te.reshape(-1, 1)).ravel()

    subject_data[sid] = dict(
        X_tr=X_tr_s, y_tr=y_tr_s,
        X_va=X_va_s, y_va=y_va_s,
        X_te=X_te_s, y_te_norm=y_te_s,
    )

# -- pool training data ------------------------------------------------------

rng = np.random.default_rng(42)
X_pool = np.concatenate([v["X_tr"] for v in subject_data.values()])
y_pool = np.concatenate([v["y_tr"] for v in subject_data.values()])
idx    = rng.permutation(len(y_pool))
X_pool, y_pool = X_pool[idx], y_pool[idx]

n_val_pool = int(len(y_pool) * VAL_FRAC)
X_tr_p, y_tr_p = X_pool[n_val_pool:], y_pool[n_val_pool:]
X_va_p, y_va_p = X_pool[:n_val_pool], y_pool[:n_val_pool]

# -- train general GRU (subject-independent, no fine-tuning) -----------------

print(f"Training general GRU on {len(y_tr_p):,} sequences (device={DEVICE}) ...")

model = GRUForcePredictor(
    input_size=N_FEAT, hidden_size=64, dense_size=64,
    output_size=1, dropout=0.2, num_layers=1,
).to(DEVICE)

opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
crit  = nn.MSELoss()

tr_loader = DataLoader(SeqDataset(X_tr_p, y_tr_p), batch_size=512, shuffle=True)
va_loader = DataLoader(SeqDataset(X_va_p, y_va_p), batch_size=1024, shuffle=False)

best_val, patience_cnt, best_state = 1e9, 0, None
for epoch in range(1, TRAIN_EPOCHS + 1):
    model.train()
    for Xb, yb in tr_loader:
        opt.zero_grad()
        loss = crit(model(Xb.to(DEVICE)), yb.to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for Xb, yb in va_loader:
            val_losses.append(crit(model(Xb.to(DEVICE)), yb.to(DEVICE)).item())
    val_loss = float(np.mean(val_losses))
    sched.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        patience_cnt = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"  epoch {epoch:3d}  val_loss={val_loss:.6f}")

model.load_state_dict(best_state)

# -- inference on per-subject test sets (general model only) -----------------

print("Evaluating general model on per-subject test sets ...")
fs_feat = GHORBANI_FS / max(1, int(round(GHORBANI_RMS_HOP_MS * GHORBANI_FS / 1000.0)))

all_actual, all_pred = [], []

for sid, arrays in sorted(subject_data.items()):
    model.eval()
    te_loader = DataLoader(
        SeqDataset(arrays["X_te"], np.zeros(len(arrays["X_te"]))),
        batch_size=1024, shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for Xb, _ in te_loader:
            preds.append(model(Xb.to(DEVICE)).cpu().numpy().ravel())
    pred_norm = np.concatenate(preds)

    # smooth (in normalized space)
    pred_norm = smooth(pred_norm, fs_feat)

    y_te_norm = arrays["y_te_norm"]
    r2_s = r2_score(y_te_norm, pred_norm)
    print(f"  S{sid}: R\u00b2={r2_s:.4f}")

    all_actual.append(y_te_norm)
    all_pred.append(pred_norm)

# -- aggregate scatter -------------------------------------------------------

y_true_all = np.concatenate(all_actual)
y_pred_all = np.concatenate(all_pred)
r2_total   = float(r2_score(y_true_all, y_pred_all))
print(f"\nCombined test R\u00b2 = {r2_total:.4f}")

# -- plot --------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_PLOT = 5000
rng_plot = np.random.default_rng(0)
idx = rng_plot.choice(len(y_true_all), min(N_PLOT, len(y_true_all)), replace=False)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_true_all[idx], y_pred_all[idx],
           alpha=0.2, s=5, color="#4C72B0")

mn = min(y_true_all.min(), y_pred_all.min())
mx = max(y_true_all.max(), y_pred_all.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Identity")

ax.set_xlabel("Actual Force (normalized, a.u.)", fontsize=12)
ax.set_ylabel("Predicted Force (normalized, a.u.)", fontsize=12)
ax.set_title(f"Scatter \u2014 GHORBANI GRU General (R\u00b2 = {r2_total:.4f})", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
fig.savefig(str(SAVE_PATH), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SAVE_PATH}")
