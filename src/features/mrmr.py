"""
MRMR (Minimum Redundancy Maximum Relevance) Channel Selection
==============================================================

Implements the greedy forward-selection algorithm from:
    Peng, H., Long, F., & Ding, C. (2005).
    "Feature selection based on mutual information: criteria of
     max-dependency, max-relevance, and min-redundancy."
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8).

The algorithm selects the best EMG channels for force prediction by
balancing two objectives:
    - Maximum relevance: each selected channel should have high mutual
      information (MI) with the target force signal.
    - Minimum redundancy: selected channels should have low MI with
      each other, avoiding duplicate information.

At each greedy step the channel maximising
    MRMR(c) = Relevance(c) - (1/|S|) * sum_{s in S} Redundancy(c, s)
is added to the selected set S.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from ..config import (
    MRMR_N_SELECT,
    MRMR_SUBSAMPLE,
    RANDOM_SEED,
    NINAPRO_CHANNEL_NAMES,
    NINAPRO_FS,
    ENVELOPE_WINDOW_MS,
)


# ---------------------------------------------------------------------------
# 1. Channel-to-target relevance
# ---------------------------------------------------------------------------

def compute_channel_relevance(X_channels, y_target, random_state=RANDOM_SEED):
    """Compute the mutual information between each channel and the target.

    Parameters
    ----------
    X_channels : np.ndarray, shape (n_samples, n_channels)
        RMS envelope values for each EMG channel.
    y_target : np.ndarray, shape (n_samples,)
        Total grip force (continuous target).
    random_state : int, optional
        Random seed for the MI estimator (default from config).

    Returns
    -------
    relevance : np.ndarray, shape (n_channels,)
        MI(channel_i, force) for every channel.
    """
    relevance = mutual_info_regression(
        X_channels, y_target, random_state=random_state
    )
    return relevance


# ---------------------------------------------------------------------------
# 2. Pairwise channel redundancy matrix
# ---------------------------------------------------------------------------

def compute_redundancy_matrix(X_channels, random_state=RANDOM_SEED):
    """Compute a symmetric pairwise MI matrix between all channels.

    Parameters
    ----------
    X_channels : np.ndarray, shape (n_samples, n_channels)
        RMS envelope values for each EMG channel.
    random_state : int, optional
        Random seed for the MI estimator (default from config).

    Returns
    -------
    redundancy : np.ndarray, shape (n_channels, n_channels)
        Symmetric matrix where entry (i, j) = MI(channel_i, channel_j).
        Diagonal entries are MI(channel_i, channel_i).
    """
    n_channels = X_channels.shape[1]
    redundancy = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i, n_channels):
            mi = mutual_info_regression(
                X_channels[:, j].reshape(-1, 1),
                X_channels[:, i],
                random_state=random_state,
            )[0]
            redundancy[i, j] = mi
            redundancy[j, i] = mi

    return redundancy


# ---------------------------------------------------------------------------
# 3. Greedy MRMR forward selection
# ---------------------------------------------------------------------------

def mrmr_select(X_channels, y_target, n_select=MRMR_N_SELECT,
                random_state=RANDOM_SEED, channel_names=None):
    """Select channels via greedy MRMR forward selection.

    The first channel is the one with the highest relevance.  Each
    subsequent channel maximises:

        MRMR(c) = relevance(c) - mean redundancy with already-selected set

    Parameters
    ----------
    X_channels : np.ndarray, shape (n_samples, n_channels)
        RMS envelope values for each EMG channel.
    y_target : np.ndarray, shape (n_samples,)
        Total grip force (continuous target).
    n_select : int, optional
        Number of channels to select (default ``MRMR_N_SELECT``).
    random_state : int, optional
        Random seed for MI estimation (default from config).
    channel_names : list of str or None, optional
        Human-readable channel names.  If *None*, names are generated
        as ``"ch_0", "ch_1", ...``.

    Returns
    -------
    result : dict
        'selected_indices'       : list[int]   -- indices of selected channels
        'selected_names'         : list[str]   -- names  of selected channels
        'relevance'              : np.ndarray  -- MI(ch_i, force) for all ch
        'redundancy_matrix'      : np.ndarray  -- pairwise MI matrix
        'mrmr_scores'            : list[dict]  -- per-step selection log
        'all_channel_mrmr_scores': np.ndarray  -- final MRMR score per channel
    """
    n_channels = X_channels.shape[1]

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    if n_select > n_channels:
        raise ValueError(
            f"n_select ({n_select}) exceeds the number of available "
            f"channels ({n_channels})."
        )

    # --- Compute relevance & redundancy -----------------------------------
    relevance = compute_channel_relevance(X_channels, y_target, random_state)
    redundancy_matrix = compute_redundancy_matrix(X_channels, random_state)

    # --- Greedy forward selection -----------------------------------------
    selected = []
    remaining = list(range(n_channels))
    mrmr_scores_log = []

    for step in range(n_select):
        best_score = -np.inf
        best_idx = None

        candidate_scores = {}

        for c in remaining:
            if len(selected) == 0:
                # First channel: pure relevance
                score = relevance[c]
            else:
                # MRMR criterion
                mean_red = np.mean([redundancy_matrix[c, s] for s in selected])
                score = relevance[c] - mean_red

            candidate_scores[c] = score

            if score > best_score:
                best_score = score
                best_idx = c

        selected.append(best_idx)
        remaining.remove(best_idx)

        step_info = {
            "step": step + 1,
            "selected_index": best_idx,
            "selected_name": channel_names[best_idx],
            "mrmr_score": best_score,
            "relevance": float(relevance[best_idx]),
            "candidate_scores": {
                channel_names[k]: float(v)
                for k, v in candidate_scores.items()
            },
        }
        mrmr_scores_log.append(step_info)

    # --- Final MRMR score for every channel (relative to full selected set) -
    all_channel_mrmr = np.zeros(n_channels)
    for c in range(n_channels):
        if len(selected) == 0:
            all_channel_mrmr[c] = relevance[c]
        else:
            others = [s for s in selected if s != c]
            if len(others) == 0:
                all_channel_mrmr[c] = relevance[c]
            else:
                mean_red = np.mean([redundancy_matrix[c, s] for s in others])
                all_channel_mrmr[c] = relevance[c] - mean_red

    return {
        "selected_indices": selected,
        "selected_names": [channel_names[i] for i in selected],
        "relevance": relevance,
        "redundancy_matrix": redundancy_matrix,
        "mrmr_scores": mrmr_scores_log,
        "all_channel_mrmr_scores": all_channel_mrmr,
    }


# ---------------------------------------------------------------------------
# 4. End-to-end MRMR analysis across subjects
# ---------------------------------------------------------------------------

def _compute_rms_envelope(emg, fs, window_ms):
    """Compute a non-overlapping RMS envelope for a multi-channel EMG signal.

    Parameters
    ----------
    emg : np.ndarray, shape (n_samples, n_channels)
        Raw (or bandpass-filtered) EMG signal.
    fs : int
        Sampling frequency in Hz.
    window_ms : int
        RMS window length in milliseconds.

    Returns
    -------
    rms : np.ndarray, shape (n_windows, n_channels)
        RMS envelope with one value per non-overlapping window.
    """
    win_len = int(fs * window_ms / 1000)
    n_samples, n_channels = emg.shape

    # Trim to an integer number of complete windows
    n_windows = n_samples // win_len
    trimmed = emg[: n_windows * win_len, :]

    # Reshape into (n_windows, win_len, n_channels) and compute RMS
    blocks = trimmed.reshape(n_windows, win_len, n_channels)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    return rms


def run_mrmr_analysis(subject_data_list, channel_names=None,
                      n_select=MRMR_N_SELECT, train_ratio=0.80):
    """Run MRMR channel selection across multiple subjects.

    IMPORTANT: Only uses the first ``train_ratio`` fraction of each
    subject's data to prevent data leakage from the test set.

    Parameters
    ----------
    subject_data_list : list of dict
        Each dict must contain:
            'emg'   : np.ndarray, shape (n_samples, n_channels) -- raw EMG
            'force' : np.ndarray, shape (n_samples,) or (n_samples, n_force)
                      -- grip force.  If 2-D, columns are summed to obtain
                         total grip force.
    channel_names : list of str or None, optional
        Human-readable names for each EMG channel.  Defaults to
        ``NINAPRO_CHANNEL_NAMES`` from the project config.
    n_select : int, optional
        Number of channels to select (default ``MRMR_N_SELECT``).
    train_ratio : float, optional
        Fraction of each subject's data to use (default 0.80).
        Prevents leakage by excluding the test portion.

    Returns
    -------
    result : dict
        Output of :func:`mrmr_select` run on the concatenated and
        (optionally) sub-sampled data.
    """
    if channel_names is None:
        channel_names = NINAPRO_CHANNEL_NAMES

    # --- Compute RMS envelopes and concatenate across subjects -------------
    all_envelopes = []
    all_forces = []

    for idx, subj in enumerate(subject_data_list):
        emg = subj["emg"]
        force = subj["force"]
        fs = subj.get("fs", NINAPRO_FS)

        # USE ONLY TRAINING PORTION to prevent data leakage
        n_train = int(len(emg) * train_ratio)
        emg = emg[:n_train]
        force_subj = force[:n_train]

        # Compute RMS envelope for EMG channels
        rms_env = _compute_rms_envelope(emg, fs, ENVELOPE_WINDOW_MS)

        # Ensure force is 1-D total grip force
        if force_subj.ndim == 2:
            force_subj = np.sum(force_subj, axis=1)

        # Downsample force to match envelope length (mean per window,
        # consistent with training pipeline in train.py / run_pipeline.py)
        win_len = int(fs * ENVELOPE_WINDOW_MS / 1000)
        n_windows = len(rms_env)
        force_ds = np.zeros(n_windows, dtype=np.float32)
        for j in range(n_windows):
            start = j * win_len
            end = min(start + win_len, len(force_subj))
            force_ds[j] = np.mean(force_subj[start:end])

        all_envelopes.append(rms_env)
        all_forces.append(force_ds)

    X_concat = np.concatenate(all_envelopes, axis=0)
    y_concat = np.concatenate(all_forces, axis=0)

    print(f"[MRMR] Concatenated data: {X_concat.shape[0]} samples, "
          f"{X_concat.shape[1]} channels")

    # --- Subsample for computational efficiency ----------------------------
    rng = np.random.RandomState(RANDOM_SEED)
    n_total = X_concat.shape[0]

    if n_total > MRMR_SUBSAMPLE:
        idx = rng.choice(n_total, size=MRMR_SUBSAMPLE, replace=False)
        idx.sort()
        X_sub = X_concat[idx]
        y_sub = y_concat[idx]
        print(f"[MRMR] Sub-sampled to {MRMR_SUBSAMPLE} points for MI "
              f"computation")
    else:
        X_sub = X_concat
        y_sub = y_concat

    # --- Run MRMR selection ------------------------------------------------
    print(f"[MRMR] Selecting {n_select} channels via MRMR ...")
    result = mrmr_select(
        X_sub, y_sub,
        n_select=n_select,
        random_state=RANDOM_SEED,
        channel_names=channel_names,
    )

    # --- Print summary -----------------------------------------------------
    print("\n" + "=" * 60)
    print("MRMR Channel Selection Results")
    print("=" * 60)

    print(f"\nRelevance MI(channel, force) for all channels:")
    for i, name in enumerate(channel_names):
        marker = " <-- SELECTED" if i in result["selected_indices"] else ""
        print(f"  {name:>25s}:  {result['relevance'][i]:.4f}{marker}")

    print(f"\nSelection order:")
    for step in result["mrmr_scores"]:
        print(f"  Step {step['step']}: {step['selected_name']} "
              f"(MRMR = {step['mrmr_score']:.4f}, "
              f"relevance = {step['relevance']:.4f})")

    print(f"\nSelected channels: {result['selected_names']}")
    print(f"Selected indices : {result['selected_indices']}")
    print("=" * 60 + "\n")

    return result
