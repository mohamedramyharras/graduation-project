"""
preprocess.py
=============
Signal preprocessing utilities for EMG-to-force prediction.

Provides:
    - bandpass_filter        : Butterworth bandpass (SOS form)
    - compute_rms_envelope   : Sliding-window RMS envelope
    - create_sequences       : Build (X, y) supervised-learning pairs
    - temporal_block_split   : Per-subject temporal train/test split
    - preprocess_dataset     : End-to-end pipeline orchestrator
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from ..config import (
    BANDPASS_LOW,
    BANDPASS_HIGH,
    BANDPASS_ORDER,
    RMS_WINDOW_MS,
    RMS_HOP_MS,
    SEQ_LEN,
    PRED_HORIZON,
    TEST_RATIO,
)


# ---------------------------------------------------------------------------
# 1. Bandpass filter
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter using second-order sections.

    Parameters
    ----------
    signal : np.ndarray
        Input array.  Shape ``(n_samples,)`` for single-channel or
        ``(n_samples, n_channels)`` for multi-channel.
    fs : float
        Sampling frequency in Hz.
    low : float
        Low cut-off frequency in Hz.
    high : float
        High cut-off frequency in Hz.
    order : int
        Filter order (applied via cascaded second-order sections).

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as *signal*.

    Raises
    ------
    ValueError
        If *signal* is empty, *low >= high*, or Nyquist constraints are
        violated.
    """
    if signal.size == 0:
        raise ValueError("Input signal is empty.")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high}).")

    nyquist = fs / 2.0
    if high >= nyquist:
        raise ValueError(
            f"High cut-off ({high} Hz) must be below Nyquist ({nyquist} Hz)."
        )

    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")

    # Handle both 1-D and 2-D inputs
    if signal.ndim == 1:
        return sosfilt(sos, signal).astype(signal.dtype)
    elif signal.ndim == 2:
        # Filter each channel independently (columns)
        out = np.empty_like(signal)
        for ch in range(signal.shape[1]):
            out[:, ch] = sosfilt(sos, signal[:, ch])
        return out
    else:
        raise ValueError(
            f"Expected 1-D or 2-D array, got {signal.ndim}-D."
        )


# ---------------------------------------------------------------------------
# 2. RMS envelope
# ---------------------------------------------------------------------------

def compute_rms_envelope(
    emg: np.ndarray,
    window_ms: float = RMS_WINDOW_MS,
    hop_ms: float = RMS_HOP_MS,
    fs: float = 2000.0,
) -> np.ndarray:
    """Compute a sliding-window RMS envelope of the EMG signal.

    Parameters
    ----------
    emg : np.ndarray
        EMG data of shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    window_ms : float
        Window length in milliseconds.
    hop_ms : float
        Hop (stride) length in milliseconds.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        RMS envelope with shape ``(n_windows,)`` or
        ``(n_windows, n_channels)``.

    Raises
    ------
    ValueError
        If the signal is shorter than one window.
    """
    if emg.ndim == 1:
        emg = emg[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    n_samples, n_channels = emg.shape
    win_len = max(1, int(round(window_ms * fs / 1000.0)))
    hop_len = max(1, int(round(hop_ms * fs / 1000.0)))

    if n_samples < win_len:
        raise ValueError(
            f"Signal length ({n_samples}) is shorter than window "
            f"length ({win_len} samples / {window_ms} ms)."
        )

    n_windows = 1 + (n_samples - win_len) // hop_len
    envelope = np.empty((n_windows, n_channels), dtype=np.float64)

    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        segment = emg[start:end, :]
        envelope[i, :] = np.sqrt(np.mean(segment ** 2, axis=0))

    if squeeze:
        envelope = envelope.squeeze(axis=1)

    return envelope


# ---------------------------------------------------------------------------
# 2b. Multi-feature extraction
# ---------------------------------------------------------------------------

def compute_multi_features(
    emg: np.ndarray,
    window_ms: float = RMS_WINDOW_MS,
    hop_ms: float = RMS_HOP_MS,
    fs: float = 2000.0,
    zc_threshold: float = 0.0,
    ssc_threshold: float = 0.0,
) -> np.ndarray:
    """Extract multiple time-domain features from EMG using a sliding window.

    Six standard features are computed per channel per window
    (Phinyomark et al., 2012):

        1. **RMS**  -- Root Mean Square (signal power)
        2. **MAV**  -- Mean Absolute Value (signal amplitude)
        3. **WL**   -- Waveform Length (signal complexity)
        4. **VAR**  -- Variance (signal power)
        5. **ZC**   -- Zero Crossings (frequency-related)
        6. **SSC**  -- Slope Sign Changes (frequency-related)

    Parameters
    ----------
    emg : np.ndarray
        EMG data, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    window_ms : float
        Window length in milliseconds.
    hop_ms : float
        Hop (stride) length in milliseconds.
    fs : float
        Sampling frequency in Hz.
    zc_threshold : float
        Amplitude threshold for zero-crossing detection (noise suppression).
    ssc_threshold : float
        Amplitude threshold for SSC detection (noise suppression).

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(n_windows, n_channels * 6)``.
        Features are interleaved per channel:
        ``[ch0_rms, ch0_mav, ch0_wl, ch0_var, ch0_zc, ch0_ssc,
          ch1_rms, ch1_mav, ...]``
    """
    N_FEAT = 6

    if emg.ndim == 1:
        emg = emg[:, np.newaxis]

    n_samples, n_channels = emg.shape
    win_len = max(1, int(round(window_ms * fs / 1000.0)))
    hop_len = max(1, int(round(hop_ms * fs / 1000.0)))

    if n_samples < win_len:
        raise ValueError(
            f"Signal length ({n_samples}) < window ({win_len} samples)."
        )

    n_windows = 1 + (n_samples - win_len) // hop_len
    features = np.empty((n_windows, n_channels * N_FEAT), dtype=np.float64)

    for i in range(n_windows):
        start = i * hop_len
        segment = emg[start : start + win_len, :]  # (win_len, n_channels)

        for ch in range(n_channels):
            x = segment[:, ch]
            b = ch * N_FEAT

            # RMS
            features[i, b] = np.sqrt(np.mean(x * x))
            # MAV
            abs_x = np.abs(x)
            features[i, b + 1] = np.mean(abs_x)
            # WL (Waveform Length)
            dx = np.diff(x)
            features[i, b + 2] = np.sum(np.abs(dx))
            # VAR
            features[i, b + 3] = np.var(x)
            # ZC (Zero Crossings)
            sign_x = np.sign(x)
            sign_changes = np.abs(np.diff(sign_x))
            if zc_threshold > 0:
                abs_diffs = np.abs(dx)
                features[i, b + 4] = np.sum(
                    (sign_changes > 0) & (abs_diffs >= zc_threshold)
                )
            else:
                features[i, b + 4] = np.sum(sign_changes > 0)
            # SSC (Slope Sign Changes)
            if len(dx) > 1:
                sign_dx = np.sign(dx)
                ssc_changes = np.abs(np.diff(sign_dx))
                if ssc_threshold > 0:
                    magnitudes = np.abs(dx[:-1]) + np.abs(dx[1:])
                    features[i, b + 5] = np.sum(
                        (ssc_changes > 0) & (magnitudes >= ssc_threshold)
                    )
                else:
                    features[i, b + 5] = np.sum(ssc_changes > 0)
            else:
                features[i, b + 5] = 0

    return features


MULTI_FEATURE_NAMES = ["RMS", "MAV", "WL", "VAR", "ZC", "SSC"]
N_FEATURES_PER_CHANNEL = len(MULTI_FEATURE_NAMES)


def compute_delta_features(features: np.ndarray, order: int = 2) -> np.ndarray:
    """Compute temporal derivatives (delta, delta-delta) of feature matrix.

    Standard technique in signal processing (analogous to MFCC deltas in
    speech).  Delta features explicitly encode the rate of change, which
    the model would otherwise have to learn implicitly from the sequence.

    Parameters
    ----------
    features : np.ndarray, shape (n_windows, n_features)
        Base feature matrix from ``compute_multi_features``.
    order : int
        Number of derivative orders:
        - 0: base features only (no-op)
        - 1: base + delta  (2x features)
        - 2: base + delta + delta-delta  (3x features)

    Returns
    -------
    np.ndarray, shape (n_windows, n_features * (1 + order))
        Concatenated [base | delta | delta-delta] along feature axis.
    """
    if order == 0:
        return features
    result = [features]
    current = features
    for _ in range(order):
        delta = np.zeros_like(current)
        delta[1:] = np.diff(current, axis=0)
        result.append(delta)
        current = delta
    return np.concatenate(result, axis=1)


# ---------------------------------------------------------------------------
# 3. Sequence creation
# ---------------------------------------------------------------------------

def create_sequences(
    envelope: np.ndarray,
    force_envelope: np.ndarray,
    seq_len: int = SEQ_LEN,
    pred_horizon: int = PRED_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised-learning input/output pairs.

    For each valid index *t*:
        X[t] = envelope[t : t + seq_len, :]
        y[t] = force_envelope[t + seq_len + pred_horizon - 1]

    Parameters
    ----------
    envelope : np.ndarray
        EMG RMS envelope, shape ``(T, n_channels)``.
    force_envelope : np.ndarray
        Force envelope (scalar per time step), shape ``(T,)``.
    seq_len : int
        Number of time steps in each input window.
    pred_horizon : int
        How many steps ahead to predict (1 = next step).

    Returns
    -------
    X : np.ndarray, shape ``(n_seq, seq_len, n_channels)``
    y : np.ndarray, shape ``(n_seq,)``

    Raises
    ------
    ValueError
        If envelopes have mismatched lengths or are too short.
    """
    if envelope.ndim == 1:
        envelope = envelope[:, np.newaxis]

    T, n_channels = envelope.shape

    if force_envelope.shape[0] != T:
        raise ValueError(
            f"Length mismatch: envelope has {T} steps, "
            f"force_envelope has {force_envelope.shape[0]}."
        )

    total_needed = seq_len + pred_horizon
    if T < total_needed:
        raise ValueError(
            f"Not enough time steps ({T}) for seq_len={seq_len} "
            f"+ pred_horizon={pred_horizon} = {total_needed}."
        )

    n_seq = T - seq_len - pred_horizon + 1
    X = np.empty((n_seq, seq_len, n_channels), dtype=envelope.dtype)
    y = np.empty(n_seq, dtype=force_envelope.dtype)

    for i in range(n_seq):
        X[i] = envelope[i : i + seq_len, :]
        y[i] = force_envelope[i + seq_len + pred_horizon - 1]

    return X, y


# ---------------------------------------------------------------------------
# 4. Temporal block split
# ---------------------------------------------------------------------------

def temporal_block_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    test_ratio: float = TEST_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-subject temporal train/test split (no shuffling).

    For every unique subject the first ``(1 - test_ratio)`` fraction of
    that subject's contiguous samples goes to the training set and the
    remainder to the test set.  Temporal ordering within each subject is
    preserved.

    Parameters
    ----------
    X : np.ndarray
        Feature array, shape ``(N, ...)``.
    y : np.ndarray
        Target array, shape ``(N,)``.
    subjects : np.ndarray
        Subject identifier per sample, shape ``(N,)``.
    test_ratio : float
        Fraction of each subject's data reserved for testing.

    Returns
    -------
    X_train, y_train, s_train, X_test, y_test, s_test : np.ndarray
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}.")

    train_idx: list[int] = []
    test_idx: list[int] = []

    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        # idx is already sorted because np.where preserves order
        split_point = int(len(idx) * (1.0 - test_ratio))
        split_point = max(1, min(split_point, len(idx) - 1))  # at least 1 in each
        train_idx.extend(idx[:split_point].tolist())
        test_idx.extend(idx[split_point:].tolist())

    train_idx = np.array(train_idx, dtype=np.intp)
    test_idx = np.array(test_idx, dtype=np.intp)

    return (X[train_idx], y[train_idx], subjects[train_idx],
            X[test_idx], y[test_idx], subjects[test_idx])


# ---------------------------------------------------------------------------
# 5. Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset(
    subject_data_list: list[dict],
    selected_channels: list[int] | None = None,
    config: dict | None = None,
    skip_bandpass: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end preprocessing pipeline.

    Parameters
    ----------
    subject_data_list : list[dict]
        Each dict must contain:
            - ``"emg"``     : np.ndarray (n_samples, n_channels)
            - ``"force"``   : np.ndarray (n_samples,)
            - ``"subject"`` : hashable subject identifier
            - ``"fs"``      : float, sampling rate in Hz
    selected_channels : list[int] or None
        Channel indices to keep.  ``None`` keeps all.
    config : dict or None
        Override default config values.  Recognised keys:
        ``bandpass_low``, ``bandpass_high``, ``bandpass_order``,
        ``rms_window_ms``, ``rms_hop_ms``, ``seq_len``, ``pred_horizon``.
    skip_bandpass : bool
        If True, skip bandpass filtering (for pre-filtered datasets like
        Ghorbani where data is already bandpass filtered at acquisition).

    Returns
    -------
    X : np.ndarray   -- shape ``(N, seq_len, n_channels)``
    y : np.ndarray   -- shape ``(N,)``
    subjects : np.ndarray -- shape ``(N,)``  subject label per sequence
    """
    cfg = {
        "bandpass_low": BANDPASS_LOW,
        "bandpass_high": BANDPASS_HIGH,
        "bandpass_order": BANDPASS_ORDER,
        "rms_window_ms": RMS_WINDOW_MS,
        "rms_hop_ms": RMS_HOP_MS,
        "seq_len": SEQ_LEN,
        "pred_horizon": PRED_HORIZON,
    }
    if config is not None:
        cfg.update(config)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_subj: list[np.ndarray] = []

    for entry in subject_data_list:
        emg = entry["emg"]
        force = entry["force"]
        subj_id = entry["subject"]
        fs = float(entry["fs"])

        # --- channel selection ---
        if selected_channels is not None:
            emg = emg[:, selected_channels]

        # --- bandpass filter (skip for pre-filtered datasets) ---
        if not skip_bandpass:
            emg = bandpass_filter(
                emg,
                fs=fs,
                low=cfg["bandpass_low"],
                high=cfg["bandpass_high"],
                order=cfg["bandpass_order"],
            )

        # --- RMS envelope (EMG) ---
        emg_env = compute_rms_envelope(
            emg,
            window_ms=cfg["rms_window_ms"],
            hop_ms=cfg["rms_hop_ms"],
            fs=fs,
        )

        # --- Downsample force to match EMG envelope length ---
        #     Use the same hop to pick force values at envelope centres.
        win_len = max(1, int(round(cfg["rms_window_ms"] * fs / 1000.0)))
        hop_len = max(1, int(round(cfg["rms_hop_ms"] * fs / 1000.0)))
        n_windows = emg_env.shape[0]
        # Take the force value at the centre of each window
        centre_indices = np.array(
            [i * hop_len + win_len // 2 for i in range(n_windows)], dtype=np.intp
        )
        centre_indices = np.clip(centre_indices, 0, len(force) - 1)
        force_ds = force[centre_indices]

        # --- create sequences ---
        Xi, yi = create_sequences(
            emg_env,
            force_ds,
            seq_len=cfg["seq_len"],
            pred_horizon=cfg["pred_horizon"],
        )

        all_X.append(Xi)
        all_y.append(yi)
        all_subj.append(np.full(len(yi), subj_id))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)

    return X, y, subjects
