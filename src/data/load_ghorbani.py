"""
load_ghorbani.py
================
Loader for the Ghorbani Grip Force Prediction Dataset (arXiv:2302.09555).

Dataset Description:
    - 10 healthy subjects (9 male, 1 female, avg age 23.8)
    - Myo armband EMG: 8 channels @ 200 Hz
    - ATI Mini45 F/T sensor: Z-axis grip force
    - Precision-type pinch grip task
    - 3 recording trials per subject (30 total files)

File Layout (Filtered data)::

    <data_dir>/
        filtered_1.csv   # Subject 1, Trial 1
        filtered_2.csv   # Subject 1, Trial 2
        filtered_3.csv   # Subject 1, Trial 3
        filtered_4.csv   # Subject 2, Trial 1
        ...
        filtered_30.csv  # Subject 10, Trial 3

Each CSV file contains:
    - Columns 0-7: EMG channels (emg0-emg7) - bandpass filtered at 20-450 Hz
    - Column 8: Force (Fz) - smoothed grip force signal
    - First row: header comment (# EMG1, EMG2, ..., FORCE)

Reference:
    Ghorbani, A., Yousefi-Koma, A., & Vedadi, A. (2023).
    "Estimation and Early Prediction of Grip Force Based on sEMG Signals
     and Deep Recurrent Neural Networks."
    Journal of the Brazilian Society of Mechanical Sciences and Engineering.
    https://doi.org/10.1007/s40430-023-04070-8

Provides:
    - load_ghorbani_trial : Load a single trial CSV file.
    - load_ghorbani_subject : Load all 3 trials for a subject.
    - load_all_ghorbani : Load all subjects and trials.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    GHORBANI_FS,
    GHORBANI_EMG_CHANNELS,
    GHORBANI_SUBJECTS,
    GHORBANI_TRIALS_PER_SUBJECT,
    GHORBANI_CHANNEL_NAMES,
    RAW_GHORBANI_DIR,
)


# ---------------------------------------------------------------------------
# Single trial loader
# ---------------------------------------------------------------------------

def load_ghorbani_trial(filepath: str | Path) -> dict:
    """
    Load a single Ghorbani dataset CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to a filtered CSV file (e.g., filtered_1.csv).

    Returns
    -------
    dict
        'emg'   : np.ndarray, shape (n_samples, 8)
        'force' : np.ndarray, shape (n_samples,)
        'fs'    : float (200 Hz)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unexpected.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Ghorbani trial file not found: {filepath}")

    # Read CSV - skip comment header line
    try:
        # Try reading with automatic header detection
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('#'):
            # Skip comment header
            data = pd.read_csv(filepath, header=None, skiprows=1)
        else:
            # No header line
            data = pd.read_csv(filepath, header=None)
        
        data = data.values.astype(np.float64)
        
    except Exception as e:
        raise ValueError(f"Failed to parse {filepath}: {e}")

    # Validate shape: expect 9 columns (8 EMG + 1 Force)
    if data.shape[1] < 9:
        raise ValueError(
            f"Expected at least 9 columns in {filepath.name}, "
            f"got {data.shape[1]}. Format: EMG0-7, Force"
        )

    # Extract EMG (columns 0-7) and Force (column 8)
    emg = data[:, :8].astype(np.float32)
    force = data[:, 8].astype(np.float32)

    # Basic validation
    if np.any(np.isnan(emg)) or np.any(np.isnan(force)):
        warnings.warn(f"NaN values detected in {filepath.name}")
        # Replace NaNs with linear interpolation (better than zero-fill for
        # time-series data); fall back to zero for leading/trailing NaNs.
        for ch in range(emg.shape[1]):
            mask = np.isnan(emg[:, ch])
            if mask.any():
                valid = np.where(~mask)[0]
                if len(valid) > 1:
                    emg[mask, ch] = np.interp(
                        np.where(mask)[0], valid, emg[valid, ch])
                else:
                    emg[mask, ch] = 0.0
        mask_f = np.isnan(force)
        if mask_f.any():
            valid_f = np.where(~mask_f)[0]
            if len(valid_f) > 1:
                force[mask_f] = np.interp(
                    np.where(mask_f)[0], valid_f, force[valid_f])
            else:
                force[mask_f] = 0.0

    return {
        "emg": emg,
        "force": force,
        "fs": float(GHORBANI_FS),
    }


# ---------------------------------------------------------------------------
# Subject loader (concatenates 3 trials)
# ---------------------------------------------------------------------------

def load_ghorbani_subject(
    data_dir: str | Path,
    subject_id: int,
    concatenate_trials: bool = True,
) -> dict:
    """
    Load all 3 trials for a single Ghorbani subject.

    File mapping:
        Subject 1: filtered_1.csv, filtered_2.csv, filtered_3.csv
        Subject 2: filtered_4.csv, filtered_5.csv, filtered_6.csv
        Subject N: filtered_{3*(N-1)+1}.csv, ..., filtered_{3*N}.csv

    Parameters
    ----------
    data_dir : str or Path
        Directory containing filtered CSV files.
    subject_id : int
        Subject number (1-10).
    concatenate_trials : bool
        If True, concatenate all trials into single arrays.
        If False, return list of trial dicts.

    Returns
    -------
    dict
        'emg'        : np.ndarray (n_samples, 8) if concatenate else list
        'force'      : np.ndarray (n_samples,) if concatenate else list
        'subject_id' : int
        'fs'         : float
        'n_trials'   : int (number of loaded trials)
    """
    data_dir = Path(data_dir)
    
    if not 1 <= subject_id <= GHORBANI_SUBJECTS:
        raise ValueError(
            f"subject_id must be 1-{GHORBANI_SUBJECTS}, got {subject_id}"
        )

    # Calculate file indices for this subject
    start_idx = (subject_id - 1) * GHORBANI_TRIALS_PER_SUBJECT + 1
    trial_files = [
        data_dir / f"filtered_{start_idx + i}.csv"
        for i in range(GHORBANI_TRIALS_PER_SUBJECT)
    ]

    trials = []
    for tf in trial_files:
        if tf.is_file():
            try:
                trial = load_ghorbani_trial(tf)
                trials.append(trial)
            except Exception as e:
                warnings.warn(f"Failed to load {tf.name}: {e}")
        else:
            warnings.warn(f"Trial file not found: {tf}")

    if not trials:
        raise FileNotFoundError(
            f"No valid trials found for subject {subject_id} in {data_dir}"
        )

    if concatenate_trials:
        # Store trial boundaries for trial-based splitting
        trial_boundaries = []
        offset = 0
        for t in trials:
            n = len(t["emg"])
            trial_boundaries.append((offset, offset + n))
            offset += n
        emg = np.concatenate([t["emg"] for t in trials], axis=0)
        force = np.concatenate([t["force"] for t in trials], axis=0)
    else:
        trial_boundaries = [(0, len(t["emg"])) for t in trials]
        emg = [t["emg"] for t in trials]
        force = [t["force"] for t in trials]

    return {
        "emg": emg,
        "force": force,
        "subject_id": subject_id,
        "fs": float(GHORBANI_FS),
        "n_trials": len(trials),
        "trial_boundaries": trial_boundaries,
    }


# ---------------------------------------------------------------------------
# Load all subjects
# ---------------------------------------------------------------------------

def load_all_ghorbani(
    data_dir: str | Path = None,
    n_subjects: int = GHORBANI_SUBJECTS,
    subjects: list[int] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Load all Ghorbani subjects from the filtered data directory.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing filtered CSV files.
        Default: RAW_GHORBANI_DIR from config.
    n_subjects : int
        Number of subjects to load (default: 10).
    subjects : list[int] or None
        Specific subject IDs to load. If None, load subjects 1 to n_subjects.
    verbose : bool
        Print loading progress.

    Returns
    -------
    list[dict]
        List of subject data dicts with keys:
        'emg', 'force', 'subject_id', 'fs', 'n_trials'

    Raises
    ------
    FileNotFoundError
        If data_dir does not exist.
    """
    if data_dir is None:
        data_dir = RAW_GHORBANI_DIR
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Ghorbani data directory not found: {data_dir}\n"
            f"Please ensure the dataset is in the correct location."
        )

    if subjects is None:
        subjects = list(range(1, min(n_subjects, GHORBANI_SUBJECTS) + 1))

    all_subjects = []
    for sid in subjects:
        try:
            subj = load_ghorbani_subject(data_dir, sid)
            all_subjects.append(subj)
            if verbose:
                print(
                    f"    Subject {sid}: EMG {subj['emg'].shape}, "
                    f"Force {subj['force'].shape}, {subj['n_trials']} trials"
                )
        except Exception as e:
            warnings.warn(f"Failed to load subject {sid}: {e}")

    if verbose:
        total_samples = sum(s["emg"].shape[0] for s in all_subjects)
        print(f"    Total: {len(all_subjects)} subjects, {total_samples:,} samples")

    return all_subjects


# ---------------------------------------------------------------------------
# Data validation and statistics
# ---------------------------------------------------------------------------

def validate_ghorbani_data(subject_data: list[dict]) -> dict:
    """
    Validate loaded Ghorbani data and compute statistics.

    Parameters
    ----------
    subject_data : list[dict]
        Output from load_all_ghorbani().

    Returns
    -------
    dict
        Validation report with statistics and potential issues.
    """
    report = {
        "n_subjects": len(subject_data),
        "total_samples": 0,
        "samples_per_subject": [],
        "emg_stats": {"min": [], "max": [], "mean": [], "std": []},
        "force_stats": {"min": [], "max": [], "mean": [], "std": []},
        "issues": [],
    }

    for subj in subject_data:
        emg = subj["emg"]
        force = subj["force"]
        
        report["total_samples"] += len(force)
        report["samples_per_subject"].append({
            "subject_id": subj["subject_id"],
            "n_samples": len(force),
            "duration_sec": len(force) / subj["fs"],
        })

        # EMG statistics
        report["emg_stats"]["min"].append(float(emg.min()))
        report["emg_stats"]["max"].append(float(emg.max()))
        report["emg_stats"]["mean"].append(float(emg.mean()))
        report["emg_stats"]["std"].append(float(emg.std()))

        # Force statistics
        report["force_stats"]["min"].append(float(force.min()))
        report["force_stats"]["max"].append(float(force.max()))
        report["force_stats"]["mean"].append(float(force.mean()))
        report["force_stats"]["std"].append(float(force.std()))

        # Check for issues
        if np.any(np.isnan(emg)):
            report["issues"].append(f"Subject {subj['subject_id']}: NaN in EMG")
        if np.any(np.isnan(force)):
            report["issues"].append(f"Subject {subj['subject_id']}: NaN in Force")
        if emg.std() < 1e-6:
            report["issues"].append(f"Subject {subj['subject_id']}: Low EMG variance")
        if force.std() < 1e-6:
            report["issues"].append(f"Subject {subj['subject_id']}: Low Force variance")

    # Aggregate stats
    report["emg_global"] = {
        "min": min(report["emg_stats"]["min"]),
        "max": max(report["emg_stats"]["max"]),
        "mean": np.mean(report["emg_stats"]["mean"]),
        "std": np.mean(report["emg_stats"]["std"]),
    }
    report["force_global"] = {
        "min": min(report["force_stats"]["min"]),
        "max": max(report["force_stats"]["max"]),
        "mean": np.mean(report["force_stats"]["mean"]),
        "std": np.mean(report["force_stats"]["std"]),
    }

    return report


def print_ghorbani_summary(subject_data: list[dict]) -> None:
    """Print a summary of the loaded Ghorbani dataset."""
    report = validate_ghorbani_data(subject_data)
    
    print("\n" + "=" * 60)
    print("GHORBANI DATASET SUMMARY")
    print("=" * 60)
    print(f"Subjects loaded:     {report['n_subjects']}")
    print(f"Total samples:       {report['total_samples']:,}")
    print(f"Sampling rate:       {GHORBANI_FS} Hz")
    print(f"EMG channels:        {GHORBANI_EMG_CHANNELS}")
    print(f"\nEMG range:           [{report['emg_global']['min']:.2f}, "
          f"{report['emg_global']['max']:.2f}]")
    print(f"Force range:         [{report['force_global']['min']:.2f}, "
          f"{report['force_global']['max']:.2f}] N")
    
    if report["issues"]:
        print(f"\nWarnings ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"  - {issue}")
    else:
        print("\nNo data quality issues detected.")
    print("=" * 60)
