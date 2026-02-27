"""
Data loading and preprocessing modules for EMG-to-Force prediction.

Available loaders:
    - load_ghorbani: Ghorbani Grip Force Dataset (arXiv:2302.09555)
    - preprocess: Signal preprocessing utilities
"""

from .load_ghorbani import load_ghorbani_trial, load_ghorbani_subject, load_all_ghorbani
from .preprocess import (
    bandpass_filter,
    compute_rms_envelope,
    create_sequences,
    temporal_block_split,
)

__all__ = [
    # Ghorbani
    "load_ghorbani_trial",
    "load_ghorbani_subject",
    "load_all_ghorbani",
    # Preprocessing
    "bandpass_filter",
    "compute_rms_envelope",
    "create_sequences",
    "temporal_block_split",
]
