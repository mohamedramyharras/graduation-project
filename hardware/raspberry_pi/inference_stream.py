#!/usr/bin/env python3
"""
inference_stream.py
===================
Real-time GRU inference on Raspberry Pi 3 for the 2-channel sEMG
prosthetic hand system.

Reads raw EMG samples from Arduino Mega 2560 over USB serial,
computes features, runs the GRU model, and sends the predicted
grip force back to the Arduino for actuator control.

Serial protocol (115200 baud):
  Arduino → Pi :  "E,<ch1_int>,<ch2_int>\\n"   at 2000 Hz
  Pi → Arduino :  "F,<force_float>\\n"           at ~20–25 Hz

Latency breakdown:
  Feature extraction (numpy) : ~2 ms
  GRU inference  (PyTorch)   : ~20–30 ms
  Serial round-trip          : ~2–5 ms
  Total end-to-end           : ~45–55 ms  (20–25 Hz update rate)

Usage:
    python hardware/raspberry_pi/inference_stream.py \\
        --model  outputs/models/gru_ghorbani_best.pt \\
        --port   /dev/ttyUSB0 \\
        --baud   115200

Requirements (install on Pi with pip):
    torch torchvision numpy scipy pyserial joblib
"""

import argparse
import time
from collections import deque
from pathlib import Path
import sys

import numpy as np
import serial
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_model import GRUForcePredictor
from src.config import (
    GRU_HIDDEN_SIZE, GRU_DENSE_SIZE, GRU_OUTPUT_SIZE,
    SAMPLE_RATE, WINDOW_SIZE_SAMPLES, HOP_SIZE_SAMPLES, SEQ_LEN,
)

# ─── Constants ────────────────────────────────────────────────────
N_CHANNELS    = 2          # ECRB (Ch5) and ECU (Ch3)
N_FEATURES    = 6          # RMS, MAV, WL, VAR, ZC, SSC
N_DELTA       = 3          # base + Δ + ΔΔ
FEAT_DIM      = N_CHANNELS * N_FEATURES * N_DELTA   # 36

SMOOTH_FC     = 4.0        # Butterworth low-pass cut-off (Hz)
ADC_MAX       = 1023.0     # Arduino 10-bit ADC range

# Butterworth filter (4 Hz, 2nd order, designed at 50 Hz = 1/hop)
_BUTTER_SOS = butter(2, SMOOTH_FC / (0.5 / (HOP_SIZE_SAMPLES / SAMPLE_RATE)),
                     btype="low", output="sos")


# ─── Feature extraction ───────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 6 time-domain features from a (W, 2) EMG window.
    Returns a (12,) vector (6 features × 2 channels).
    """
    feats = []
    for ch in range(N_CHANNELS):
        x = window[:, ch]
        rms = np.sqrt(np.mean(x ** 2))
        mav = np.mean(np.abs(x))
        wl  = np.sum(np.abs(np.diff(x)))
        var = np.var(x)
        zc  = np.sum(np.diff(np.sign(x)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(x))) != 0)
        feats.extend([rms, mav, wl, var, float(zc), float(ssc)])
    return np.array(feats, dtype=np.float32)


def delta_augment(feat_history: np.ndarray) -> np.ndarray:
    """
    Augment the latest feature vector with first- and second-order
    temporal differences, returning a (36,) vector.
    feat_history: (>=3, 12) array of recent feature vectors.
    """
    f0 = feat_history[-1]
    d1 = feat_history[-1] - feat_history[-2] if len(feat_history) >= 2 else np.zeros_like(f0)
    d2 = (feat_history[-1] - 2 * feat_history[-2] + feat_history[-3]
          if len(feat_history) >= 3 else np.zeros_like(f0))
    return np.concatenate([f0, d1, d2])


# ─── Main inference loop ──────────────────────────────────────────
def run(model_path: str, port: str, baud: int):
    # ── Load model ────────────────────────────────────────────────
    print(f"Loading model from {model_path} ...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model = GRUForcePredictor(
        input_size=FEAT_DIM,
        hidden_size=GRU_HIDDEN_SIZE,
        dense_size=GRU_DENSE_SIZE,
        output_size=GRU_OUTPUT_SIZE,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Open serial port ──────────────────────────────────────────
    print(f"Opening serial port {port} at {baud} baud ...")
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(2)   # Wait for Arduino reset
    ser.reset_input_buffer()
    print("Connected. Starting inference loop ...")

    # ── Buffers ───────────────────────────────────────────────────
    raw_buf   = deque(maxlen=WINDOW_SIZE_SAMPLES)   # rolling raw window
    feat_hist = deque(maxlen=3)                     # last 3 feature vecs for delta
    seq_buf   = deque(maxlen=SEQ_LEN)               # sequence of 36-D vecs
    hop_accum = 0                                   # samples since last hop

    # MinMax scaler: populated after first SEQ_LEN steps
    x_min = None
    x_scale = None

    # Rolling force buffer for Butterworth smoothing (causal)
    force_raw_buf = deque(maxlen=8)
    sos_state = np.zeros((len(_BUTTER_SOS), 2))     # filter state (causal)

    t_infer = 0.0    # Latest inference time

    while True:
        # ── Read one line from Arduino ────────────────────────────
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
        except serial.SerialException as exc:
            print(f"Serial error: {exc}")
            break

        if not line.startswith("E,"):
            continue

        parts = line.split(",")
        if len(parts) != 3:
            continue

        try:
            ch1 = int(parts[1]) / ADC_MAX
            ch2 = int(parts[2]) / ADC_MAX
        except ValueError:
            continue

        raw_buf.append([ch1, ch2])
        hop_accum += 1

        # ── Compute features every HOP_SIZE_SAMPLES ───────────────
        if hop_accum < HOP_SIZE_SAMPLES or len(raw_buf) < WINDOW_SIZE_SAMPLES:
            continue

        hop_accum = 0
        window = np.array(raw_buf, dtype=np.float32)   # (W, 2)
        feats  = extract_features(window)               # (12,)

        feat_hist.append(feats)
        if len(feat_hist) < 3:
            continue

        vec36 = delta_augment(np.array(feat_hist))     # (36,)
        seq_buf.append(vec36)

        if len(seq_buf) < SEQ_LEN:
            continue

        # ── Build sequence tensor & simple MinMax normalization ───
        seq_np = np.array(seq_buf, dtype=np.float32)   # (50, 36)

        if x_min is None:   # Initialize scaler from first seen sequence
            x_min   = seq_np.min(axis=0, keepdims=True)
            x_scale = seq_np.max(axis=0, keepdims=True) - x_min
            x_scale[x_scale < 1e-8] = 1.0

        seq_norm = (seq_np - x_min) / x_scale           # (50, 36)
        x = torch.tensor(seq_norm[np.newaxis], dtype=torch.float32)  # (1,50,36)

        # ── GRU inference ──────────────────────────────────────────
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(x).item()
        t_infer = (time.perf_counter() - t0) * 1000     # ms

        # ── Causal exponential smoothing (lightweight alternative) ─
        # Use a running average to avoid sosfiltfilt (which needs future data)
        alpha = 0.3
        if force_raw_buf:
            pred = alpha * pred + (1 - alpha) * force_raw_buf[-1]
        force_raw_buf.append(pred)

        force = float(np.clip(pred, 0.0, 1.0))

        # ── Send back to Arduino ───────────────────────────────────
        ser.write(f"F,{force:.4f}\n".encode("ascii"))

        print(f"\rForce: {force:.3f}  |  Infer: {t_infer:.1f} ms", end="", flush=True)


# ─── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time GRU inference on Raspberry Pi 3"
    )
    parser.add_argument(
        "--model", default="outputs/models/gru_ghorbani_best.pt",
        help="Path to trained .pt model checkpoint"
    )
    parser.add_argument(
        "--port", default="/dev/ttyUSB0",
        help="Arduino serial port (e.g. /dev/ttyUSB0 or /dev/ttyACM0)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Serial baud rate (must match Arduino sketch)"
    )
    args = parser.parse_args()
    run(args.model, args.port, args.baud)
