# Abstract

## A Lightweight 2-Channel sEMG-Based Grip Force Controller for 3D-Printed Prosthetic Hands Using Gated Recurrent Units

Proportional grip force control is essential for functional upper-limb prostheses, yet most EMG-based systems require multiple electrodes or per-user calibration, limiting affordability and ease of use. We present a complete end-to-end system for real-time grip force prediction from surface electromyography (sEMG) signals, deployed on an InMoov 3D-printed prosthetic hand actuated via Arduino-controlled linear actuators.

Our approach combines Minimum Redundancy Maximum Relevance (MRMR) channel selection with a lightweight Gated Recurrent Unit (GRU) neural network. MRMR identifies two optimal EMG channels — ECRB (Extensor Carpi Radialis Brevis, Ch5) and ECU (Extensor Carpi Ulnaris, Ch3) — from an 8-channel Myo armband array. These muscles, located on the posterior forearm, are biomechanically active during grip force generation through co-activation of wrist extensors (the "extensor paradox"). A 36-dimensional feature vector (6 time-domain features x 2 channels x 3 delta orders) feeds a single-layer GRU with 23,809 trainable parameters.

The model is trained in a fully subject-independent manner — pooled data from nine healthy participants, no per-user calibration required — making it suitable for plug-and-play prosthetic deployment. On the publicly available Ghorbani grip-force dataset (Myo armband, 200 Hz, 10 subjects), the proposed method achieves:

| Metric | Value |
|--------|-------|
| R² | 0.778 ± 0.062 |
| NRMSE% | 11.5 ± 1.8 |
| RMSE | 0.103 ± 0.011 |
| MAE | 0.078 ± 0.009 |
| Pearson r | 0.890 ± 0.033 |
| 95% CI (R²) | [0.730, 0.826] |

Four of nine subjects exceed R² = 0.80. The GRU outperforms both MLP (R² = 0.730) and Ridge regression (R² = 0.690) baselines. Subject 8 is excluded as a data-quality outlier (all models fail: GRU R² = 0.434, Ridge R² = 0.213, MLP R² = 0.166).

The trained model is exported as a C header file for deployment on an Arduino Mega 2560, which drives five linear actuators controlling the InMoov hand's fingers proportionally to the predicted grip force. The system requires only two MyoWare 2.0 surface EMG sensors, making it practical for low-cost, accessible prosthetic applications.

**Keywords:** surface EMG, grip force prediction, MRMR channel selection, GRU, prosthetic hand, InMoov, embedded inference, subject-independent, Arduino
