# sEMG-Controlled Prosthetic Hand

## Proportional Grip Force Control from 2 Surface EMG Channels Using a Subject-Independent GRU Model

A complete end-to-end system for real-time grip force prediction from surface electromyography (sEMG) signals, deployed on an InMoov 3D-printed prosthetic hand actuated via Arduino-controlled linear actuators.

The AI model uses only **2 optimally-selected EMG channels** (MRMR criterion) and achieves **R² = 0.778** in a fully subject-independent setting — no per-user calibration required.

---

## System Overview

```
  MyoWare 2.0         Signal         Feature         GRU Neural        Force         Arduino       InMoov
  sEMG Sensors   -->  Processing --> Extraction  --> Network      --> Prediction --> Mega 2560 --> Prosthetic
  (2 channels)        (RMS, 80Hz)    (36 features)   (23,809 params)   (0-1 norm)    (PWM)         Hand
       |                                                                                |
   ECRB (Ch5)                                                                    Linear Actuators
   ECU  (Ch3)                                                                    + Nylon Strings
```

**Electrode Placement**: Two electrodes over the posterior forearm —
ECRB (Extensor Carpi Radialis Brevis) and ECU (Extensor Carpi Ulnaris),
the primary wrist extensors biomechanically active during grip force generation.

---

## Key Results

Evaluated on the Ghorbani Grip Force Dataset (Myo armband, 200 Hz, 9 subjects).

| Metric | GRU (Ours) | MLP Baseline | Ridge Baseline |
|--------|-----------|-------------|---------------|
| **R²** | **0.778 +/- 0.062** | 0.730 +/- 0.086 | 0.690 +/- 0.084 |
| **NRMSE%** | **11.5 +/- 1.8** | 12.6 +/- 2.1 | 13.6 +/- 2.2 |
| **Pearson r** | **0.890 +/- 0.033** | 0.873 +/- 0.036 | 0.847 +/- 0.043 |
| **MAE** | **0.078 +/- 0.009** | 0.085 +/- 0.011 | 0.095 +/- 0.014 |

- **Subject-independent**: one model works for all users, no calibration needed
- **Minimal hardware**: 2 EMG electrodes (vs. 4-10 in prior work)
- **Embedded-friendly**: 23,809 parameters, runs on Arduino Mega 2560

---

## Hardware

| Component | Specification |
|-----------|--------------|
| Prosthetic Hand | InMoov (3D-printed, PLA/ABS) |
| Actuation | 5 linear actuators + nylon strings |
| Microcontroller | Arduino Mega 2560 |
| EMG Sensors | MyoWare 2.0 Muscle Sensors (x2) |
| Communication | HC-05 Bluetooth (calibration) |
| EMG Channels | ECRB (Ch5) + ECU (Ch3) |

The hand is actuated by nylon strings connected to linear actuators, with grip force proportionally controlled by the predicted EMG-to-force mapping.

---

## Model Architecture

```
Input: 50 timesteps x 36 features
       (2 channels x 6 features x 3 delta orders)
            |
    GRU (hidden=64, 1 layer)
            |
    ReLU -> Dropout(0.2)
            |
    FC(64) -> ReLU -> Dropout(0.2)
            |
    FC(1) -> Predicted Grip Force
            |
    Butterworth LP (4 Hz, zero-phase)
            |
    Smoothed Force Output
```

- **Parameters**: 23,809 (embedded-deployable)
- **Features**: RMS, MAV, WL, VAR, ZC, SSC + delta + delta-delta
- **Channel selection**: MRMR (Peng et al., 2005)
- **Training**: Subject-independent (pooled data, no per-user calibration)

---

## Project Structure

```
sEMG-Prosthetic-Hand/
|
+-- src/
|   +-- config.py                          # Central configuration
|   +-- data/
|   |   +-- load_ghorbani.py               # Dataset loader
|   |   +-- preprocess.py                  # Feature extraction pipeline
|   +-- features/
|   |   +-- mrmr.py                        # MRMR channel selection
|   +-- models/
|   |   +-- gru_model.py                   # GRU architecture
|   |   +-- run_evaluation.py              # Evaluation pipeline
|   +-- visualization/
|       +-- generate_paper_figures.py      # Publication figures
|       +-- generate_scatter_ghorbani.py   # Scatter plot
|
+-- hardware/
|   +-- arduino/
|   |   +-- emg_force_inference.ino        # Real-time inference on Arduino
|   |   +-- calibration.h                  # Bluetooth calibration protocol
|   |   +-- model_weights.h               # Exported GRU weights
|   +-- mobile_app/
|       +-- README.md                      # Calibration app specification
|
+-- paper/
|   +-- main.tex                           # IEEE LaTeX manuscript
|   +-- refs.bib                           # BibTeX references
|   +-- generate_docx.py                   # Word manuscript generator
|   +-- generate_pptx.py                   # Presentation generator
|
+-- data/
|   +-- ghorbani_raw/                      # Dataset (not tracked in git)
|
+-- outputs/
|   +-- figures/                           # Generated figures (300 DPI)
|   +-- models/                            # MRMR results, model checkpoints
|   +-- results/                           # Evaluation JSON
|
+-- docs/                                  # Project documentation
+-- notebooks/                             # Exploratory Jupyter notebooks
+-- requirements.txt
+-- LICENSE
+-- README.md
```

---

## Installation

### Prerequisites

```bash
python -m pip install torch numpy scipy scikit-learn matplotlib python-docx python-pptx
```

### Dataset

Clone the Ghorbani Grip Force Dataset into `data/ghorbani_raw/`:
```bash
git clone https://github.com/Atusa-gh/GrippingForcePrediction data/ghorbani_raw
```

---

## Usage

### Run Evaluation

```bash
# Full evaluation (9 subjects, S8 excluded)
python -m src.models.run_evaluation --exclude-subjects 8

# Quick evaluation (skip ablation studies)
python -m src.models.run_evaluation --exclude-subjects 8 --skip-ablation --skip-loso --skip-cross-trial
```

### Generate Figures

```bash
# All publication figures (300 DPI)
python -m src.visualization.generate_paper_figures

# Scatter plot only
python src/visualization/generate_scatter_ghorbani.py
```

### Generate Publication Documents

```bash
# Word manuscript
python paper/generate_docx.py

# PowerPoint presentation
python paper/generate_pptx.py

# LaTeX (requires pdflatex + bibtex)
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Arduino Deployment

Upload `hardware/arduino/emg_force_inference.ino` to Arduino Mega 2560.
Connect MyoWare 2.0 sensors to analog pins A0 (ECRB) and A1 (ECU).
See `hardware/arduino/calibration.h` for Bluetooth calibration protocol.

---

## Dataset

**Ghorbani Grip Force Dataset** (Ghorbani et al., 2023)
- 10 healthy subjects (9 male, 1 female; mean age 23.8 years)
- Myo armband: 8 channels at 200 Hz (pre-filtered)
- ATI Mini-45 F/T sensor: z-axis grip force
- 3 trials per subject, isometric precision gripping
- Subject 8 excluded (data-quality outlier — all models fail)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{semg_prosthetic_hand_2026,
  title   = {A Lightweight 2-Channel sEMG-Based Grip Force Controller
             for 3D-Printed Prosthetic Hands Using Gated Recurrent Units},
  author  = {[Author Name(s)]},
  year    = {2026},
  note    = {GitHub: https://github.com/[username]/sEMG-Prosthetic-Hand}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
