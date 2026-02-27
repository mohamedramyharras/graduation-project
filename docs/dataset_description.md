# Dataset Description

## Ghorbani Grip Force Dataset

### Overview

The Ghorbani Grip Force Dataset is a publicly available benchmark for EMG-based grip force prediction research. It provides synchronized surface EMG and grip force measurements during isometric precision gripping tasks.

**Citation**: Ghorbani, A., Attarbashi, Z.S., & Lowenau, J. (2023). Estimation and Early Prediction of Grip Force Based on sEMG Signals and Deep Recurrent Neural Networks. arXiv:2302.09555.

**Source**: https://github.com/Atusa-gh/GrippingForcePrediction

### Acquisition Details

| Parameter | Value |
|-----------|-------|
| Subjects | 10 healthy (9 male, 1 female) |
| Age range | Mean 23.8 years |
| EMG sensors | Myo armband (8 channels) |
| EMG sampling rate | 200 Hz (pre-filtered) |
| Force sensor | ATI Mini-45 F/T sensor |
| Force axis | Z-axis (grip force) |
| Trials | 3 recordings per subject |
| Task | Isometric precision gripping |

### EMG Channel Placement (Myo Armband)

The Myo armband contains 8 equally-spaced EMG channels worn around the proximal forearm. The channels correspond to the following approximate anatomical locations:

| Channel | Muscle | Abbreviation |
|---------|--------|-------------|
| 0 | Flexor Carpi Radialis | FCR |
| 1 | Palmaris Longus | PL |
| 2 | Flexor Carpi Ulnaris | FCU |
| **3** | **Extensor Carpi Ulnaris** | **ECU** |
| 4 | Extensor Digitorum | ED |
| **5** | **Extensor Carpi Radialis Brevis** | **ECRB** |
| 6 | Brachioradialis | BR |
| 7 | Pronator Teres | PT |

**Bold** channels (Ch3 ECU and Ch5 ECRB) are the two MRMR-selected channels used in this project.

### MRMR Channel Selection Results

| Channel | Muscle | Relevance (MI) | MRMR Score |
|---------|--------|----------------|------------|
| **Ch5** | **ECRB** | **0.07705** | **0.06527** (selected 1st) |
| **Ch3** | **ECU** | **0.03031** | **0.01853** (selected 2nd) |
| Ch6 | BR | 0.03713 | -0.00946 |
| Ch4 | ED | 0.02775 | -0.06051 |
| Ch2 | FCU | 0.02086 | -0.00377 |
| Ch1 | PL | 0.03019 | 0.01867 |
| Ch7 | PT | 0.01289 | -0.01518 |
| Ch0 | FCR | 0.01182 | -0.00756 |

ECRB has the highest relevance (MI = 0.077) and highest MRMR score, making it the clear first choice. ECU is selected second because it provides complementary information (low redundancy with ECRB) while maintaining reasonable relevance.

### Force Measurements

- Single-axis grip force (ATI Mini-45 z-axis)
- Continuous force during isometric precision gripping
- Force values normalized to [0, 1] range per subject (MinMaxScaler fit on training data only)

### Data Format

- MATLAB `.mat` files (loaded via custom Python loader)
- Variables: `emg` (N x 8), `force` (N x 1), sampling rate metadata
- Typical recording length: several minutes per trial

---

## Subject 8 Outlier

Subject 8 is identified as a data-quality outlier based on consistently poor performance across ALL model families:

| Model | S8 R² | Group Mean R² (9 subjects) |
|-------|-------|---------------------------|
| GRU | 0.434 | 0.778 |
| Ridge | 0.213 | 0.690 |
| MLP | 0.166 | 0.730 |

The fact that even simple linear regression (Ridge) fails indicates a fundamental data-quality issue — likely poor electrode contact, unusual muscle anatomy, or recording artifacts — rather than a model limitation. Subject 8 is excluded from primary statistics but reported separately for transparency.

---

## Data Splitting Protocol

| Split | Proportion | Method |
|-------|-----------|--------|
| Training | ~68% | Temporal first portion |
| Validation | ~12% | Random from training block |
| Test | 20% | Temporal last portion |
| Gap | 10 sequences | Between train/val and test (prevent leakage) |

The temporal split ensures the model is never tested on data from the same time segment it trained on, which is critical for time-series evaluation.
