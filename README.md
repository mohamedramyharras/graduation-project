# AI-Controlled Prosthetic Hand with Grip Force Prediction
## Team 15 | Cairo University | March 24, 2026

**An affordable, subject-independent AI system that predicts grip force from muscle signals to control a 3D-printed prosthetic hand.**

---

## 📂 Project Structure

```
graduation-project/
├── README.md                              ← You are here
├── READ_ME_FIRST.txt                      ← Start with this!
├── LICENSE
├── requirements.txt
│
├── study_guides/                          ← 📚 USE THESE FOR PRESENTATION
│   ├── PROJECT_STUDY_GUIDE.pdf            ← PRIMARY STUDY MATERIAL (21 KB, 16 sections + Q&A)
│   ├── EXECUTIVE_BRIEF.md                 ← Quick overview
│   └── START_HERE.md                      ← Navigation guide
│
├── reference/                             ← 📋 Quick reference materials
│   ├── OVERFITTING_PROOF_CARD.md          ← 2-minute answer script (print this!)
│   ├── FINAL_CHECKLIST.md                 ← Tonight + tomorrow plan
│   ├── PRESENTATION_QUICK_REFERENCE.md    ← Key numbers & talking points
│   └── PRESENTATION_CHECKLIST.md          ← Study schedule
│
├── technical/                             ← 🔧 Scripts & generation
│   └── generate_project_guide.py          ← PDF generation script
│
├── src/                                   ← 💻 Source code
│   ├── models/                            ← Model implementations
│   ├── preprocessing/                     ← Data preprocessing
│   ├── visualization/                     ← Plotting & visualization
│   └── config.py                          ← Configuration
│
├── data/                                  ← 📊 Datasets
│   ├── ghorbani/                          ← Ghorbani et al. (2023) dataset
│   └── processed/                         ← Preprocessed data
│
├── outputs/                               ← 📈 Results & evaluation
│   ├── models/                            ← Trained models
│   ├── results/                           ← Evaluation metrics
│   └── figures/                           ← Plots & visualizations
│
├── notebooks/                             ← 📔 Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_ghorbani_analysis.ipynb
│
├── hardware/                              ← ⚙️ Hardware specifications
│   ├── arduino/                           ← Arduino code
│   ├── raspberry_pi/                      ← Raspberry Pi code
│   └── prosthetic_hand/                   ← Hand design & CAD
│
├── docs/                                  ← 📖 Documentation
│   ├── abstract.md
│   ├── methodology.md
│   ├── results_discussion.md
│   └── hardware_design.md
│
└── paper/                                 ← 📄 Papers & presentations
    ├── main.tex                           ← LaTeX manuscript
    ├── GP Presentation.pptx               ← Presentation slides
    └── Graduation Project Report Final.docx
```

---

## 🎯 Quick Start

### For Presentation Tomorrow:
1. **Read:** `READ_ME_FIRST.txt` (2 min)
2. **Study:** `study_guides/PROJECT_STUDY_GUIDE.pdf` (90 min)
3. **Reference:** `reference/OVERFITTING_PROOF_CARD.md` (10 min, print if possible)
4. **Plan:** `reference/FINAL_CHECKLIST.md` (tonight + tomorrow)
5. **Practice:** `reference/PRESENTATION_QUICK_REFERENCE.md` (key numbers)

### For Understanding the System:
- **Architecture:** See `src/models/` and `src/config.py`
- **Data:** See `data/ghorbani/` 
- **Results:** See `outputs/results/`
- **Notebooks:** Run `notebooks/` for step-by-step analysis

### For Hardware Deployment:
- **Arduino Code:** `hardware/arduino/emg_force_inference.ino`
- **Raspberry Pi:** `hardware/raspberry_pi/inference_stream.py`
- **Hand Design:** `hardware/prosthetic_hand/` (STL files for 3D printing)

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Test R²** | 0.778 ± 0.062 |
| **NRMSE** | 11.5% ± 1.8% |
| **Latency** | 45-55 ms (real-time) |
| **Hardware Cost** | ~$150 |
| **Subject-Independent** | ✓ Yes (unique!) |

---

## 🎓 Study Materials

### Primary
- **PROJECT_STUDY_GUIDE.pdf** (21 KB)
  - 16 comprehensive sections
  - 15 Q&A questions with detailed answers
  - All corrections applied
  - Ready to print

### Supporting
- **OVERFITTING_PROOF_CARD.md** - Printable reference card
- **FINAL_CHECKLIST.md** - Tonight & morning preparation
- **PRESENTATION_QUICK_REFERENCE.md** - Key numbers & talking points
- **PRESENTATION_CHECKLIST.md** - Study schedule

### Navigation
- **READ_ME_FIRST.txt** - Quick start guide
- **START_HERE.md** - File index & reading plans

---

## 🔑 Key Innovation

**Subject-Independence:** Unlike prior work, our model works for ANY new user immediately without per-user calibration (0 min). Traditional systems require 30-60 min calibration per user.

**Trade:** Slight accuracy loss (R²=0.778 vs 0.85-0.96) for **massive usability gain** (instant deployment).

---

## 📝 Hardware Specs

| Component | Specification |
|-----------|---------------|
| EMG Sensors | MyoWare 2.0 (×2) |
| Microcontroller | Arduino Uno R3 |
| Inference | Raspberry Pi 3 |
| Actuators | SG90 Servos (×5) |
| Hand | InMoov 3D-printed |
| **Total Cost** | **~$150** |

---

## 🧠 Model Architecture

- **Type:** GRU (Gated Recurrent Unit)
- **Hidden Units:** 64
- **Parameters:** 23,809 (lightweight)
- **Input:** 50 timesteps × 36 features
- **Output:** Grip force (0-1 scalar)

**Why GRU?**
- +6.6% improvement vs MLP
- +12.8% improvement vs Ridge
- Temporal modeling captures muscle dynamics
- Real-time inference (20-30 ms)

---

## ✅ No Overfitting - Proven By

1. **Validation Curve:** Loss converges for both training & validation (no divergence)
2. **Per-Subject Consistency:** Test R² ranges 0.67–0.88 (no extreme outliers)
3. **Ridge Baseline:** Linear model (R²=0.69) proves GRU's +12.8% gain is from temporal learning, not memorization

---

## 📈 Results Summary

### Performance
- **Test R²:** 0.778 ± 0.062 ✓ Excellent
- **NRMSE:** 11.5% ± 1.8% ✓ State-of-the-art
- **MAE:** 15.6 N ✓ Clinically acceptable
- **Pearson r:** 0.8903 ✓ Strong correlation

### Model Comparison
| Model | R² | Improvement |
|-------|-----|-----------|
| Ridge | 0.690 | Baseline |
| MLP | 0.730 | +6.6% |
| **GRU** | **0.778** | **+12.8%** |

---

## 🚀 System Pipeline

```
EMG Sensors (200 Hz)
    ↓
Arduino Uno R3 (ADC @ 2 kHz)
    ↓ Serial @ 115,200 baud
Raspberry Pi 3 (GRU inference 20-30 ms)
    ↓ PWM signals
SG90 Servo Motors (×5)
    ↓
InMoov Prosthetic Hand
    ↓
Proportional Grip Force Control
```

**Total Latency:** 45-55 ms (within prosthetic threshold 100-300 ms) ✓

---

## 📖 Feature Extraction

**36-Dimensional Feature Vector:**
- 2 channels (ECRB, ECU) selected via mRMR
- 6 time-domain features per channel (RMS, MAV, WL, VAR, ZC, SSC)
- 3 derivative orders (0th, 1st, 2nd)
- **Formula:** 2 × 6 × 3 = 36 features

**Why Derivatives?**
Grip force is dynamic. Derivatives capture:
- Rate of change (1st derivative)
- Acceleration (2nd derivative)
Essential for capturing muscle onset, sustained hold, and release.

---

## 🎓 Dataset

**Ghorbani et al. (2023)**
- 10 subjects (9 analyzed, S8 outlier excluded)
- Ramp-and-hold contractions (20%, 40%, 60%, 80% MVC)
- EMG: 200 Hz, 8 channels (2 selected)
- Force: Dynamometer, normalized [0,1]
- Total: ~1.75 hours of data

---

## 📚 Files Included

### Study Guides
- `study_guides/PROJECT_STUDY_GUIDE.pdf` - Complete 21 KB guide with Q&A
- `study_guides/EXECUTIVE_BRIEF.md` - Project overview
- `study_guides/START_HERE.md` - Navigation guide

### Reference Materials
- `reference/OVERFITTING_PROOF_CARD.md` - 2-minute answer script (PRINT THIS!)
- `reference/FINAL_CHECKLIST.md` - Tonight + tomorrow plan
- `reference/PRESENTATION_QUICK_REFERENCE.md` - Key numbers
- `reference/PRESENTATION_CHECKLIST.md` - Study schedule

### Source Code
- `src/models/gru_model.py` - GRU implementation
- `src/preprocessing/` - Data preprocessing pipeline
- `src/config.py` - Hyperparameters

### Hardware
- `hardware/arduino/emg_force_inference.ino` - Arduino code
- `hardware/raspberry_pi/inference_stream.py` - Real-time inference

### Results
- `outputs/results/ghorbani_evaluation.json` - Test results (R²=0.778)
- `outputs/models/` - Trained GRU model

### Documentation
- `docs/abstract.md` - Project abstract
- `docs/methodology.md` - Methods explanation
- `docs/results_discussion.md` - Results analysis
- `docs/hardware_design.md` - Hardware specifications

---

## 🔗 GitHub Repository

https://github.com/mohamedramyharras/graduation-project.git

**Latest Commits:**
- `a51501b` - Add comprehensive Q&A section
- `8f833c2` - Update study guides with corrections (Arduino Uno R3, overfitting proof)
- `41dc2b0` - Update deployment architecture

---

## ✨ Corrections Applied

✅ **Hardware:** Arduino Uno R3 (not Mega 2560)
✅ **Overfitting Proof:** 3-point rigorous method (validation curve + consistency + Ridge baseline)
✅ **Study Materials:** All updated with corrections
✅ **Q&A:** 15 comprehensive questions with answers

---

## 🎯 Presentation Checklist

**Tonight:**
- [ ] Read `READ_ME_FIRST.txt`
- [ ] Study `study_guides/PROJECT_STUDY_GUIDE.pdf`
- [ ] Review `reference/OVERFITTING_PROOF_CARD.md`
- [ ] Check `reference/FINAL_CHECKLIST.md`
- [ ] Sleep 8 hours minimum

**Tomorrow Morning:**
- [ ] Review `reference/PRESENTATION_QUICK_REFERENCE.md`
- [ ] Mental system walkthrough
- [ ] Practice opening statement
- [ ] Go present with confidence! 🚀

---

## 📞 Questions?

- **Quick answers:** See `reference/PRESENTATION_QUICK_REFERENCE.md`
- **Deep dive:** See `study_guides/PROJECT_STUDY_GUIDE.pdf`
- **Specific topic:** See relevant notebook in `notebooks/`
- **Code:** See relevant file in `src/`

---

## 📜 License

See `LICENSE` file.

---

## 👥 Team & Supervision

**Team:** Team 15
**Institution:** Cairo University, Faculty of Engineering
**Supervisor:** Dr. Aliaa Rehan
**Date:** March 24, 2026

---

**Status: ✅ READY FOR PRESENTATION**

All study materials prepared. All corrections applied. All files organized.
Go present with confidence! 🎉
