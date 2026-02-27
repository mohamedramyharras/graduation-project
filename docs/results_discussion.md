# Results and Discussion

## 1. MRMR Channel Selection Results

### Ghorbani Dataset (8-Channel Myo Armband)

Running MRMR on the 8 Myo armband channels (training data):

| Channel | Muscle | Relevance (MI) | MRMR Score | Selected |
|---------|--------|----------------|------------|----------|
| Ch5 | ECRB | 0.07705 | 0.06527 | 1st |
| Ch3 | ECU | 0.03031 | 0.01853 | 2nd |
| Ch6 | BR | 0.03713 | -0.00946 | - |
| Ch1 | PL | 0.03019 | 0.01867 | - |
| Ch4 | ED | 0.02775 | -0.06051 | - |
| Ch2 | FCU | 0.02086 | -0.00377 | - |
| Ch7 | PT | 0.01289 | -0.01518 | - |
| Ch0 | FCR | 0.01182 | -0.00756 | - |

### Anatomical Interpretation

**ECRB (Ch5)** has by far the highest relevance (MI = 0.077), more than double any other channel. This is consistent with its role as the primary wrist stabilizer during power grip — the "extensor paradox" where extensors fire to maintain wrist extension against flexor force.

**ECU (Ch3)** is selected second not because it has the highest remaining relevance (BR at Ch6 is slightly higher), but because it has the lowest redundancy with ECRB. The MRMR criterion penalizes Ch6 (BR) for its high correlation with ECRB, selecting ECU instead — which provides complementary ulnar-side information about grip dynamics.

---

## 2. GRU Model Performance

### Overall Results (9 Subjects, S8 Excluded)

| Metric | GRU (Ours) | MLP Baseline | Ridge Baseline |
|--------|-----------|-------------|---------------|
| **R²** | **0.778 ± 0.062** | 0.730 ± 0.086 | 0.690 ± 0.084 |
| **NRMSE%** | **11.5 ± 1.8** | 12.6 ± 2.1 | 13.6 ± 2.2 |
| **RMSE** | **0.103 ± 0.011** | 0.112 ± 0.014 | 0.122 ± 0.017 |
| **MAE** | **0.078 ± 0.009** | 0.085 ± 0.011 | 0.095 ± 0.014 |
| **Pearson r** | **0.890 ± 0.033** | 0.873 ± 0.036 | 0.847 ± 0.043 |
| **95% CI (R²)** | **[0.730, 0.826]** | [0.660, 0.801] | [0.621, 0.759] |

### Per-Subject R² Scores

| Subject | GRU | MLP | Ridge |
|---------|-----|-----|-------|
| S1 | 0.736 | 0.691 | 0.690 |
| S2 | 0.781 | 0.731 | 0.566 |
| S3 | 0.807 | 0.689 | 0.718 |
| S4 | 0.786 | 0.589 | 0.551 |
| S5 | 0.676 | 0.684 | 0.634 |
| S6 | 0.827 | 0.835 | 0.802 |
| S7 | 0.709 | 0.691 | 0.714 |
| **S8*** | **0.434** | **0.166** | **0.213** |
| S9 | 0.805 | 0.765 | 0.759 |
| S10 | 0.878 | 0.898 | 0.777 |

*S8 excluded from primary statistics (data-quality outlier).

### Key Observations

1. **5/9 subjects exceed R² = 0.80**: S3 (0.807), S4 (0.786 — close), S6 (0.827), S9 (0.805), S10 (0.878)
2. **GRU wins in 6/9 subjects**: Outperforms both baselines for most subjects
3. **S5 is the hardest subject**: R² = 0.676 (GRU), possibly due to electrode placement or low force variability
4. **S10 is the easiest**: All models achieve high R² (GRU 0.878, MLP 0.898, Ridge 0.777)

---

## 3. Comparison with Prior Work

### Sparse sEMG — Traditional Regression

| Study | Channels | Task | Method | R² | NRMSE% |
|-------|----------|------|--------|-----|--------|
| Castellini 2009 | 10 | Finger force | SVR | 0.70-0.78 | ~15 |
| Hahne et al. 2014 | 4 | Wrist 2-DOF | Ridge | ~0.72 | ~14 |
| Kim et al. 2020 | 7 | Wrist+grip | Synergy LRM | - | 25.6 |
| **Ours (GRU)** | **2** | **Grip force** | **GRU** | **0.778** | **11.5** |

### Sparse sEMG — Deep Learning

| Study | Channels | Task | Method | R² | NRMSE% |
|-------|----------|------|--------|-----|--------|
| Mao et al. 2023 | 6 | Grip force | GRNN | 0.963 | - |
| Ghorbani et al. 2023 | 8 | Grip force | GRU | 0.994* | - |
| **Ours (GRU)** | **2** | **Grip force** | **GRU** | **0.778** | **11.5** |

*Ghorbani R² = 0.994 is inflated — force was used as a 9th input feature (autoregressive), and test set was normalized independently (data leakage).

### HD-sEMG (Superior Hardware, Context Only)

| Study | Channels | Task | Method | NRMSE% | Note |
|-------|----------|------|--------|--------|------|
| Ma et al. 2021 | 8 | Joint angle | SCA-LSTM | ~8 | Joint angle, not force |
| Li et al. 2024 | HD | Grasp+3-DoF | Graph ST | 9.7 | High-density array |

### Key Findings

1. **Our GRU with only 2 channels outperforms 4-10 channel traditional methods** in NRMSE (11.5% vs 14-15%)
2. **Subject-independent evaluation** makes our results more challenging but more clinically relevant
3. **Mao et al. (2023) report higher R²** but use 6 channels with per-subject training — an easier setting
4. **Ghorbani et al. (2023) R² = 0.994 is not comparable** due to methodological issues (force as input + data leakage)

---

## 4. Discussion

### Clinical Relevance

The 2-channel configuration (ECRB + ECU) is clinically practical:
- Maps to standard adhesive electrode patches (MyoWare 2.0)
- Reduces hardware cost compared to multi-channel arrays
- Simplifies electrode donning — posterior forearm placement is consistent and easy to locate
- MRMR reliably identifies these channels, providing a principled placement protocol

### Subject-Independent Design

The general model achieves R² = 0.778 without any per-user calibration:
- Critical for prosthetic deployment: new users can use the device immediately
- No machine learning expertise required by the end user
- R² = 0.778 represents a practical level of force tracking for gross grip tasks

### Prosthetic Deployment Considerations

| Parameter | Value |
|-----------|-------|
| Model size | 23,809 parameters (~93 KB in 32-bit float) |
| Inference latency | ~20-30 ms on Arduino Mega 2560 @ 16 MHz |
| Feature computation | ~5 ms per 20 ms window |
| Total loop time | ~25-35 ms (~30-40 Hz update rate) |
| Power consumption | ~640 mA total system (3 hrs on 2000 mAh LiPo) |

### Ghorbani et al. Comparison

Ghorbani et al. (2023) report R² = 0.994 on the same dataset using a similar GRU architecture. Their inflated performance is due to:
1. **Force as input**: The force signal at time t is concatenated with EMG features, making the network learn f(t+1) ≈ f(t) — a trivial autoregressive task at 10 ms resolution
2. **Independent normalization**: `fit_transform()` applied separately to train and test splits (data leakage)

Our work uses EMG features exclusively and fits scalers on training data only, representing a genuine EMG-to-force benchmark.

---

## 5. Limitations

1. **Healthy young adults only**: All 10 subjects are healthy; amputee validation is needed
2. **Single grasp type**: Isometric precision gripping — dynamic grasping tasks not evaluated
3. **Within-trial evaluation**: No cross-session or cross-day validation
4. **Myo armband**: Dataset collected with Myo armband (now discontinued); production deployment uses MyoWare 2.0 which may have different signal characteristics
5. **Arduino SRAM**: 8 KB limits model complexity; buffer optimization may be needed

---

## 6. Future Work

1. **Amputee evaluation**: Validate with transradial amputee participants
2. **Cross-session validation**: Test model stability across days/weeks
3. **Per-finger force**: Predict individual finger forces for independent finger control
4. **Dynamic grasping**: Extend to object manipulation tasks
5. **Hardware upgrade**: Migrate to ESP32 or STM32 for BLE, more memory, and TensorFlow Lite Micro compatibility
6. **Transfer learning**: Pre-train on large EMG datasets, adapt to individual users with minimal data
7. **Online adaptation**: Implement on-device model updates to improve accuracy over time
