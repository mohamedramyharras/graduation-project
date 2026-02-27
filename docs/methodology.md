# Methodology

## 1. System Architecture

The complete system pipeline for prosthetic grip force control:

```
MyoWare 2.0 → Signal Processing → Feature Extraction → GRU Model → Smoothing → Arduino Mega → InMoov Hand
(2 channels)    (RMS envelope)     (36 features)       (23,809 params)  (4 Hz LP)    (PWM)         (actuators)
```

### Hardware Components
- **EMG Sensors**: 2x MyoWare 2.0 (production deployment) or Myo armband channels 3 and 5 (dataset validation)
- **Microcontroller**: Arduino Mega 2560 (256 KB Flash, 8 KB SRAM)
- **Actuators**: 5 linear actuators with nylon string tendons
- **Prosthetic**: InMoov 3D-printed hand (open-source design by Gael Langevin)
- **Communication**: HC-05 Bluetooth module for calibration

---

## 2. Forearm Anatomy and Electrode Placement

### Target Muscles

The two selected EMG channels correspond to muscles in the posterior forearm compartment:

**ECRB (Extensor Carpi Radialis Brevis) — Channel 5**
- **Origin**: Lateral epicondyle of humerus (common extensor origin)
- **Insertion**: Base of 3rd metacarpal (dorsal surface)
- **Function**: Wrist extension and radial deviation
- **Grip role**: Co-activates during power grip to stabilize the wrist in slight extension — the "extensor paradox" where extensors fire during a gripping (flexion) task

**ECU (Extensor Carpi Ulnaris) — Channel 3**
- **Origin**: Lateral epicondyle of humerus + posterior border of ulna
- **Insertion**: Base of 5th metacarpal
- **Function**: Wrist extension and ulnar deviation
- **Grip role**: Provides ulnar stabilization during grip, complementing ECRB's radial action

### Why These Muscles for Grip Force?

During power grip, the finger flexors (FDS, FDP) generate the primary gripping force. However, the wrist extensors (ECRB, ECU) must co-activate to prevent the wrist from flexing under the flexor load. This "extensor paradox" means that grip force is reliably encoded in the extensor EMG signals, often with less noise than flexor recordings (which are deeper and more prone to cross-talk).

### Electrode Placement Protocol

1. Place electrodes over the muscle bellies at the proximal third of the forearm
2. Align electrodes parallel to muscle fiber direction
3. ECRB electrode: lateral posterior forearm, approximately 5 cm distal to lateral epicondyle
4. ECU electrode: medial posterior forearm, over the ulnar border
5. Standard Ag/AgCl disposable electrodes with conductive gel

---

## 3. Feature Extraction

### 3.1 RMS Envelope

The pre-filtered EMG signal (Myo armband provides bandpass-filtered output at 200 Hz) is processed using a sliding window:

- **Window size**: 100 ms (20 samples at 200 Hz)
- **Hop size**: 20 ms (4 samples) — 80% overlap
- **Output rate**: 50 Hz feature vectors

### 3.2 Time-Domain Features

Six features are computed per channel per window (Phinyomark et al., 2012):

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| RMS | sqrt(mean(x²)) | Signal power / amplitude |
| MAV | mean(\|x\|) | Mean absolute amplitude |
| WL | sum(\|x[i+1] - x[i]\|) | Waveform complexity |
| VAR | mean(x²) | Signal variance |
| ZC | count(x[i] * x[i+1] < 0) | Zero crossings (frequency info) |
| SSC | count of slope sign changes | Spectral content indicator |

### 3.3 Delta Augmentation

Each feature vector is augmented with temporal derivatives:

- **Base features**: 2 channels x 6 features = 12
- **Delta (1st order)**: Difference between consecutive windows = 12
- **Delta-delta (2nd order)**: Difference of deltas = 12
- **Total**: 36 features per time step

Delta features explicitly encode the rate of change of muscle activation, capturing the electromechanical dynamics of the EMG-to-force relationship.

---

## 4. MRMR Channel Selection

### Algorithm: Minimum Redundancy Maximum Relevance (Peng et al., 2005)

MRMR balances two objectives:
1. **Maximum Relevance**: Each selected channel should have high mutual information (MI) with grip force
2. **Minimum Redundancy**: Selected channels should have low MI with each other

The MRMR score for candidate channel c given already-selected set S:

```
MRMR(c) = I(c; force) - (1/|S|) * sum(I(c; s))  for s in S
```

### Selection Results

From 8 Myo armband channels, MRMR selects:
1. **Ch5 (ECRB)**: Highest relevance (MI = 0.077), highest MRMR score (0.065)
2. **Ch3 (ECU)**: Good relevance (MI = 0.030), low redundancy with ECRB (MRMR = 0.019)

---

## 5. GRU Neural Network

### Architecture

```
Input: (batch, 50, 36)  — 50 timesteps x 36 features = 1.0 second context
  |
GRU(input_size=36, hidden_size=64, num_layers=1)
  |
Last hidden state → ReLU → Dropout(0.2)
  |
FC(64, 64) → ReLU → Dropout(0.2)
  |
FC(64, 1) → Predicted grip force
```

**Total parameters**: 23,809 (embedded-deployable on Arduino Mega)

### Why GRU?

- **Temporal modeling**: Captures the 30-100 ms electromechanical delay between EMG activation and force production
- **Compact**: Fewer parameters than LSTM (no separate cell state), fits in microcontroller memory
- **Causal**: No bidirectional processing — compatible with real-time inference

---

## 6. Training Procedure

### Subject-Independent (General) Training

A single model is trained on pooled normalized data from all 9 primary subjects:

| Parameter | Value |
|-----------|-------|
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam (lr=0.001, weight_decay=5e-5) |
| Batch size | 512 |
| Max epochs | 50 |
| Early stopping | Patience = 12 (on validation loss) |
| Gradient clipping | Max norm = 1.0 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Random seed | 42 |

### Key Design Choice: No Per-User Calibration

The model is trained once on pooled data and used directly for all subjects without fine-tuning. This subject-independent approach is critical for prosthetic deployment because:
- New users can start immediately without a calibration session
- No machine learning expertise required by the end user
- Consistent behavior across sessions
- Simpler embedded implementation

---

## 7. Prediction Smoothing

Raw model predictions are post-processed with a zero-phase Butterworth low-pass filter:

- **Filter**: 2nd-order Butterworth
- **Cutoff**: 4 Hz
- **Implementation**: scipy.signal.sosfiltfilt (zero-phase, no group delay)

This removes high-frequency prediction jitter while preserving the bandwidth of realistic grip force trajectories (voluntary force changes are typically < 3-4 Hz).

---

## 8. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² | 1 - SS_res/SS_tot | Proportion of variance explained (1.0 = perfect) |
| RMSE | sqrt(mean((y - y_hat)²)) | Average prediction error magnitude |
| NRMSE% | RMSE / (y_max - y_min) x 100 | Normalized error as percentage |
| MAE | mean(\|y - y_hat\|) | Average absolute error |
| Pearson r | corr(y, y_hat) | Linear correlation |

All metrics computed on held-out test set (last 20% of each subject, temporal split).

---

## 9. Embedded Deployment

### Model Export

Trained PyTorch weights are exported as C float arrays in `model_weights.h`:
- GRU weight matrices (input-hidden and hidden-hidden)
- GRU bias vectors
- Dense layer weights and biases
- MinMaxScaler parameters

### Real-Time Inference on Arduino Mega 2560

1. Sample 2 EMG channels via ADC (analog pins A0, A1)
2. Compute feature windows every 20 ms
3. Maintain sliding buffer of 50 feature vectors
4. Run GRU forward pass (matrix multiplications + sigmoid/tanh activations)
5. Apply Butterworth smoothing
6. Map predicted force (0-1) to PWM signals (0-255) for linear actuators

### Memory Budget

| Component | SRAM (bytes) |
|-----------|-------------|
| GRU weights (PROGMEM) | 0 (stored in Flash) |
| Sequence buffer (50 x 36 x 4) | 7,200 |
| Hidden state (64 x 4) | 256 |
| Dense intermediate (64 x 4) | 256 |
| Arduino core + libs | ~2,000 |
| **Total SRAM** | **~7,712** |
| **Available SRAM** | **8,192** |

GRU weights stored in PROGMEM (Flash, 256 KB available) using `pgm_read_float_near()`.

---

## References

- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. arXiv:1406.1078.
- Ghorbani, A., et al. (2023). Estimation and Early Prediction of Grip Force. arXiv:2302.09555.
- Peng, H., et al. (2005). Feature selection based on mutual information. IEEE TPAMI, 27(8), 1226-1238.
- Phinyomark, A., et al. (2012). Feature reduction and selection for EMG signal classification. Expert Syst. Appl., 39(8), 7420-7431.
- Criswell, E. (2011). Cram's Introduction to Surface Electromyography. Jones & Bartlett Learning.
- Langevin, G. (2017). InMoov: Open-Source 3D Printed Life-Size Robot. https://inmoov.fr
