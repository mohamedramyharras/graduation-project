"""
EMG-to-Force Prediction: Central Configuration
All paths, hyperparameters, and constants in one place.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_NINAPRO_DIR = DATA_DIR / "raw"
RAW_HYSER_DIR = DATA_DIR / "hyser"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# ── NinaPro DB2 ───────────────────────────────────────────────────────
NINAPRO_FS = 2000                # Sampling rate (Hz)
NINAPRO_EMG_CHANNELS = 10       # Channels 1-10 (11-12 biceps/triceps excluded)
NINAPRO_FORCE_CHANNELS = 6      # Summed to total grip force
NINAPRO_SUBJECTS = 10

NINAPRO_CHANNEL_NAMES = [
    "Forearm_1", "Forearm_2", "Forearm_3", "Forearm_4",
    "Forearm_5", "Forearm_6", "Forearm_7", "Forearm_8",
    "Flexor_Digitorum", "Extensor_Digitorum",
]

# Subject metadata: (age, gender, height_cm, weight_kg, handedness)
# gender: 0=male, 1=female | handedness: 0=right, 1=left
NINAPRO_SUBJECT_META = {
    1:(29,0,187,75,0), 2:(29,0,183,75,0), 3:(31,0,174,69,0), 4:(30,1,154,50,1),
    5:(25,0,175,70,0), 6:(35,0,172,79,0), 7:(27,0,187,92,0), 8:(45,0,173,73,0),
    9:(23,0,172,63,0), 10:(34,0,173,84,0), 11:(32,1,150,54,0), 12:(29,0,184,90,0),
    13:(30,0,182,70,1), 14:(30,1,173,59,0), 15:(30,0,169,58,0), 16:(34,0,173,76,0),
    17:(29,0,175,70,0), 18:(30,1,169,90,0), 19:(31,1,158,52,0), 20:(26,1,155,52,0),
    21:(32,0,170,75,0), 22:(28,1,162,54,1), 23:(25,0,170,66,0), 24:(28,0,170,73,0),
    25:(31,0,168,70,1), 26:(30,0,186,90,1), 27:(29,0,170,65,0), 28:(29,1,160,61,0),
    29:(27,0,171,64,0), 30:(30,0,173,68,0), 31:(29,0,185,98,0), 32:(28,0,173,72,0),
    33:(25,0,183,71,0), 34:(31,0,192,78,0), 35:(24,1,170,52,0), 36:(27,1,155,44,0),
    37:(34,0,190,105,0), 38:(30,1,163,62,0), 39:(31,0,183,96,0), 40:(31,0,173,65,0),
}

# ── Hyser (PhysioNet) ─────────────────────────────────────────────────
HYSER_FS = 2048                  # EMG sampling rate (Hz)
HYSER_FORCE_FS = 100             # Force sampling rate (Hz)
HYSER_EMG_CHANNELS = 256         # HD-sEMG grid
HYSER_FORCE_CHANNELS = 5        # Per-finger: thumb, index, middle, ring, little
HYSER_SUBJECTS = 20
HYSER_SESSIONS = 2
HYSER_FINGERS = 5
HYSER_SAMPLES_PER_FINGER = 3

# ── Ghorbani Dataset (arXiv:2302.09555) ────────────────────────────────
# Grip force prediction using Myo armband (8 channels) @ 200 Hz
# Reference: Ghorbani et al., "Estimation and Early Prediction of 
#            Grip Force Based on sEMG Signals and Deep RNNs", 2023
GHORBANI_FS = 200                # EMG sampling rate (Hz) - Myo armband
GHORBANI_EMG_CHANNELS = 8        # Myo armband channels (emg0-emg7)
GHORBANI_FORCE_CHANNELS = 1      # Single-axis grip force (Fz from ATI F/T sensor)
GHORBANI_SUBJECTS = 10           # 10 healthy subjects (9 male, 1 female)
GHORBANI_TRIALS_PER_SUBJECT = 3  # 3 recordings per subject
RAW_GHORBANI_DIR = DATA_DIR / "ghorbani_raw" / "Dataset" / "Filtered"

# Channel names for Myo armband (anatomically ordered around forearm)
GHORBANI_CHANNEL_NAMES = [
    "Myo_Ch0_FCR",     # Flexor Carpi Radialis region
    "Myo_Ch1_PL",      # Palmaris Longus region
    "Myo_Ch2_FCU",     # Flexor Carpi Ulnaris region
    "Myo_Ch3_ECU",     # Extensor Carpi Ulnaris region
    "Myo_Ch4_ED",      # Extensor Digitorum region
    "Myo_Ch5_ECRB",    # Extensor Carpi Radialis Brevis region
    "Myo_Ch6_BR",      # Brachioradialis region
    "Myo_Ch7_PT",      # Pronator Teres region
]

# Ghorbani subject demographic info (age, gender: 0=M/1=F)
GHORBANI_SUBJECT_META = {
    1: (24, 0), 2: (23, 0), 3: (25, 0), 4: (24, 0), 5: (22, 0),
    6: (24, 0), 7: (23, 1), 8: (25, 0), 9: (24, 0), 10: (24, 0),
}

# ── Preprocessing ─────────────────────────────────────────────────────
BANDPASS_LOW = 20                # Hz
BANDPASS_HIGH = 450              # Hz (NinaPro/Hyser - adjust for lower fs datasets)
BANDPASS_ORDER = 4
ENVELOPE_WINDOW_MS = 25          # RMS envelope window (ms)
ENVELOPE_HOP_MS = 25             # Non-overlapping hop (ms)
RMS_WINDOW_MS = ENVELOPE_WINDOW_MS   # Alias used by preprocess module
RMS_HOP_MS = ENVELOPE_HOP_MS        # Alias used by preprocess module
PRED_HORIZON = 1                     # Prediction horizon (1 = next step)

# Ghorbani-specific: Data is pre-filtered, use lower high cutoff if filtering needed
# Ghorbani Nyquist = 100 Hz, so max high cutoff is ~95 Hz
GHORBANI_BANDPASS_HIGH = 95      # Hz (safe margin below Nyquist)
GHORBANI_DATA_PREFILTERED = True # Dataset comes pre-filtered, skip bandpass

# Ghorbani-specific preprocessing (200 Hz needs larger RMS window for smooth envelope)
# At 200 Hz, 25ms window = only 5 samples (too noisy); 100ms = 20 samples (good balance)
GHORBANI_RMS_WINDOW_MS = 100     # 20 samples at 200 Hz - balances smoothing vs detail
GHORBANI_RMS_HOP_MS = 20         # 4 samples hop -> 80% overlap for smooth envelope

# ── MRMR ──────────────────────────────────────────────────────────────
MRMR_N_SELECT = 2                # Number of channels to select
MRMR_SUBSAMPLE = 20000           # Max samples for MI computation (speed)

# ── GRU Model ─────────────────────────────────────────────────────────
SEQ_LEN = 50                    # Timesteps per sequence (50 x 25ms = 1.25s)
GRU_HIDDEN_SIZE = 50
GRU_DENSE_SIZE = 100
GRU_INPUT_SIZE = 2               # After MRMR selection
GRU_OUTPUT_SIZE = 1              # Total grip force

# ── Training ──────────────────────────────────────────────────────────
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 25
VAL_SPLIT = 0.15                 # 15% of training for validation
TEST_SPLIT = 0.20                # Last 20% of each subject = test
TEST_RATIO = TEST_SPLIT              # Alias used by preprocess module
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42
