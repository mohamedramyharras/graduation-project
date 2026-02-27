# Mobile Calibration App — Design Specification

## Overview

A Bluetooth-based mobile application that calibrates the EMG-to-Force prosthetic hand
for each individual user. The app communicates with the Arduino Mega via HC-05/HC-06
Bluetooth module to adjust force prediction parameters in real-time.

## Why Calibration Is Needed

EMG signals vary significantly between individuals due to:
- Differences in muscle mass and skin impedance
- Electrode placement variations
- Limb anatomy and residual muscle strength (for amputees)
- Sensor gain and environmental noise

A per-user calibration step maps the model's raw force predictions to the user's
actual grip force range, ensuring accurate and comfortable prosthetic control.

## Calibration Protocol

### Step 1: Rest Baseline
1. User relaxes forearm muscles completely
2. App records 5 seconds of resting EMG → computes `CAL_OFFSET`
3. This offset is subtracted from all future predictions to zero out noise

### Step 2: Maximum Voluntary Contraction (MVC)
1. User grips a dynamometer or squeezes as hard as comfortable
2. App records 5 seconds of maximum grip EMG → computes `FORCE_MAX`
3. All predictions are normalized to this maximum

### Step 3: Proportional Control Verification
1. App displays a target force level (e.g., 25%, 50%, 75%)
2. User tries to match the target by adjusting grip strength
3. App computes `CAL_SCALE` to linearize the response
4. Visual feedback shows predicted force vs target in real-time

### Step 4: Save and Apply
1. Calibration parameters are sent to Arduino via Bluetooth
2. Parameters are optionally saved to phone storage for quick reload
3. User can recalibrate at any time

## Bluetooth Communication Protocol

The app communicates with the Arduino over Serial1 (9600 baud) using
simple text commands terminated by newline (`\n`).

### Commands (App → Arduino)

| Command | Description | Example |
|---------|-------------|---------|
| `CAL_SCALE:<float>` | Set force scaling factor | `CAL_SCALE:1.35` |
| `CAL_OFFSET:<float>` | Set force offset (baseline) | `CAL_OFFSET:-0.02` |
| `FORCE_MAX:<float>` | Set maximum force value | `FORCE_MAX:0.85` |
| `STATUS` | Request current parameters | `STATUS` |
| `RESET` | Reset to factory defaults | `RESET` |

### Responses (Arduino → App)

| Response | Description | Example |
|----------|-------------|---------|
| `OK:SCALE=<float>` | Scale updated | `OK:SCALE=1.35` |
| `OK:OFFSET=<float>` | Offset updated | `OK:OFFSET=-0.02` |
| `OK:FMAX=<float>` | Force max updated | `OK:FMAX=0.85` |
| `SCALE=<f>,OFFSET=<f>,FMAX=<f>` | Status report | `SCALE=1.35,OFFSET=-0.02,FMAX=0.85` |
| `OK:RESET` | Parameters reset | `OK:RESET` |

## App UI Mockup

```
┌──────────────────────────────┐
│   EMG Prosthetic Calibrator  │
├──────────────────────────────┤
│                              │
│  Connection: ● Connected     │
│  Device: HC-05 (98:D3:...)   │
│                              │
│  ┌────────────────────────┐  │
│  │  Live Force:  ████░░  │  │
│  │  Predicted:   0.62     │  │
│  │  Target:      0.50     │  │
│  └────────────────────────┘  │
│                              │
│  [1. Record Rest Baseline]   │
│  [2. Record Max Grip     ]   │
│  [3. Verify Proportional ]   │
│  [4. Save Calibration    ]   │
│                              │
│  ── Current Parameters ──    │
│  Scale:   1.35               │
│  Offset: -0.02               │
│  Max:     0.85               │
│                              │
│  [Reset to Defaults]         │
│                              │
└──────────────────────────────┘
```

## Recommended Technology Stack

| Component | Option |
|-----------|--------|
| Framework | MIT App Inventor (simplest), Flutter, or React Native |
| Bluetooth | Classic Bluetooth SPP (Serial Port Profile) |
| Module | HC-05 or HC-06 paired with phone |
| Storage | SharedPreferences (Android) or UserDefaults (iOS) |

**MIT App Inventor** is recommended for graduation projects as it allows
rapid prototyping without extensive mobile development experience.

## Implementation Notes

- The HC-05 module must be pre-paired with the phone (default PIN: 1234)
- Baud rate must match Arduino Serial1 configuration (9600)
- Commands must end with `\n` (newline) for `readStringUntil('\n')` parsing
- The app should implement a 200ms debounce on parameter updates to avoid
  flooding the serial buffer
- Force readings can be streamed from Arduino to app by adding a
  `STREAM:ON/OFF` command (not yet implemented in the Arduino sketch)
