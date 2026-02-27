# Hardware Design

## 1. System Architecture

```
+--------------+    Bluetooth     +--------------+
|  Mobile App  |<---------------->|  HC-05/HC-06 |
| (Calibration)|    (Serial)      |   Module     |
+--------------+                  +------+-------+
                                         | TX1/RX1
                                         |
+--------------+    Analog        +------+-------+    PWM         +--------------+
|  MyoWare 2.0 |---------------->|              |-------------->| Thumb Actuator|
|  ECRB (Ch5)  |    A0           |              |    D3         +--------------+
+--------------+                  |              |
                                  |   Arduino    |    PWM         +--------------+
+--------------+    Analog        |   Mega 2560  |-------------->| Index Actuator|
|  MyoWare 2.0 |---------------->|              |    D5         +--------------+
|  ECU  (Ch3)  |    A1           |              |
+--------------+                  |  (GRU Model  |    PWM         +--------------+
                                  |   Inference)  |-------------->|Middle Actuator|
                                  |              |    D6         +--------------+
                                  |              |
                                  |              |    PWM         +--------------+
                                  |              |-------------->| Ring Actuator |
                                  |              |    D9         +--------------+
                                  |              |
                                  |              |    PWM         +--------------+
                                  |              |-------------->|Little Actuator|
                                  +--------------+    D10        +--------------+
```

## 2. InMoov Hand Design

The InMoov is an open-source, 3D-printable humanoid robot designed by Gael Langevin (2017). The hand assembly is used as the prosthetic platform in this project.

### Key Specifications

| Property | Value |
|----------|-------|
| Material | PLA or PETG (3D printed) |
| Degrees of freedom | 5 (one per finger) |
| Actuation | Tendon-driven via linear actuators |
| Weight (hand only) | ~300-400g |
| Grip force | Up to ~5N per finger (actuator dependent) |
| Design files | https://inmoov.fr |

### Actuation Mechanism

Each finger is controlled by a nylon string (tendon) connected to a linear actuator:
- Linear actuator extends → nylon string pulls → finger flexes (grips)
- Linear actuator retracts → spring returns finger to open position
- Grip force is proportional to actuator extension (controlled by PWM duty cycle)

The GRU model output (0-1 normalized force) maps directly to PWM values (0-255), providing continuous proportional grip force control.

## 3. EMG Sensors

### MyoWare 2.0 Muscle Sensors

| Property | Value |
|----------|-------|
| Output | Analog envelope (0-Vs) or raw EMG |
| Supply voltage | 3.3V or 5V |
| Gain | Adjustable via onboard potentiometer |
| Electrodes | Snap-on Ag/AgCl disposable |
| Size | 52.3 x 20.5 mm |
| Interface | Single analog output pin |

### Electrode Placement

Based on MRMR channel selection results and forearm anatomy:

**Sensor 1 (A0) — ECRB (Extensor Carpi Radialis Brevis)**
- Location: Lateral posterior forearm, ~5 cm distal to lateral epicondyle
- Muscle function: Wrist extension + radial deviation
- Grip relevance: Highest MI with grip force (0.077), primary wrist stabilizer during grip

**Sensor 2 (A1) — ECU (Extensor Carpi Ulnaris)**
- Location: Medial posterior forearm, over ulnar border
- Muscle function: Wrist extension + ulnar deviation
- Grip relevance: Complementary to ECRB, provides ulnar stabilization

Both electrodes are placed over the muscle bellies at the proximal third of the forearm, aligned parallel to muscle fiber direction, following standard sEMG placement guidelines (De Luca, 1997; Criswell, 2011).

## 4. Linear Actuators

### Specifications

| Property | Recommended Value |
|----------|------------------|
| Type | L12 Actuonix or equivalent micro linear actuator |
| Stroke | 30-50mm |
| Force | 20-50N |
| Speed | 10-20mm/s |
| Voltage | 6V or 12V |
| Control | PWM via motor driver (L298N or DRV8833) |
| Feedback | Built-in potentiometer (optional) |

### Wiring

Each actuator requires:
1. **PWM signal** from Arduino (pins D3, D5, D6, D9, D10)
2. **Motor driver** (L298N or DRV8833) between Arduino and actuator
3. **External power supply** (do NOT power actuators from Arduino 5V)

### Motor Driver Configuration

```
Arduino PWM Pin --> Motor Driver IN --> Actuator +
Arduino GND     --> Motor Driver GND
External 6-12V  --> Motor Driver VCC --> Actuator -
```

One L298N module can drive 2 actuators (dual H-bridge). Three modules are needed for 5 actuators.

## 5. Arduino Mega 2560

### Why Mega 2560

| Requirement | Mega 2560 Capability |
|-------------|---------------------|
| PWM outputs (5 needed) | 15 PWM pins available |
| Analog inputs (2 needed) | 16 analog inputs |
| Serial ports (2 needed: USB + BT) | 4 hardware serial ports |
| Flash memory | 256 KB (sufficient for GRU weights ~93 KB) |
| SRAM | 8 KB (sufficient for inference buffers ~7.7 KB) |
| Clock speed | 16 MHz (sufficient for ~50 Hz inference) |

### Pin Assignment

| Pin | Function | Connected To |
|-----|----------|-------------|
| A0 | Analog In | MyoWare Sensor 1 (ECRB) |
| A1 | Analog In | MyoWare Sensor 2 (ECU) |
| D3 | PWM Out | Thumb actuator (via motor driver) |
| D5 | PWM Out | Index actuator (via motor driver) |
| D6 | PWM Out | Middle actuator (via motor driver) |
| D9 | PWM Out | Ring actuator (via motor driver) |
| D10 | PWM Out | Little actuator (via motor driver) |
| TX1 (D18) | Serial1 TX | HC-05 RXD |
| RX1 (D19) | Serial1 RX | HC-05 TXD |
| 5V | Power | EMG sensors, HC-05 VCC |
| GND | Ground | Common ground |
| VIN | Power In | 7.4V LiPo battery |

### Memory Budget

| Component | Flash (bytes) | SRAM (bytes) |
|-----------|---------------|--------------|
| GRU weights (23,809 floats x 4) | ~93,000 | 0 (PROGMEM) |
| Sequence buffer (50 x 36 x 4) | - | 7,200 |
| Hidden state (64 x 4) | - | 256 |
| Dense intermediate (64 x 4) | - | 256 |
| Arduino core + libs | ~15,000 | ~2,000 |
| **Total** | **~108,000** | **~9,712** |
| **Available** | **262,144** | **8,192** |

**Note**: SRAM is tight. The GRU weights MUST be stored in PROGMEM (Flash) and read with `pgm_read_float_near()`. The sequence buffer may need to be optimized (e.g., circular buffer storing only the latest window rather than all 50 timesteps simultaneously).

## 6. Bluetooth Module (HC-05)

### Wiring

| HC-05 Pin | Arduino Pin |
|-----------|-------------|
| VCC | 5V |
| GND | GND |
| TXD | RX1 (D19) |
| RXD | TX1 (D18) via voltage divider* |

*HC-05 RXD is 3.3V logic. Use a voltage divider (1k + 2k) to step down from Arduino's 5V TX1.

### Configuration

| Parameter | Value |
|-----------|-------|
| Baud rate | 9600 (default) |
| Pairing PIN | 1234 (default) |
| Mode | Slave (waits for phone connection) |

### Calibration Protocol

Via Bluetooth, a mobile app sends calibration commands:
1. **REST**: User relaxes — records baseline EMG offset
2. **MVC**: User performs maximum voluntary contraction — records scale factor
3. **VERIFY**: User grips at ~50% — confirms calibration accuracy
4. Parameters stored in Arduino EEPROM for persistence across power cycles

## 7. Power Supply

### Recommended: 7.4V 2S LiPo Battery

| Property | Value |
|----------|-------|
| Voltage | 7.4V nominal (8.4V charged) |
| Capacity | 2000-3000 mAh |
| Discharge rate | 10C minimum |
| Connection | XT60 connector to Arduino VIN |

### Estimated Battery Life

| Component | Current Draw |
|-----------|-------------|
| Arduino Mega | ~80 mA |
| 2x MyoWare sensors | ~20 mA |
| HC-05 | ~40 mA |
| 5x Actuators (avg) | ~500 mA |
| **Total** | **~640 mA** |

With a 2000 mAh battery: ~3 hours of continuous operation.

## 8. Assembly Checklist

1. Print InMoov hand and forearm components (PLA, 0.2mm layer height)
2. Install 5 linear actuators with nylon tendon attachments
3. Mount Arduino Mega in forearm electronics bay
4. Wire 3x motor drivers (L298N) to Arduino PWM pins
5. Connect actuators to motor driver outputs
6. Attach 2x MyoWare 2.0 sensors to forearm (ECRB and ECU positions)
7. Wire EMG analog outputs to A0 (ECRB) and A1 (ECU)
8. Connect HC-05 to Serial1 (TX1/RX1) with voltage divider
9. Upload firmware (`emg_force_inference.ino`) with trained model weights
10. Connect LiPo battery to VIN
11. Pair phone with HC-05 and run calibration app
12. Perform calibration sequence (REST -> MVC -> VERIFY)
