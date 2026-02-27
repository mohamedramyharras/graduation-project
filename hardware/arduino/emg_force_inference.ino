/*
 * emg_force_inference.ino
 * =======================
 * Arduino Mega 2560 firmware for the 2-channel sEMG prosthetic hand system.
 *
 * Role in the split-processing architecture:
 *   1. Sample two MyoWare 2.0 EMG channels at 2000 Hz (ADC)
 *   2. Send raw ADC values to Raspberry Pi 3 over USB serial (115200 baud)
 *   3. Receive predicted grip force [0,1] from Raspberry Pi 3
 *   4. Map force to PWM and drive 5 linear actuators for InMoov fingers
 *   5. Handle optional Bluetooth calibration commands
 *
 * The Raspberry Pi 3 handles all computation:
 *   feature extraction → 36-D vector → GRU inference → Butterworth filter
 *
 * Wiring:
 *   A0       : EMG Channel 1 (ECRB, Ch5 in Myo labeling)
 *   A1       : EMG Channel 2 (ECU,  Ch3 in Myo labeling)
 *   D3  (PWM): Thumb actuator
 *   D5  (PWM): Index actuator
 *   D6  (PWM): Middle actuator
 *   D9  (PWM): Ring actuator
 *   D10 (PWM): Little actuator
 *   TX1/RX1  : HC-05 Bluetooth module (Serial1, 9600 baud)
 *   USB      : Raspberry Pi 3 (Serial, 115200 baud)
 *
 * Serial protocol (USB, 115200 baud):
 *   Arduino → Pi : "E,<ch1_raw>,<ch2_raw>\n"  (every 500 µs = 2000 Hz)
 *   Pi → Arduino : "F,<force_0_to_1>\n"        (every ~50 ms)
 *
 * Author : [Your Name]
 * Date   : 2026
 */

// ─── Pin Configuration ────────────────────────────────────────────
#define EMG_CH1_PIN   A0
#define EMG_CH2_PIN   A1

#define THUMB_PIN      3
#define INDEX_PIN      5
#define MIDDLE_PIN     6
#define RING_PIN       9
#define LITTLE_PIN    10

// ─── Sampling ─────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ  2000            // 2 kHz raw EMG
#define SAMPLE_PERIOD_US (1000000UL / SAMPLE_RATE_HZ)  // 500 µs

// ─── Force state ──────────────────────────────────────────────────
volatile float grip_force = 0.0f;       // Latest force from Pi [0, 1]
float cal_scale  = 1.0f;
float cal_offset = 0.0f;

// ─── Timing ───────────────────────────────────────────────────────
unsigned long last_sample_us = 0;

// ═══════════════════════════════════════════════════════════════════
void setup() {
    // USB serial to Raspberry Pi 3
    Serial.begin(115200);

    // Bluetooth module for calibration
    Serial1.begin(9600);

    // Actuator PWM pins
    pinMode(THUMB_PIN,  OUTPUT);
    pinMode(INDEX_PIN,  OUTPUT);
    pinMode(MIDDLE_PIN, OUTPUT);
    pinMode(RING_PIN,   OUTPUT);
    pinMode(LITTLE_PIN, OUTPUT);

    // Ensure all actuators start at rest
    drive_actuators(0.0f);

    Serial.println("READY");
}

// ═══════════════════════════════════════════════════════════════════
void loop() {
    unsigned long now_us = micros();

    // ── 1. Sample EMG at 2000 Hz and stream to Pi ─────────────────
    if (now_us - last_sample_us >= SAMPLE_PERIOD_US) {
        last_sample_us = now_us;

        int ch1 = analogRead(EMG_CH1_PIN);   // 0-1023
        int ch2 = analogRead(EMG_CH2_PIN);   // 0-1023

        // Send compact CSV line: E,<ch1>,<ch2>
        Serial.print('E');
        Serial.print(',');
        Serial.print(ch1);
        Serial.print(',');
        Serial.println(ch2);
    }

    // ── 2. Receive force prediction from Pi ───────────────────────
    if (Serial.available()) {
        // Expected format: "F,<float>\n"
        char tag = Serial.read();
        if (tag == 'F' && Serial.read() == ',') {
            float f = Serial.parseFloat();
            grip_force = constrain(f * cal_scale + cal_offset, 0.0f, 1.0f);
            drive_actuators(grip_force);
        }
    }

    // ── 3. Bluetooth calibration commands ─────────────────────────
    handle_bluetooth();
}

// ─── Drive Actuators ─────────────────────────────────────────────
/*
 * Maps normalized grip force [0,1] to PWM [0,255] for all five
 * finger actuators.  All fingers move together for power grip;
 * extend this with per-finger gains for individual finger control.
 */
void drive_actuators(float force) {
    int pwm = constrain((int)(force * 255.0f), 0, 255);
    analogWrite(THUMB_PIN,  pwm);
    analogWrite(INDEX_PIN,  pwm);
    analogWrite(MIDDLE_PIN, pwm);
    analogWrite(RING_PIN,   pwm);
    analogWrite(LITTLE_PIN, pwm);
}

// ─── Bluetooth Calibration Handler ───────────────────────────────
/*
 * Accepts simple text commands over Serial1 (HC-05):
 *   SCALE:<float>   — multiply raw force by scale factor
 *   OFFSET:<float>  — add constant offset
 *   RESET           — restore defaults
 *   STATUS          — print current parameters
 */
void handle_bluetooth() {
    if (!Serial1.available()) return;

    String cmd = Serial1.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("SCALE:")) {
        cal_scale = cmd.substring(6).toFloat();
        Serial1.print("OK:SCALE="); Serial1.println(cal_scale);
    }
    else if (cmd.startsWith("OFFSET:")) {
        cal_offset = cmd.substring(7).toFloat();
        Serial1.print("OK:OFFSET="); Serial1.println(cal_offset);
    }
    else if (cmd == "RESET") {
        cal_scale = 1.0f; cal_offset = 0.0f;
        Serial1.println("OK:RESET");
    }
    else if (cmd == "STATUS") {
        Serial1.print("SCALE=");  Serial1.print(cal_scale);
        Serial1.print(",OFFSET="); Serial1.println(cal_offset);
    }
}
