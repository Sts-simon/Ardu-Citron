#include <ESP32Servo.h>
#include <math.h>
#include "config.h"
#include "imu.h"
#include "kalman.h"
#include "pid.h"
#include "mixer.h"
#include "failsafe.h"
#include "buzzer.h"
#include "flight_mode.h"
#include "imu_health.h"
#include "protocol.h"
#include "rc_input.h"
#include "servo_output.h"

// ------------------------------------------------------------
// Instances globales des modules
// ------------------------------------------------------------
IMU imu;
Buzzer buzzer;
Failsafe failsafe;

KalmanFilter kalmanRoll (KALMAN_Q_ANGLE, KALMAN_Q_BIAS, KALMAN_R_MEASURE);
KalmanFilter kalmanPitch(KALMAN_Q_ANGLE, KALMAN_Q_BIAS, KALMAN_R_MEASURE);
KalmanFilter kalmanYaw  (KALMAN_Q_ANGLE, KALMAN_Q_BIAS, KALMAN_R_MEASURE);

PID pidRoll (Kp_roll,  Ki_roll,  Kd_roll,  PID_INTEGRAL_LIMIT, PID_OUTPUT_LIMIT, PID_DERIVATIVE_ALPHA);
PID pidPitch(Kp_pitch, Ki_pitch, Kd_pitch, PID_INTEGRAL_LIMIT, PID_OUTPUT_LIMIT, PID_DERIVATIVE_ALPHA);
PID pidYaw  (Kp_yaw,   Ki_yaw,   Kd_yaw,   PID_INTEGRAL_LIMIT, PID_OUTPUT_LIMIT, PID_DERIVATIVE_ALPHA);

Servo servo1, servo2, servo3;
HardwareSerial PiSerial(2);
VisionFrameParser visionParser;
VisionPoseFrame latestVisionFrame{};
bool visionFrameSeen = false;
bool armSwitchWasLow = false;
uint16_t lastVisionSequence = 0;
FlightModeManager flightModes(FAILSAFE_TIMEOUT_MS, FAILSAFE_TIMEOUT_MS);
ImuHealthMonitor imuHealth(IMU_STALE_TIMEOUT_US, IMU_MAX_CONSECUTIVE_ERRORS);
RcInput rcInput;

ServoOutputLimiter servo1Output({SERVO_MAX_TRAVEL_1_DEG, SERVO_MAX_SPEED_DEG_PER_S, SERVO_DEADBAND_DEG, SERVO_TRIM_1_DEG, SERVO_DIR_1, SERVO_MIN_US, SERVO_CENTER_US, SERVO_MAX_US});
ServoOutputLimiter servo2Output({SERVO_MAX_TRAVEL_2_DEG, SERVO_MAX_SPEED_DEG_PER_S, SERVO_DEADBAND_DEG, SERVO_TRIM_2_DEG, SERVO_DIR_2, SERVO_MIN_US, SERVO_CENTER_US, SERVO_MAX_US});
ServoOutputLimiter servo3Output({SERVO_MAX_TRAVEL_3_DEG, SERVO_MAX_SPEED_DEG_PER_S, SERVO_DEADBAND_DEG, SERVO_TRIM_3_DEG, SERVO_DIR_3, SERVO_MIN_US, SERVO_CENTER_US, SERVO_MAX_US});

// ------------------------------------------------------------
// Etat partage entre taches
// ------------------------------------------------------------
static float currentRoll  = 0.0f;
static float currentPitch = 0.0f;
static float currentYaw   = 0.0f;
static float currentYawRateDps = 0.0f;

static float rollCmdDeg  = 0.0f;
static float pitchCmdDeg = 0.0f;
static float yawCmdDeg   = 0.0f;

// ------------------------------------------------------------
// Erreur bloquante de demarrage : signale par un bip rapide en boucle
// ------------------------------------------------------------
void haltOnError() {
  pinMode(PIN_BUZZER, OUTPUT);
  while (true) {
    tone(PIN_BUZZER, 3000, 100);
    delay(300); // Acceptable ici : etat d'erreur fatal, avant tout vol
  }
}

void writeServo(Servo &servo, ServoOutputLimiter &output, float commandDeg) {
  servo.writeMicroseconds(output.update(commandDeg, 1.0f / FREQ_SERVO_HZ));
}

// ------------------------------------------------------------
// Taches (une fonction par frequence de fonctionnement)
// ------------------------------------------------------------

// Lecture IMU + fusion Kalman (couplees, meme frequence : FREQ_IMU_HZ)
void taskImuKalman(float dt) {
  IMUSample s;
  if (!imu.read(s)) {
    imuHealth.recordRead(false, micros());
    return; // Le superviseur déclenche le défaut après erreurs répétées ou données périmées.
  }
  imuHealth.recordRead(true, s.timestamp_us);

  float accelRoll  = computeAccelAngleRoll(s.accelY_g, s.accelZ_g);
  float accelPitch = computeAccelAnglePitch(s.accelX_g, s.accelY_g, s.accelZ_g);

  currentRoll  = kalmanRoll.update(s.gyroX_dps, accelRoll, dt);
  currentPitch = kalmanPitch.update(s.gyroY_dps, accelPitch, dt);

  // Le lacet n'a pas de reference absolue avec un MPU6050 seul : on
  // integre uniquement le gyroscope (derive attendue, cf cahier des
  // charges). kalmanYaw.updateExternal(...) permettra plus tard de le
  // recaler via la fusion avec le Raspberry Pi (~10Hz, vision).
  currentYawRateDps = s.gyroZ_dps;
  currentYaw = kalmanYaw.predict(s.gyroZ_dps, dt);
}

// Boucle PID (FREQ_PID_HZ)
void taskPid(float dt) {
  if (!flightModes.controlEnabled()) {
    rollCmdDeg = pitchCmdDeg = yawCmdDeg = 0.0f;
    pidRoll.reset(); pidPitch.reset(); pidYaw.reset();
    return;
  }
  const PilotCommand command = flightModes.command();
  rollCmdDeg  = pidRoll.compute (command.rollDeg,  currentRoll,  dt);
  pitchCmdDeg = pidPitch.compute(command.pitchDeg, currentPitch, dt);
  yawCmdDeg   = pidYaw.compute  (command.yawRateDps, currentYawRateDps, dt);
}

// Sortie servos, apres mixage (FREQ_SERVO_HZ)
void taskServoOutput() {
  float rCmd = rollCmdDeg;
  float pCmd = pitchCmdDeg;
  float yCmd = yawCmdDeg;

#if ENABLE_FAILSAFE
  if (failsafe.isActive() || !flightModes.controlEnabled()) {
    // Etat sur : servos centres, aucune commande PID appliquee.
    rCmd = 0.0f;
    pCmd = 0.0f;
    yCmd = 0.0f;
  }
#endif

  ServoCommandsDeg mixed = Mixer::mix(rCmd, pCmd, yCmd);

  writeServo(servo1, servo1Output, mixed.servo1Deg);
  writeServo(servo2, servo2Output, mixed.servo2Deg);
  writeServo(servo3, servo3Output, mixed.servo3Deg);
}

// Parse les trames de vision en continu, sans attente bloquante ni fusion
// de pose dans les consignes de vol tant que les repères ne sont pas validés.
void taskUartBridge() {
#if ENABLE_UART_BRIDGE
  while (PiSerial.available() > 0) {
    VisionPoseFrame candidate;
    if (visionParser.feed(static_cast<uint8_t>(PiSerial.read()), candidate) &&
        (!visionFrameSeen || isNewerSequence(candidate.sequence, lastVisionSequence))) {
      latestVisionFrame = candidate;
      lastVisionSequence = candidate.sequence;
      visionFrameSeen = true;
    }
  }
#endif
}

void taskRcAndModes() {
  RcFrame frame;
  const uint32_t nowMs = millis();
  if (rcInput.read(frame, nowMs)) {
    flightModes.notifyRcFrame(true, nowMs);
    failsafe.notifySignalReceived();
    if (frame.disarmRequested) {
      armSwitchWasLow = true;
      flightModes.disarm();
      failsafe.begin();
    } else {
      if (!flightModes.submitCommand(frame.command, nowMs)) flightModes.notifyCommandLost();
      if (frame.armRequested && armSwitchWasLow) {
        armSwitchWasLow = false; // Un cycle interrupteur bas->haut est requis pour chaque armement.
        flightModes.requestArm(nowMs);
      }
    }
  } else {
    flightModes.notifyRcFrame(false, nowMs);
  }

  const bool healthy = imuHealth.update(micros());
  const FlightMode mode = flightModes.update(nowMs, healthy);
  if (mode == FlightMode::FAILSAFE || mode == FlightMode::SENSOR_FAULT) failsafe.trigger();
}

// ------------------------------------------------------------
// Setup
// ------------------------------------------------------------
void setup() {
  Serial.begin(UART_BAUD_RATE);

#if ENABLE_BUZZER
  buzzer.begin();
#endif

  if (!imu.begin()) {
    // MPU6050 introuvable ou non fonctionnel : etat d'erreur fatal.
    haltOnError();
  }
  PiSerial.begin(UART_BAUD_RATE, SERIAL_8N1, PIN_UART_PI_RX, PIN_UART_PI_TX);
  imuHealth.begin(micros());
  rcInput.begin();

  // Servos : plage d'impulsion definie dans config.h
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  servo3.setPeriodHertz(50);
  servo1.attach(PIN_SERVO_1, SERVO_MIN_US, SERVO_MAX_US);
  servo2.attach(PIN_SERVO_2, SERVO_MIN_US, SERVO_MAX_US);
  servo3.attach(PIN_SERVO_3, SERVO_MIN_US, SERVO_MAX_US);
  servo1.writeMicroseconds(SERVO_CENTER_US);
  servo2.writeMicroseconds(SERVO_CENTER_US);
  servo3.writeMicroseconds(SERVO_CENTER_US);

#if ENABLE_FAILSAFE
  failsafe.begin();
#endif
  delay(5000);
  // --- Calibration IMU (planeur immobile) ---
  // Bip court toutes les 500 ms pendant CALIBRATION_DURATION_MS, en
  // parallele de l'accumulation des echantillons IMU. Tout est pilote
  // par millis()/micros(), sans delay() bloquant a l'interieur.
  imu.calibrationReset();
#if ENABLE_BUZZER
  buzzer.startCalibrationPattern();
#endif

  uint32_t calibStartMs = millis();
  uint32_t lastImuSampleUs = micros();
  const uint32_t sampleIntervalUs = (CALIBRATION_DURATION_MS * 1000UL) / CALIBRATION_SAMPLE_COUNT;

  while (millis() - calibStartMs < CALIBRATION_DURATION_MS) {
#if ENABLE_BUZZER
    buzzer.update();
#endif
    if (micros() - lastImuSampleUs >= sampleIntervalUs) {
      lastImuSampleUs = micros();
      imu.calibrationSample();
    }
  }
  imu.calibrationFinalize();
  if (!imu.calibrationValid(CALIBRATION_MIN_SAMPLE_COUNT)) {
    flightModes.completeStartup(false);
    failsafe.trigger();
    haltOnError();
  }

#if ENABLE_BUZZER
  buzzer.stop();
  buzzer.playReadyBeep();
  while (buzzer.isActive()) {
    buzzer.update(); // Attente active non bloquante du bip long de fin
  }
#endif

  // Initialisation des filtres de Kalman a partir d'une premiere lecture
  // post-calibration, pour eviter un transitoire de convergence au demarrage.
  IMUSample firstSample;
  if (!imu.read(firstSample)) {
    flightModes.completeStartup(false);
    failsafe.trigger();
    haltOnError();
  }
  kalmanRoll.setAngle(computeAccelAngleRoll(firstSample.accelY_g, firstSample.accelZ_g));
  kalmanPitch.setAngle(computeAccelAnglePitch(firstSample.accelX_g, firstSample.accelY_g, firstSample.accelZ_g));
  kalmanYaw.setAngle(0.0f); // Pas de reference absolue au demarrage (derive assumee)
  imuHealth.recordRead(true, firstSample.timestamp_us);
  flightModes.completeStartup(true);

  // Le système reste désarmé jusqu'à des trames RC valides et une action armement.
}

// ------------------------------------------------------------
// Loop - scheduler cooperatif non bloquant, plusieurs frequences
// ------------------------------------------------------------
void loop() {
  uint32_t nowUs = micros();

  // Tache IMU + Kalman (500 Hz par defaut)
  static uint32_t lastImuUs = 0;
  if (nowUs - lastImuUs >= PERIOD_IMU_US) {
    float dt = lastImuUs == 0 ? 1.0f / FREQ_IMU_HZ : (nowUs - lastImuUs) / 1000000.0f;
    lastImuUs = nowUs;
    taskImuKalman(dt);
  }

  // Tache PID (200 Hz par defaut)
  static uint32_t lastPidUs = 0;
  if (nowUs - lastPidUs >= PERIOD_PID_US) {
    float dt = lastPidUs == 0 ? 1.0f / FREQ_PID_HZ : (nowUs - lastPidUs) / 1000000.0f;
    lastPidUs = nowUs;
    taskPid(dt);
  }

  // Tache sortie servos (50 Hz par defaut)
  static uint32_t lastServoUs = 0;
  if (nowUs - lastServoUs >= PERIOD_SERVO_US) {
    lastServoUs = nowUs;
    taskServoOutput();
  }

#if ENABLE_UART_BRIDGE
  // Parseur UART Pi (100 Hz par defaut, sans attente bloquante)
  static uint32_t lastUartUs = 0;
  if (nowUs - lastUartUs >= PERIOD_UART_RPI_US) {
    lastUartUs = nowUs;
    taskUartBridge();
  }
#endif

#if ENABLE_BUZZER
  // Tache buzzer : independante, mise a jour a chaque passage de boucle
  buzzer.update();
#endif

  taskRcAndModes();
}