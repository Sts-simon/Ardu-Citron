#include <ESP32Servo.h>
#include <math.h>
#include "config.h"
#include "imu.h"
#include "kalman.h"
#include "pid.h"
#include "mixer.h"
#include "failsafe.h"
#include "buzzer.h"

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

// ------------------------------------------------------------
// Etat partage entre taches
// ------------------------------------------------------------
static float currentRoll  = 0.0f;
static float currentPitch = 0.0f;
static float currentYaw   = 0.0f;

static float rollCmdDeg  = 0.0f;
static float pitchCmdDeg = 0.0f;
static float yawCmdDeg   = 0.0f;

// Memoire de position servo pour la limitation de vitesse (slew rate)
static float prevServo1Deg = 0.0f;
static float prevServo2Deg = 0.0f;
static float prevServo3Deg = 0.0f;

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

// ------------------------------------------------------------
// Applique zone morte, limitation de debattement, limitation de
// vitesse (slew rate), inversion et trim a une commande de mixage.
// ------------------------------------------------------------
float applyServoLimits(float commandDeg, float trimDeg, int dir, float &prevDeg, float maxTravelDeg) {
  float cmd = commandDeg;

  // Zone morte : evite les micro-corrections permanentes
  if (fabs(cmd) < SERVO_DEADBAND_DEG) {
    cmd = 0.0f;
  }

  // Limitation du debattement (propre a ce servo)
  cmd = constrain(cmd, -maxTravelDeg, maxTravelDeg);

  // Limitation de vitesse (slew rate), si activee (0 = desactivee)
  if (SERVO_MAX_SPEED_DEG_PER_S > 0.0f) {
    float maxDelta = SERVO_MAX_SPEED_DEG_PER_S / (float)FREQ_SERVO_HZ;
    float delta = constrain(cmd - prevDeg, -maxDelta, maxDelta);
    cmd = prevDeg + delta;
  }

  prevDeg = cmd;

  // Inversion de sens + trim logiciel (applique apres la limitation)
  return dir * cmd + trimDeg;
}

void writeServo(Servo &servo, float commandDeg, float trimDeg, int dir, float &prevDeg, float maxTravelDeg) {
  float finalDeg = applyServoLimits(commandDeg, trimDeg, dir, prevDeg, maxTravelDeg);

  // Mapping lineaire symetrique degres -> microsecondes, propre a ce servo
  // (maxTravelDeg = debattement mecanique reel de CE servo -> SERVO_MIN/MAX_US)
  float usPerDeg = (float)(SERVO_MAX_US - SERVO_CENTER_US) / maxTravelDeg;
  int pulseUs = SERVO_CENTER_US + (int)(finalDeg * usPerDeg);
  pulseUs = constrain(pulseUs, SERVO_MIN_US, SERVO_MAX_US);

  servo.writeMicroseconds(pulseUs);
}

// ------------------------------------------------------------
// Taches (une fonction par frequence de fonctionnement)
// ------------------------------------------------------------

// Lecture IMU + fusion Kalman (couplees, meme frequence : FREQ_IMU_HZ)
void taskImuKalman(float dt) {
  IMUSample s;
  if (!imu.read(s)) {
    return; // Erreur I2C ponctuelle : on garde la derniere estimation valide
  }

  float accelRoll  = computeAccelAngleRoll(s.accelY_g, s.accelZ_g);
  float accelPitch = computeAccelAnglePitch(s.accelX_g, s.accelY_g, s.accelZ_g);

  currentRoll  = kalmanRoll.update(s.gyroX_dps, accelRoll, dt);
  currentPitch = kalmanPitch.update(s.gyroY_dps, accelPitch, dt);

  // Le lacet n'a pas de reference absolue avec un MPU6050 seul : on
  // integre uniquement le gyroscope (derive attendue, cf cahier des
  // charges). kalmanYaw.updateExternal(...) permettra plus tard de le
  // recaler via la fusion avec le Raspberry Pi (~10Hz, vision).
  currentYaw = kalmanYaw.predict(s.gyroZ_dps, dt);
}

// Boucle PID (FREQ_PID_HZ)
void taskPid(float dt) {
  rollCmdDeg  = pidRoll.compute (SETPOINT_ROLL_DEG,  currentRoll,  dt);
  pitchCmdDeg = pidPitch.compute(SETPOINT_PITCH_DEG, currentPitch, dt);
  yawCmdDeg   = pidYaw.compute  (SETPOINT_YAW_DEG,   currentYaw,   dt);
}

// Sortie servos, apres mixage (FREQ_SERVO_HZ)
void taskServoOutput() {
  float rCmd = rollCmdDeg;
  float pCmd = pitchCmdDeg;
  float yCmd = yawCmdDeg;

#if ENABLE_FAILSAFE
  if (failsafe.isActive()) {
    // Etat sur : servos centres, aucune commande PID appliquee.
    rCmd = 0.0f;
    pCmd = 0.0f;
    yCmd = 0.0f;
  }
#endif

  ServoCommandsDeg mixed = Mixer::mix(rCmd, pCmd, yCmd);

  writeServo(servo1, mixed.servo1Deg, SERVO_TRIM_1_DEG, SERVO_DIR_1, prevServo1Deg, SERVO_MAX_TRAVEL_1_DEG);
  writeServo(servo2, mixed.servo2Deg, SERVO_TRIM_2_DEG, SERVO_DIR_2, prevServo2Deg, SERVO_MAX_TRAVEL_2_DEG);
  writeServo(servo3, mixed.servo3Deg, SERVO_TRIM_3_DEG, SERVO_DIR_3, prevServo3Deg, SERVO_MAX_TRAVEL_3_DEG);
}

// Pont UART reserve pour le futur Raspberry Pi (FREQ_UART_RPI_HZ)
// Ne fait rien pour l'instant : prepare uniquement la structure d'accueil.
void taskUartBridge() {
#if ENABLE_UART_BRIDGE
  // Format de trame prevu (a definir precisement plus tard, ex: binaire
  // structure ou texte JSON), a deserialiser dans une RaspberryPiFrame
  // (voir config.h), puis a utiliser par exemple ainsi :
  //
  //   RaspberryPiFrame frame;
  //   if (parseFrameFromSerial(frame)) {
  //     kalmanYaw.updateExternal(frame.rotationYaw_deg, KALMAN_R_EXTERNAL);
  //   }
  //
  // Pour l'instant, on se contente de vider le buffer serie si des
  // octets arrivent, pour eviter tout debordement futur.
  while (Serial.available() > 0) {
    Serial.read();
  }
#endif
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
  if (imu.read(firstSample)) {
    kalmanRoll.setAngle(computeAccelAngleRoll(firstSample.accelY_g, firstSample.accelZ_g));
    kalmanPitch.setAngle(computeAccelAnglePitch(firstSample.accelX_g, firstSample.accelY_g, firstSample.accelZ_g));
  }
  kalmanYaw.setAngle(0.0f); // Pas de reference absolue au demarrage (derive assumee)

  // Le planeur peut etre lance : le bip long vient de se terminer.
}

// ------------------------------------------------------------
// Loop - scheduler cooperatif non bloquant, plusieurs frequences
// ------------------------------------------------------------
void loop() {
  uint32_t nowUs = micros();

  // Tache IMU + Kalman (500 Hz par defaut)
  static uint32_t lastImuUs = 0;
  if (nowUs - lastImuUs >= PERIOD_IMU_US) {
    float dt = (nowUs - lastImuUs) / 1000000.0f;
    lastImuUs = nowUs;
    taskImuKalman(dt);
  }

  // Tache PID (200 Hz par defaut)
  static uint32_t lastPidUs = 0;
  if (nowUs - lastPidUs >= PERIOD_PID_US) {
    float dt = (nowUs - lastPidUs) / 1000000.0f;
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
  // Tache UART future Raspberry Pi (10 Hz par defaut, reservee)
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

#if ENABLE_FAILSAFE
  // Tache failsafe : independante, mise a jour a chaque passage de boucle
  failsafe.update();
#endif
}