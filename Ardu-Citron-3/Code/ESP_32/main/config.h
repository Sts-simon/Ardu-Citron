#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ============================================================
//  CONFIGURATION GENERALE DU CONTROLEUR DE VOL - PLANEUR RC
// ============================================================
//  Toutes les constantes du systeme sont centralisees ici.
//  Modifier ce fichier permet d'ajuster tout le comportement
//  du controleur sans toucher au reste du code.
// ============================================================

// ------------------------------------------------------------
// ACTIVATION DES MODULES
// ------------------------------------------------------------
#define ENABLE_KALMAN        true   // Filtre de Kalman pour l'estimation d'attitude
#define ENABLE_FAILSAFE      true   // Architecture de failsafe (prete, RC pas encore branche)
#define ENABLE_BUZZER        true   // Retours sonores (calibration, etats)
#define ENABLE_UART_BRIDGE   true   // Protocole binaire versionne sur UART2 (Pi)

// ------------------------------------------------------------
// BROCHAGE
// ------------------------------------------------------------
// I2C (MPU6050)
constexpr uint8_t PIN_I2C_SDA = 22;
constexpr uint8_t PIN_I2C_SCL = 23;
constexpr uint32_t I2C_CLOCK_HZ = 400000; // I2C fast mode

// Servos
constexpr uint8_t PIN_SERVO_1 = 26;
constexpr uint8_t PIN_SERVO_2 = 27;
constexpr uint8_t PIN_SERVO_3 = 14;

// Buzzer passif
constexpr uint8_t PIN_BUZZER = 13;

// UART2 dedie au Raspberry Pi : conserve UART0 pour flash/console de debug.
constexpr uint8_t PIN_UART_PI_RX = 16;
constexpr uint8_t PIN_UART_PI_TX = 17;

// ------------------------------------------------------------
// CONSIGNES D'ATTITUDE (setpoints) - tres facilement modifiables
// ------------------------------------------------------------
constexpr float SETPOINT_ROLL_DEG  = 0.0f;
constexpr float SETPOINT_PITCH_DEG = 0.0f;
constexpr float SETPOINT_YAW_DEG   = 0.0f; 

// ------------------------------------------------------------
// FREQUENCES DE FONCTIONNEMENT (Hz) -> conversion automatique en periodes (us)
// ------------------------------------------------------------
constexpr uint32_t FREQ_IMU_HZ      = 500;  // Lecture MPU6050
constexpr uint32_t FREQ_KALMAN_HZ   = 500;  // Fusion Kalman (couplee a la lecture IMU)
constexpr uint32_t FREQ_PID_HZ      = 200;  // Boucle PID
constexpr uint32_t FREQ_SERVO_HZ    = 50;   // Sortie servos
constexpr uint32_t FREQ_UART_RPI_HZ = 100;  // Drain du parseur Pi (sans attente bloquante)
constexpr uint32_t FREQ_BUZZER_HZ   = 10;  // Frequence de mise a jour de la tache buzzer

constexpr uint32_t PERIOD_IMU_US      = 1000000UL / FREQ_IMU_HZ;
constexpr uint32_t PERIOD_KALMAN_US   = 1000000UL / FREQ_KALMAN_HZ;
constexpr uint32_t PERIOD_PID_US      = 1000000UL / FREQ_PID_HZ;
constexpr uint32_t PERIOD_SERVO_US    = 1000000UL / FREQ_SERVO_HZ;
constexpr uint32_t PERIOD_UART_RPI_US = 1000000UL / FREQ_UART_RPI_HZ;
constexpr uint32_t PERIOD_BUZZER_US   = 1000000UL / FREQ_BUZZER_HZ;

// ------------------------------------------------------------
// DEMARRAGE / CALIBRATION
// ------------------------------------------------------------
constexpr uint32_t CALIBRATION_DURATION_MS    = 10000; // 10 s, planeur immobile
constexpr uint32_t CALIBRATION_BEEP_PERIOD_MS = 500;  // Bip court toutes les 500 ms
constexpr uint32_t CALIBRATION_BEEP_DURATION_MS = 60;
constexpr uint16_t CALIBRATION_SAMPLE_COUNT   = 400;  // Echantillons moyens demandes
constexpr uint16_t CALIBRATION_MIN_SAMPLE_COUNT = 360; // Refus si plus de 10% sont perdus

// ------------------------------------------------------------
// IMU - MPU6050
// ------------------------------------------------------------
constexpr uint8_t MPU6050_I2C_ADDR = 0x68;

// Sensibilites correspondant a la configuration choisie dans imu.cpp
constexpr float ACCEL_SENSITIVITY_LSB_PER_G  = 16384.0f; // Plage +-2g
constexpr float GYRO_SENSITIVITY_LSB_PER_DPS = 65.5f;    // Plage +-500 dps

// ------------------------------------------------------------
// FILTRE DE KALMAN
// ------------------------------------------------------------
constexpr float KALMAN_Q_ANGLE = 0.001f;  // Bruit de process sur l'angle
constexpr float KALMAN_Q_BIAS  = 0.003f;  // Bruit de process sur le biais gyro
constexpr float KALMAN_R_MEASURE  = 0.03f; // Bruit de mesure accelerometre

// Bruit de mesure pour une correction externe future (vision / Raspberry Pi, ~10Hz)
constexpr float KALMAN_R_EXTERNAL = 0.05f;

// ------------------------------------------------------------
// GAINS PID - regroupes ici pour un reglage rapide
// ------------------------------------------------------------
constexpr float Kp_roll = 1.2f;
constexpr float Ki_roll = 0.05f;
constexpr float Kd_roll = 0.08f;

constexpr float Kp_pitch = 1.4f;
constexpr float Ki_pitch = 0.06f;
constexpr float Kd_pitch = 0.10f;

constexpr float Kp_yaw = 0.8f;
constexpr float Ki_yaw = 0.02f;
constexpr float Kd_yaw = 0.05f;

// Limites communes aux 3 PID
constexpr float PID_OUTPUT_LIMIT     = 45.0f; // deg, commande max envoyee au mixeur
constexpr float PID_INTEGRAL_LIMIT   = 20.0f; // anti-windup : limite de la somme integrale
constexpr float PID_DERIVATIVE_ALPHA = 0.2f;  // filtre passe-bas sur la derivee (0-1)

// ------------------------------------------------------------
// SERVOS
// ------------------------------------------------------------
constexpr int SERVO_CENTER_US = 1500;
constexpr int SERVO_MIN_US    = 1000;
constexpr int SERVO_MAX_US    = 2000;

constexpr float SERVO_DEADBAND_DEG    = 1.0f;  // Zone morte +-1 deg

// Debattement max PAR SERVO (deg), correspond a SERVO_MIN/MAX_US pour CE servo.
// A regler individuellement selon la mecanique reelle de chaque gouverne
// (longueur de bras de servo, position du guignol, etc.).
constexpr float SERVO_MAX_TRAVEL_1_DEG = 10.0f;
constexpr float SERVO_MAX_TRAVEL_2_DEG = 10.f;
constexpr float SERVO_MAX_TRAVEL_3_DEG = 10.0f;

constexpr float SERVO_MAX_SPEED_DEG_PER_S = 50.0f; // 0 = pas de limitation de vitesse

// Trims logiciels (deg), ajoutes apres mixage
constexpr float SERVO_TRIM_1_DEG = 0.0f;
constexpr float SERVO_TRIM_2_DEG = 0.0f;
constexpr float SERVO_TRIM_3_DEG = 0.0f;

// Inversion de sens (+1 normal, -1 inverse)
constexpr int SERVO_DIR_1 = -1;
constexpr int SERVO_DIR_2 = +1;
constexpr int SERVO_DIR_3 = +1;

// ------------------------------------------------------------
// MIXAGE
// ------------------------------------------------------------

enum class MixingPreset : uint8_t {
  STANDARD_AIL_ELEV_RUD, // Servo1=Aileron(roll), Servo2=Profondeur(pitch), Servo3=Derive(yaw)
  VTAIL,                 // Servo1/2 = queue en V (melange pitch+yaw), Servo3 = aileron (roll)
  ELEVON                 // Servo1/2 = elevons (melange roll+pitch), Servo3 = derive (yaw)
};

constexpr MixingPreset ACTIVE_MIXING_PRESET = MixingPreset::STANDARD_AIL_ELEV_RUD;

// ------------------------------------------------------------
// FAILSAFE
// ------------------------------------------------------------
// Lecture RC PWM active sur GPIO34/35/32/33. L'armement est interdit tant que
// les quatre voies valides n'ont pas été vues dans les 250 dernières ms.
constexpr bool RC_RECEIVER_CONNECTED = true;
constexpr uint32_t FAILSAFE_TIMEOUT_MS = 500; // Duree sans signal avant declenchement

// ------------------------------------------------------------
// UART2 - PROTOCOLE VERSIONNE RASPBERRY PI
// ------------------------------------------------------------
constexpr uint32_t UART_BAUD_RATE = 115200;
constexpr uint32_t IMU_STALE_TIMEOUT_US = 100000;
constexpr uint8_t IMU_MAX_CONSECUTIVE_ERRORS = 5;

#endif // CONFIG_H
