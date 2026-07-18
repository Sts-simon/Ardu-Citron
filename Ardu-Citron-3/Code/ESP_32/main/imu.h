#ifndef IMU_H
#define IMU_H

#include <Arduino.h>

// ============================================================
//  MODULE IMU - Pilote MPU6050 (acces registre direct via I2C)
// ============================================================
//  Gere l'initialisation du capteur, la calibration des offsets
//  au demarrage (planeur immobile) et la lecture des donnees
//  brutes converties en unites physiques (g et deg/s).
// ============================================================

struct IMUSample {
  float accelX_g;
  float accelY_g;
  float accelZ_g;
  float gyroX_dps;
  float gyroY_dps;
  float gyroZ_dps;
  uint32_t timestamp_us;
};

struct IMUOffsets {
  float accelX_g  = 0.0f;
  float accelY_g  = 0.0f;
  float accelZ_g  = 0.0f;
  float gyroX_dps = 0.0f;
  float gyroY_dps = 0.0f;
  float gyroZ_dps = 0.0f;
};

class IMU {
public:
  // Initialise le bus I2C et configure le MPU6050 (DLPF, plages, sample rate).
  // Retourne false si le capteur ne repond pas correctement.
  bool begin();

  // --- Calibration non bloquante ---
  // A appeler en sequence depuis le setup() : reset() une fois, puis
  // sample() de facon reguliere pendant CALIBRATION_DURATION_MS
  // (en parallele du bip du buzzer), puis finalize() une fois.
  void calibrationReset();
  void calibrationSample();
  void calibrationFinalize();

  // Lit un echantillon et lui applique les offsets de calibration.
  // Retourne false en cas d'erreur de communication I2C.
  bool read(IMUSample &sample);

  const IMUOffsets &getOffsets() const { return _offsets; }

private:
  IMUOffsets _offsets;

  // Accumulateurs de calibration
  double _sumAx = 0, _sumAy = 0, _sumAz = 0;
  double _sumGx = 0, _sumGy = 0, _sumGz = 0;
  uint32_t _calibSamples = 0;

  bool writeRegister(uint8_t reg, uint8_t value);
  bool readRegisters(uint8_t startReg, uint8_t *buffer, uint8_t length);
  bool readRaw(int16_t &ax, int16_t &ay, int16_t &az,
               int16_t &gx, int16_t &gy, int16_t &gz);
};

#endif // IMU_H
