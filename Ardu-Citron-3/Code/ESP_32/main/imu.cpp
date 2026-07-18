#include "imu.h"
#include "config.h"
#include <Wire.h>

// ------------------------------------------------------------
// Registres MPU6050 utilises par ce pilote
// ------------------------------------------------------------
namespace {
  constexpr uint8_t REG_PWR_MGMT_1   = 0x6B;
  constexpr uint8_t REG_SMPLRT_DIV   = 0x19;
  constexpr uint8_t REG_CONFIG       = 0x1A; // DLPF
  constexpr uint8_t REG_GYRO_CONFIG  = 0x1B;
  constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
  constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
  constexpr uint8_t REG_WHO_AM_I     = 0x75;
}

bool IMU::writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_I2C_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
}

bool IMU::readRegisters(uint8_t startReg, uint8_t *buffer, uint8_t length) {
  Wire.beginTransmission(MPU6050_I2C_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) {
    return false; // Repeated start pour ne pas relacher le bus entre write et read
  }
  uint8_t received = Wire.requestFrom((int)MPU6050_I2C_ADDR, (int)length, (int)true);
  if (received != length) {
    return false;
  }
  for (uint8_t i = 0; i < length; i++) {
    buffer[i] = Wire.read();
  }
  return true;
}

bool IMU::readRaw(int16_t &ax, int16_t &ay, int16_t &az,
                   int16_t &gx, int16_t &gy, int16_t &gz) {
  uint8_t raw[14];
  if (!readRegisters(REG_ACCEL_XOUT_H, raw, 14)) {
    return false;
  }
  ax = (int16_t)((raw[0] << 8) | raw[1]);
  ay = (int16_t)((raw[2] << 8) | raw[3]);
  az = (int16_t)((raw[4] << 8) | raw[5]);
  // raw[6..7] = temperature, non utilisee ici
  gx = (int16_t)((raw[8]  << 8) | raw[9]);
  gy = (int16_t)((raw[10] << 8) | raw[11]);
  gz = (int16_t)((raw[12] << 8) | raw[13]);
  return true;
}

bool IMU::begin() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, I2C_CLOCK_HZ);

  // Verification de presence du capteur (non bloquant si different de 0x68 :
  // certains clones renvoient une valeur legerement differente).
  uint8_t whoAmI = 0;
  if (!readRegisters(REG_WHO_AM_I, &whoAmI, 1)) {
    return false;
  }

  // Sortie du mode sleep, horloge = PLL sur gyro X (plus stable que l'oscillateur interne)
  if (!writeRegister(REG_PWR_MGMT_1, 0x01)) return false;
  delay(10);

  // DLPF ~44 Hz : bon compromis bruit / latence pour un planeur
  if (!writeRegister(REG_CONFIG, 0x03)) return false;

  // Sample rate divider : avec DLPF actif, la base gyro est a 1kHz.
  // Fs = 1000 / (1 + SMPLRT_DIV) -> div = 1 donne 500 Hz, aligne sur FREQ_IMU_HZ.
  if (!writeRegister(REG_SMPLRT_DIV, 0x01)) return false;

  // Gyroscope : plage +-500 dps
  if (!writeRegister(REG_GYRO_CONFIG, 0x08)) return false;

  // Accelerometre : plage +-2g
  if (!writeRegister(REG_ACCEL_CONFIG, 0x00)) return false;

  delay(10);
  return true;
}

void IMU::calibrationReset() {
  _sumAx = _sumAy = _sumAz = 0.0;
  _sumGx = _sumGy = _sumGz = 0.0;
  _calibSamples = 0;
}

void IMU::calibrationSample() {
  // A appeler de facon reguliere (non bloquant) pendant la phase de
  // calibration. Le planeur est suppose immobile durant cet appel.
  int16_t ax, ay, az, gx, gy, gz;
  if (!readRaw(ax, ay, az, gx, gy, gz)) {
    return; // Echantillon perdu, on ne casse pas la sequence de calibration
  }
  _sumAx += ax / ACCEL_SENSITIVITY_LSB_PER_G;
  _sumAy += ay / ACCEL_SENSITIVITY_LSB_PER_G;
  _sumAz += az / ACCEL_SENSITIVITY_LSB_PER_G;
  _sumGx += gx / GYRO_SENSITIVITY_LSB_PER_DPS;
  _sumGy += gy / GYRO_SENSITIVITY_LSB_PER_DPS;
  _sumGz += gz / GYRO_SENSITIVITY_LSB_PER_DPS;
  _calibSamples++;
}

void IMU::calibrationFinalize() {
  if (_calibSamples == 0) {
    return; // Aucun echantillon valide : on garde les offsets par defaut (0)
  }
  _offsets.accelX_g  = (float)(_sumAx / _calibSamples);
  _offsets.accelY_g  = (float)(_sumAy / _calibSamples);
  // Sur Z, l'accelerometre mesure 1g au repos (axe vertical) : on ne retire
  // que l'ecart par rapport a 1g, pas la valeur totale.
  _offsets.accelZ_g  = (float)(_sumAz / _calibSamples) - 1.0f;
  _offsets.gyroX_dps = (float)(_sumGx / _calibSamples);
  _offsets.gyroY_dps = (float)(_sumGy / _calibSamples);
  _offsets.gyroZ_dps = (float)(_sumGz / _calibSamples);
}

bool IMU::read(IMUSample &sample) {
  int16_t ax, ay, az, gx, gy, gz;
  if (!readRaw(ax, ay, az, gx, gy, gz)) {
    return false;
  }

  sample.accelX_g = (ax / ACCEL_SENSITIVITY_LSB_PER_G) - _offsets.accelX_g;
  sample.accelY_g = (ay / ACCEL_SENSITIVITY_LSB_PER_G) - _offsets.accelY_g;
  sample.accelZ_g = (az / ACCEL_SENSITIVITY_LSB_PER_G) - _offsets.accelZ_g;

  sample.gyroX_dps = (gx / GYRO_SENSITIVITY_LSB_PER_DPS) - _offsets.gyroX_dps;
  sample.gyroY_dps = (gy / GYRO_SENSITIVITY_LSB_PER_DPS) - _offsets.gyroY_dps;
  sample.gyroZ_dps = (gz / GYRO_SENSITIVITY_LSB_PER_DPS) - _offsets.gyroZ_dps;

  sample.timestamp_us = micros();
  return true;
}
