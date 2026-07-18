#include "kalman.h"
#include <math.h>

KalmanFilter::KalmanFilter(float qAngle, float qBias, float rMeasure)
  : _qAngle(qAngle), _qBias(qBias), _rMeasure(rMeasure) {
}

void KalmanFilter::setAngle(float angle) {
  _angle = angle;
}

void KalmanFilter::predictInternal(float newRate, float dt) {
  // Le taux de rotation reel est le taux mesure moins le biais estime.
  _rate = newRate - _bias;
  _angle += dt * _rate;

  // Propagation de la covariance d'erreur (modele standard angle/biais).
  _cov[0][0] += dt * (dt * _cov[1][1] - _cov[0][1] - _cov[1][0] + _qAngle);
  _cov[0][1] -= dt * _cov[1][1];
  _cov[1][0] -= dt * _cov[1][1];
  _cov[1][1] += _qBias * dt;
}

float KalmanFilter::correctInternal(float measuredAngle, float R) {
  float S  = _cov[0][0] + R;
  float K0 = _cov[0][0] / S;
  float K1 = _cov[1][0] / S;

  float y = measuredAngle - _angle; // Innovation
  _angle += K0 * y;
  _bias  += K1 * y;

  float P00_temp = _cov[0][0];
  float P01_temp = _cov[0][1];

  _cov[0][0] -= K0 * P00_temp;
  _cov[0][1] -= K0 * P01_temp;
  _cov[1][0] -= K1 * P00_temp;
  _cov[1][1] -= K1 * P01_temp;

  return _angle;
}

float KalmanFilter::update(float newRate, float newAngle, float dt) {
  predictInternal(newRate, dt);
  return correctInternal(newAngle, _rMeasure);
}

float KalmanFilter::predict(float newRate, float dt) {
  predictInternal(newRate, dt);
  return _angle;
}

float KalmanFilter::updateExternal(float measuredAngle, float rExternal) {
  // Correction pure, sans nouvelle prediction (la prediction a deja ete
  // faite par ailleurs, ex: dans la tache IMU/Kalman a 500Hz). Prevu pour
  // etre appele depuis le futur pont UART Raspberry Pi, a ~10Hz.
  return correctInternal(measuredAngle, rExternal);
}

// ------------------------------------------------------------
// Calcul des angles a partir de l'accelerometre
// ------------------------------------------------------------
float computeAccelAngleRoll(float ay_g, float az_g) {
  return atan2f(ay_g, az_g) * 180.0f / (float)M_PI;
}

float computeAccelAnglePitch(float ax_g, float ay_g, float az_g) {
  return atan2f(-ax_g, sqrtf(ay_g * ay_g + az_g * az_g)) * 180.0f / (float)M_PI;
}
