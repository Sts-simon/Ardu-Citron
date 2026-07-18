#include "pid.h"

PID::PID(float kp, float ki, float kd,
         float integralLimit, float outputLimit, float derivativeFilterAlpha)
  : _kp(kp), _ki(ki), _kd(kd),
    _integralLimit(integralLimit), _outputLimit(outputLimit),
    _derivativeAlpha(derivativeFilterAlpha) {
}

void PID::reset() {
  _integral = 0.0f;
  _prevMeasurement = 0.0f;
  _filteredDerivative = 0.0f;
  _firstRun = true;
}

float PID::compute(float setpoint, float measurement, float dt) {
  if (dt <= 0.0f) {
    return 0.0f; // Protection contre un dt invalide
  }

  float error = setpoint - measurement;

  // --- Terme integral avec anti-windup (limitation directe de la somme) ---
  _integral += error * dt;
  _integral = constrain(_integral, -_integralLimit, _integralLimit);

  // --- Terme derive, calcule sur la mesure (pas sur l'erreur) pour eviter ---
  // --- les a-coups lors d'un changement brutal de consigne, puis filtre  ---
  // --- passe-bas pour attenuer le bruit du capteur.                     ---
  float rawDerivative = 0.0f;
  if (!_firstRun) {
    rawDerivative = -(measurement - _prevMeasurement) / dt;
  }
  _filteredDerivative = _derivativeAlpha * rawDerivative
                        + (1.0f - _derivativeAlpha) * _filteredDerivative;

  _prevMeasurement = measurement;
  _firstRun = false;

  float output = _kp * error + _ki * _integral + _kd * _filteredDerivative;

  // --- Limitation de sortie + anti-windup complementaire : si la sortie ---
  // --- sature, on retire la contribution qui vient d'etre ajoutee a    ---
  // --- l'integrale dans le sens de la saturation, pour ne pas accumuler ---
  // --- inutilement pendant que la commande est bloquee.                ---
  if (output > _outputLimit) {
    output = _outputLimit;
    if (error > 0.0f) {
      _integral -= error * dt;
    }
  } else if (output < -_outputLimit) {
    output = -_outputLimit;
    if (error < 0.0f) {
      _integral -= error * dt;
    }
  }

  return output;
}
