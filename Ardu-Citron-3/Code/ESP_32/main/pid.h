#ifndef PID_H
#define PID_H

#include <Arduino.h>

// ============================================================
//  MODULE PID - Correcteur PID generique avec anti-windup
// ============================================================
//  Une instance independante par axe (roll, pitch, yaw). Inclut :
//   - anti-windup (limitation de l'integrale + back-off en saturation)
//   - derivee sur la mesure, filtree passe-bas (reduit le bruit)
//   - sortie limitee
// ============================================================

class PID {
public:
  PID(float kp, float ki, float kd,
      float integralLimit, float outputLimit, float derivativeFilterAlpha);

  // Calcule la commande de sortie pour un pas de temps dt (en secondes).
  float compute(float setpoint, float measurement, float dt);

  // Reinitialise l'etat interne (integrale, derivee filtree...).
  void reset();

private:
  float _kp, _ki, _kd;
  float _integralLimit;
  float _outputLimit;
  float _derivativeAlpha; // Coefficient du filtre passe-bas sur la derivee (0-1)

  float _integral = 0.0f;
  float _prevMeasurement = 0.0f;
  float _filteredDerivative = 0.0f;
  bool _firstRun = true;
};

#endif // PID_H
