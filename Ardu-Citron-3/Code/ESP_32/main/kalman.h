#ifndef KALMAN_H
#define KALMAN_H

#include <Arduino.h>

// ============================================================
//  MODULE KALMAN - Filtre de Kalman 2 etats (angle / biais gyro)
// ============================================================
//  Une instance par axe (roll, pitch, yaw). Fusionne l'integration
//  gyroscopique avec une mesure d'angle absolue (issue de
//  l'accelerometre pour roll/pitch). Prevu pour accepter en plus,
//  a terme, une correction externe a ~10 Hz (vision / Raspberry Pi),
//  utile en particulier pour corriger la derive du lacet.
// ============================================================

class KalmanFilter {
public:
  KalmanFilter(float qAngle, float qBias, float rMeasure);

  // Definit l'angle courant (utile a l'initialisation, pour partir
  // d'une estimation coherente plutot que de 0).
  void setAngle(float angle);

  // Etape complete : prediction (integration gyro) + correction avec
  // une mesure d'angle (typiquement calculee depuis l'accelerometre).
  float update(float newRate, float newAngle, float dt);

  // Prediction seule, sans mesure de correction interne. Utilise pour
  // le lacet, qui n'a pas de reference absolue avec un MPU6050 seul.
  float predict(float newRate, float dt);

  // Correction externe optionnelle (ex: vision / Raspberry Pi, ~10Hz).
  // A appeler en plus de predict()/update(), avec un bruit de mesure
  // rExternal propre a la source de la mesure.
  float updateExternal(float measuredAngle, float rExternal);

  float getAngle() const { return _angle; }
  float getBias()  const { return _bias; }

private:
  float _qAngle;
  float _qBias;
  float _rMeasure;

  float _angle = 0.0f;
  float _bias  = 0.0f;
  float _rate  = 0.0f;

  // Matrice de covariance de l'erreur d'estimation (2x2)
  // NB: nommee "_cov" et non "_P" car "_P" est une macro definie par
  // ctype.h (classe caractere "punctuation"), incluse transitivement par
  // Arduino.h -> WCharacter.h -> ctype.h sur le core ESP32. L'utiliser
  // comme nom de membre provoque une erreur de preprocesseur silencieuse.
  float _cov[2][2] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

  void predictInternal(float newRate, float dt);
  float correctInternal(float measuredAngle, float R);
};

// Utilitaires : calcul de l'angle roll/pitch a partir de l'accelerometre
// (convention : X = avant, Y = droite, Z = bas ; a adapter si le MPU6050
// est monte differemment sur la cellule).
float computeAccelAngleRoll(float ay_g, float az_g);
float computeAccelAnglePitch(float ax_g, float ay_g, float az_g);

#endif // KALMAN_H
