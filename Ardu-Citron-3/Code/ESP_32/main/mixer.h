#ifndef MIXER_H
#define MIXER_H

#include <Arduino.h>
#include "config.h"

// ============================================================
//  MODULE MIXER - Mixage des commandes PID vers les servos
// ============================================================
//  Les PID ne pilotent jamais directement les servos : ils
//  produisent des commandes Roll / Pitch / Yaw. Ce module les
//  transforme en 3 commandes servo, selon le preset de mixage
//  choisi dans config.h (ACTIVE_MIXING_PRESET). Facilement
//  extensible : ajouter un cas dans le switch pour un nouveau
//  preset (ex: aile volante, canard...).
// ============================================================

struct ServoCommandsDeg {
  float servo1Deg;
  float servo2Deg;
  float servo3Deg;
};

class Mixer {
public:
  // Transforme les commandes PID (degres) en commandes servos (degres),
  // selon ACTIVE_MIXING_PRESET defini dans config.h.
  static ServoCommandsDeg mix(float rollCmdDeg, float pitchCmdDeg, float yawCmdDeg);
};

#endif // MIXER_H
