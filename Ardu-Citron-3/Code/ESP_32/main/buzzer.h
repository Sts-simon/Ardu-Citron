#ifndef BUZZER_H
#define BUZZER_H

#include <Arduino.h>

// ============================================================
//  MODULE BUZZER - Retours sonores non bloquants
// ============================================================
//  Gere le buzzer passif sans jamais utiliser delay(). S'appuie
//  sur tone()/noTone() (non bloquants sur ESP32 Arduino core) et
//  sur une petite machine a etats pilotee par millis().
// ============================================================

enum class BuzzerPattern : uint8_t {
  NONE,
  CALIBRATION_BEEP, // Bip court repete toutes les CALIBRATION_BEEP_PERIOD_MS
  READY_BEEP        // Bip long unique, signale la fin de la calibration
};

class Buzzer {
public:
  void begin();

  // A appeler regulierement (tache independante, non bloquante).
  void update();

  // Demarre le motif de bips courts repetes (phase de calibration).
  void startCalibrationPattern();

  // Joue un bip long unique (fin de calibration / pret a voler).
  void playReadyBeep();

  // Coupe le buzzer et revient a l'etat inactif.
  void stop();

  // true tant qu'un motif est en cours (utile pour attendre la fin du
  // bip long avant de continuer, par exemple).
  bool isActive() const { return _pattern != BuzzerPattern::NONE; }

private:
  BuzzerPattern _pattern = BuzzerPattern::NONE;
  uint32_t _patternStartMs = 0;
  uint32_t _lastBeepMs = 0;
};

#endif // BUZZER_H
