#include "buzzer.h"
#include "config.h"

namespace {
  constexpr unsigned int CALIBRATION_BEEP_FREQ_HZ = 2200;
  constexpr unsigned int READY_BEEP_FREQ_HZ       = 1500;
  constexpr uint32_t READY_BEEP_DURATION_MS       = 800;
}

void Buzzer::begin() {
  pinMode(PIN_BUZZER, OUTPUT);
  noTone(PIN_BUZZER);
  _pattern = BuzzerPattern::NONE;
}

void Buzzer::startCalibrationPattern() {
  _pattern = BuzzerPattern::CALIBRATION_BEEP;
  _patternStartMs = millis();
  // On declenche immediatement un premier bip pour un retour instantane.
  _lastBeepMs = millis() - CALIBRATION_BEEP_PERIOD_MS;
}

void Buzzer::playReadyBeep() {
  _pattern = BuzzerPattern::READY_BEEP;
  _patternStartMs = millis();
  tone(PIN_BUZZER, READY_BEEP_FREQ_HZ, READY_BEEP_DURATION_MS);
}

void Buzzer::stop() {
  noTone(PIN_BUZZER);
  _pattern = BuzzerPattern::NONE;
}

void Buzzer::update() {
  uint32_t now = millis();

  switch (_pattern) {

    case BuzzerPattern::NONE:
      // Rien a faire
      break;

    case BuzzerPattern::CALIBRATION_BEEP:
      if (now - _lastBeepMs >= CALIBRATION_BEEP_PERIOD_MS) {
        _lastBeepMs = now;
        tone(PIN_BUZZER, CALIBRATION_BEEP_FREQ_HZ, CALIBRATION_BEEP_DURATION_MS);
      }
      break;

    case BuzzerPattern::READY_BEEP:
      // tone() avec duree s'arrete tout seul ; on repasse a NONE une fois
      // la duree ecoulee pour permettre au code appelant de detecter la fin
      // via isActive().
      if (now - _patternStartMs >= READY_BEEP_DURATION_MS) {
        _pattern = BuzzerPattern::NONE;
      }
      break;
  }
}
