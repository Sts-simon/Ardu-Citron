#include "failsafe.h"
#include "config.h"

void Failsafe::begin() {
  _lastSignalMs = millis();
  _active = false;
}

void Failsafe::notifySignalReceived() {
  _lastSignalMs = millis();
  _active = false;
}

bool Failsafe::update() {
  if (!RC_RECEIVER_CONNECTED) {
    // Aucun recepteur RC branche pour le moment (etape actuelle : maintien
    // d'attitude seul). Le module reste pret mais inactif, pour ne pas
    // declencher un failsafe permanent faute de signal. A retirer cette
    // condition (ou passer RC_RECEIVER_CONNECTED a true) des que la lecture
    // RC reelle sera en place.
    _active = false;
    return false;
  }

  if (millis() - _lastSignalMs > FAILSAFE_TIMEOUT_MS) {
    _active = true;
  }
  return _active;
}
