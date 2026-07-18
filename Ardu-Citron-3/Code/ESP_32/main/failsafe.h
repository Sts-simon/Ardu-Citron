#ifndef FAILSAFE_H
#define FAILSAFE_H

#include <Arduino.h>

// ============================================================
//  MODULE FAILSAFE - Architecture de securite
// ============================================================
//  Prete des maintenant, meme si le recepteur RC n'est pas encore
//  cable (voir RC_RECEIVER_CONNECTED dans config.h). Une fois la
//  lecture RC reelle implementee, il suffira d'appeler
//  notifySignalReceived() a chaque trame RC valide pour activer
//  la detection de perte de communication.
// ============================================================

class Failsafe {
public:
  void begin();

  // A appeler a chaque reception d'un signal RC valide.
  // (Reserve : sera cable au futur recepteur RC.)
  void notifySignalReceived();

  // A appeler regulierement. Retourne true si le failsafe est actif
  // (perte de communication detectee).
  bool update();

  bool isActive() const { return _active; }

private:
  uint32_t _lastSignalMs = 0;
  bool _active = false;
};

#endif // FAILSAFE_H
