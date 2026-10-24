#ifndef RC_INPUT_H
#define RC_INPUT_H

#include <Arduino.h>
#include "flight_mode.h"

struct RcFrame {
  PilotCommand command;
  bool armRequested;
  bool disarmRequested;
  bool manualSelected;
  bool autoRequested;
  float directServo1Deg;
  float directServo2Deg;
  float directServo3Deg;
};

class RcInput {
public:
  RcInput();
  void begin();
  bool read(RcFrame &frame, uint32_t nowMs);

private:
  static void IRAM_ATTR onEdge0();
  static void IRAM_ATTR onEdge1();
  static void IRAM_ATTR onEdge2();
  static void IRAM_ATTR onEdge3();
  static void IRAM_ATTR onEdge4();
  static void IRAM_ATTR capture(uint8_t channel);

  static volatile uint32_t _riseUs[5];
  static volatile uint32_t _pulseUs[5];
  static volatile uint32_t _updatedMs[5];
};

#endif // RC_INPUT_H
