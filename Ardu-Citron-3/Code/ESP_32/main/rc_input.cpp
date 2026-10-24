#include "rc_input.h"
#include "config.h"

namespace {
constexpr uint8_t RC_PINS[4] = {34, 35, 32, 33};
constexpr uint32_t RC_MIN_US = 900;
constexpr uint32_t RC_MAX_US = 2100;
constexpr uint32_t RC_STALE_MS = 250;
constexpr float RC_MAX_ANGLE_DEG = 30.0f;
constexpr float RC_MAX_YAW_RATE_DPS = 90.0f;
constexpr uint32_t ARM_LOW_US = 1300;
constexpr uint32_t ARM_HIGH_US = 1700;
}

volatile uint32_t RcInput::_riseUs[4] = {};
volatile uint32_t RcInput::_pulseUs[4] = {};
volatile uint32_t RcInput::_updatedMs[4] = {0, 0, 0, 0};

RcInput::RcInput() = default;

void IRAM_ATTR RcInput::capture(uint8_t channel) {
  const uint32_t nowUs = micros();
  if (digitalRead(RC_PINS[channel])) {
    _riseUs[channel] = nowUs;
  } else {
      const uint32_t width = nowUs - _riseUs[channel];
    if (width >= RC_MIN_US && width <= RC_MAX_US) {
      _pulseUs[channel] = width;
      _updatedMs[channel] = millis();
    }
  }
}

void IRAM_ATTR RcInput::onEdge0() { capture(0); }
void IRAM_ATTR RcInput::onEdge1() { capture(1); }
void IRAM_ATTR RcInput::onEdge2() { capture(2); }
void IRAM_ATTR RcInput::onEdge3() { capture(3); }

void RcInput::begin() {
  for (uint8_t i = 0; i < 4; ++i) pinMode(RC_PINS[i], INPUT);
  attachInterrupt(digitalPinToInterrupt(RC_PINS[0]), onEdge0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RC_PINS[1]), onEdge1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RC_PINS[2]), onEdge2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RC_PINS[3]), onEdge3, CHANGE);
}

bool RcInput::read(RcFrame &frame, uint32_t nowMs) {
  uint32_t pulse[4];
  uint32_t updated[4];
  noInterrupts();
  for (uint8_t i = 0; i < 4; ++i) {
    pulse[i] = _pulseUs[i];
    updated[i] = _updatedMs[i];
  }
  interrupts();

  for (uint8_t i = 0; i < 4; ++i) {
    if (pulse[i] < RC_MIN_US || pulse[i] > RC_MAX_US || nowMs - updated[i] > RC_STALE_MS) return false;
  }
  const auto normalized = [](uint32_t value) {
    const float centered = (static_cast<float>(value) - 1500.0f) / 500.0f;
    const float bounded = centered < -1.0f ? -1.0f : (centered > 1.0f ? 1.0f : centered);
    return bounded * RC_MAX_ANGLE_DEG;
  };
  frame.command.rollDeg = normalized(pulse[0]);
  frame.command.pitchDeg = normalized(pulse[1]);
  const float yawCentered = (static_cast<float>(pulse[2]) - 1500.0f) / 500.0f;
  const float boundedYaw = yawCentered < -1.0f ? -1.0f : (yawCentered > 1.0f ? 1.0f : yawCentered);
  frame.command.yawRateDps = boundedYaw * RC_MAX_YAW_RATE_DPS;
  frame.disarmRequested = pulse[3] < ARM_LOW_US;
  frame.armRequested = pulse[3] > ARM_HIGH_US;
  return true;
}
