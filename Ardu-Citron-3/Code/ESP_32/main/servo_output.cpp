#include "servo_output.h"

#include <math.h>

namespace {
float clampf(float value, float low, float high) {
  return value < low ? low : (value > high ? high : value);
}
}

ServoOutputLimiter::ServoOutputLimiter(const ServoOutputConfig &config) : _config(config) {
}

void ServoOutputLimiter::reset() {
  _previousCommandDeg = 0.0f;
}

int ServoOutputLimiter::update(float commandDeg, float dtSeconds) {
  if (!_config.maxTravelDeg || !isfinite(_config.maxTravelDeg) ||
      !isfinite(_config.maxSpeedDegPerSec) || !isfinite(_config.deadbandDeg) ||
      !isfinite(_config.trimDeg) || _config.maxTravelDeg < 0.0f ||
      _config.direction == 0 || _config.minPulseUs >= _config.centerPulseUs ||
      _config.centerPulseUs >= _config.maxPulseUs || !isfinite(commandDeg) ||
      !isfinite(dtSeconds) || dtSeconds <= 0.0f) return _config.centerPulseUs;

  float target = fabsf(commandDeg) < _config.deadbandDeg ? 0.0f :
                 clampf(commandDeg, -_config.maxTravelDeg, _config.maxTravelDeg);
  if (_config.maxSpeedDegPerSec > 0.0f) {
    const float maxDelta = _config.maxSpeedDegPerSec * dtSeconds;
    target = _previousCommandDeg + clampf(target - _previousCommandDeg, -maxDelta, maxDelta);
  }
  _previousCommandDeg = target;

  const float servoDeg = clampf(_config.direction * target + _config.trimDeg,
                                -_config.maxTravelDeg, _config.maxTravelDeg);
  const float pulseOffset = servoDeg >= 0.0f
      ? servoDeg * (_config.maxPulseUs - _config.centerPulseUs) / _config.maxTravelDeg
      : servoDeg * (_config.centerPulseUs - _config.minPulseUs) / _config.maxTravelDeg;
  const long pulse = lroundf(_config.centerPulseUs + pulseOffset);
  return static_cast<int>(pulse < _config.minPulseUs ? _config.minPulseUs :
                          (pulse > _config.maxPulseUs ? _config.maxPulseUs : pulse));
}
