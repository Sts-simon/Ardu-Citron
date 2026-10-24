#include "imu_health.h"

ImuHealthMonitor::ImuHealthMonitor(uint32_t staleTimeoutUs, uint8_t maxConsecutiveErrors)
  : _staleTimeoutUs(staleTimeoutUs), _maxConsecutiveErrors(maxConsecutiveErrors) {
}

void ImuHealthMonitor::begin(uint32_t nowUs) {
  _lastGoodUs = nowUs;
  _consecutiveErrors = 0;
  _seenSample = false;
  _faulted = false;
}

void ImuHealthMonitor::recordRead(bool success, uint32_t nowUs) {
  if (success) {
    _seenSample = true;
    _lastGoodUs = nowUs;
    _consecutiveErrors = 0;
    return;
  }
  if (_consecutiveErrors < UINT8_MAX) ++_consecutiveErrors;
  if (_consecutiveErrors >= _maxConsecutiveErrors) _faulted = true;
}

bool ImuHealthMonitor::update(uint32_t nowUs) {
  if (_seenSample && nowUs - _lastGoodUs > _staleTimeoutUs) _faulted = true;
  return healthy();
}
