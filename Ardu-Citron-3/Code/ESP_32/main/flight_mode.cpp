#include "flight_mode.h"

#include <math.h>

FlightModeManager::FlightModeManager(uint32_t linkTimeoutMs, uint32_t commandTimeoutMs,
                                     float maxCommandAngleDeg, float maxYawRateDps)
  : _linkTimeoutMs(linkTimeoutMs), _commandTimeoutMs(commandTimeoutMs),
    _maxCommandAngleDeg(maxCommandAngleDeg), _maxYawRateDps(maxYawRateDps) {
}

bool FlightModeManager::completeStartup(bool sensorHealthy) {
  if (_mode != FlightMode::STARTUP) return false;
  _sensorHealthy = sensorHealthy;
  _mode = sensorHealthy ? FlightMode::DISARMED : FlightMode::SENSOR_FAULT;
  return sensorHealthy;
}

bool FlightModeManager::submitCommand(const PilotCommand &command, uint32_t nowMs) {
  if (!isfinite(command.rollDeg) || !isfinite(command.pitchDeg) || !isfinite(command.yawRateDps) ||
      fabsf(command.rollDeg) > _maxCommandAngleDeg ||
      fabsf(command.pitchDeg) > _maxCommandAngleDeg ||
      fabsf(command.yawRateDps) > _maxYawRateDps ||
      _mode == FlightMode::STARTUP || _mode == FlightMode::SENSOR_FAULT) return false;
  _command = command;
  _lastCommandMs = nowMs;
  _hasCommand = true;
  return true;
}

void FlightModeManager::notifyCommandLost() {
  _hasCommand = false;
}

void FlightModeManager::notifyRcFrame(bool valid, uint32_t nowMs) {
  if (valid) {
    _lastRcMs = nowMs;
    _rcSeen = true;
  }
}

bool FlightModeManager::requestArm(uint32_t nowMs) {
  if (_mode != FlightMode::DISARMED || !_sensorHealthy || !_rcSeen || !_hasCommand ||
      nowMs - _lastRcMs > _linkTimeoutMs || nowMs - _lastCommandMs > _commandTimeoutMs) return false;
  _mode = FlightMode::STABILIZE;
  return true;
}

void FlightModeManager::disarm() {
  if (_mode == FlightMode::STABILIZE || _mode == FlightMode::FAILSAFE) {
    _mode = _sensorHealthy ? FlightMode::DISARMED : FlightMode::SENSOR_FAULT;
    _hasCommand = false;
  }
}

FlightMode FlightModeManager::update(uint32_t nowMs, bool sensorHealthy) {
  _sensorHealthy = sensorHealthy;
  if (!sensorHealthy && _mode != FlightMode::STARTUP) {
    _mode = FlightMode::SENSOR_FAULT;
    _hasCommand = false;
  } else if (_mode == FlightMode::STABILIZE &&
             (!_rcSeen || !_hasCommand || nowMs - _lastRcMs > _linkTimeoutMs ||
              nowMs - _lastCommandMs > _commandTimeoutMs)) {
    _mode = FlightMode::FAILSAFE;
    _hasCommand = false;
  }
  return _mode;
}
