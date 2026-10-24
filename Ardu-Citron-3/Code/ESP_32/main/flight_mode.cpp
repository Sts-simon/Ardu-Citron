#include "flight_mode.h"

#include <math.h>

FlightModeManager::FlightModeManager(uint32_t linkTimeoutMs, uint32_t commandTimeoutMs,
                                     float maxCommandAngleDeg, float maxYawRateDps)
  : _linkTimeoutMs(linkTimeoutMs), _commandTimeoutMs(commandTimeoutMs),
    _maxCommandAngleDeg(maxCommandAngleDeg), _maxYawRateDps(maxYawRateDps) {}

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

void FlightModeManager::notifyCommandLost() { _hasCommand = false; }

void FlightModeManager::notifyRcFrame(bool valid, uint32_t nowMs) {
  if (valid) {
    _lastRcMs = nowMs;
    _rcSeen = true;
  }
}

void FlightModeManager::setPilotMode(FlightMode requested) {
  if (requested != FlightMode::MANUAL && requested != FlightMode::STABILIZE &&
      requested != FlightMode::AUTO) return;
  if (requested == FlightMode::AUTO && !_autoAllowed) requested = FlightMode::STABILIZE;
  _selectedMode = requested;
  _autoSelected = requested == FlightMode::AUTO;
  if (_mode == FlightMode::AUTO && requested != FlightMode::AUTO) {
    _mode = requested; // RC switch takeover is immediate; update loop checks RC freshness.
    _hasAutoCommand = false;
  } else if ((_mode == FlightMode::MANUAL || _mode == FlightMode::STABILIZE) &&
             requested != FlightMode::AUTO) {
    _mode = requested;
  }
}

void FlightModeManager::configureAuto(bool enabled, float minimumNavigationQuality) {
  _autoAllowed = enabled && isfinite(minimumNavigationQuality) &&
                 minimumNavigationQuality >= 0.0f && minimumNavigationQuality <= 1.0f;
  if (isfinite(minimumNavigationQuality)) _minimumAutoQuality = minimumNavigationQuality;
  if (!_autoAllowed) {
    _autoSelected = false;
    if (_mode == FlightMode::AUTO) {
      _mode = (_hasCommand && _rcSeen) ? FlightMode::STABILIZE : FlightMode::FAILSAFE;
    }
    _hasAutoCommand = false;
  }
}

void FlightModeManager::notifyAutoCommandLost() {
  _hasAutoCommand = false;
  _autoExpired = true;
}

bool FlightModeManager::submitAutoCommand(const PilotCommand &command, uint32_t nowMs,
                                          uint16_t validityMs, float navigationQuality) {
  if (!_autoAllowed || _selectedMode != FlightMode::AUTO || _mode == FlightMode::STARTUP ||
      _mode == FlightMode::SENSOR_FAULT || !isfinite(command.rollDeg) ||
      !isfinite(command.pitchDeg) || !isfinite(command.yawRateDps) ||
      !isfinite(navigationQuality) || fabsf(command.rollDeg) > 25.0f ||
      fabsf(command.pitchDeg) > 8.0f || fabsf(command.yawRateDps) > 25.0f ||
      validityMs == 0 || validityMs > _commandTimeoutMs || validityMs > 500 ||
      navigationQuality < _minimumAutoQuality || navigationQuality > 1.0f) return false;
  if (_hasAutoCommand && !_autoExpired && nowMs - _lastAutoCommandMs < _autoFreshWindowMs &&
      (fabsf(command.rollDeg - _autoCommand.rollDeg) > 0.01f ||
       fabsf(command.pitchDeg - _autoCommand.pitchDeg) > 0.01f ||
       fabsf(command.yawRateDps - _autoCommand.yawRateDps) > 0.01f)) return false;
  _autoCommand = command;
  _autoQuality = navigationQuality;
  _lastAutoCommandMs = nowMs; // Use ESP32 receive time, never the Pi clock domain.
  _autoValidityMs = validityMs;
  _hasAutoCommand = true;
  _autoExpired = false;
  return true;
}

bool FlightModeManager::requestArm(uint32_t nowMs) {
  if (_mode != FlightMode::DISARMED || !_sensorHealthy || !_rcSeen || !_hasCommand ||
      nowMs - _lastRcMs > _linkTimeoutMs || nowMs - _lastCommandMs > _commandTimeoutMs) return false;
  if (_selectedMode == FlightMode::AUTO && _autoAllowed) {
    if (!_hasAutoCommand || nowMs - _lastAutoCommandMs > _autoValidityMs ||
        _autoQuality < _minimumAutoQuality) return false;
    _mode = FlightMode::AUTO;
  } else {
    _mode = _selectedMode;
  }
  return true;
}

bool FlightModeManager::requestAuto(uint32_t nowMs) {
  if (!_autoAllowed || _selectedMode != FlightMode::AUTO ||
      (_mode != FlightMode::MANUAL && _mode != FlightMode::STABILIZE) ||
      !_sensorHealthy || !_rcSeen || !_hasCommand || !_hasAutoCommand ||
      nowMs - _lastRcMs > _linkTimeoutMs || nowMs - _lastCommandMs > _commandTimeoutMs ||
      nowMs - _lastAutoCommandMs > _autoValidityMs || _autoQuality < _minimumAutoQuality) return false;
  _mode = FlightMode::AUTO;
  return true;
}

void FlightModeManager::disarm() {
  if (_mode == FlightMode::MANUAL || _mode == FlightMode::STABILIZE ||
      _mode == FlightMode::AUTO || _mode == FlightMode::FAILSAFE) {
    _mode = _sensorHealthy ? FlightMode::DISARMED : FlightMode::SENSOR_FAULT;
    _hasCommand = false;
    _hasAutoCommand = false;
  }
}

FlightMode FlightModeManager::update(uint32_t nowMs, bool sensorHealthy) {
  _sensorHealthy = sensorHealthy;
  if (!sensorHealthy && _mode != FlightMode::STARTUP) {
    _mode = FlightMode::SENSOR_FAULT;
    _hasCommand = _hasAutoCommand = false;
  } else if (_mode == FlightMode::AUTO &&
             (!_autoAllowed || !_autoSelected || !_rcSeen || !_hasCommand ||
              nowMs - _lastRcMs > _linkTimeoutMs || nowMs - _lastCommandMs > _commandTimeoutMs ||
              !_hasAutoCommand || nowMs - _lastAutoCommandMs > _autoValidityMs ||
              _autoQuality < _minimumAutoQuality)) {
    // Stale navigation falls back to live RC stabilization. If RC is stale too,
    // fail closed instead of continuing on either stale command.
    _mode = (_hasCommand && _rcSeen && nowMs - _lastRcMs <= _linkTimeoutMs &&
             nowMs - _lastCommandMs <= _commandTimeoutMs) ? FlightMode::STABILIZE : FlightMode::FAILSAFE;
    _selectedMode = FlightMode::STABILIZE;
    _hasAutoCommand = false;
  } else if ((_mode == FlightMode::MANUAL || _mode == FlightMode::STABILIZE) &&
             (!_rcSeen || nowMs - _lastRcMs > _linkTimeoutMs ||
              (_mode == FlightMode::STABILIZE && (!_hasCommand || nowMs - _lastCommandMs > _commandTimeoutMs)))) {
    _mode = FlightMode::FAILSAFE;
    _hasCommand = false;
  }
  return _mode;
}
