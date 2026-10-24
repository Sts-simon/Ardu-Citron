#ifndef FLIGHT_MODE_H
#define FLIGHT_MODE_H

#include <stdint.h>

enum class FlightMode : uint8_t {
  STARTUP,
  DISARMED,
  MANUAL,
  STABILIZE,
  AUTO,
  FAILSAFE,
  SENSOR_FAULT
};

struct PilotCommand {
  float rollDeg;
  float pitchDeg;
  float yawRateDps;
};

class FlightModeManager {
public:
  explicit FlightModeManager(uint32_t linkTimeoutMs = 500,
                             uint32_t commandTimeoutMs = 500,
                             float maxCommandAngleDeg = 30.0f,
                             float maxYawRateDps = 90.0f);

  // Startup is released only after calibration and a valid initial IMU read.
  bool completeStartup(bool sensorHealthy);
  bool submitCommand(const PilotCommand &command, uint32_t nowMs);
  void notifyCommandLost();
  void notifyRcFrame(bool valid, uint32_t nowMs);
  void setPilotMode(FlightMode requested);
  void configureAuto(bool enabled, float minimumNavigationQuality = 0.5f);
  void notifyAutoCommandLost();
  bool submitAutoCommand(const PilotCommand &command, uint32_t nowMs, uint16_t validityMs,
                         float navigationQuality);
  bool requestArm(uint32_t nowMs);
  bool requestAuto(uint32_t nowMs);
  void disarm();
  FlightMode update(uint32_t nowMs, bool sensorHealthy);

  FlightMode mode() const { return _mode; }
  bool controlEnabled() const { return _mode == FlightMode::STABILIZE || _mode == FlightMode::AUTO; }
  bool manualMode() const { return _mode == FlightMode::MANUAL; }
  bool autoEnabled() const { return _mode == FlightMode::AUTO; }
  PilotCommand command() const { return _mode == FlightMode::AUTO ? _autoCommand : _command; }
  bool hasCommand() const { return _hasCommand; }

private:
  uint32_t _linkTimeoutMs;
  uint32_t _commandTimeoutMs;
  float _maxCommandAngleDeg;
  float _maxYawRateDps;
  uint32_t _autoFreshWindowMs = 100;
  uint32_t _lastRcMs = 0;
  uint32_t _lastCommandMs = 0;
  uint32_t _lastAutoCommandMs = 0;
  uint32_t _autoValidityMs = 0;
  float _minimumAutoQuality = 0.5f;
  float _autoQuality = 0.0f;
  bool _rcSeen = false;
  bool _hasCommand = false;
  bool _autoSelected = false;
  bool _autoAllowed = false;
  bool _hasAutoCommand = false;
  bool _autoExpired = false;
  bool _sensorHealthy = false;
  FlightMode _mode = FlightMode::STARTUP;
  FlightMode _selectedMode = FlightMode::STABILIZE;
  PilotCommand _command{0.0f, 0.0f, 0.0f};
  PilotCommand _autoCommand{0.0f, 0.0f, 0.0f};
};

#endif // FLIGHT_MODE_H
