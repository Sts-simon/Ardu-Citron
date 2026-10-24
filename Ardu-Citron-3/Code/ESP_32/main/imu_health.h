#ifndef IMU_HEALTH_H
#define IMU_HEALTH_H

#include <stdint.h>

class ImuHealthMonitor {
public:
  explicit ImuHealthMonitor(uint32_t staleTimeoutUs = 100000,
                            uint8_t maxConsecutiveErrors = 5);
  void begin(uint32_t nowUs);
  void recordRead(bool success, uint32_t nowUs);
  bool update(uint32_t nowUs);
  bool healthy() const { return _seenSample && !_faulted; }
  bool faulted() const { return _faulted; }
  uint8_t consecutiveErrors() const { return _consecutiveErrors; }
  uint32_t lastGoodSampleUs() const { return _lastGoodUs; }

private:
  uint32_t _staleTimeoutUs;
  uint8_t _maxConsecutiveErrors;
  uint32_t _lastGoodUs = 0;
  uint8_t _consecutiveErrors = 0;
  bool _seenSample = false;
  bool _faulted = false;
};

#endif // IMU_HEALTH_H
