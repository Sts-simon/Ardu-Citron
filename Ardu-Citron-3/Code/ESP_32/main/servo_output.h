#ifndef SERVO_OUTPUT_H
#define SERVO_OUTPUT_H

struct ServoOutputConfig {
  float maxTravelDeg;
  float maxSpeedDegPerSec;
  float deadbandDeg;
  float trimDeg;
  int direction;
  int minPulseUs;
  int centerPulseUs;
  int maxPulseUs;
};

class ServoOutputLimiter {
public:
  explicit ServoOutputLimiter(const ServoOutputConfig &config);
  int update(float commandDeg, float dtSeconds);
  void reset();
  float limitedCommandDeg() const { return _previousCommandDeg; }

private:
  ServoOutputConfig _config;
  float _previousCommandDeg = 0.0f;
};

#endif // SERVO_OUTPUT_H
