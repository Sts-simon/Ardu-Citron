#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../main/flight_mode.h"
#include "../main/imu_health.h"
#include "../main/kalman.h"
#include "../main/mixer.h"
#include "../main/pid.h"
#include "../main/protocol.h"
#include "../main/servo_output.h"

static void testKalmanAndPid() {
  KalmanFilter filter(0.001f, 0.003f, 0.03f);
  filter.setAngle(10.0f);
  assert(fabsf(filter.update(0.0f, 10.0f, 0.005f) - 10.0f) < 0.1f);
  const float before = filter.getAngle();
  filter.updateExternal(before + 2.0f, 0.05f);
  assert(filter.getAngle() > before);

  PID pid(1.0f, 0.5f, 0.1f, 2.0f, 5.0f, 0.2f);
  assert(fabsf(pid.compute(3.0f, 0.0f, 0.01f) - 3.015f) < 0.01f);
  assert(pid.compute(1000.0f, 0.0f, 0.01f) <= 5.0f);
  assert(pid.compute(0.0f, 0.0f, 0.0f) == 0.0f);
  pid.reset();
}

static void testMixAndServo() {
  const ServoCommandsDeg standard = Mixer::mix(3.0f, -4.0f, 5.0f);
  assert(standard.servo1Deg == 3.0f && standard.servo2Deg == -4.0f && standard.servo3Deg == 5.0f);

  ServoOutputConfig config{10.0f, 20.0f, 1.0f, 0.0f, 1, 1000, 1500, 2000};
  ServoOutputLimiter servo(config);
  assert(servo.update(10.0f, 0.02f) == 1520); // limited to 0.4 degrees per 20 ms
  assert(servo.update(100.0f, 1.0f) == 2000); // bounded at full travel
  assert(servo.update(NAN, 0.02f) == 1500);
}

static void testImuHealthAndModes() {
  ImuHealthMonitor health(100, 3);
  health.begin(0);
  assert(!health.healthy());
  health.recordRead(true, 10);
  assert(health.update(100));
  assert(!health.update(111));

  health.begin(0);
  health.recordRead(true, 1);
  health.recordRead(false, 2);
  health.recordRead(false, 3);
  health.recordRead(false, 4);
  assert(health.faulted());
  health.recordRead(true, 5);
  assert(!health.healthy()); // fault is latched until explicit reinitialization

  FlightModeManager manager(100, 100, 30.0f);
  assert(!manager.completeStartup(false));
  assert(manager.mode() == FlightMode::SENSOR_FAULT);

  FlightModeManager modes(100, 100, 30.0f);
  assert(modes.completeStartup(true));
  assert(modes.mode() == FlightMode::DISARMED);
  PilotCommand neutral{0.0f, 0.0f, 0.0f};
  assert(modes.submitCommand(neutral, 10));
  assert(!modes.submitCommand(PilotCommand{NAN, 0.0f, 0.0f}, 10));
  assert(!modes.submitCommand(PilotCommand{31.0f, 0.0f, 0.0f}, 10));
  assert(!modes.submitCommand(PilotCommand{0.0f, 0.0f, 91.0f}, 10));
  assert(!modes.requestArm(10));
  modes.notifyRcFrame(false, 10);
  assert(!modes.requestArm(10));
  modes.notifyRcFrame(true, 10);
  assert(modes.requestArm(10));
  assert(modes.controlEnabled());
  assert(modes.update(50, false) == FlightMode::SENSOR_FAULT);
  assert(!modes.controlEnabled());
  modes.disarm();
  assert(modes.mode() == FlightMode::SENSOR_FAULT);

  FlightModeManager timeoutModes(100, 100, 30.0f);
  assert(timeoutModes.completeStartup(true));
  assert(timeoutModes.submitCommand(neutral, 10));
  timeoutModes.notifyRcFrame(true, 10);
  assert(timeoutModes.requestArm(10));
  assert(timeoutModes.update(111, true) == FlightMode::FAILSAFE);
  assert(!timeoutModes.controlEnabled());
  assert(!timeoutModes.requestArm(112)); // explicit disarm/re-arm is required
  timeoutModes.disarm();
  assert(timeoutModes.mode() == FlightMode::DISARMED);
}

static void testProtocol() {
  VisionPoseFrame sent{65535, 0xFFFFFFFEu, 1.25f, -2.5f, 0.75f, 10.0f, -20.0f, 179.0f, 0.9f};
  uint8_t bytes[VISION_FRAME_SIZE];
  assert(encodeVisionFrame(sent, bytes, sizeof(bytes)));
  assert(!encodeVisionFrame(sent, bytes, sizeof(bytes) - 1));

  VisionFrameParser parser;
  VisionPoseFrame received{};
  assert(!parser.feed(0x00, received));
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    const bool decoded = parser.feed(bytes[i], received);
    assert(decoded == (i == sizeof(bytes) - 1));
  }
  assert(received.sequence == sent.sequence);
  assert(received.senderTimestampMs == sent.senderTimestampMs);
  assert(fabsf(received.positionY_m - sent.positionY_m) < 1e-6f);
  assert(fabsf(received.rotationYaw_deg - sent.rotationYaw_deg) < 1e-6f);
  assert(fabsf(received.detectionQuality - sent.detectionQuality) < 1e-6f);
  assert(isNewerSequence(0, 65535));
  assert(!isNewerSequence(65535, 0));

  bytes[20] ^= 0x80;
  VisionFrameParser damaged;
  bool decoded = false;
  for (size_t i = 0; i < sizeof(bytes); ++i) decoded |= damaged.feed(bytes[i], received);
  assert(!decoded && damaged.rejectedFrames() > 0);
  assert(!encodeVisionFrame(VisionPoseFrame{0, 0, NAN, 0, 0, 0, 0, 0, 1}, bytes, sizeof(bytes)));

  uint8_t stream[VISION_FRAME_SIZE + 3] = {0xA5, 0xA5, 0x00};
  VisionPoseFrame valid{42, 10, 0, 0, 0, 0, 0, 0, 0.5f};
  assert(encodeVisionFrame(valid, stream + 3, sizeof(stream) - 3));
  VisionFrameParser resync;
  decoded = false;
  for (size_t i = 0; i < sizeof(stream); ++i) decoded |= resync.feed(stream[i], received);
  assert(decoded && received.sequence == 42);
}

int main() {
  testKalmanAndPid();
  testMixAndServo();
  testImuHealthAndModes();
  testProtocol();
  puts("Firmware native tests: all passed");
  return 0;
}
