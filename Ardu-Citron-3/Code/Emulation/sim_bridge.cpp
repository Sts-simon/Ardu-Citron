#include <cmath>
#include <cstdint>

#include "../ESP_32/main/flight_mode.h"
#include "../ESP_32/main/imu_health.h"
#include "../ESP_32/main/kalman.h"
#include "../ESP_32/main/mixer.h"
#include "../ESP_32/main/pid.h"
#include "../ESP_32/main/protocol.h"
#include "../ESP_32/main/servo_output.h"

namespace {
constexpr float Q_ANGLE = 0.001f, Q_BIAS = 0.003f, R_ACCEL = 0.03f;
constexpr float PID_LIMIT = 45.0f, PID_INT_LIMIT = 20.0f, DERIV_ALPHA = 0.2f;
constexpr uint32_t LINK_TIMEOUT_MS = 500, IMU_STALE_US = 100000;
constexpr uint8_t IMU_MAX_ERRORS = 5;
constexpr float MAX_ROLL_PITCH_DEG = 30.0f, MAX_YAW_RATE_DPS = 90.0f;
}

struct SimController {
  KalmanFilter rollFilter{Q_ANGLE, Q_BIAS, R_ACCEL};
  KalmanFilter pitchFilter{Q_ANGLE, Q_BIAS, R_ACCEL};
  KalmanFilter yawFilter{Q_ANGLE, Q_BIAS, R_ACCEL};
  PID rollPid{1.2f, 0.05f, 0.08f, PID_INT_LIMIT, PID_LIMIT, DERIV_ALPHA};
  PID pitchPid{1.4f, 0.06f, 0.10f, PID_INT_LIMIT, PID_LIMIT, DERIV_ALPHA};
  PID yawPid{0.8f, 0.02f, 0.05f, PID_INT_LIMIT, PID_LIMIT, DERIV_ALPHA};
  ServoOutputLimiter servo1{{10.0f, 50.0f, 1.0f, 0.0f, -1, 1000, 1500, 2000}};
  ServoOutputLimiter servo2{{10.0f, 50.0f, 1.0f, 0.0f, 1, 1000, 1500, 2000}};
  ServoOutputLimiter servo3{{10.0f, 50.0f, 1.0f, 0.0f, 1, 1000, 1500, 2000}};
  FlightModeManager modes{LINK_TIMEOUT_MS, LINK_TIMEOUT_MS, MAX_ROLL_PITCH_DEG, MAX_YAW_RATE_DPS};
  ImuHealthMonitor health{IMU_STALE_US, IMU_MAX_ERRORS};
  uint32_t lastGoodImuUs = 0;
  bool initialized = false;
  float estimatedRollDeg = 0.0f;
  float estimatedPitchDeg = 0.0f;
  float estimatedYawDeg = 0.0f;
  float estimatedYawRateDps = 0.0f;
  float actuatorRollDeg = 0.0f;
  float actuatorPitchDeg = 0.0f;
  float actuatorYawDeg = 0.0f;
  float pulse1Us = 1500.0f;
  float pulse2Us = 1500.0f;
  float pulse3Us = 1500.0f;
  FlightMode requestedMode = FlightMode::STABILIZE;
  bool autoArmed = false;
  bool autoAllowed = false;
  uint16_t lastAutoSequence = 0;
};

extern "C" {

SimController *sim_controller_create() { return new SimController(); }
void sim_controller_destroy(SimController *c) { delete c; }

void sim_controller_configure_auto(SimController *c, int enabled, float minimumQuality) {
  if (!c) return;
  c->autoAllowed = enabled != 0;
  c->modes.configureAuto(c->autoAllowed, minimumQuality);
}

int sim_controller_initialize(SimController *c, float rollDeg, float pitchDeg, uint32_t nowUs) {
  if (!c || !std::isfinite(rollDeg) || !std::isfinite(pitchDeg)) return 0;
  c->rollFilter.setAngle(rollDeg);
  c->pitchFilter.setAngle(pitchDeg);
  c->yawFilter.setAngle(0.0f);
  c->health.begin(nowUs);
  c->health.recordRead(true, nowUs);
  c->lastGoodImuUs = nowUs;
  c->initialized = c->modes.completeStartup(true);
  return c->initialized ? 1 : 0;
}

int sim_controller_step(SimController *c, float dt, uint32_t nowMs,
                        float rollRateDps, float pitchRateDps, float yawRateDps,
                        float accelRollDeg, float accelPitchDeg,
                        int imuUpdated, int imuValid, int rcUpdated, int rcValid,
                        int rcArm, int rcDisarm, int controlTick, int servoTick,
                        float commandRollDeg, float commandPitchDeg, float commandYawRateDps,
                        float *output, uint32_t nowUs, int modeSwitch,
                        const uint8_t *guidanceBytes, size_t guidanceLength,
                        int directManual, float directServo1, float directServo2, float directServo3) {
  if (!c || !output || !std::isfinite(dt) || dt <= 0.0f) return 0;
  if (!c->initialized) return 0;

  if (imuUpdated) {
    c->health.recordRead(imuValid != 0, nowUs);
    if (imuValid) {
      c->lastGoodImuUs = nowUs;
      c->estimatedRollDeg = c->rollFilter.update(rollRateDps, accelRollDeg, dt);
      c->estimatedPitchDeg = c->pitchFilter.update(pitchRateDps, accelPitchDeg, dt);
      c->estimatedYawRateDps = yawRateDps;
      c->estimatedYawDeg = c->yawFilter.predict(yawRateDps, dt);
    }
  }

  const FlightMode selected = modeSwitch == 2 && c->autoAllowed ? FlightMode::AUTO :
                              (modeSwitch == 0 ? FlightMode::MANUAL : FlightMode::STABILIZE);
  c->requestedMode = selected;
  c->modes.setPilotMode(selected);

  if (guidanceBytes && guidanceLength) {
    GuidanceFrameParser parser;
    GuidanceCommandFrame frame{};
    for (size_t i = 0; i < guidanceLength; ++i) {
      if (!parser.feed(guidanceBytes[i], frame)) continue;
      if (c->autoArmed && !isNewerSequence(frame.sequence, c->lastAutoSequence)) continue;
      c->lastAutoSequence = frame.sequence;
      if (c->autoAllowed && c->requestedMode == FlightMode::AUTO) {
        const PilotCommand command{frame.rollDeg, frame.pitchDeg, frame.yawRateDps};
        if (c->modes.submitAutoCommand(command, nowMs, frame.validityMs, frame.navigationQuality))
          c->autoArmed = true;
        else
          c->modes.notifyAutoCommandLost();
      }
    }
  }

  if (rcUpdated && rcValid) {
    c->modes.notifyRcFrame(true, nowMs);
    if (rcDisarm) {
      c->modes.disarm();
      c->autoArmed = false;
    } else {
      if (!c->modes.submitCommand(PilotCommand{commandRollDeg, commandPitchDeg, commandYawRateDps}, nowMs))
        c->modes.notifyCommandLost();
      if (rcArm) {
        c->modes.setPilotMode(c->requestedMode);
        c->modes.requestArm(nowMs);
      }
      if (c->requestedMode == FlightMode::AUTO && c->autoArmed) c->modes.requestAuto(nowMs);
    }
  } else if (rcUpdated) {
    c->modes.notifyRcFrame(false, nowMs);
  }

  const FlightMode mode = c->modes.update(nowMs, c->health.update(nowUs));
  const bool enabled = c->modes.controlEnabled();
  if (controlTick) {
    if (enabled) {
      const PilotCommand command = c->modes.command();
      c->actuatorRollDeg = c->rollPid.compute(command.rollDeg, c->estimatedRollDeg, dt);
      c->actuatorPitchDeg = c->pitchPid.compute(command.pitchDeg, c->estimatedPitchDeg, dt);
      c->actuatorYawDeg = c->yawPid.compute(command.yawRateDps, c->estimatedYawRateDps, dt);
    } else {
      c->rollPid.reset(); c->pitchPid.reset(); c->yawPid.reset();
      c->actuatorRollDeg = c->actuatorPitchDeg = c->actuatorYawDeg = 0.0f;
    }
  }
  if (servoTick) {
    if (mode == FlightMode::MANUAL) {
      c->pulse1Us = static_cast<float>(c->servo1.update(directManual ? directServo1 : 0.0f, dt));
      c->pulse2Us = static_cast<float>(c->servo2.update(directManual ? directServo2 : 0.0f, dt));
      c->pulse3Us = static_cast<float>(c->servo3.update(directManual ? directServo3 : 0.0f, dt));
    } else {
      const ServoCommandsDeg mixed = Mixer::mix(enabled ? c->actuatorRollDeg : 0.0f,
                                                  enabled ? c->actuatorPitchDeg : 0.0f,
                                                  enabled ? c->actuatorYawDeg : 0.0f);
      c->pulse1Us = static_cast<float>(c->servo1.update(mixed.servo1Deg, dt));
      c->pulse2Us = static_cast<float>(c->servo2.update(mixed.servo2Deg, dt));
      c->pulse3Us = static_cast<float>(c->servo3.update(mixed.servo3Deg, dt));
    }
  }

  output[0] = c->estimatedRollDeg;
  output[1] = c->estimatedPitchDeg;
  output[2] = c->estimatedYawDeg;
  output[3] = c->estimatedYawRateDps;
  output[4] = c->actuatorRollDeg;
  output[5] = c->actuatorPitchDeg;
  output[6] = c->actuatorYawDeg;
  output[7] = c->pulse1Us;
  output[8] = c->pulse2Us;
  output[9] = c->pulse3Us;
  output[10] = static_cast<float>(static_cast<uint8_t>(mode));
  output[11] = enabled || mode == FlightMode::MANUAL ? 1.0f : 0.0f;
  output[12] = c->health.healthy() ? 1.0f : 0.0f;
  return 1;
}
}
