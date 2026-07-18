#include "mixer.h"

ServoCommandsDeg Mixer::mix(float rollCmdDeg, float pitchCmdDeg, float yawCmdDeg) {
  ServoCommandsDeg out{0.0f, 0.0f, 0.0f};

  switch (ACTIVE_MIXING_PRESET) {

    case MixingPreset::STANDARD_AIL_ELEV_RUD:
      // Servo1 = Aileron (roll), Servo2 = Profondeur (pitch), Servo3 = Derive (yaw)
      out.servo1Deg = rollCmdDeg;
      out.servo2Deg = pitchCmdDeg;
      out.servo3Deg = yawCmdDeg;
      break;

    case MixingPreset::VTAIL:
      // Queue en V : les deux servos de queue melangent pitch et yaw.
      // Servo3 reste disponible pour un aileron independant si present.
      out.servo1Deg = pitchCmdDeg + yawCmdDeg;
      out.servo2Deg = pitchCmdDeg - yawCmdDeg;
      out.servo3Deg = rollCmdDeg;
      break;

    case MixingPreset::ELEVON:
      // Elevons : melange roll+pitch sur les deux servos d'aile.
      // Servo3 reste disponible pour une derive si presente.
      out.servo1Deg = pitchCmdDeg + rollCmdDeg;
      out.servo2Deg = pitchCmdDeg - rollCmdDeg;
      out.servo3Deg = yawCmdDeg;
      break;
  }

  return out;
}
