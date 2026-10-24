#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

// Wire format is documented in Code/ESP_32/README.md. Floats are IEEE-754
// binary32, serialized explicitly little-endian; never transmit a C++ struct.
constexpr uint8_t VISION_FRAME_SYNC_0 = 0xA5;
constexpr uint8_t VISION_FRAME_SYNC_1 = 0x5A;
constexpr uint8_t VISION_PROTOCOL_VERSION = 1;
constexpr uint8_t VISION_MESSAGE_POSE = 1;
constexpr uint8_t GUIDANCE_MESSAGE_SETPOINT = 2;
constexpr uint8_t VISION_PAYLOAD_SIZE = 34;
constexpr size_t VISION_FRAME_SIZE = 41;
constexpr uint8_t GUIDANCE_PAYLOAD_SIZE = 24;
constexpr size_t GUIDANCE_FRAME_SIZE = 31;
constexpr uint16_t GUIDANCE_MAX_VALIDITY_MS = 500;

struct GuidanceCommandFrame {
  uint16_t sequence;
  uint32_t senderTimestampMs;
  float rollDeg;
  float pitchDeg;
  float yawRateDps;
  uint16_t validityMs;
  float navigationQuality;
};

struct VisionPoseFrame {
  uint16_t sequence;
  uint32_t senderTimestampMs; // Sender clock only; not comparable to ESP32 millis().
  float positionX_m;
  float positionY_m;
  float positionZ_m;
  float rotationRoll_deg;
  float rotationPitch_deg;
  float rotationYaw_deg;
  float detectionQuality;
};

uint16_t visionCrc16(const uint8_t *data, size_t length);
bool encodeVisionFrame(const VisionPoseFrame &frame, uint8_t *output, size_t capacity);
bool encodeGuidanceFrame(const GuidanceCommandFrame &frame, uint8_t *output, size_t capacity);
bool isNewerSequence(uint16_t candidate, uint16_t previous);

class GuidanceFrameParser {
public:
  bool feed(uint8_t byte, GuidanceCommandFrame &frame);
  uint32_t rejectedFrames() const { return _rejectedFrames; }
  void reset();

private:
  uint8_t _buffer[GUIDANCE_FRAME_SIZE] = {};
  size_t _used = 0;
  uint32_t _rejectedFrames = 0;
  void discardFirstByte();
  void seekSync();
  bool decode(GuidanceCommandFrame &frame) const;
};

class VisionFrameParser {
public:
  // Returns true only when one complete, semantically valid frame is decoded.
  bool feed(uint8_t byte, VisionPoseFrame &frame);
  uint32_t rejectedFrames() const { return _rejectedFrames; }
  void reset();

private:
  uint8_t _buffer[VISION_FRAME_SIZE] = {};
  size_t _used = 0;
  uint32_t _rejectedFrames = 0;

  void discardFirstByte();
  void seekSync();
  bool decode(VisionPoseFrame &frame) const;
};

#endif // PROTOCOL_H
