#include "protocol.h"

#include <math.h>
#include <limits>
#include <string.h>

namespace {
constexpr size_t CRC_START = 2;
constexpr size_t CRC_OFFSET = VISION_FRAME_SIZE - 2;
constexpr size_t HEADER_SIZE = 5;
constexpr size_t GUIDANCE_CRC_OFFSET = GUIDANCE_FRAME_SIZE - 2;

void putU16(uint8_t *out, uint16_t value) {
  out[0] = static_cast<uint8_t>(value & 0xFFu);
  out[1] = static_cast<uint8_t>(value >> 8);
}

void putU32(uint8_t *out, uint32_t value) {
  for (uint8_t i = 0; i < 4; ++i) out[i] = static_cast<uint8_t>(value >> (8u * i));
}

uint16_t getU16(const uint8_t *in) {
  return static_cast<uint16_t>(in[0]) | (static_cast<uint16_t>(in[1]) << 8);
}

uint32_t getU32(const uint8_t *in) {
  uint32_t value = 0;
  for (uint8_t i = 0; i < 4; ++i) value |= static_cast<uint32_t>(in[i]) << (8u * i);
  return value;
}

void putFloat(uint8_t *out, float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  putU32(out, bits);
}

float getFloat(const uint8_t *in) {
  const uint32_t bits = getU32(in);
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

bool validGuidance(const GuidanceCommandFrame &frame) {
  return isfinite(frame.rollDeg) && isfinite(frame.pitchDeg) && isfinite(frame.yawRateDps) &&
         isfinite(frame.navigationQuality) && fabsf(frame.rollDeg) <= 25.0f &&
         fabsf(frame.pitchDeg) <= 8.0f && fabsf(frame.yawRateDps) <= 25.0f &&
         frame.validityMs > 0 && frame.validityMs <= GUIDANCE_MAX_VALIDITY_MS &&
         frame.navigationQuality >= 0.0f && frame.navigationQuality <= 1.0f;
}

bool validFrame(const VisionPoseFrame &frame) {
  return isfinite(frame.positionX_m) && isfinite(frame.positionY_m) &&
         isfinite(frame.positionZ_m) && isfinite(frame.rotationRoll_deg) &&
         isfinite(frame.rotationPitch_deg) && isfinite(frame.rotationYaw_deg) &&
         isfinite(frame.detectionQuality) && frame.rotationRoll_deg >= -180.0f &&
         frame.rotationRoll_deg <= 180.0f && frame.rotationPitch_deg >= -180.0f &&
         frame.rotationPitch_deg <= 180.0f && frame.rotationYaw_deg >= -180.0f &&
         frame.rotationYaw_deg <= 180.0f && frame.detectionQuality >= 0.0f &&
         frame.detectionQuality <= 1.0f;
}
}

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "UART protocol requires IEEE-754 binary32 floats");

uint16_t visionCrc16(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000u) ? static_cast<uint16_t>((crc << 1) ^ 0x1021u)
                             : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

bool encodeVisionFrame(const VisionPoseFrame &frame, uint8_t *output, size_t capacity) {
  if (output == nullptr || capacity < VISION_FRAME_SIZE || !validFrame(frame)) return false;
  output[0] = VISION_FRAME_SYNC_0;
  output[1] = VISION_FRAME_SYNC_1;
  output[2] = VISION_PROTOCOL_VERSION;
  output[3] = VISION_MESSAGE_POSE;
  output[4] = VISION_PAYLOAD_SIZE;
  putU16(output + 5, frame.sequence);
  putU32(output + 7, frame.senderTimestampMs);
  putFloat(output + 11, frame.positionX_m);
  putFloat(output + 15, frame.positionY_m);
  putFloat(output + 19, frame.positionZ_m);
  putFloat(output + 23, frame.rotationRoll_deg);
  putFloat(output + 27, frame.rotationPitch_deg);
  putFloat(output + 31, frame.rotationYaw_deg);
  putFloat(output + 35, frame.detectionQuality);
  putU16(output + CRC_OFFSET, visionCrc16(output + CRC_START, CRC_OFFSET - CRC_START));
  return true;
}

bool encodeGuidanceFrame(const GuidanceCommandFrame &frame, uint8_t *output, size_t capacity) {
  if (output == nullptr || capacity < GUIDANCE_FRAME_SIZE || !validGuidance(frame)) return false;
  output[0] = VISION_FRAME_SYNC_0;
  output[1] = VISION_FRAME_SYNC_1;
  output[2] = VISION_PROTOCOL_VERSION;
  output[3] = GUIDANCE_MESSAGE_SETPOINT;
  output[4] = GUIDANCE_PAYLOAD_SIZE;
  putU16(output + 5, frame.sequence);
  putU32(output + 7, frame.senderTimestampMs);
  putFloat(output + 11, frame.rollDeg);
  putFloat(output + 15, frame.pitchDeg);
  putFloat(output + 19, frame.yawRateDps);
  putU16(output + 23, frame.validityMs);
  putFloat(output + 25, frame.navigationQuality);
  putU16(output + GUIDANCE_CRC_OFFSET,
         visionCrc16(output + CRC_START, GUIDANCE_CRC_OFFSET - CRC_START));
  return true;
}

bool isNewerSequence(uint16_t candidate, uint16_t previous) {
  const uint16_t difference = static_cast<uint16_t>(candidate - previous);
  return difference != 0 && difference < 0x8000u;
}

void VisionFrameParser::reset() {
  _used = 0;
}

void VisionFrameParser::discardFirstByte() {
  if (_used > 1) memmove(_buffer, _buffer + 1, _used - 1);
  if (_used > 0) --_used;
}

void VisionFrameParser::seekSync() {
  while (_used > 0 && _buffer[0] != VISION_FRAME_SYNC_0) discardFirstByte();
  while (_used >= 2 && _buffer[1] != VISION_FRAME_SYNC_1) discardFirstByte();
}

void GuidanceFrameParser::reset() { _used = 0; }

void GuidanceFrameParser::discardFirstByte() {
  if (_used > 1) memmove(_buffer, _buffer + 1, _used - 1);
  if (_used > 0) --_used;
}

void GuidanceFrameParser::seekSync() {
  while (_used > 0 && _buffer[0] != VISION_FRAME_SYNC_0) discardFirstByte();
  while (_used >= 2 && _buffer[1] != VISION_FRAME_SYNC_1) discardFirstByte();
}

bool GuidanceFrameParser::decode(GuidanceCommandFrame &frame) const {
  if (_buffer[2] != VISION_PROTOCOL_VERSION || _buffer[3] != GUIDANCE_MESSAGE_SETPOINT ||
      _buffer[4] != GUIDANCE_PAYLOAD_SIZE) return false;
  if (getU16(_buffer + GUIDANCE_CRC_OFFSET) !=
      visionCrc16(_buffer + CRC_START, GUIDANCE_CRC_OFFSET - CRC_START)) return false;
  frame.sequence = getU16(_buffer + 5);
  frame.senderTimestampMs = getU32(_buffer + 7);
  frame.rollDeg = getFloat(_buffer + 11);
  frame.pitchDeg = getFloat(_buffer + 15);
  frame.yawRateDps = getFloat(_buffer + 19);
  frame.validityMs = getU16(_buffer + 23);
  frame.navigationQuality = getFloat(_buffer + 25);
  return validGuidance(frame);
}

bool GuidanceFrameParser::feed(uint8_t byte, GuidanceCommandFrame &frame) {
  if (_used == GUIDANCE_FRAME_SIZE) discardFirstByte();
  _buffer[_used++] = byte;
  seekSync();
  if (_used >= HEADER_SIZE &&
      (_buffer[2] != VISION_PROTOCOL_VERSION || _buffer[3] != GUIDANCE_MESSAGE_SETPOINT ||
       _buffer[4] != GUIDANCE_PAYLOAD_SIZE)) {
    ++_rejectedFrames;
    discardFirstByte();
    seekSync();
  }
  if (_used == GUIDANCE_FRAME_SIZE) {
    if (decode(frame)) {
      _used = 0;
      return true;
    }
    ++_rejectedFrames;
    discardFirstByte();
    seekSync();
  }
  return false;
}

bool VisionFrameParser::decode(VisionPoseFrame &frame) const {
  if (_buffer[2] != VISION_PROTOCOL_VERSION || _buffer[3] != VISION_MESSAGE_POSE ||
      _buffer[4] != VISION_PAYLOAD_SIZE) return false;
  if (getU16(_buffer + CRC_OFFSET) != visionCrc16(_buffer + CRC_START, CRC_OFFSET - CRC_START)) return false;

  frame.sequence = getU16(_buffer + 5);
  frame.senderTimestampMs = getU32(_buffer + 7);
  frame.positionX_m = getFloat(_buffer + 11);
  frame.positionY_m = getFloat(_buffer + 15);
  frame.positionZ_m = getFloat(_buffer + 19);
  frame.rotationRoll_deg = getFloat(_buffer + 23);
  frame.rotationPitch_deg = getFloat(_buffer + 27);
  frame.rotationYaw_deg = getFloat(_buffer + 31);
  frame.detectionQuality = getFloat(_buffer + 35);
  return validFrame(frame);
}

bool VisionFrameParser::feed(uint8_t byte, VisionPoseFrame &frame) {
  if (_used == VISION_FRAME_SIZE) discardFirstByte();
  _buffer[_used++] = byte;
  seekSync();

  if (_used >= HEADER_SIZE &&
      (_buffer[2] != VISION_PROTOCOL_VERSION || _buffer[3] != VISION_MESSAGE_POSE ||
       _buffer[4] != VISION_PAYLOAD_SIZE)) {
    ++_rejectedFrames;
    discardFirstByte();
    seekSync();
  }

  if (_used == VISION_FRAME_SIZE) {
    if (decode(frame)) {
      _used = 0;
      return true;
    }
    ++_rejectedFrames;
    discardFirstByte();
    seekSync();
  }
  return false;
}
