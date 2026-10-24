#!/usr/bin/env python3
"""Version 1 Pi -> ESP32 UART pose frame codec; standard library only."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

SYNC = b"\xA5\x5A"
VERSION = 1
MESSAGE_POSE = 1
MESSAGE_GUIDANCE = 2
PAYLOAD_SIZE = 34
FRAME_SIZE = 41
GUIDANCE_PAYLOAD_SIZE = 24
GUIDANCE_FRAME_SIZE = 31
GUIDANCE_MAX_VALIDITY_MS = 500


@dataclass(frozen=True)
class GuidanceCommand:
    sequence: int
    sender_timestamp_ms: int
    roll_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    validity_ms: int
    navigation_quality: float


@dataclass(frozen=True)
class VisionPose:
    sequence: int
    sender_timestamp_ms: int
    x_m: float
    y_m: float
    z_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    quality: float


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _validate(pose: VisionPose) -> None:
    if not 0 <= pose.sequence <= 0xFFFF or not 0 <= pose.sender_timestamp_ms <= 0xFFFFFFFF:
        raise ValueError("sequence/timestamp outside wire range")
    values = (pose.x_m, pose.y_m, pose.z_m, pose.roll_deg, pose.pitch_deg, pose.yaw_deg, pose.quality)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose contains NaN/Inf")
    if any(not -180.0 <= value <= 180.0 for value in values[3:6]):
        raise ValueError("pose angle outside [-180,180]")
    if not 0.0 <= pose.quality <= 1.0:
        raise ValueError("quality outside [0,1]")


def encode_pose(pose: VisionPose) -> bytes:
    _validate(pose)
    payload = struct.pack(
        "<HI7f", pose.sequence, pose.sender_timestamp_ms,
        pose.x_m, pose.y_m, pose.z_m, pose.roll_deg, pose.pitch_deg, pose.yaw_deg, pose.quality,
    )
    header = SYNC + bytes((VERSION, MESSAGE_POSE, PAYLOAD_SIZE)) + payload
    return header + struct.pack("<H", crc16_ccitt_false(header[2:]))


def _validate_guidance(command: GuidanceCommand) -> None:
    if not 0 <= command.sequence <= 0xFFFF or not 0 <= command.sender_timestamp_ms <= 0xFFFFFFFF:
        raise ValueError("sequence/timestamp outside wire range")
    values = (command.roll_deg, command.pitch_deg, command.yaw_rate_dps, command.navigation_quality)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("guidance command contains NaN/Inf")
    if abs(command.roll_deg) > 25.0 or abs(command.pitch_deg) > 8.0 or abs(command.yaw_rate_dps) > 25.0:
        raise ValueError("guidance setpoint exceeds configured flight envelope")
    if not 0 < command.validity_ms <= GUIDANCE_MAX_VALIDITY_MS:
        raise ValueError("guidance validity outside range")
    if not 0.0 <= command.navigation_quality <= 1.0:
        raise ValueError("navigation quality outside [0,1]")


def encode_guidance(command: GuidanceCommand) -> bytes:
    _validate_guidance(command)
    payload = struct.pack("<HI3fHf", command.sequence, command.sender_timestamp_ms,
                          command.roll_deg, command.pitch_deg, command.yaw_rate_dps,
                          command.validity_ms, command.navigation_quality)
    header = SYNC + bytes((VERSION, MESSAGE_GUIDANCE, GUIDANCE_PAYLOAD_SIZE)) + payload
    return header + struct.pack("<H", crc16_ccitt_false(header[2:]))


def decode_guidance(frame: bytes) -> GuidanceCommand:
    if len(frame) != GUIDANCE_FRAME_SIZE or frame[:2] != SYNC:
        raise ValueError("invalid guidance frame size or sync")
    if frame[2:5] != bytes((VERSION, MESSAGE_GUIDANCE, GUIDANCE_PAYLOAD_SIZE)):
        raise ValueError("unsupported guidance version, type or payload length")
    expected = struct.unpack_from("<H", frame, GUIDANCE_FRAME_SIZE - 2)[0]
    if crc16_ccitt_false(frame[2:-2]) != expected:
        raise ValueError("guidance CRC mismatch")
    command = GuidanceCommand(*struct.unpack_from("<HI3fHf", frame, 5))
    _validate_guidance(command)
    return command


def decode_pose(frame: bytes) -> VisionPose:
    if len(frame) != FRAME_SIZE or frame[:2] != SYNC:
        raise ValueError("invalid frame size or sync")
    if frame[2:5] != bytes((VERSION, MESSAGE_POSE, PAYLOAD_SIZE)):
        raise ValueError("unsupported frame version, type or payload length")
    expected = struct.unpack_from("<H", frame, FRAME_SIZE - 2)[0]
    if crc16_ccitt_false(frame[2:-2]) != expected:
        raise ValueError("CRC mismatch")
    pose = VisionPose(*struct.unpack_from("<HI7f", frame, 5))
    _validate(pose)
    return pose


class GuidanceStreamDecoder:
    """Incremental parser for the fixed-length guidance setpoint frame."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.rejected = 0

    def feed(self, data: bytes) -> list[GuidanceCommand]:
        self._buffer.extend(data)
        output: list[GuidanceCommand] = []
        while True:
            start = self._buffer.find(SYNC)
            if start < 0:
                keep = 1 if self._buffer.endswith(SYNC[:1]) else 0
                self._buffer[:] = self._buffer[-keep:] if keep else b""
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 5:
                break
            if self._buffer[2:5] != bytes((VERSION, MESSAGE_GUIDANCE, GUIDANCE_PAYLOAD_SIZE)):
                self.rejected += 1
                del self._buffer[0]
                continue
            if len(self._buffer) < GUIDANCE_FRAME_SIZE:
                break
            try:
                output.append(decode_guidance(bytes(self._buffer[:GUIDANCE_FRAME_SIZE])))
                del self._buffer[:GUIDANCE_FRAME_SIZE]
            except ValueError:
                self.rejected += 1
                del self._buffer[0]
        if len(self._buffer) > GUIDANCE_FRAME_SIZE:
            del self._buffer[:-GUIDANCE_FRAME_SIZE]
        return output


class StreamDecoder:
    """Incremental bounded parser; rejects noise/corruption and recovers sync."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.rejected = 0

    def feed(self, data: bytes) -> list[VisionPose]:
        self._buffer.extend(data)
        output: list[VisionPose] = []
        while True:
            start = self._buffer.find(SYNC)
            if start < 0:
                keep = 1 if self._buffer.endswith(SYNC[:1]) else 0
                self._buffer[:] = self._buffer[-keep:] if keep else b""
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 5:
                break
            if self._buffer[2:5] != bytes((VERSION, MESSAGE_POSE, PAYLOAD_SIZE)):
                self.rejected += 1
                del self._buffer[0]
                continue
            if len(self._buffer) < FRAME_SIZE:
                break
            candidate = bytes(self._buffer[:FRAME_SIZE])
            try:
                output.append(decode_pose(candidate))
                del self._buffer[:FRAME_SIZE]
            except ValueError:
                self.rejected += 1
                del self._buffer[0]
        if len(self._buffer) > FRAME_SIZE:
            del self._buffer[:-FRAME_SIZE]
        return output
