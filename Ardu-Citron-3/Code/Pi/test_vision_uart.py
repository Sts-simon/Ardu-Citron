#!/usr/bin/env python3
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vision_uart import (GuidanceCommand, GuidanceStreamDecoder, StreamDecoder, VisionPose,
                         crc16_ccitt_false, decode_guidance, decode_pose, encode_guidance, encode_pose)


class VisionUartTests(unittest.TestCase):
    def setUp(self):
        self.pose = VisionPose(65535, 0xFFFFFFFE, 1.25, -2.5, 0.75, 10.0, -20.0, 179.0, 0.9)

    def test_encode_decode_and_incremental_delivery(self):
        encoded = encode_pose(self.pose)
        self.assertEqual(len(encoded), 41)
        decoder = StreamDecoder()
        self.assertEqual(decoder.feed(b"noise\xA5"), [])
        self.assertEqual(decoder.feed(encoded[1:13]), [])
        decoded = decoder.feed(encoded[13:])
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].sequence, self.pose.sequence)
        self.assertEqual(decoded[0].sender_timestamp_ms, self.pose.sender_timestamp_ms)
        self.assertAlmostEqual(decoded[0].quality, self.pose.quality, places=6)
        root = pathlib.Path(__file__).resolve().parents[1] / "ESP_32"
        with tempfile.TemporaryDirectory(prefix="ardu-citron-golden-") as temp:
            executable = pathlib.Path(temp) / "protocol-fixture"
            subprocess.run([
                "g++", "-std=c++17", "-I", str(root / "test/native"),
                "-I", str(root / "main"), str(root / "test/native/protocol_fixture.cpp"),
                str(root / "main/protocol.cpp"), "-o", str(executable),
            ], check=True)
            firmware_hex = subprocess.check_output([str(executable)], text=True).strip()
            self.assertEqual(firmware_hex, encoded.hex())

    def test_corruption_is_rejected_then_parser_recovers(self):
        damaged = bytearray(encode_pose(self.pose))
        damaged[20] ^= 0x80
        valid = encode_pose(VisionPose(0, 5, 0, 0, 0, 0, 0, 0, 0.5))
        decoder = StreamDecoder()
        decoded = decoder.feed(bytes(damaged) + valid)
        self.assertEqual([pose.sequence for pose in decoded], [0])
        self.assertGreater(decoder.rejected, 0)

    def test_guidance_command_cross_language_roundtrip_and_bounds(self):
        command = GuidanceCommand(9, 1234, 12.0, -3.0, 8.0, 150, 0.95)
        encoded = encode_guidance(command)
        self.assertEqual(len(encoded), 31)
        decoded_command = decode_guidance(encoded)
        self.assertEqual(decoded_command.sequence, command.sequence)
        self.assertEqual(decoded_command.validity_ms, command.validity_ms)
        for actual, expected in ((decoded_command.roll_deg, command.roll_deg),
                                 (decoded_command.pitch_deg, command.pitch_deg),
                                 (decoded_command.yaw_rate_dps, command.yaw_rate_dps),
                                 (decoded_command.navigation_quality, command.navigation_quality)):
            self.assertAlmostEqual(actual, expected, places=6)
        parser = GuidanceStreamDecoder()
        self.assertEqual(parser.feed(b"noise" + encoded[:9]), [])
        streamed = parser.feed(encoded[9:])
        self.assertEqual(len(streamed), 1)
        self.assertEqual(streamed[0].sequence, command.sequence)
        self.assertAlmostEqual(streamed[0].navigation_quality, command.navigation_quality, places=6)
        with self.assertRaises(ValueError):
            encode_guidance(GuidanceCommand(0, 0, 26.0, 0.0, 0.0, 100, 0.9))
        with self.assertRaises(ValueError):
            decode_guidance(encoded[:-1] + bytes([encoded[-1] ^ 1]))
        root = pathlib.Path(__file__).resolve().parents[1] / "ESP_32"
        with tempfile.TemporaryDirectory(prefix="ardu-citron-guidance-") as temp:
            executable = pathlib.Path(temp) / "guidance-fixture"
            subprocess.run([
                "g++", "-std=c++17", "-I", str(root / "test/native"),
                "-I", str(root / "main"), str(root / "test/native/guidance_fixture.cpp"),
                str(root / "main/protocol.cpp"), "-o", str(executable),
            ], check=True)
            firmware_hex = subprocess.check_output([str(executable)], text=True).strip()
            self.assertEqual(firmware_hex, encoded.hex())

    def test_semantic_invalidity_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_pose(VisionPose(0, 0, 0, 0, 0, 0, 0, 0, math.nan))
        invalid = bytearray(encode_pose(self.pose))
        invalid[35:39] = b"\x00\x00\xC0\x7F"  # quality = NaN
        invalid[-2:] = struct.pack("<H", crc16_ccitt_false(invalid[2:-2]))
        with self.assertRaises(ValueError):
            decode_pose(bytes(invalid))


if __name__ == "__main__":
    unittest.main()
